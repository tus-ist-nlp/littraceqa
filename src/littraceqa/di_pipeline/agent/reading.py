"""The iterative agent: retrieve, read, re-search for whatever is missing.

Two earlier baselines each had only one half of this, and both were removed:

* Iterate without checking the content. Stopping on "number of papers returned >=
  threshold" is satisfied immediately at retrieve_top_k=20, so the loop never took
  a second round.
* Let an LLM read the candidates and pick, but only once. It never searches again
  when what it has is not enough.

ReadingAgent joins the two. Iteration only becomes meaningful once the stopping
condition is "do the papers the LLM could actually confirm answer the question".

    1. split the question into subqueries
    2. retrieve with each subquery
    3. show the top candidates' chunks to an LLM **in full** (1800 chars by
       default) and have it pick the papers that truly carry evidence, plus which
       chunks that evidence is in
    4. if the LLM says it is still short, ask what is missing, re-split, go to 2
    5. stop when it is sufficient, or at max_steps

Evidence objects are built from the chosen chunks (see agent/evidence.py).
**Answer (freeform / multiple_choice / table) is never filled in**: generating
answers and choosing which papers to submit both belong to the reading team, so
this agent hands over the candidate list and the evidence and stops there. The
`paper_ids` the reading LLM returns are used only as the stopping condition, for
evidence, and as the anchors of ranking B — never to choose the submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.contracts import Answer, Evidence, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever, to_gold_papers
from littraceqa.di_pipeline.retrieve.paper_expander import PaperExpander


# How many candidate papers to keep in the prediction file. The headline metric is
# recall@20, but re-running is expensive, so keep enough to compute recall@50 later.
CANDIDATE_PAPERS_LIMIT = 50

# How many subqueries `_decompose()` asks for. **Not split by task_family.**
#
# Asking for 1-3 on single-paper questions and 3-6 on multi-paper ones bought an
# average of 0.58 extra subqueries, and cost an LLM call per query just to guess
# task_family (production input does not carry it). The two estimators were also
# indistinguishable in accuracy (0.670 vs 0.673 over 55 queries), so there was no
# basis for branching. Fixing the count at 4 sits between the measured averages and
# removes one LLM call.
#
# **The count is enforced on the return value, not just requested in the prompt.**
# `_decompose()` mostly complies (3.3 on average) but does overshoot (up to 6), and
# `_refine()`, which used to state no count at all, ballooned to 8.2-9.3 on average
# and 20 at worst. One subquery = one retrieval = one reranker pass over pool_k
# chunks, so this directly sets the wall-clock cost of a run.
SUBQUERY_COUNT = 4

# Prepended to every prompt that asks for subqueries.
#
# **Without it the LLM writes Google queries.** 29-41% of the subqueries `_refine()`
# produced carried web operators (`site:arxiv.org`, `filetype:pdf`, ...), affecting
# 30-39 of 55 queries. They go to a local BM25 and faiss index, where such operators
# match nothing, so rounds 2 and 3 of the loop were retrieving pure noise.
# (`_decompose()` showed 0%, but the same misunderstanding is possible, so both
# prompts carry the note.)
CORPUS_NOTE = (
    "The subqueries are sent to a local search index built over the full text of the "
    "papers (BM25 + dense embeddings). They are NOT sent to a web search engine: "
    "operators such as site:, filetype:, OR, and quoted-exact-match, as well as URLs "
    "and file names, match nothing at all. Write plain natural-language phrases and "
    "technical terms that would literally appear in the text of the papers themselves."
)


@dataclass
class SubqueryRun:
    """One subquery's results, **in the order retrieval returned them**.

    The loop's `chunks: dict[chunk_id -> RetrievalResult]` is a max merge (same
    chunk, keep the higher score), which destroys which subquery ranked it where.
    **This keeps the provenance of the ranking** for uses that need it later, such
    as the offline replay basis `scripts/run_search.py --dump-runs` writes.
    `chunks` remains the lookup by chunk_id (hallucination checks and evidence).
    """

    step: int
    subquery: str
    results: list[RetrievalResult] = field(default_factory=list)


@dataclass(frozen=True)
class CombineConfig:
    """How ranking A (question->paper) and B (paper->paper) are fused with RRF.

    **Only ReadingAgent uses these** (`_combine_rrf`). An expander is responsible
    for nothing but returning neighbours of the anchors it is handed, so it does
    not hold any of these values.
    """

    # Whether depth of rank or presence in both lists weighs more. **Smaller k
    # favours the top of each list.**
    #
    # At k=60 a paper at rank r in both lists scores 2/(61+r) while A's top hit
    # scores 1/61, and 2/(61+r) > 1/61 holds for r < 61 — with lists of 50, **being
    # in both wins no matter how deep**. Papers the reranker had put first were
    # being overtaken by a paper sitting 40th in A and 40th in B. k=10 moves the
    # threshold to r < 11, restoring the intended meaning: being in both wins only
    # near the top of both.
    #
    # Measured over 55 queries with full-length A: ecr@5 0.781 -> 0.806,
    # ecr@20 0.932 -> 0.948, nothing regressed. **Pair it with neighbors=100** —
    # neither change does anything on its own.
    rrf_k: int = 60
    # Weight and offset for the B side. Plain RRF (1.0 / 0) was best on all four
    # bases: lowering the weight drops B-only papers below A's tail and the fusion
    # stops meaning anything, raising it lets B take over the candidate list, and an
    # offset just walks back toward positional insertion.
    related_weight: float = 1.0
    related_offset: int = 0
    # How many top candidates seed ranking B. **Do not raise this** — 3 was worse on
    # all four bases. Putting the 2nd and 3rd candidates at the head of B earns them
    # the same two-term bonus and pushes down papers that belong above them.
    anchors: int = 1
    # "verdict" adds **the papers the reading LLM confirmed** to the anchors (see
    # `_anchor_papers`). None uses only the top `anchors` candidates.
    anchor_from: str | None = None


@dataclass(frozen=True)
class ReadingConfig:
    """The numbers that shape ReadingAgent's behaviour.

    **Kept apart from the collaborators (retriever / llm / paper_expander).** Those
    say *what* to use; these say *how* to use it and get swept during experiments.
    ReadingAgent takes the two as separate arguments.

    **This dataclass is the single definition of which params exist.** There used
    to be a `from_params()` that took a dict of yaml keys and rejected unknown ones
    by name; with the yaml gone, every call site names the fields directly and a
    misspelling is a TypeError at the constructor.
    """

    # ---- the iteration loop ----
    max_steps: int = 3
    # Cap on subqueries per step. **Used both to ask in the prompt and to truncate
    # the response** (the LLM does not always respect the requested count).
    # Raising it does not improve retrieval; see the note on SUBQUERY_COUNT.
    subquery_count: int = SUBQUERY_COUNT
    # How many **chunks** (not papers) one subquery's retrieval returns. Distinct
    # from the retriever's per_index_k / pool_k: this is how many of the reranker's
    # already-scored pool_k results are handed back. Raising it costs no extra
    # reranker inference — it only picks up results that were being discarded.
    retrieve_top_k: int = 20

    # ---- what the reading LLM gets to see ----
    max_candidates: int = 15
    chunks_per_paper: int = 2
    snippet_chars: int = 1800

    # ---- submission ----
    # **Choosing which papers to submit belongs to the reading team**, so this agent
    # passes the candidate ranking through and only cuts it to `max_papers`.
    max_papers: int = 10

    # ---- a paper's representative score ----
    # **Chunk types excluded from the representative score** (`["table"]` measured
    # best). Table chunks are dense with numbers and short labels, so one table can
    # spike a paper that is not the question's topic. See `to_gold_papers` in
    # retrieve/hybrid.py. **The chunk pool handed to the reader is unchanged**, so
    # tables still reach evidence as before.
    paper_score_skip_chunk_types: tuple[str, ...] = ()


class ReadingAgent:
    """Read the candidates, settle the evidence, search again if it is not enough.

    `__init__` names only the **collaborators** (retriever / llm / paper_expander);
    every number lives in `ReadingConfig` and `CombineConfig`, passed as one object
    each. Omitting either uses its defaults.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        *,
        # Paper-to-paper expansion (retrieve/paper_expander.py), there to recover
        # peer gold papers the question never names. None leaves the candidate list
        # exactly as retrieval ranked it.
        paper_expander: PaperExpander | None = None,
        # A/B fusion settings; unused when no expander is given.
        combine: CombineConfig | None = None,
        config: ReadingConfig | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.paper_expander = paper_expander
        self.combine = combine or CombineConfig()
        self.config = config or ReadingConfig()
        # Read by scripts/run_search.py --dump-runs. Deliberately not in
        # Prediction.trace, which would bloat the submission file.
        self.last_runs: list[SubqueryRun] = []

    def run(self, query: Query) -> Prediction:
        # The venue/year constraint is taken **once, from the original question**.
        # Subqueries from _decompose() often drop terms like "NAACL 2025", so
        # extracting from them would simply never fire. Reused across all steps.
        attribute_filter = self._extract_attribute_filter(query)
        # Only pass the argument when a constraint was found; retrieve() takes
        # (query, top_k), so always passing it would break simpler retrievers.
        retrieve_kwargs = (
            {} if attribute_filter is None else {"attribute_filter": attribute_filter}
        )

        subqueries = self._decompose(query, attribute_filter)
        tried: list[str] = []
        chunks: dict[str, RetrievalResult] = {}
        runs: list[SubqueryRun] = []
        verdict: dict | None = None
        trace: list[dict] = []

        for step in range(self.config.max_steps):
            tried.extend(subqueries)
            for subquery in subqueries:
                results = self._retrieve(subquery, retrieve_kwargs)
                runs.append(SubqueryRun(step=step, subquery=subquery, results=results))
                for result in results:
                    # When several subqueries hit the same chunk, keep the higher
                    # score. Last-write-wins would let subquery 3's low score
                    # overwrite subquery 1's top hit, dragging the paper ranking in
                    # _candidate_papers toward whichever subquery ran last.
                    previous = chunks.get(result.chunk_id)
                    if previous is None or result.score > previous.score:
                        chunks[result.chunk_id] = result

            # `chunks` was built as a max merge (same chunk_id, keep the higher
            # score), so the accumulator itself is the merged ranking.
            candidates = self._candidate_papers(list(chunks.values()))
            new_verdict = self._read_and_judge(query, candidates, chunks)
            if new_verdict is not None:
                verdict = new_verdict

            trace.append(
                {
                    "step": step,
                    "subqueries": subqueries,
                    # Recorded so it is possible to tell afterwards whether the filter fired.
                    "attribute_filter": (
                        None
                        if attribute_filter is None
                        else {"venue": attribute_filter.venue, "year": attribute_filter.year}
                    ),
                    "n_chunks": len(chunks),
                    "n_candidates": len(candidates),
                    "selected": [] if verdict is None else verdict["paper_ids"],
                    "sufficient": None if verdict is None else verdict["sufficient"],
                    "missing": None if verdict is None else verdict["missing"],
                }
            )

            # This is what separates the loop from "stop at N results": it stops on
            # whether the papers the LLM could confirm are enough, not on a count.
            if verdict is not None and verdict["sufficient"]:
                break
            if step == self.config.max_steps - 1:
                break

            missing = "" if verdict is None else verdict["missing"]
            subqueries = self._refine(query, missing, tried, attribute_filter)
            if not subqueries:
                break

        self.last_runs = runs
        return self._build_prediction(query, verdict, chunks, trace)

    # ---- one retrieval ------------------------------------------------------

    def _retrieve(self, subquery: str, retrieve_kwargs: dict) -> list[RetrievalResult]:
        """Retrieve with one subquery and return the top `retrieve_top_k` results."""
        return list(
            self.retriever.retrieve(subquery, self.config.retrieve_top_k, **retrieve_kwargs)
        )

    # ---- 0. extract the attribute constraint --------------------------------

    def _extract_attribute_filter(self, query: Query):
        """Take the venue/year constraint the question states. None if disabled.

        The extractor belongs to the retriever (`pipeline.build_retriever()` hands
        it one). Without it this returns None and retrieve() takes its normal path.
        """
        extractor = getattr(self.retriever, "attribute_extractor", None)
        if extractor is None:
            return None
        attribute_filter = extractor.extract(query.question)
        # Empty becomes None so retrieve() is not even given the argument, keeping
        # unconstrained questions on exactly the original code path.
        return None if attribute_filter.is_empty() else attribute_filter

    def _constraint_note(self, attribute_filter) -> str:
        """Wording that tells the subquery prompt about the constraint; empty if none.

        The actual narrowing is done by attribute_filter on the results, so this is
        the constraint **as a search term**. A title_abstract chunk's text really
        does begin `[ACL 2025] Title...` (preprocess/mineru_chunker.py), so leading
        with the same tag pulls BM25 toward that venue. `_decompose()` sometimes
        keeps the venue on its own but `_refine()` dropped it, so both say it now.
        """
        if attribute_filter is None:
            return ""
        tag = " ".join(
            str(part)
            for part in (attribute_filter.venue, attribute_filter.year)
            if part is not None
        )
        if not tag:
            return ""
        return (
            f"The question limits the search to {tag}. Begin every subquery with the "
            f'tag "[{tag}]" and keep the rest of the subquery about the content: the '
            "title/abstract text of each paper in the index literally starts with that tag."
        )

    # ---- 1. split -----------------------------------------------------------

    def _decompose(self, query: Query, attribute_filter=None) -> list[str]:
        """Split the question into search subqueries.

        Splitting only multi-paper questions is possible, but even single-paper ones
        want separate terms for "which paper" and "which table inside it", so this
        always splits.

        **The count is fixed at `subquery_count`, never branched on task_family.**
        The LLM overshoots the request (up to 6 observed), so the result is cut too.
        """
        prompt = "\n".join(
            part
            for part in (
                "You are helping to decompose a research question into search "
                "subqueries against a scientific paper corpus.",
                f"Question: {query.question}",
                # One wording covers both single-paper and multi-paper questions.
                # Committing to either loses one of the two useful behaviours when
                # the guess is wrong: paraphrases that land the paper in focus, or a
                # split that goes after each paper separately.
                "The evidence may live in a single paper or be spread across several. "
                "Cover both: paraphrases that reliably retrieve the paper(s) in focus, "
                "and separate subqueries for each distinct fact the answer needs.",
                self._constraint_note(attribute_filter),
                CORPUS_NOTE,
                f"Decompose it into {self.config.subquery_count} short, self-contained search "
                "subqueries.",
                'Respond with JSON only, in the form {"subqueries": ["...", "..."]}.',
            )
            if part
        )
        subqueries = self._ask_for_list(prompt, "subqueries")[: self.config.subquery_count]
        return subqueries or [query.question]

    # ---- 2. assemble the candidates -----------------------------------------

    def _candidate_papers(
        self, results: list[RetrievalResult]
    ) -> list[tuple[str, list[RetrievalResult]]]:
        """Group the merged ranking by paper and return the top-scoring papers."""
        by_paper: dict[str, list[RetrievalResult]] = {}
        for result in results:
            by_paper.setdefault(result.paper_id, []).append(result)

        for results in by_paper.values():
            results.sort(key=lambda r: r.score, reverse=True)

        ranked = sorted(
            by_paper.items(), key=lambda item: item[1][0].score, reverse=True
        )
        return [
            (paper_id, results[: self.config.chunks_per_paper])
            for paper_id, results in ranked[: self.config.max_candidates]
        ]

    # ---- 3. read and judge --------------------------------------------------

    def _read_and_judge(
        self,
        query: Query,
        candidates: list[tuple[str, list[RetrievalResult]]],
        chunks: dict[str, RetrievalResult],
    ) -> dict | None:
        """Have the LLM read the candidate chunks and pick the evidence."""
        if not candidates:
            return None

        listing = "\n\n".join(
            self._format_paper(paper_id, results)
            for paper_id, results in candidates
        )
        prompt = (
            "You are reading excerpts from papers returned by a search and selecting "
            "only the papers that are truly needed as evidence to answer the "
            "question.\n\n"
            f"Question: {query.question}\n"
            f"{self._format_answer_spec(query)}\n\n"
            "Candidates (most relevant first; each chunk is an excerpt of a paper's "
            "body, table, or figure caption):\n"
            f"{listing}\n\n"
            "After reading the excerpts, determine the following.\n"
            "1. Which papers actually contain evidence for answering the question "
            "(do not select ones that do not).\n"
            "2. For each paper, which chunk_ids are the evidence.\n"
            "3. Whether this fully answers the question. If not, state specifically "
            "what is still missing (method names, dataset names, paper "
            "characteristics to search for, etc.).\n\n"
            "Do not invent any paper_id / chunk_id that is not in the candidate list.\n"
            "Respond with JSON only, in the following form:\n"
            '{"papers": [{"paper_id": "...", "evidence_chunk_ids": ["..."]}], '
            '"sufficient": true, "missing": ""}'
        )

        parsed = self._ask_for_json(prompt)
        if parsed is None:
            return None

        papers = parsed.get("papers")
        if not isinstance(papers, list):
            return None

        candidate_ids = {paper_id for paper_id, _ in candidates}
        paper_ids: list[str] = []
        evidence_chunk_ids: list[str] = []
        for item in papers:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id", ""))
            if paper_id not in candidate_ids or paper_id in paper_ids:
                continue
            paper_ids.append(paper_id)
            for chunk_id in item.get("evidence_chunk_ids") or []:
                chunk_id = str(chunk_id)
                # Reject hallucinations: the chunk must exist and belong to that paper.
                result = chunks.get(chunk_id)
                if result is not None and result.paper_id == paper_id:
                    evidence_chunk_ids.append(chunk_id)

        if not paper_ids:
            return None

        # No truncation here; submission is the top `max_papers` of the candidate list.
        return {
            "paper_ids": paper_ids,
            "evidence_chunk_ids": evidence_chunk_ids,
            "sufficient": bool(parsed.get("sufficient")),
            "missing": str(parsed.get("missing") or ""),
        }

    def _format_paper(self, paper_id: str, results: list[RetrievalResult]) -> str:
        head = results[0]
        title = (head.metadata or {}).get("title", "")
        venue = (head.metadata or {}).get("venue", "")
        year = (head.metadata or {}).get("year", "")
        lines = [f"[paper_id: {paper_id}] {title} ({venue} {year})"]
        for result in results:
            metadata = result.metadata or {}
            where = [f"type={result.chunk_type}"]
            for key in ("page", "section", "table_id", "figure_id", "equation_id"):
                if metadata.get(key) is not None:
                    where.append(f"{key}={metadata[key]}")
            lines.append(f"  - chunk_id: {result.chunk_id} ({', '.join(where)})")
            lines.append(f"    {result.text[: self.config.snippet_chars]}")
        return "\n".join(lines)

    def _format_answer_spec(self, query: Query) -> str:
        parts = [f"Answer format: {', '.join(query.answer_types) or '(unspecified)'}"]
        if query.table_schema:
            columns = ", ".join(str(c.get("name")) for c in query.table_schema)
            parts.append(f"Answer table columns: {columns}")
        return "\n".join(parts)

    # ---- 4. re-split from what is missing ------------------------------------

    def _refine(
        self,
        query: Query,
        missing: str,
        tried: list[str],
        attribute_filter=None,
    ) -> list[str]:
        """Build the next subqueries from what the reader said was missing."""
        tried_text = "\n".join(f"- {sq}" for sq in dict.fromkeys(tried))
        missing_text = missing or "(No specific note from the LLM. Search from a different angle.)"
        prompt = "\n\n".join(
            part
            for part in (
                f"Original question: {query.question}",
                "The search so far still lacks sufficient evidence. What is missing:\n"
                f"{missing_text}",
                "Search subqueries already tried (do not repeat the same or similar ones):\n"
                f"{tried_text}",
                # Omit this and the LLM starts adding site: / filetype: as its idea
                # of "a narrower search" (29-41% observed). See CORPUS_NOTE.
                CORPUS_NOTE,
                self._constraint_note(attribute_filter),
                # **State a count or the LLM emits as many as it likes** (8.2-9.3 on
                # average, 20 at worst). The extras degenerate into a cross product
                # of method name x phrasing; q_021's step 1 listed 20 variants of
                # "SimLingo ... Bench2Drive Base". Each subquery costs a reranker
                # pass over pool_k chunks, so this multiplies the run time directly.
                f"Propose at most {self.config.subquery_count} new search subqueries to fill "
                "this gap. Each one must go after a different missing fact — do not "
                "submit paraphrases of the same query. "
                "If further searching is unlikely to find anything, return an empty list.\n"
                'Respond with JSON only, in the form {"subqueries": ["...", "..."]}.',
            )
            if part
        )
        # The prompt's cap is not always respected, so truncate here as well.
        return self._ask_for_list(prompt, "subqueries")[: self.config.subquery_count]

    # ---- 5. assemble the prediction ------------------------------------------

    def _build_prediction(
        self,
        query: Query,
        verdict: dict | None,
        chunks: dict[str, RetrievalResult],
        trace: list[dict],
    ) -> Prediction:
        # The chunks accumulated during the loop, collapsed to a paper ranking.
        # This is "what retrieval found", before any truncation, so recall@k analysis
        # must read this rather than gold_papers (which comes after the cut and
        # therefore mixes retrieval quality with selection).
        merged = list(chunks.values())
        if self.paper_expander is not None:
            # Expansion enters only through the A/B RRF fusion (positional
            # insertion lost on every metric and was deleted; see CLAUDE.md).
            #
            # Fusion uses the **full-length** A ranking, before the cut to 50: a
            # paper sitting at rank 51 cannot be lifted by B if it was already
            # discarded. The cut happens after fusion.
            candidate_papers = to_gold_papers(
                merged, skip_chunk_types=self.config.paper_score_skip_chunk_types
            )
            if candidate_papers:
                # Only used with `anchor_from: verdict`. When verdict is None this
                # is empty and `_anchor_papers()` falls back to the top candidates.
                candidate_papers = self._combine_rrf(
                    candidate_papers,
                    trace,
                    [] if verdict is None else verdict["paper_ids"],
                )
            candidate_papers = candidate_papers[:CANDIDATE_PAPERS_LIMIT]
        else:
            candidate_papers = to_gold_papers(
                merged,
                max_papers=CANDIDATE_PAPERS_LIMIT,
                skip_chunk_types=self.config.paper_score_skip_chunk_types,
            )

        # **No selection happens here.** The retrieval ranking is passed through and
        # choosing is left to the reading team (the reader's `paper_ids` are used
        # only as the stopping condition and for evidence).
        evidence: list[Evidence] = []
        paper_ids = candidate_papers[: self.config.max_papers]

        evidence_results: list[RetrievalResult] = []
        if verdict is not None:
            # Do not emit evidence for papers the cut removed.
            kept = set(paper_ids)
            evidence_results = [
                chunks[chunk_id]
                for chunk_id in dict.fromkeys(verdict["evidence_chunk_ids"])
                if chunks[chunk_id].paper_id in kept
            ]
            evidence = [evidence_from_result(r) for r in evidence_results]

        # **No answer is generated** (freeform / multiple_choice / table belong to
        # the reading team). An empty Answer goes out as is, saving one LLM call.
        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=evidence,
            answer=Answer(),
            trace=trace,
            candidate_papers=candidate_papers,
        )

    # ---- 6. fuse rankings A and B (paper-to-paper expansion) -----------------

    def _anchor_papers(
        self, candidate_papers: list[str], verdict_papers: list[str] | None
    ) -> list[str]:
        """The anchors of ranking B.

        By default the **top `anchors` candidates**. With `anchor_from: "verdict"`,
        the papers the reading LLM confirmed by reading their text (the `paper_ids`
        from `_read_and_judge()`) are added to them.

        **The top candidate always stays first.** Using only the LLM's confirmations
        pushed it out of B's head, where papers present in both rankings overtook it
        and **single-paper cr@1 fell 0.923 -> 0.885**. Taking the union leaves single
        completely unchanged and keeps the gain.

        **What helps is the count, not the precision.** The LLM's confirmations are
        gold 76% of the time (52/68), *below* the top candidate's 85% (47/55). They
        help anyway because one anchor can only expand one topic cluster. 16 of 55
        queries end up with more than one anchor, and **14 of those are multi-paper**,
        which is why single-paper queries see no side effect.

        Measured across four bases (multi-paper recall, single-paper cr@1 unchanged
        in all of them): @5 moves by -0.006 to +0.076, @10 by +0.030 to +0.082,
        @20 by 0.000 to +0.023. **The gain concentrates at @10** because that is
        where the misses are: ordering a multi-paper query's evidence-backed gold by
        rank, the first sits at median rank 1 while the second, third and fourth sit
        at 4, 8 and 14.
        """
        anchors = candidate_papers[: self.combine.anchors]
        if self.combine.anchor_from != "verdict" or not verdict_papers:
            return anchors
        return list(dict.fromkeys(anchors + [p for p in verdict_papers if p]))

    def _combine_rrf(
        self,
        candidate_papers: list[str],
        trace: list[dict],
        verdict_papers: list[str] | None = None,
    ) -> list[str]:
        """Fuse two rankings with RRF.

        * **A (question->paper)**: retrieval itself. BM25 + embeddings -> RRF -> reranker.
        * **B (paper->paper)**: proximity between papers (SPECTER2 / bibliographic
          coupling / full-text MLT, fused with RRF). **Never passed through the
          reranker** — it judges "does this answer the question" and would always
          demote the peer gold papers the question does not name. Keeping B away
          from it is the whole point.

            score(p) = 1 / (k + rank_A) + w_B / (k + offset + rank_B)

        Two things separate this from the positional insertion it replaced:
        **papers present in both rankings gain** (A 30th x B 3rd behaves like a top
        hit), and no fixed count has to be chosen. Insertion lost on every metric.
        An earlier attempt to mix by score broke badly (cr@20 0.822 -> 0.773) because
        it added the reranker's absolute score to the expander's improvised one; RRF
        looks only at ranks and cannot have that problem.

        **Plain RRF (weight 1.0, offset 0) measured best**: over 55 queries
        cr@20 0.789 -> 0.879, ecr@20 0.868 -> 0.926, multi@20 0.601 -> 0.770,
        cr@50 0.832 -> 0.917, with single-paper cr@20 staying at 1.000. Lowering the
        weight drops B-only papers below A's tail and the fusion stops meaning
        anything (0.817 at w=0.5); raising it lets B take over (0.830 at w=2.0).
        offset 0 is likewise best (0.839 at offset 15) — in a rank fusion the
        "insertion depth" that mattered for positional insertion is carried by the
        overlap bonus instead.
        """
        anchors = self._anchor_papers(candidate_papers, verdict_papers)
        related = self.paper_expander.rank(anchors)
        if not related:
            return candidate_papers

        # **Put the anchors themselves at the head of B.** Every expander excludes
        # an anchor from its own neighbours, so otherwise an anchor holds only A's
        # 1/(k+1) and is overtaken by every paper that appears in both. Measured: on
        # two single-paper queries the top candidate, which *was* the gold paper,
        # fell out of the top 20. A paper is also by definition its own nearest
        # neighbour, so first place in B is where it belongs.
        related = anchors + [p for p in related if p not in anchors]

        k = self.combine.rrf_k
        offset = self.combine.related_offset
        scores = {paper_id: 1.0 / (k + rank + 1) for rank, paper_id in enumerate(candidate_papers)}
        for rank, paper_id in enumerate(related):
            scores[paper_id] = scores.get(paper_id, 0.0) + self.combine.related_weight / (
                k + offset + rank + 1
            )
        # Ties break by insertion order (A's rank first, then B's) since sorted is stable.
        fused = sorted(scores, key=lambda paper_id: -scores[paper_id])

        before = {paper_id: rank for rank, paper_id in enumerate(candidate_papers)}
        head = fused[: self.config.max_candidates]
        trace.append(
            {
                "paper_fusion": {
                    "anchor": candidate_papers[0],
                    # The anchors actually used: the top `anchors` candidates by
                    # default, plus the reader's confirmations under `verdict`.
                    "anchors": anchors,
                    "n_a": len(candidate_papers),
                    "n_b": len(set(related)),
                    # Papers that were only in B yet reached the visible head (newly found).
                    "b_only_promoted": [p for p in head if p not in before],
                    # Papers outside A's visible head that got lifted (reordering at work).
                    "promoted": [
                        p for p in head if before.get(p, 0) >= self.config.max_candidates
                    ],
                }
            }
        )
        return fused

    # ---- thin wrappers around the LLM call -----------------------------------

    def _ask_for_json(self, prompt: str) -> dict | None:
        try:
            return parse_json_object(self.llm(prompt))
        except Exception:
            return None

    def _ask_for_list(self, prompt: str, key: str) -> list[str]:
        parsed = self._ask_for_json(prompt)
        if not parsed:
            return []
        values = parsed.get(key)
        if not isinstance(values, list):
            return []
        return [str(v) for v in values if v]
