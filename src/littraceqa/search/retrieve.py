"""Retrieval: the venue/year filter, paper-level RRF, and the retriever itself.

The three used to be separate modules. They are one file because there is one of
each — the fuser is always paper-level RRF, the filter is always the venue/year
one — so the seams only cost hops to read.


Each index is queried, the runs are fused into one ranking, and the reranker
blends its own ordering into that (see `_blend_rerank`).

When the question names a venue ("Which NAACL 2025 papers ..."), each index is
asked for extra results and the surplus is dropped by metadata afterwards (see
attribute_filter.py). The indexes need no changes and every index benefits alike.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Collection
from dataclasses import dataclass, replace
from pathlib import Path

from littraceqa.search.contracts import RetrievalResult
from littraceqa.search.reranker import Qwen3Reranker


# ============================================================================
# The venue/year attribute filter
# ============================================================================
#
# Narrow results by the venue and year the question states.
#
# Some LitTraceQA questions name their search scope outright ("Which NAACL 2025
# papers ...", "Among ICML 2025 papers ..."). Five of 55 validation queries do, and
# in those the gold papers satisfied the constraint 18 times out of 18 — when the
# constraint is stated, filtering on it is always right.
#
# No index changes are needed. ``RetrievalResult.metadata`` already carries venue
# and year (see metadata_base in ``preprocess.py``), so dropping
# results afterwards works identically for every index.
#
# **It only fires when exactly one venue can be extracted.** Why:
#
# * Filtering on the year alone buys little: the corpus is 91.3% 2025 and 8.7% 2024.
# * Some questions target every venue explicitly ("Across all venues, among
#   2025 ..."), and their gold spanned iccv / neurips / icml.
# * A cited paper's venue can leak in ("Which CVPR 2025 papers cite UniAD (...,
#   CVPR2023)"), so finding two or more venues means giving up.
#
# When nothing is extracted an empty AttributeFilter is returned and the caller
# takes its normal path, so questions that name no venue behave exactly as before.
#
# **Regular expressions are enough.** Across the 55 validation queries nothing was
# missed: 5 fired and 2 `all venues` cases were correctly declined. An LLM fallback
# for aliases such as "NIPS" was built and removed after measuring zero gain.

# The venues present in the corpus (all 9 values of venue in
# data/paper_metadata.jsonl). Looked up lower-cased to absorb spelling variants.
_VENUES = ("NeurIPS", "ICLR", "EMNLP", "ACL", "ICML", "CVPR", "ICCV", "ECCV", "NAACL")

# Questions that explicitly target every venue; no venue must be extracted.
_ALL_VENUES_RE = re.compile(r"\ball\s+venues\b", re.I)

# Only a year adjacent to the venue (separated by space, comma or apostrophe),
# so "CVPR 2025 papers cite UniAD (..., CVPR2023)" is not dragged to the distant year.
_YEAR_RE = r"(20\d{2})"


@dataclass(frozen=True)
class AttributeFilter:
    """The attribute constraint applied to results. Empty means no constraint."""

    venue: str | None = None
    year: int | None = None

    def is_empty(self) -> bool:
        return self.venue is None and self.year is None

    def matches(self, metadata: dict | None) -> bool:
        """Does this chunk's metadata satisfy the constraint?"""
        metadata = metadata or {}
        if self.venue is not None and metadata.get("venue") != self.venue:
            return False
        if self.year is not None and metadata.get("year") != self.year:
            return False
        return True


class AttributeExtractor:
    """Builds an AttributeFilter from a question and reports its selectivity.

    Selectivity is computed from paper counts in paper_metadata.jsonl rather than
    chunk counts. Chunks per paper do not vary much by venue, so the ratio is close
    enough for working back to a fetch size.
    """

    def __init__(self, paper_metadata: str | Path):
        self._venue_by_lower = {v.lower(): v for v in _VENUES}
        self._total = 0
        self._counts: dict[tuple[str | None, int | None], int] = {}
        self._load(Path(paper_metadata))
        # Match venues on word boundaries so ACL does not match inside NAACL.
        self._venue_re = re.compile(
            r"\b(" + "|".join(re.escape(v) for v in _VENUES) + r")\b", re.I
        )

    def _load(self, path: Path) -> None:
        papers: list[tuple[str, int]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                venue = record.get("venue")
                year = record.get("year")
                papers.append((venue, int(year) if year is not None else None))

        self._total = len(papers)
        counts: dict[tuple[str | None, int | None], int] = {}
        for venue, year in papers:
            # Count (venue, None), (venue, year) and (None, year) so selectivity
            # can be looked up for a constraint missing either part.
            for key in ((venue, None), (venue, year), (None, year)):
                counts[key] = counts.get(key, 0) + 1
        self._counts = counts

    def exists(self, venue: str | None, year: int | None) -> bool:
        """Does the corpus contain at least one paper with this (venue, year)?"""
        if venue is None and year is None:
            return False
        return self._counts.get((venue, year), 0) > 0

    def extract(self, question: str) -> AttributeFilter:
        """Extract the constraint from a question; an empty AttributeFilter if there is none."""
        if not question or _ALL_VENUES_RE.search(question):
            return AttributeFilter()

        found = {self._venue_by_lower[m.group(1).lower()] for m in self._venue_re.finditer(question)}
        if len(found) != 1:
            # Neither zero venues nor two or more (a cited venue leaked in) is
            # usable, so both decline.
            return AttributeFilter()
        venue = next(iter(found))

        return AttributeFilter(venue=venue, year=self._adjacent_year(question, venue))

    def _adjacent_year(self, question: str, venue: str) -> int | None:
        """Return only a year adjacent to the venue, ignoring years elsewhere.

        In "Which CVPR 2025 papers cite UniAD (Planning-oriented ..., CVPR2023)" the
        CVPR2023 also reads as adjacent, so finding more than one means declining.
        """
        pattern = re.compile(r"\b" + re.escape(venue) + r"\b[\s,'’]*" + _YEAR_RE, re.I)
        years = {int(m.group(1)) for m in pattern.finditer(question)}
        if len(years) != 1:
            return None
        year = next(iter(years))
        # A year absent from the corpus would always filter to nothing, so the
        # year is dropped in that case.
        if self._counts.get((venue, year), 0) == 0:
            return None
        return year

    def selectivity(self, attribute_filter: AttributeFilter) -> float:
        """Fraction of papers satisfying the constraint, floored to avoid dividing by zero."""
        if attribute_filter.is_empty() or self._total == 0:
            return 1.0
        matched = self._counts.get((attribute_filter.venue, attribute_filter.year), 0)
        if matched <= 0:
            return 1.0
        return matched / self._total


def filter_results(results: list, attribute_filter: AttributeFilter) -> list:
    """Keep only the results that satisfy the constraint."""
    if attribute_filter.is_empty():
        return list(results)
    return [r for r in results if attribute_filter.matches(r.metadata)]


# ============================================================================
# Paper-level RRF
# ============================================================================
#
# Paper-level RRF: one vote per paper when fusing several indexes.
#
# Fusing per chunk,
#
#     s(c) = sum_i  w_i / (k + chunk_rank_i(c))
#
# gives **every chunk of a paper its own independent vote**. Long papers and
# table-heavy papers simply have more chunks, so they occupy the top for reasons
# unrelated to "is this paper close to the question". The metric is per paper
# (`candidate_recall`), so that distortion lands straight on the score.
#
# Here the collapse to papers happens **first**, then the fusion:
#
#     s(p) = sum_i  w_i / (k + paper_rank_i(p))
#     paper_rank_i(p) = where p first appears in run i (dense rank, 0-based)
#
# **Within one run, a paper gets one vote no matter how many of its chunks hit.**
#
# The output is still a list of **chunks** (the reranker, the reading agent and
# evidence all work on chunk_id). Papers order the list, chunks within a paper break
# the tie, and `score` carries **a value that reproduces the same order when
# re-sorted downstream** (the accumulation in agent.py, `_candidate_papers`
# and `to_gold_papers` all re-sort by score, so an order carried only by list
# position would be thrown away).
#
# **A paper may contribute at most `chunks_per_paper` chunks.** Without that limit a
# single paper with 100 chunks eats `pool_k` and the reranker only ever sees one
# paper. The default of 3 is slightly above the 2 chunks per paper the reading agent
# displays.
#
# **Pseudo chunks from `bm25s_paper` (`{paper_id}#paper`) are never chosen as a
# paper's representative.** Their chunk_id does not exist, so they cannot be used as
# evidence, and their text is the whole paper, so they cannot be read. They still
# count for ranking papers — that is what the index is for — while a real chunk is
# used as the representative.

# Pseudo chunks from the paper-level index: used for ranking, never as a representative.
PAPER_LEVEL_SOURCES = frozenset({"bm25s_paper"})

# Tiny offset that preserves chunk order within a paper. Kept far below the gap
# between paper scores (1/(k+r) - 1/(k+r+1) ~ 1/(k+r)^2, i.e. 1.5e-5 even at
# k=60, r=200) so it can never disturb the ordering across papers.
_CHUNK_ORDER_EPS = 1e-9


def is_paper_level(result: RetrievalResult) -> bool:
    """Is this a pseudo chunk produced by the paper-level index?"""
    return result.source in PAPER_LEVEL_SOURCES or result.chunk_id.endswith("#paper")


def paper_rrf_fuse(
    runs: list[list[RetrievalResult]],
    top_k: int,
    k: int = 60,
    chunks_per_paper: int = 3,
) -> list[RetrievalResult]:
    """Paper-level RRF. The body of `PaperRRFFuser.fuse()`, exposed as a function.

    **Every run counts the same.** There used to be a per-source `weights` dict for
    weighting one index above another; sweeping it never beat equal weights, and
    the final configuration passes nothing, so the ranks alone decide.
    """
    paper_scores: dict[str, float] = {}
    # Chunk-level RRF scores, used only to order chunks within a paper.
    chunk_scores: dict[str, float] = {}
    chunk_of: dict[str, RetrievalResult] = {}
    chunks_of_paper: dict[str, list[str]] = {}

    for run in runs:
        # paper_id -> dense rank in this run
        seen_papers: dict[str, int] = {}
        for rank, result in enumerate(run):
            chunk_scores[result.chunk_id] = chunk_scores.get(result.chunk_id, 0.0) + 1.0 / (
                k + rank + 1
            )
            if result.chunk_id not in chunk_of:
                chunk_of[result.chunk_id] = result
                chunks_of_paper.setdefault(result.paper_id, []).append(result.chunk_id)
            # **One vote per paper per run.** Only its first appearance sets the rank.
            if result.paper_id not in seen_papers:
                seen_papers[result.paper_id] = len(seen_papers)
        for paper_id, paper_rank in seen_papers.items():
            paper_scores[paper_id] = paper_scores.get(paper_id, 0.0) + 1.0 / (
                k + paper_rank + 1
            )

    ordered_papers = sorted(paper_scores, key=lambda p: (-paper_scores[p], p))

    fused: list[RetrievalResult] = []
    for paper_id in ordered_papers:
        candidates = sorted(
            chunks_of_paper.get(paper_id, []),
            # Pseudo chunks go last, so a real chunk becomes the representative when one exists.
            key=lambda cid: (is_paper_level(chunk_of[cid]), -chunk_scores[cid], cid),
        )
        for offset, chunk_id in enumerate(candidates[:chunks_per_paper]):
            fused.append(
                dataclasses.replace(
                    chunk_of[chunk_id],
                    score=paper_scores[paper_id] - offset * _CHUNK_ORDER_EPS,
                )
            )
            if len(fused) >= top_k:
                return fused
    return fused


class PaperRRFFuser:
    def __init__(self, k: int = 60, chunks_per_paper: int = 3):
        self.k = k
        self.chunks_per_paper = chunks_per_paper

    def fuse(
        self, runs: list[list[RetrievalResult]], top_k: int
    ) -> list[RetrievalResult]:
        return paper_rrf_fuse(
            runs, top_k, k=self.k, chunks_per_paper=self.chunks_per_paper
        )


# ============================================================================
# The retriever
# ============================================================================


@dataclass(frozen=True)
class RerankBlend:
    """Blend the reranker's ranking with the pre-rerank one instead of replacing it.

    With `rerank_blend=None` the reranker **replaces** the order outright. But it
    judges "does this answer the question", so it always demotes peer gold papers
    the question never names (the same reason `_combine_rrf` in agent.py
    keeps ranking B away from the reranker). Replacing leaves the search ranking
    exposed to that.
    """

    original_weight: float = 0.6
    rerank_weight: float = 0.4
    rrf_k: int = 60
    # Keep the pre-fusion top N as a *set* at the front, so nothing below rank N
    # can walk into the head unconditionally.
    protect_top: int = 0


@dataclass(frozen=True)
class SeedExpansion:
    """Append the top paper's first `query_chars` characters to the question and query again.

    **A question does not know what a paper calls itself.** One gold paper never
    says `reference-free`; it calls itself a `Direct Alignment Algorithm` — words
    the question does not contain, so querying with the question alone can never
    reach it. This borrows the corpus's own vocabulary from the top hit. No LLM.
    """

    query_chars: int = 512


class HybridRetriever:
    def __init__(
        self,
        indexers: list,
        fuser: PaperRRFFuser,
        # None returns the fused ranking as is (no reranking).
        reranker: Qwen3Reranker | None = None,
        per_index_k: int = 100,
        pool_k: int | None = None,
        attribute_extractor: AttributeExtractor | None = None,
        fetch_safety: float = 1.5,
        max_fetch_k: int = 5000,
        min_filtered_results: int = 10,
        rerank_blend: RerankBlend | None = None,
        seed_expansion: SeedExpansion | None = None,
        anchor_store: object | None = None,
    ):
        self.indexers = indexers
        self.fuser = fuser
        self.reranker = reranker
        self.per_index_k = per_index_k
        # Size of the candidate pool handed to the reranker; defaults to top_k*3.
        self.pool_k = pool_k
        # None lets the reranker replace the order; a RerankBlend mixes it with the
        # pre-rerank ranking instead (see _blend_rerank).
        self.rerank_blend = rerank_blend
        # None skips seed expansion (a single query per subquery); a SeedExpansion
        # appends the top paper's vocabulary and queries again (see _seed_expand).
        self.seed_expansion = seed_expansion
        # ChunkStore used to look up the anchor's title+abstract (seed expansion only).
        self.anchor_store = anchor_store
        # None disables the attribute filter entirely.
        self.attribute_extractor = attribute_extractor
        self.fetch_safety = fetch_safety
        self.max_fetch_k = max_fetch_k
        self.min_filtered_results = min_filtered_results

    def retrieve(
        self,
        query: str,
        top_k: int,
        attribute_filter: AttributeFilter | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve.

        Passing `attribute_filter` narrows the results by that constraint. Without
        it, the extractor (if any) derives one from `query` itself — a fallback for
        callers that send a raw question. **A production run never takes that
        path**: ReadingAgent extracts once from the original question and passes it
        in, because subqueries drop the venue name (see `_extract_attribute_filter`
        in agent.py).
        """
        if not self.indexers:
            return []

        if attribute_filter is None and self.attribute_extractor is not None:
            attribute_filter = self.attribute_extractor.extract(query)

        runs = self._run_indexers(query, attribute_filter)
        if self.reranker is not None:
            fuse_k = self.pool_k if self.pool_k is not None else top_k * 3
        else:
            fuse_k = top_k
        fused = self.fuser.fuse(runs, top_k=fuse_k)
        # Seed expansion goes **before** the reranker: running the reranker twice
        # would double inference, while querying the indexes twice is cheap. The
        # reranker still runs once, on the original query.
        fused = self._seed_expand(query, fused, attribute_filter, fuse_k)

        if self.reranker is None:
            return fused[:top_k]
        if self.rerank_blend is None:
            return self.reranker.rerank(query, fused, top_k)
        # Blending needs the full ranking before truncation, hence len(fused).
        # **This costs no extra inference**: Qwen3Reranker already scores every
        # candidate and only then cuts to top_k (see reranker.py).
        reranked = self.reranker.rerank(query, fused, len(fused))
        return self._blend_rerank(fused, reranked)[:top_k]

    def _blend_rerank(
        self, fused: list[RetrievalResult], reranked: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Blend the pre-rerank ranking with the reranker's ranking via RRF.

            score(c) = w_orig / (k + rank_fused) + w_rerank / (k + rank_reranked)

        **Only ranks are used, never scores.** RRF scores and the reranker's yes
        probability are on different scales and cannot be added (`_combine_rrf`
        uses ranks for the same reason).

        `protect_top` keeps the **set** of the pre-fusion top N at the front
        (ordered by the blended result), so nothing below it can walk into the head.

        **The blended rank is written back into `score`.** Everything downstream
        re-sorts by `score` (the accumulation in agent.py,
        `_candidate_papers`, `to_gold_papers`), so a ranking carried only by list
        order would be thrown away. `Qwen3Reranker.rerank` overwrites score for the
        same reason.
        """
        blend = self.rerank_blend or RerankBlend()
        k = blend.rrf_k
        w_orig = blend.original_weight
        w_rerank = blend.rerank_weight
        protect_top = blend.protect_top

        scores: dict[str, float] = {}
        for rank, result in enumerate(fused):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + w_orig / (k + rank + 1)
        for rank, result in enumerate(reranked):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + w_rerank / (k + rank + 1)

        # **The protection has to live in `score` too.** Moving items to the front
        # of the list is undone the moment something downstream re-sorts by score.
        # Adding (max score + 1) keeps their relative RRF order while pinning the
        # group above everything else.
        if protect_top and scores:
            boost = max(scores.values()) + 1.0
            for result in fused[:protect_top]:
                scores[result.chunk_id] += boost

        # Take the objects the reranker returned (only score is overwritten below).
        # Nothing can be in the reranked list but not the fused one; merging both
        # sides just guarantees nothing is dropped.
        by_id = {result.chunk_id: result for result in fused}
        by_id.update({result.chunk_id: result for result in reranked})
        ordered = sorted(by_id.values(), key=lambda r: -scores[r.chunk_id])
        return [replace(r, score=scores[r.chunk_id]) for r in ordered]

    def _seed_expand(
        self,
        query: str,
        fused: list[RetrievalResult],
        attribute_filter: AttributeFilter | None,
        fuse_k: int,
    ) -> list[RetrievalResult]:
        """Query again with the top paper's vocabulary, then fuse with the first ranking.

        **Pseudo relevance feedback, not LLM query rewriting.**

            expanded = original question + the top paper's title+abstract, first `query_chars`

        A question does not know what a paper calls itself. The gold paper for
        q_022 (AlphaPO) never says `reference-free`; it says `Direct Alignment
        Algorithm` and `reward shape` — words absent from the question, so querying
        with the question alone can never reach it. **Borrowing the corpus's own
        vocabulary from the top hit** is what this does.

        An LLM rewriting variant was tried and removed: the LLM turned candidate
        titles into queries verbatim and just re-fetched papers already in hand
        (subqueries containing 80%+ of a candidate's title words went from 1% to
        16%). Concatenating mechanically cannot fail that way, and it always keeps
        the original question.

        **Fusion is left to `self.fuser`**, so with the paper-level fuser the two
        rounds mix per paper — seed expansion and paper-level RRF compose freely.

        **It runs before the reranker**, so reranker inference stays at one pass;
        the only added cost is one more sweep over the indexes.
        """
        if not self.seed_expansion or not fused:
            return fused
        anchor_text = self._anchor_text(fused[0])
        if not anchor_text:
            return fused
        query_chars = self.seed_expansion.query_chars
        expanded = f"{query}\n{anchor_text[:query_chars]}"
        runs = self._run_indexers(expanded, attribute_filter)
        expanded_fused = self.fuser.fuse(runs, top_k=fuse_k)
        if not expanded_fused:
            return fused
        # Fuse the two rounds as two runs. Only ranks matter, so the scale
        # difference between the two RRF score sets is irrelevant.
        return self.fuser.fuse([fused, expanded_fused], top_k=fuse_k)

    def _anchor_text(self, anchor: RetrievalResult) -> str:
        """The anchor's title+abstract, falling back to the matched chunk's text.

        A `title_abstract` chunk (`{paper_id}#c0000`) has the form
        `"[venue year] title\\n" + abstract`, so truncating from
        the front yields exactly title + abstract (preprocess.py).
        """
        store = self.anchor_store
        if store is not None:
            try:
                for chunk in store.load_paper(anchor.paper_id):
                    if chunk.get("chunk_type") == "title_abstract":
                        return str(chunk.get("text") or "")
            except Exception:  # noqa: BLE001 - retrieval continues even without the text
                pass
        # No ChunkStore (tests) or the paper was not found. The paper-level index's
        # pseudo chunk holds the full paper text, so truncating from the front still
        # yields title + abstract.
        title = str((anchor.metadata or {}).get("title") or "")
        if title and not anchor.text.startswith("["):
            return f"{title}\n{anchor.text}"
        return anchor.text

    def _run_indexers(
        self, query: str, attribute_filter: AttributeFilter | None
    ) -> list[list[RetrievalResult]]:
        """Query every index; with a constraint, over-fetch and then drop."""
        if attribute_filter is None or attribute_filter.is_empty():
            return [indexer.search(query, self.per_index_k) for indexer in self.indexers]

        fetch_k = self._fetch_k(attribute_filter)
        runs = []
        for indexer in self.indexers:
            raw = indexer.search(query, fetch_k)
            kept = filter_results(raw, attribute_filter)
            # If filtering drains the run, fall back to the unfiltered one. Better
            # to tolerate noise than to lose recall to a misextraction or to having
            # fetched too few (fail-open).
            if len(kept) < self.min_filtered_results:
                kept = raw
            runs.append(kept[: self.per_index_k])
        return runs

    def _fetch_k(self, attribute_filter: AttributeFilter) -> int:
        """Work back from the selectivity so per_index_k survives the filter."""
        selectivity = 1.0
        if self.attribute_extractor is not None:
            selectivity = self.attribute_extractor.selectivity(attribute_filter)
        if selectivity <= 0:
            return self.max_fetch_k
        needed = int(self.per_index_k / selectivity * self.fetch_safety)
        return max(self.per_index_k, min(needed, self.max_fetch_k))


def to_gold_papers(
    results: list[RetrievalResult],
    max_papers: int | None = None,
    skip_chunk_types: Collection[str] = (),
) -> list[str]:
    """Collapse a chunk ranking into a paper ranking.

    Chunk types listed in `skip_chunk_types` **contribute 0 to a paper's
    representative score**. A paper made only of those sinks to the bottom of
    ranking A but **stays in the candidate list**, so ranking B can still lift it
    back. `"table"` is the measured best: table chunks are dense with numbers and
    short labels, so BM25 and the reranker both score them highly on word overlap
    alone, and **a single table can spike a paper that is not the question's topic**.

    Sweeping a weight instead (score = max(non-table, w x table)) is exactly
    equivalent for every w <= 0.85, so the rule itself is what matters, not a
    threshold — which is why this is a boolean with **no free parameter**.

    **The chunk pool is untouched.** Only the representative score changes; table
    chunks still reach the reading LLM and can still be emitted as `evidence`
    (table is the most common gold `primary_evidence_type`, 17 queries, so they
    must not be dropped). Showing tables to the reader in fact raises its
    confirmation rate (71% vs 51%).

    Dropping `figure` / `equation_algorithm` alongside makes things **worse**, so
    only `table` is skipped. **A paper is represented by its single best chunk**
    (max), which is what makes the distortion possible in the first place: summing
    the chunks instead makes the effect nearly vanish. That was measured, not left
    behind as a knob — there is no aggregation parameter.

    **Do not add a "use the table score for papers that only have tables"
    fallback.** It looks kind and loses in measurement (multi@5 0.758 -> 0.720).
    488 papers have nothing but tables, and **sinking them is precisely what helps**.
    """
    skip = set(skip_chunk_types)
    scores: dict[str, float] = {}
    for result in results:
        value = 0.0 if result.chunk_type in skip else result.score
        scores[result.paper_id] = max(scores.get(result.paper_id, value), value)

    # Ties break by insertion order (i.e. order in the fused ranking) since sorted is stable.
    papers = sorted(scores, key=lambda paper_id: scores[paper_id], reverse=True)
    if max_papers is not None:
        papers = papers[:max_papers]
    return papers
