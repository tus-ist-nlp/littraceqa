"""The retriever: several indexes, a fuser, and (optionally) a reranker.

Each index is queried, the runs are fused into one ranking, and the reranker
blends its own ordering into that (see `_blend_rerank`).

When the question names a venue ("Which NAACL 2025 papers ..."), each index is
asked for extra results and the surplus is dropped by metadata afterwards (see
attribute_filter.py). The indexes need no changes and every index benefits alike.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.attribute_filter import (
    AttributeExtractor,
    AttributeFilter,
    filter_results,
)
from littraceqa.di_pipeline.retrieve.paper_rrf import PaperRRFFuser
from littraceqa.di_pipeline.retrieve.reranker import Qwen3Reranker


@dataclass(frozen=True)
class RerankBlend:
    """Blend the reranker's ranking with the pre-rerank one instead of replacing it.

    With `rerank_blend=None` the reranker **replaces** the order outright. But it
    judges "does this answer the question", so it always demotes peer gold papers
    the question never names (the same reason `_combine_rrf` in agent/reading.py
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
        in agent/reading.py).
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
        # candidate and only then cuts to top_k (see retrieve/reranker.py).
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
        re-sorts by `score` (the accumulation in agent/reading.py,
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
        the front yields exactly title + abstract (preprocess/mineru_chunker.py).
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
