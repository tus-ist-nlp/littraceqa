"""Paper-level RRF: one vote per paper when fusing several indexes.

Fusing per chunk,

    s(c) = sum_i  w_i / (k + chunk_rank_i(c))

gives **every chunk of a paper its own independent vote**. Long papers and
table-heavy papers simply have more chunks, so they occupy the top for reasons
unrelated to "is this paper close to the question". The metric is per paper
(`candidate_recall`), so that distortion lands straight on the score.

Here the collapse to papers happens **first**, then the fusion:

    s(p) = sum_i  w_i / (k + paper_rank_i(p))
    paper_rank_i(p) = where p first appears in run i (dense rank, 0-based)

**Within one run, a paper gets one vote no matter how many of its chunks hit.**

The output is still a list of **chunks** (the reranker, the reading agent and
evidence all work on chunk_id). Papers order the list, chunks within a paper break
the tie, and `score` carries **a value that reproduces the same order when
re-sorted downstream** (the accumulation in agent/reading.py, `_candidate_papers`
and `to_gold_papers` all re-sort by score, so an order carried only by list
position would be thrown away).

**A paper may contribute at most `chunks_per_paper` chunks.** Without that limit a
single paper with 100 chunks eats `pool_k` and the reranker only ever sees one
paper. The default of 3 is slightly above the 2 chunks per paper the reading agent
displays.

**Pseudo chunks from `bm25s_paper` (`{paper_id}#paper`) are never chosen as a
paper's representative.** Their chunk_id does not exist, so they cannot be used as
evidence, and their text is the whole paper, so they cannot be read. They still
count for ranking papers — that is what the index is for — while a real chunk is
used as the representative.
"""

from __future__ import annotations

import dataclasses

from littraceqa.di_pipeline.contracts import RetrievalResult

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
    weights: dict[str, float] | None = None,
    chunks_per_paper: int = 3,
) -> list[RetrievalResult]:
    """Paper-level RRF. The body of `PaperRRFFuser.fuse()`, exposed as a function."""
    weights = weights or {}
    paper_scores: dict[str, float] = {}
    # Chunk-level RRF scores, used only to order chunks within a paper.
    chunk_scores: dict[str, float] = {}
    chunk_of: dict[str, RetrievalResult] = {}
    chunks_of_paper: dict[str, list[str]] = {}

    for run in runs:
        # paper_id -> (dense rank in this run, source of the chunk that first surfaced it)
        seen_papers: dict[str, tuple[int, str]] = {}
        for rank, result in enumerate(run):
            weight = weights.get(result.source, 1.0)
            chunk_scores[result.chunk_id] = chunk_scores.get(result.chunk_id, 0.0) + weight / (
                k + rank + 1
            )
            if result.chunk_id not in chunk_of:
                chunk_of[result.chunk_id] = result
                chunks_of_paper.setdefault(result.paper_id, []).append(result.chunk_id)
            # **One vote per paper per run.** Only its first appearance sets the rank.
            if result.paper_id not in seen_papers:
                seen_papers[result.paper_id] = (len(seen_papers), result.source)
        for paper_id, (paper_rank, source) in seen_papers.items():
            # A paper's vote is weighted by the index that produced the run (1 run = 1 index).
            paper_scores[paper_id] = paper_scores.get(paper_id, 0.0) + weights.get(
                source, 1.0
            ) / (k + paper_rank + 1)

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
    def __init__(
        self,
        k: int = 60,
        weights: dict[str, float] | None = None,
        chunks_per_paper: int = 3,
    ):
        self.k = k
        self.weights = weights or {}
        self.chunks_per_paper = chunks_per_paper

    def fuse(
        self, runs: list[list[RetrievalResult]], top_k: int
    ) -> list[RetrievalResult]:
        return paper_rrf_fuse(
            runs,
            top_k,
            k=self.k,
            weights=self.weights,
            chunks_per_paper=self.chunks_per_paper,
        )
