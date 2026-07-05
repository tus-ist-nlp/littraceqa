"""RRF (Reciprocal Rank Fusion) による複数 Indexer の検索結果統合。

複数の Indexer (bm25s, faiss, ...) が返す RetrievalResult のリスト（run）を、
Reciprocal Rank Fusion で1つのランキングに統合する。
"""

from __future__ import annotations

import dataclasses

from litqa.contracts import RetrievalResult
from litqa.registry import register


@register("fuser", "rrf")
class RRFFuser:
    def __init__(self, k: int = 60, weights: dict[str, float] | None = None):
        self.k = k
        self.weights = weights or {}

    def fuse(
        self,
        runs: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        first_seen: dict[str, RetrievalResult] = {}

        for run in runs:
            for rank, result in enumerate(run):
                weight = self.weights.get(result.source, 1.0)
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (
                    self.k + rank + 1
                )
                if result.chunk_id not in first_seen:
                    first_seen[result.chunk_id] = result

        fused = [
            dataclasses.replace(first_seen[chunk_id], score=score)
            for chunk_id, score in scores.items()
        ]
        fused.sort(key=lambda result: result.score, reverse=True)
        return fused[:top_k]
