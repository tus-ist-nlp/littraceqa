"""reranker を使わない設定用の placeholder 実装。"""

from __future__ import annotations

from litqa.contracts import RetrievalResult
from litqa.registry import register


@register("reranker", "none")
class NoneReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        return candidates[:top_k]
