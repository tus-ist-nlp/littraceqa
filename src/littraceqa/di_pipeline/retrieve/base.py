"""Retriever パイプラインの各コンポーネントが実装すべき Protocol 定義。"""

from __future__ import annotations

from typing import Protocol

from littraceqa.di_pipeline.contracts import RetrievalResult, SearchHints


class Fuser(Protocol):
    def fuse(self, runs: list[list[RetrievalResult]], top_k: int) -> list[RetrievalResult]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]: ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        hints: SearchHints | None = None,
    ) -> list[RetrievalResult]: ...
