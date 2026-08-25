"""Retriever パイプラインの各コンポーネントが実装すべき Protocol 定義。"""

from __future__ import annotations

from typing import Protocol

from littraceqa.di_pipeline.contracts import RetrievalResult


class Fuser(Protocol):
    def fuse(self, runs: list[list[RetrievalResult]], top_k: int) -> list[RetrievalResult]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]: ...


class Retriever(Protocol):
    # 実装は任意で attribute_filter キーワード引数を受け付けてよい
    # （retrieve/attribute_filter.py の AttributeFilter）。呼び出し側は制約が
    # 取れたときだけ渡すので、受け付けない実装もそのまま動く。
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]: ...
