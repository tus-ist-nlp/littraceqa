"""1回だけ検索して gold_papers を返すシンプルなエージェント。反復なし・LLM呼び出しなし。"""

from __future__ import annotations

from litqa.contracts import Answer, Prediction, Query
from litqa.registry import register
from litqa.retrieve.hybrid import HybridRetriever, to_gold_papers


@register("agent", "simple")
class SimpleAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 20,
        max_papers: int | None = None,
    ):
        if max_papers is not None and max_papers <= 0:
            raise ValueError("max_papers must be positive or None")
        self.retriever = retriever
        self.top_k = top_k
        self.max_papers = max_papers

    def run(self, query: Query) -> Prediction:
        results = self.retriever.retrieve(query.question, self.top_k)
        paper_ids = to_gold_papers(results, max_papers=self.max_papers)
        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=[],
            answer=Answer(),
        )
