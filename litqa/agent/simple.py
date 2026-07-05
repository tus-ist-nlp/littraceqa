"""1回だけ検索して gold_papers を返すシンプルなエージェント。反復なし・LLM呼び出しなし。"""

from __future__ import annotations

from litqa.contracts import Answer, Prediction, Query
from litqa.registry import register
from litqa.retrieve.hybrid import HybridRetriever, to_gold_papers

_CUTOFF_BY_TASK_FAMILY = {
    "hidden_source_single_paper": 2,
    "multi_paper": 5,
}


@register("agent", "simple")
class SimpleAgent:
    def __init__(self, retriever: HybridRetriever, top_k: int = 20):
        self.retriever = retriever
        self.top_k = top_k

    def run(self, query: Query) -> Prediction:
        results = self.retriever.retrieve(query.question, self.top_k)
        paper_ids = to_gold_papers(results)
        cutoff = _CUTOFF_BY_TASK_FAMILY.get(query.task_family)
        if cutoff is not None:
            paper_ids = paper_ids[:cutoff]
        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=[],
            answer=Answer(),
        )
