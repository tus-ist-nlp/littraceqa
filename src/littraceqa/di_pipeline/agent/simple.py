"""1回だけ検索して gold_papers を返すシンプルなエージェント。反復なし。

task_family は本番入力に無いので、llm を渡せば LLM で推定し、渡さなければ
ヒューリスティックで推定する（src/littraceqa/di_pipeline/agent/task_family.py 参照）。
"""

from __future__ import annotations

from littraceqa.di_pipeline.agent.task_family import CUTOFF_BY_TASK_FAMILY, TaskFamilyClassifier
from littraceqa.di_pipeline.contracts import Answer, Prediction, Query
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.base import Retriever
from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers


@register("agent", "simple")
class SimpleAgent:
    def __init__(
        self,
        retriever: Retriever,
        top_k: int = 20,
        llm: LLMClient | None = None,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.task_family = TaskFamilyClassifier(llm)

    def run(self, query: Query) -> Prediction:
        results = self.retriever.retrieve(query.question, self.top_k)
        paper_ids = to_gold_papers(results)
        cutoff = CUTOFF_BY_TASK_FAMILY.get(self.task_family.infer(query))
        if cutoff is not None:
            paper_ids = paper_ids[:cutoff]
        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=[],
            answer=Answer(),
        )
