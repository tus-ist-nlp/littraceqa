"""本番入力（query_id / question / answer_types / table_schema の4つだけ）でも
task_family を解決して cutoff を効かせられることを確認するテスト。"""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.agent.task_family import MULTI, SINGLE, TaskFamilyClassifier
from littraceqa.di_pipeline.contracts import Query, RetrievalResult
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _query(**kwargs) -> Query:
    base = {
        "query_id": "q_1",
        "question": "What is X?",
        "answer_types": ["freeform"],
    }
    base.update(kwargs)
    return Query(**base)


class _StubRetriever:
    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"p{i}#c0",
                paper_id=f"p{i}",
                score=1.0 / (i + 1),
                text=f"paper {i}",
                chunk_type="text_span",
                metadata={},
            )
            for i in range(10)
        ]


def test_query_from_dict_accepts_production_input():
    """本番入力の4フィールドだけで Query が作れ、欠けている2つは None になる。"""
    query = Query.from_dict(
        {
            "query_id": "q_20",
            "question": "Which papers reference MCTS?",
            "answer_types": ["table"],
            "table_schema": [{"name": "Paper Title", "type": "string", "is_row_key": True}],
        }
    )
    assert query.task_family is None
    assert query.primary_evidence_type is None
    assert query.table_schema == [
        {"name": "Paper Title", "type": "string", "is_row_key": True}
    ]


def test_input_task_family_wins_over_inference():
    """入力に task_family があれば LLM を呼ばずにそれを使う。"""
    llm = FakeLLM(responses=['{"task_family": "multi_paper"}'])
    classifier = TaskFamilyClassifier(llm)
    assert classifier.infer(_query(task_family=SINGLE)) == SINGLE
    assert llm.calls == []


def test_llm_infers_task_family_when_absent():
    llm = FakeLLM(responses=['{"task_family": "multi_paper"}'])
    classifier = TaskFamilyClassifier(llm)
    assert classifier.infer(_query()) == MULTI
    assert len(llm.calls) == 1


def test_llm_result_is_cached_per_query():
    """IterativeAgent は1クエリ中に何度も引くので、LLM 呼び出しは1回に抑える。"""
    llm = FakeLLM(responses=['{"task_family": "multi_paper"}'])
    classifier = TaskFamilyClassifier(llm)
    query = _query()
    for _ in range(3):
        assert classifier.infer(query) == MULTI
    assert len(llm.calls) == 1


