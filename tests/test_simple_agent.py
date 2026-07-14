"""Tests for task-label-independent simple retrieval."""

from __future__ import annotations

from litqa.agent.simple import SimpleAgent
from litqa.contracts import Query, RetrievalResult


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"p{index}#c0000",
                paper_id=f"p{index}",
                score=float(10 - index),
                text="text",
                chunk_type="text_span",
                metadata={},
            )
            for index in range(6)
        ]


def _query(task_family: str | None) -> Query:
    return Query(
        query_id="q1",
        question="What is reported?",
        answer_types=["freeform"],
        task_family=task_family,
    )


def test_fixed_paper_limit_does_not_depend_on_legacy_task_family():
    agent = SimpleAgent(retriever=_FakeRetriever(), top_k=20, max_papers=3)

    without_label = agent.run(_query(None))
    single_label = agent.run(_query("hidden_source_single_paper"))
    multi_label = agent.run(_query("multi_paper"))

    expected = ["p0", "p1", "p2"]
    assert [item["paper_id"] for item in without_label.gold_papers] == expected
    assert [item["paper_id"] for item in single_label.gold_papers] == expected
    assert [item["paper_id"] for item in multi_label.gold_papers] == expected
