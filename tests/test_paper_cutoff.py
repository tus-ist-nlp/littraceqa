"""提出本数の決め方（paper_cutoff）が全エージェントで揃うことのテスト。

論文集合は F1 で採点されるので、提出本数がスコアを支配する。ここを揃えないと、
比較実験で「エージェントの賢さ」ではなく「本数の決め方」の差を測ってしまう。
"""

from __future__ import annotations

import json

import pytest

from litqa.agent.reading import ReadingAgent
from litqa.agent.simple import SimpleAgent
from litqa.agent.task_family import TaskFamilyClassifier, apply_paper_cutoff
from litqa.agent.verifying import VerifyingAgent
from litqa.contracts import Query, RetrievalResult
from litqa.llm.fake import FakeLLM


def _query(task_family="multi_paper") -> Query:
    return Query(
        query_id="q_1",
        question="Which papers report FID?",
        answer_types=["table"],
        task_family=task_family,
    )


class _StubRetriever:
    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"p{i}#c00",
                paper_id=f"p{i}",
                score=1.0 / (i + 1),
                text=f"body of p{i}",
                chunk_type="text_span",
                metadata={"title": f"Paper {i}"},
            )
            for i in range(12)
        ][:top_k]


def _judge_all_12(sufficient=True):
    """LLM が候補12本すべてを「根拠だ」と言い張るケース。"""
    return json.dumps(
        {
            "papers": [
                {"paper_id": f"p{i}", "evidence_chunk_ids": [f"p{i}#c00"]}
                for i in range(12)
            ],
            "sufficient": sufficient,
            "missing": "",
        }
    )


def test_apply_paper_cutoff_task_family_mode():
    classifier = TaskFamilyClassifier()
    papers = [f"p{i}" for i in range(12)]
    assert (
        len(apply_paper_cutoff(papers, _query("multi_paper"), classifier, "task_family", 10))
        == 5
    )
    assert (
        len(
            apply_paper_cutoff(
                papers, _query("hidden_source_single_paper"), classifier, "task_family", 10
            )
        )
        == 2
    )


def test_apply_paper_cutoff_llm_mode_caps_at_max_papers():
    classifier = TaskFamilyClassifier()
    papers = [f"p{i}" for i in range(12)]
    assert len(apply_paper_cutoff(papers, _query(), classifier, "llm", 10)) == 10
    assert len(apply_paper_cutoff(papers[:3], _query(), classifier, "llm", 10)) == 3


def test_unknown_cutoff_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown paper_cutoff"):
        apply_paper_cutoff(["p0"], _query(), TaskFamilyClassifier(), "oops", 10)


@pytest.mark.parametrize("task_family,expected", [("multi_paper", 5), ("hidden_source_single_paper", 2)])
def test_all_agents_submit_the_same_count_under_task_family_cutoff(task_family, expected):
    """paper_cutoff=task_family なら、LLM が何本選ぼうと全エージェントが同じ本数を出す。

    これが揃っていないと simple → verifying → reading の引き算が成立しない。
    """
    query = _query(task_family)

    simple = SimpleAgent(_StubRetriever(), top_k=20)
    verifying = VerifyingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[json.dumps({"paper_ids": [f"p{i}" for i in range(12)]})]),
        top_k=20,
        paper_cutoff="task_family",
    )
    reading = ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(
            responses=[json.dumps({"subqueries": ["sq"]}), _judge_all_12()]
        ),
        top_k=20,
        max_steps=1,
        paper_cutoff="task_family",
    )

    counts = {
        "simple": len(simple.run(query).gold_papers),
        "verifying": len(verifying.run(query).gold_papers),
        "reading": len(reading.run(query).gold_papers),
    }
    assert counts == {"simple": expected, "verifying": expected, "reading": expected}


def test_llm_cutoff_lets_the_llm_decide_the_count():
    """paper_cutoff=llm なら LLM が選んだ本数がそのまま出る（max_papers で頭打ち）。"""
    reading = ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[json.dumps({"subqueries": ["sq"]}), _judge_all_12()]),
        top_k=20,
        max_steps=1,
        paper_cutoff="llm",
        max_papers=10,
    )
    prediction = reading.run(_query())
    assert len(prediction.gold_papers) == 10  # 12本選ばれたが max_papers で頭打ち


def test_evidence_is_dropped_for_papers_cut_by_the_cutoff():
    """打ち切りで落ちた論文の evidence は提出しない。"""
    reading = ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[json.dumps({"subqueries": ["sq"]}), _judge_all_12()]),
        top_k=20,
        max_steps=1,
        paper_cutoff="task_family",  # multi_paper → 5本に切られる
    )
    prediction = reading.run(_query("multi_paper"))
    kept = {p["paper_id"] for p in prediction.gold_papers}
    assert len(kept) == 5
    assert {e.paper_id for e in prediction.evidence} <= kept
