"""VerifyingAgent の内容ベース選択ロジックのテスト。"""

from __future__ import annotations

import json

from littraceqa.di_pipeline.agent.verifying import VerifyingAgent
from littraceqa.di_pipeline.contracts import Query, RetrievalResult
from littraceqa.di_pipeline.llm.fake import FakeLLM


class FakeRetriever:
    """query 文字列 -> RetrievalResult 一覧 の対応表で応答する検索スタブ。"""

    def __init__(self, responses: dict[str, list[RetrievalResult]]):
        self.responses = responses

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self.responses.get(query, [])


class RaisingLLM:
    def __call__(self, prompt: str) -> str:
        raise RuntimeError("llm unavailable")


def _result_with_score(paper_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper_id}#c0000",
        paper_id=paper_id,
        score=score,
        text=f"text of {paper_id}",
        chunk_type="text_span",
        metadata={},
    )


def _query(task_family: str = "multi_paper", question: str = "What is X?") -> Query:
    return Query(
        query_id="q1",
        task_family=task_family,
        primary_evidence_type="text_span",
        question=question,
        answer_types=["freeform"],
    )


def test_selects_low_ranked_candidate_that_llm_judges_relevant():
    """順位が低くてもLLMが根拠ありと判定すれば最終提出に含まれる。"""
    retriever = FakeRetriever(
        {
            "What is X?": [
                _result_with_score("pTop", 0.9),
                _result_with_score("pMid", 0.5),
                _result_with_score("pLow", 0.1),
            ]
        }
    )
    llm = FakeLLM(responses=[json.dumps({"paper_ids": ["pTop", "pLow"]})])
    agent = VerifyingAgent(retriever=retriever, llm=llm, top_k=20)

    prediction = agent.run(_query())

    assert {g["paper_id"] for g in prediction.gold_papers} == {"pTop", "pLow"}


def test_ignores_hallucinated_paper_ids_not_in_candidates():
    retriever = FakeRetriever(
        {"What is X?": [_result_with_score("p1", 0.9), _result_with_score("p2", 0.5)]}
    )
    llm = FakeLLM(responses=[json.dumps({"paper_ids": ["p1", "pGhost"]})])
    agent = VerifyingAgent(retriever=retriever, llm=llm, top_k=20)

    prediction = agent.run(_query())

    assert [g["paper_id"] for g in prediction.gold_papers] == ["p1"]


def test_falls_back_to_rank_cutoff_when_llm_output_unparseable():
    papers = [_result_with_score(f"p{i}", 1.0 - i * 0.01) for i in range(10)]
    retriever = FakeRetriever({"What is X?": papers})
    llm = FakeLLM(responses=["not json at all"])
    agent = VerifyingAgent(retriever=retriever, llm=llm, top_k=20)

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.gold_papers) == 5  # multi_paper の固定カットオフ


def test_falls_back_to_rank_cutoff_when_llm_selects_only_hallucinated_ids():
    papers = [_result_with_score(f"p{i}", 1.0 - i * 0.01) for i in range(10)]
    retriever = FakeRetriever({"What is X?": papers})
    llm = FakeLLM(responses=[json.dumps({"paper_ids": ["pGhost"]})])
    agent = VerifyingAgent(retriever=retriever, llm=llm, top_k=20)

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.gold_papers) == 5


def test_falls_back_to_rank_cutoff_when_llm_raises():
    papers = [_result_with_score(f"p{i}", 1.0 - i * 0.01) for i in range(3)]
    retriever = FakeRetriever({"What is X?": papers})
    agent = VerifyingAgent(retriever=retriever, llm=RaisingLLM(), top_k=20)

    prediction = agent.run(_query("hidden_source_single_paper"))

    assert len(prediction.gold_papers) == 2  # hidden_source_single_paper の固定カットオフ


def test_no_candidates_returns_empty_prediction():
    retriever = FakeRetriever({})
    llm = FakeLLM(responses=["should not be called"])
    agent = VerifyingAgent(retriever=retriever, llm=llm, top_k=20)

    prediction = agent.run(_query())

    assert prediction.gold_papers == []


def test_only_top_max_candidates_are_shown_to_llm():
    papers = [_result_with_score(f"p{i}", 1.0 - i * 0.01) for i in range(5)]
    retriever = FakeRetriever({"What is X?": papers})
    llm = FakeLLM(responses=[json.dumps({"paper_ids": []})])
    agent = VerifyingAgent(retriever=retriever, llm=llm, top_k=20, max_candidates=2)

    agent.run(_query())

    prompt = llm.calls[0]
    assert "p0" in prompt
    assert "p1" in prompt
    assert "p2" not in prompt
