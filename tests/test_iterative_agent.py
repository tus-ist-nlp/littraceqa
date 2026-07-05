"""IterativeAgent の反復検索ロジックのテスト。"""

from __future__ import annotations

import json

import pytest

from litqa.agent.iterative import IterativeAgent, _is_enumeration_or_comparison
from litqa.contracts import Query, RetrievalResult
from litqa.llm.fake import FakeLLM


class FakeRetriever:
    """query 文字列 -> RetrievalResult 一覧 の対応表で応答する検索スタブ。"""

    def __init__(self, responses: dict[str, list[RetrievalResult]]):
        self.responses = responses
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.queries.append(query)
        return self.responses.get(query, [])


class RaisingLLM:
    def __call__(self, prompt: str) -> str:
        raise RuntimeError("llm unavailable")


def _result(paper_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper_id}#c0000",
        paper_id=paper_id,
        score=1.0,
        text=f"text of {paper_id}",
        chunk_type="text_span",
        metadata={},
    )


def _result_with_score(paper_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper_id}#c0000",
        paper_id=paper_id,
        score=score,
        text=f"text of {paper_id}",
        chunk_type="text_span",
        metadata={},
    )


def _query(task_family: str, question: str = "What is X?") -> Query:
    return Query(
        query_id="q1",
        task_family=task_family,
        primary_evidence_type="text_span",
        question=question,
        answer_types=["freeform"],
    )


def test_single_paper_skips_decompose_and_stops_after_first_hit():
    retriever = FakeRetriever({"What is X?": [_result("p1")]})
    llm = FakeLLM(responses=["should not be used"])
    agent = IterativeAgent(retriever=retriever, llm=llm, max_steps=3, top_k=20)

    prediction = agent.run(_query("hidden_source_single_paper"))

    assert llm.calls == []
    assert [g["paper_id"] for g in prediction.gold_papers] == ["p1"]
    assert len(prediction.trace) == 1


def test_multi_paper_iterates_until_sufficient():
    retriever = FakeRetriever(
        {
            "q1": [_result("p1"), _result("p2")],
            "q2": [_result("p3")],
            "q3": [_result("p4")],
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1", "q2"]}),
            json.dumps({"subqueries": ["q3"]}),
        ]
    )
    agent = IterativeAgent(retriever=retriever, llm=llm, max_steps=3, top_k=20)

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.trace) == 2
    assert prediction.trace[0]["n_found"] == 3
    assert prediction.trace[1]["n_found"] == 4
    assert {g["paper_id"] for g in prediction.gold_papers} == {"p1", "p2", "p3", "p4"}
    assert len(llm.calls) == 2


def test_llm_failure_falls_back_to_question_as_is():
    retriever = FakeRetriever({"What is X?": [_result("p1")]})
    agent = IterativeAgent(retriever=retriever, llm=RaisingLLM(), max_steps=3, top_k=20)

    prediction = agent.run(_query("multi_paper", question="What is X?"))

    assert retriever.queries == ["What is X?"]
    assert [g["paper_id"] for g in prediction.gold_papers] == ["p1"]


def test_stops_when_no_new_papers_found():
    retriever = FakeRetriever(
        {
            "q1": [_result("p1")],
            "q2": [_result("p1")],  # 既知の paper のみ -> 増分なし
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1"]}),
            json.dumps({"subqueries": ["q2"]}),
            json.dumps({"subqueries": ["should-not-be-called"]}),
        ]
    )
    agent = IterativeAgent(retriever=retriever, llm=llm, max_steps=5, top_k=20)

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.trace) == 2
    assert len(llm.calls) == 2  # decompose + 1回目の refine のみ。2回目の refine は呼ばれない
    assert [g["paper_id"] for g in prediction.gold_papers] == ["p1"]


@pytest.mark.parametrize(
    "task_family,cutoff",
    [("hidden_source_single_paper", 2), ("multi_paper", 5)],
)
def test_cutoff_by_task_family(task_family, cutoff):
    papers = [_result(f"p{i}") for i in range(10)]
    retriever = FakeRetriever({"What is X?": papers})
    llm = FakeLLM(responses=[json.dumps({"subqueries": []})])
    agent = IterativeAgent(retriever=retriever, llm=llm, max_steps=1, top_k=20)

    prediction = agent.run(_query(task_family))

    assert len(prediction.gold_papers) == cutoff


def test_stagnation_patience_tolerates_one_zero_growth_step():
    retriever = FakeRetriever(
        {
            "q1": [_result("p1"), _result("p2")],
            "q2": [_result("p1"), _result("p2")],  # 増分なし（1回だけなら許容したい）
            "q3": [_result("p3"), _result("p4")],  # 増分あり -> 4件に到達して停止
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1"]}),
            json.dumps({"subqueries": ["q2"]}),
            json.dumps({"subqueries": ["q3"]}),
        ]
    )
    agent = IterativeAgent(
        retriever=retriever,
        llm=llm,
        max_steps=3,
        top_k=20,
        stagnation_patience=2,
    )

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.trace) == 3
    assert "q3" in retriever.queries
    assert {g["paper_id"] for g in prediction.gold_papers} == {"p1", "p2", "p3", "p4"}


def test_growth_threshold_counts_small_positive_growth_as_stagnant():
    retriever = FakeRetriever(
        {
            "q1": [_result("p1"), _result("p2")],
            "q2": [_result("p3")],  # 2件->3件、成長率0.5（growth_threshold以下なので停滞扱い）
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1"]}),
            json.dumps({"subqueries": ["q2"]}),
        ]
    )
    agent = IterativeAgent(
        retriever=retriever,
        llm=llm,
        max_steps=5,
        top_k=20,
        growth_threshold=0.5,
    )

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.trace) == 2
    assert {g["paper_id"] for g in prediction.gold_papers} == {"p1", "p2", "p3"}


def test_refine_prompt_includes_tried_subquery_history():
    retriever = FakeRetriever(
        {
            "q1": [_result("p1")],
            "q2": [_result("p2")],
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1"]}),
            json.dumps({"subqueries": ["q2"]}),
            json.dumps({"subqueries": []}),
        ]
    )
    agent = IterativeAgent(retriever=retriever, llm=llm, max_steps=5, top_k=20)

    agent.run(_query("multi_paper"))

    first_refine_prompt = llm.calls[1]
    second_refine_prompt = llm.calls[2]
    assert "q1" in first_refine_prompt
    assert "q2" not in first_refine_prompt
    assert "q1" in second_refine_prompt
    assert "q2" in second_refine_prompt


def test_refine_prompt_orders_snippets_by_score_desc_not_insertion_order():
    retriever = FakeRetriever(
        {
            "q1": [_result_with_score("pLow", 0.1)],
            "q2": [_result_with_score("pHigh", 0.9)],
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1", "q2"]}),
            json.dumps({"subqueries": []}),
        ]
    )
    agent = IterativeAgent(retriever=retriever, llm=llm, max_steps=3, top_k=20)

    agent.run(_query("multi_paper"))

    refine_prompt = llm.calls[1]
    assert refine_prompt.index("pHigh") < refine_prompt.index("pLow")


def test_dynamic_sufficiency_overrides_fixed_threshold():
    retriever = FakeRetriever(
        {
            "q1": [_result("p1")],
            "q2": [_result("p2")],
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1", "q2"]}),
            json.dumps({"n_papers": 2}),
        ]
    )
    agent = IterativeAgent(
        retriever=retriever, llm=llm, max_steps=3, top_k=20, dynamic_sufficiency=True
    )

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.trace) == 1
    assert len(llm.calls) == 2
    assert {g["paper_id"] for g in prediction.gold_papers} == {"p1", "p2"}


def test_dynamic_sufficiency_estimate_also_raises_final_cutoff():
    """見積もり本数が固定カットオフ(multi_paper=5)を超える場合、最終提出も見積もりに従うべき

    (以前は反復停止の判定にしか使われず、最終提出は固定5件で切り捨てられていた)。
    """
    retriever = FakeRetriever({"q1": [_result(f"p{i}") for i in range(6)]})
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1"]}),
            json.dumps({"n_papers": 6}),
        ]
    )
    agent = IterativeAgent(
        retriever=retriever, llm=llm, max_steps=3, top_k=20, dynamic_sufficiency=True
    )

    prediction = agent.run(_query("multi_paper"))

    assert len(prediction.gold_papers) == 6


def test_dynamic_sufficiency_falls_back_to_fixed_dict_on_unparseable_estimate():
    retriever = FakeRetriever(
        {
            "q1": [_result("p1")],
            "q2": [_result("p2")],
        }
    )
    llm = FakeLLM(
        responses=[
            json.dumps({"subqueries": ["q1", "q2"]}),
            "not json at all",
            json.dumps({"subqueries": []}),
        ]
    )
    agent = IterativeAgent(
        retriever=retriever, llm=llm, max_steps=3, top_k=20, dynamic_sufficiency=True
    )

    prediction = agent.run(_query("multi_paper"))

    # 固定辞書(multi_paper=4)にフォールバックしたため、2件では不十分と判定して refine まで呼ばれる
    assert len(llm.calls) == 3
    assert {g["paper_id"] for g in prediction.gold_papers} == {"p1", "p2"}


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which papers propose contrastive pretraining?", True),
        ("List all papers that evaluate on SQuAD.", True),
        ("Compare BERT and RoBERTa across benchmarks.", True),
        ("How many papers use reinforcement learning?", True),
        ("What is the accuracy of model X on dataset Y?", False),
    ],
)
def test_is_enumeration_or_comparison_detects_expected_patterns(question, expected):
    assert _is_enumeration_or_comparison(question) is expected


def test_enumeration_fanout_widens_subquery_count_hint_in_prompt():
    retriever = FakeRetriever({})
    llm = FakeLLM(responses=[json.dumps({"subqueries": []})])
    agent = IterativeAgent(
        retriever=retriever, llm=llm, max_steps=1, top_k=20, enumeration_fanout=True
    )

    agent.run(_query("multi_paper", question="Which papers propose contrastive pretraining?"))

    assert "4〜6" in llm.calls[0]


def test_enumeration_fanout_off_by_default_keeps_original_hint():
    retriever = FakeRetriever({})
    llm = FakeLLM(responses=[json.dumps({"subqueries": []})])
    agent = IterativeAgent(retriever=retriever, llm=llm, max_steps=1, top_k=20)

    agent.run(_query("multi_paper", question="Which papers propose contrastive pretraining?"))

    assert "2〜4" in llm.calls[0]
