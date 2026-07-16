"""ReadingAgent のテスト。

一番大事なのは「LLM が足りないと言ったら本当に再検索するか」。IterativeAgent は
ここが本数カウントだったため反復が回らなかった（iterative.py の docstring 参照）。
"""

from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.agent.reading import ReadingAgent
from littraceqa.di_pipeline.contracts import Query, RetrievalResult
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _query(**kwargs) -> Query:
    # task_family を明示して TaskFamilyClassifier が LLM を呼ばないようにする。
    # そうしないと分類用のプロンプトが FakeLLM の応答を1つ消費してしまい、
    # エージェント本体のロジックを見るテストにならない。
    base = {
        "query_id": "q_1",
        "question": "Which papers report FID on CIFAR-10?",
        "answer_types": ["table"],
        "task_family": "multi_paper",
    }
    base.update(kwargs)
    return Query(**base)


def _result(i: int, paper: str, score: float, **metadata) -> RetrievalResult:
    base = {"title": f"Paper {paper}", "venue": "NeurIPS", "year": 2025}
    base.update(metadata)
    return RetrievalResult(
        chunk_id=f"{paper}#c{i:02d}",
        paper_id=paper,
        score=score,
        text=f"body of {paper} chunk {i}",
        chunk_type=base.pop("chunk_type", "text_span"),
        metadata=base,
    )


class _StubRetriever:
    """サブクエリごとに違う論文を返すスタブ。呼ばれたクエリを記録する。"""

    def __init__(self, by_query: dict[str, list[RetrievalResult]] | None = None):
        self.by_query = by_query or {}
        self.calls: list[str] = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append(question)
        if question in self.by_query:
            return self.by_query[question][:top_k]
        return [_result(0, f"p{i}", 1.0 / (i + 1)) for i in range(5)][:top_k]


def _judge(papers, sufficient, missing=""):
    return json.dumps(
        {
            "papers": [
                {"paper_id": p, "evidence_chunk_ids": [f"{p}#c00"]} for p in papers
            ],
            "sufficient": sufficient,
            "missing": missing,
        }
    )


def _subqueries(*values):
    return json.dumps({"subqueries": list(values)})


def test_loop_iterates_when_llm_says_insufficient():
    """LLM が sufficient=false を返したら、不足分で再検索して2周目に入る。

    IterativeAgent はここが「本数 >= しきい値」だったため、初回検索で20件返った
    時点で常に打ち切られ、_refine が一度も呼ばれなかった。
    """
    retriever = _StubRetriever(
        {
            "step2-sq": [_result(0, "pX", 9.0)],
        }
    )
    llm = FakeLLM(
        responses=[
            _subqueries("step1-sq"),  # _decompose
            _judge(["p0"], sufficient=False, missing="ECM-XL の FID がまだ無い"),
            _subqueries("step2-sq"),  # _refine
            _judge(["p0", "pX"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=3, top_k=5)
    prediction = agent.run(_query())

    assert len(prediction.trace) == 2, "2ステップ回っていない"
    assert prediction.trace[0]["sufficient"] is False
    assert prediction.trace[0]["missing"] == "ECM-XL の FID がまだ無い"
    assert prediction.trace[1]["sufficient"] is True
    # 不足を埋めるための再検索が実際に走っている
    assert "step2-sq" in retriever.calls
    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0", "pX"]


def test_stops_as_soon_as_llm_says_sufficient():
    retriever = _StubRetriever()
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=3, top_k=5)
    prediction = agent.run(_query())

    assert len(prediction.trace) == 1
    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0"]


def test_respects_max_steps():
    """LLM がずっと sufficient=false でも max_steps で止まる。"""
    retriever = _StubRetriever()
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=False, missing="まだ足りない"),
            _subqueries("sq2"),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=2, top_k=5)
    prediction = agent.run(_query())
    assert len(prediction.trace) == 2


def test_builds_evidence_from_cited_chunks():
    """LLM が根拠として挙げたチャンクの metadata から Evidence を組む。"""
    retriever = _StubRetriever(
        {
            "sq": [
                _result(0, "p0", 5.0, chunk_type="table", page=6, table_id="Table 4"),
                _result(1, "p0", 4.0, chunk_type="text_span", page=2),
            ]
        }
    )
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, top_k=5)
    prediction = agent.run(_query())

    assert len(prediction.evidence) == 1
    evidence = prediction.evidence[0]
    assert evidence.paper_id == "p0"
    assert evidence.source_type == "table"
    assert evidence.locator.page == 6
    assert evidence.locator.table_id == "Table 4"


def test_reads_full_chunk_text_not_a_200_char_stub():
    """LLM に渡す抜粋は snippet_chars まで。200文字ではタイトルしか読めない。"""
    long_text = "x" * 5000
    result = _result(0, "p0", 1.0)
    result.text = long_text
    retriever = _StubRetriever({"sq": [result]})
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, top_k=5, snippet_chars=1800)
    agent.run(_query())

    judge_prompt = llm.calls[1]
    assert "x" * 1800 in judge_prompt
    assert "x" * 1801 not in judge_prompt


def test_drops_hallucinated_paper_and_chunk_ids():
    """候補一覧に無い paper_id / chunk_id は捨てる。"""
    retriever = _StubRetriever({"sq": [_result(0, "p0", 1.0)]})
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            json.dumps(
                {
                    "papers": [
                        {"paper_id": "p0", "evidence_chunk_ids": ["p0#c00", "p0#c99"]},
                        {"paper_id": "ghost", "evidence_chunk_ids": ["ghost#c00"]},
                    ],
                    "sufficient": True,
                    "missing": "",
                }
            ),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, top_k=5)
    prediction = agent.run(_query())

    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0"]
    assert len(prediction.evidence) == 1  # 実在しない p0#c99 は落ちる


@pytest.mark.parametrize(
    "task_family,expected", [("hidden_source_single_paper", 2), ("multi_paper", 5)]
)
def test_falls_back_to_cutoff_when_llm_output_is_unusable(task_family, expected):
    """LLM が一度も使える判定を返さなければ、順位カットオフで出す。"""
    retriever = _StubRetriever()
    llm = FakeLLM(responses=["not json"])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, top_k=5)
    prediction = agent.run(_query(task_family=task_family))

    assert len(prediction.gold_papers) == expected
    assert prediction.evidence == []
