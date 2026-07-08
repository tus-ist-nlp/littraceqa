"""compute_modality_hit_rates() のモダリティ別 hit/miss 集計ロジックのテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from litqa.contracts import RetrievalResult
from measure_modality_gap import compute_modality_hit_rates


def _result(paper_id: str, page: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper_id}#c{page:04d}",
        paper_id=paper_id,
        score=1.0,
        text="",
        chunk_type="text_span",
        metadata={"page": page},
    )


def _evidence(source_type: str, paper_id: str, page: int | None, **locator_extra) -> dict:
    return {
        "evidence_id": f"ev_{source_type}",
        "paper_id": paper_id,
        "source_type": source_type,
        "locator": {"page": page, **locator_extra},
    }


def test_hit_and_miss_are_counted_per_source_type():
    gold_records = [
        {
            "query_id": "q1",
            "question": "q1?",
            "evidence": [_evidence("figure", "p1", page=2, figure_id="Figure 1")],
        },
        {
            "query_id": "q2",
            "question": "q2?",
            "evidence": [_evidence("equation_algorithm", "p2", page=9, equation_id="Equation 3")],
        },
    ]

    def retrieve_fn(question: str) -> list[RetrievalResult]:
        if question == "q1?":
            return [_result("p1", 2)]
        return [_result("p2", 1)]

    stats = compute_modality_hit_rates(gold_records, retrieve_fn)

    assert stats["figure"]["total"] == 1
    assert stats["figure"]["hit"] == 1
    assert stats["figure"]["misses"] == []

    assert stats["equation_algorithm"]["total"] == 1
    assert stats["equation_algorithm"]["hit"] == 0
    assert stats["equation_algorithm"]["misses"] == [("q2", "ev_equation_algorithm", "Equation 3")]


def test_no_page_locator_is_excluded_from_hit_and_miss():
    gold_records = [
        {
            "query_id": "q1",
            "question": "q1?",
            "evidence": [_evidence("citation_context", "p1", page=None, citation_id="c1")],
        }
    ]

    stats = compute_modality_hit_rates(gold_records, retrieve_fn=lambda q: [])

    assert stats["citation_context"]["total"] == 0
    assert stats["citation_context"]["hit"] == 0
    assert stats["citation_context"]["no_page_locator"] == 1
    assert stats["citation_context"]["misses"] == []


def test_multiple_evidence_items_reuse_a_single_retrieval_call():
    gold_records = [
        {
            "query_id": "q1",
            "question": "q1?",
            "evidence": [
                _evidence("figure", "p1", page=2, figure_id="Figure 1"),
                _evidence("text_span", "p1", page=3, section="Method"),
            ],
        }
    ]

    calls: list[str] = []

    def retrieve_fn(question: str) -> list[RetrievalResult]:
        calls.append(question)
        return [_result("p1", 2), _result("p1", 3)]

    stats = compute_modality_hit_rates(gold_records, retrieve_fn)

    assert calls == ["q1?"]
    assert stats["figure"]["hit"] == 1
    assert stats["text_span"]["hit"] == 1
