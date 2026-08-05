from __future__ import annotations

from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.contracts import RetrievalResult


def _result(**metadata) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="p1#alg0002",
        paper_id="p1",
        score=1.0,
        text="Algorithm body",
        chunk_type="equation_algorithm",
        metadata=metadata,
    )


def test_evidence_from_result_preserves_algorithm_id():
    evidence = evidence_from_result(
        _result(page=4, section="Method", algorithm_id="Algorithm 2")
    )

    assert evidence.locator.page == 4
    assert evidence.locator.section == "Method"
    assert evidence.locator.algorithm_id == "Algorithm 2"
    assert evidence.locator.equation_id is None


def test_evidence_from_result_keeps_equation_and_algorithm_ids_distinct():
    evidence = evidence_from_result(
        _result(
            page=5,
            equation_id="Equation 6",
            algorithm_id="Algorithm 2",
            citation_id="24",
        )
    )

    assert evidence.locator.equation_id == "Equation 6"
    assert evidence.locator.algorithm_id == "Algorithm 2"
    assert evidence.locator.citation_id == "24"
