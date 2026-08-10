from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.paper_tables import (
    PaperEvidenceDocument,
    PaperTable,
)
from littraceqa.di_pipeline.select.citation_table_coverage import (
    CitationTableOpenSetRefiner,
    citation_table_candidate_ids,
    parse_citation_table_condition,
)
from littraceqa.di_pipeline.select.selector import PaperSelection
from littraceqa.di_pipeline.select.table_coverage import EvidenceCoverageRefiner

_QUESTION = (
    "Which ACL 2024 papers cite BaseNet (Base Networks, ICML2020) and use it "
    "as a baseline in their main comparison table?"
)


class StubDocumentSource:
    def __init__(self, documents: dict[str, PaperEvidenceDocument]) -> None:
        self.documents = documents

    def document(self, paper_id: str) -> PaperEvidenceDocument:
        return self.documents.get(paper_id, PaperEvidenceDocument(paper_id, "", (), ()))

    def tables(self, paper_id: str) -> tuple[PaperTable, ...]:
        return self.document(paper_id).tables


def _query(question: str = _QUESTION) -> Query:
    return Query(
        query_id="q",
        question=question,
        answer_types=["table"],
        table_schema=[
            {"name": "Paper Title", "type": "string", "is_row_key": True}
        ],
    )


def _table(paper_id: str, method: str, *, table_id: str = "Table 1") -> PaperTable:
    rows = (
        ("Method", "Score", "Accuracy"),
        (method, "10.0", "20.0"),
        ("Our model", "30.0", "40.0"),
    )
    return PaperTable(paper_id, table_id, "Main comparison results", rows, "")


def _document(
    paper_id: str,
    method: str = "BaseNet [7]",
    reference: str = "[7] A. Author. Base Networks. ICML, 2020.",
    *,
    tables: tuple[PaperTable, ...] | None = None,
) -> PaperEvidenceDocument:
    return PaperEvidenceDocument(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        text_blocks=(),
        tables=tables if tables is not None else (_table(paper_id, method),),
        reference_entries=(reference,),
    )


def _metadata(*paper_ids: str) -> dict[str, dict[str, object]]:
    return {
        paper_id: {"venue": "ACL", "year": 2024}
        for paper_id in paper_ids
    }


def _selection(paper_id: str = "p1") -> PaperSelection:
    return PaperSelection((paper_id,), expected_count=1, reason="open_set")


def test_parses_a_strict_citation_table_condition():
    condition = parse_citation_table_condition(_query())

    assert condition is not None
    assert (condition.venue, condition.year) == ("ACL", 2024)
    assert condition.alias == "BaseNet"
    assert condition.cited_title == "Base Networks"
    assert (condition.cited_venue, condition.cited_year) == ("ICML", 2020)


def test_parses_the_outer_scope_before_a_spaced_citation_venue():
    query = _query(
        "Which ICML 2024 papers cite BaseNet (Base Networks, ACL 2020) and "
        "use it as a baseline in their main comparison table?"
    )

    condition = parse_citation_table_condition(query)

    assert condition is not None
    assert (condition.venue, condition.year) == ("ICML", 2024)
    assert (condition.cited_venue, condition.cited_year) == ("ACL", 2020)


def test_candidate_ids_include_only_supported_questions():
    queries = {
        "supported": _query(),
        "other": _query("Which ACL 2024 papers report a comparison table?"),
    }
    rankings = {"supported": ["p1", "p2"], "other": ["p3"]}

    assert citation_table_candidate_ids(queries, rankings) == {"p1", "p2"}


def test_expands_to_all_verified_papers_in_retrieval_order():
    source = StubDocumentSource(
        {
            "p1": _document("p1"),
            "p2": _document(
                "p2",
                "BaseNet-Large [8]",
                "[8] B. Author. Base Networks. ICML, 2020.",
            ),
            "citation-only": _document("citation-only", tables=()),
        }
    )
    refiner = CitationTableOpenSetRefiner(
        source,
        _metadata("p1", "p2", "citation-only"),
    )

    result = refiner.refine(
        _query(),
        ["p2", "citation-only", "p1"],
        _selection("p1"),
    )

    assert result.paper_ids == ("p2", "p1")
    assert result.expected_count == 2
    assert result.reason == "open_set+citation_table_coverage"


def test_accepts_an_exact_markerless_baseline_with_one_matching_reference():
    source = StubDocumentSource(
        {
            "p1": _document("p1"),
            "p2": _document("p2", "BaseNet"),
        }
    )
    result = CitationTableOpenSetRefiner(
        source,
        _metadata("p1", "p2"),
    ).refine(_query(), ["p1", "p2"], _selection())

    assert result.paper_ids == ("p1", "p2")


def test_accepts_a_reference_with_the_venue_written_in_full():
    full_venue_question = _query(
        "Which CVPR 2025 papers cite BaseNet "
        "(Base Networks, CVPR2023) and use it as a baseline in their main "
        "comparison table?"
    )
    full_venue_reference = (
        "[7] A. Author. Base Networks. In Proceedings of the IEEE/CVF "
        "Conference on Computer Vision and Pattern Recognition, 2023."
    )
    source = StubDocumentSource(
        {
            "p1": _document("p1", reference=full_venue_reference),
            "p2": _document("p2", reference=full_venue_reference),
        }
    )
    metadata = {
        paper_id: {"venue": "CVPR", "year": 2025}
        for paper_id in ("p1", "p2")
    }

    result = CitationTableOpenSetRefiner(source, metadata).refine(
        full_venue_question,
        ["p1", "p2"],
        _selection(),
    )

    assert result.paper_ids == ("p1", "p2")


def test_accepts_one_title_year_match_when_the_venue_has_an_ocr_split():
    question = _query(
        "Which CVPR 2025 papers cite BaseNet "
        "(Base Networks, CVPR2023) and use it as a baseline in their main "
        "comparison table?"
    )
    reference = (
        "[7] A. Author. Base Networks. In Proceedings of the Conference on "
        "Com puter Vision and Pattern Recognition, 2023."
    )
    source = StubDocumentSource(
        {
            "p1": _document("p1", reference=reference),
            "p2": _document("p2", reference=reference),
        }
    )
    metadata = {
        paper_id: {"venue": "CVPR", "year": 2025}
        for paper_id in ("p1", "p2")
    }

    result = CitationTableOpenSetRefiner(source, metadata).refine(
        question,
        ["p1", "p2"],
        _selection(),
    )

    assert result.paper_ids == ("p1", "p2")


def test_rejects_a_reference_from_the_wrong_cited_venue():
    question = _query(
        "Which CVPR 2025 papers cite BaseNet "
        "(Base Networks, CVPR2023) and use it as a baseline in their main "
        "comparison table?"
    )
    wrong_reference = "[7] A. Author. Base Networks. ICML, 2023."
    source = StubDocumentSource(
        {
            "p1": _document("p1", reference=wrong_reference),
            "p2": _document("p2", reference=wrong_reference),
        }
    )
    metadata = {
        paper_id: {"venue": "CVPR", "year": 2025}
        for paper_id in ("p1", "p2")
    }
    original = _selection()

    result = CitationTableOpenSetRefiner(source, metadata).refine(
        question,
        ["p1", "p2"],
        original,
    )

    assert result is original


def test_combined_evidence_refiner_runs_citation_coverage_last():
    source = StubDocumentSource(
        {"p1": _document("p1"), "p2": _document("p2", "BaseNet")}
    )

    result = EvidenceCoverageRefiner(
        source,
        evidence_source=source,
        paper_metadata=_metadata("p1", "p2"),
    ).refine(_query(), ["p1", "p2"], _selection())

    assert result.paper_ids == ("p1", "p2")


def test_rejects_reference_only_table_only_and_caption_only_candidates():
    caption_only = PaperTable(
        "caption",
        "Table 1",
        "BaseNet comparison",
        (("Method", "Score"), ("Other", "10.0"), ("Ours", "20.0")),
        "",
    )
    source = StubDocumentSource(
        {
            "seed": _document("seed"),
            "reference": _document("reference", tables=()),
            "table": _document("table", reference="[7] A different paper."),
            "caption": _document("caption", tables=(caption_only,)),
        }
    )
    original = _selection("seed")

    result = CitationTableOpenSetRefiner(
        source,
        _metadata("seed", "reference", "table", "caption"),
    ).refine(_query(), ["seed", "reference", "table", "caption"], original)

    assert result is original


def test_rejects_an_embedded_alias_and_a_mismatched_citation_number():
    source = StubDocumentSource(
        {
            "seed": _document("seed"),
            "embedded": _document("embedded", "MMTL-BaseNet [7]"),
            "wrong-number": _document(
                "wrong-number",
                "BaseNet [7]",
                "[8] A. Author. Base Networks. ICML, 2020.",
            ),
        }
    )
    original = _selection("seed")

    result = CitationTableOpenSetRefiner(
        source,
        _metadata("seed", "embedded", "wrong-number"),
    ).refine(_query(), ["seed", "embedded", "wrong-number"], original)

    assert result is original


def test_rejects_wrong_metadata_and_late_tables():
    late_tables = (
        _table("late", "Other [1]"),
        _table("late", "Other [1]", table_id="Table 2"),
        _table("late", "BaseNet [7]", table_id="Table 3"),
    )
    source = StubDocumentSource(
        {
            "seed": _document("seed"),
            "wrong-venue": _document("wrong-venue"),
            "late": _document("late", tables=late_tables),
        }
    )
    metadata = _metadata("seed", "late")
    metadata["wrong-venue"] = {"venue": "EMNLP", "year": 2024}
    original = _selection("seed")

    result = CitationTableOpenSetRefiner(source, metadata).refine(
        _query(),
        ["seed", "wrong-venue", "late"],
        original,
    )

    assert result is original


def test_falls_back_when_the_seed_is_unverified_or_too_many_papers_match():
    documents = {f"p{index}": _document(f"p{index}") for index in range(1, 4)}
    source = StubDocumentSource(documents)
    refiner = CitationTableOpenSetRefiner(
        source,
        _metadata(*documents),
        max_papers=2,
    )

    unverified = _selection("missing")
    assert refiner.refine(_query(), list(documents), unverified) is unverified

    too_many = _selection("p1")
    assert refiner.refine(_query(), list(documents), too_many) is too_many


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"candidate_limit": True}, TypeError),
        ({"candidate_limit": 0}, ValueError),
        ({"max_papers": "10"}, TypeError),
        ({"max_papers": 1}, ValueError),
    ],
)
def test_rejects_invalid_limits(kwargs, error):
    with pytest.raises(error):
        CitationTableOpenSetRefiner(StubDocumentSource({}), {}, **kwargs)


def test_unsupported_schema_and_question_forms_do_not_trigger():
    wrong_schema = Query(
        query_id="q",
        question=_QUESTION,
        answer_types=["table"],
        table_schema=[{"name": "Method", "type": "string", "is_row_key": True}],
    )
    missing_conjunction = _query(
        "Which ACL 2024 papers cite BaseNet (Base Networks, ICML2020)?"
    )

    assert parse_citation_table_condition(wrong_schema) is None
    assert parse_citation_table_condition(missing_conjunction) is None
