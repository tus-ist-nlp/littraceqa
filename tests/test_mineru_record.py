from __future__ import annotations

from littraceqa.mineru_record import (
    coarse_locator,
    readable_image_path,
    record_source_type,
    submission_evidence_eligible,
)


def _record(chunk_type: str, **metadata):
    return {
        "paper_id": "p1",
        "chunk_id": "p1#1",
        "chunk_type": chunk_type,
        "text": "evidence",
        "metadata": metadata,
    }


def test_title_and_reference_chunks_map_to_official_source_types():
    assert record_source_type(_record("title_abstract", page=1)) == "text_span"
    assert (
        record_source_type(_record("text_span", page=9, section="References"))
        == "citation_context"
    )


def test_table_locator_and_submission_eligibility_share_one_contract():
    valid = _record("table", page=3, table_id="Table 2")
    missing_id = _record("table", page=3)

    assert coarse_locator(valid) == {"page": 3, "table_id": "Table 2"}
    assert submission_evidence_eligible(valid) is True
    assert submission_evidence_eligible(missing_id) is False


def test_boolean_and_zero_pages_are_not_valid_submission_locators():
    assert submission_evidence_eligible(_record("text_span", page=True)) is False
    assert submission_evidence_eligible(_record("text_span", page=0)) is False


def test_readable_image_path_requires_an_existing_table_or_figure_image(tmp_path):
    image = tmp_path / "table.png"
    image.write_bytes(b"image")

    assert readable_image_path(_record("table", page=1, image_path=str(image))) == str(
        image
    )
    assert readable_image_path(_record("text_span", page=1, image_path=str(image))) == ""
    assert readable_image_path(_record("figure", page=1, image_path="missing.png")) == ""
