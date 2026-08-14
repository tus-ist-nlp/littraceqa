from __future__ import annotations

import base64
import os

import littraceqa.mineru_record as mineru_record

from littraceqa.mineru_record import (
    ImageValidationError,
    coarse_locator,
    query_aware_prediction_locator,
    readable_image_path,
    record_source_type,
    recover_split_caption_table_records,
    split_caption_table_locator_overrides,
    submission_evidence_eligible,
    validate_image_file,
)


VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
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


def test_page_wins_and_section_is_the_official_location_fallback():
    assert coarse_locator(
        _record("text_span", page=3, section="  Results  ")
    ) == {"page": 3}
    section_only = _record("text_span", section="  Results  ")
    assert coarse_locator(section_only) == {"section": "Results"}
    assert submission_evidence_eligible(section_only) is True


def test_all_official_object_ids_are_preserved_in_minimal_locators():
    assert coarse_locator(
        _record("figure", page=2, figure_id=" Figure 4 ")
    ) == {"page": 2, "figure_id": "Figure 4"}
    assert coarse_locator(
        _record("equation_algorithm", page=5, equation_id=" Equation 6 ")
    ) == {"page": 5, "equation_id": "Equation 6"}
    assert coarse_locator(
        _record("equation_algorithm", page=6, algorithm_id=" Algorithm 2 ")
    ) == {"page": 6, "algorithm_id": "Algorithm 2"}
    assert coarse_locator(
        _record("citation_context", citation_id=" 24 ")
    ) == {"citation_id": "24"}
    assert coarse_locator(_record("table", page=3, table_id=2)) == {
        "page": 3,
        "table_id": "2",
    }


def test_equation_id_takes_official_precedence_over_algorithm_id():
    record = _record(
        "equation_algorithm",
        section="Method",
        equation_id="Equation 3",
        algorithm_id="Algorithm 1",
    )

    assert coarse_locator(record) == {
        "section": "Method",
        "equation_id": "Equation 3",
    }
    assert submission_evidence_eligible(record) is True


def test_citation_object_id_is_sufficient_when_page_and_section_are_absent():
    assert submission_evidence_eligible(
        _record("citation_context", citation_id="Citation 24")
    ) is True
    assert submission_evidence_eligible(_record("citation_context")) is False


def test_table_and_figure_accept_section_fallback_but_require_visible_object_id():
    assert submission_evidence_eligible(
        _record("table", section="Results", table_id="Table 2")
    ) is True
    assert submission_evidence_eligible(
        _record("table", table_id="Table 2")
    ) is False
    assert submission_evidence_eligible(
        _record("figure", page=2, figure_id='""')
    ) is False


def test_boolean_and_zero_pages_are_not_valid_submission_locators():
    assert submission_evidence_eligible(_record("text_span", page=True)) is False
    assert submission_evidence_eligible(_record("text_span", page=0)) is False


def test_query_aware_prediction_locator_recovers_later_merged_table_caption():
    record = _record(
        "table",
        page=6,
        table_id="Table 3",
        title="500x C ompressor: Generalized Prompt Compression",
    )
    record["text"] = """[ACL 2025] 500x C ompressor
Table 3: In-domain QA results on ArxivQA. F1 deltas compare 500xCompressor and ICAE.
Table 4: Cross-domain QA results on NaturalQuestions (NaturalQ), RACE, and TriviaQA.
| Dataset | NaturalQ | RACE |
| Ours500→1 | 41.36 | 21.37 |
| ICAE500→1 | 26.65 | 14.24 |
| Absolute delta | 14.70 | 7.12 |
"""

    locator = query_aware_prediction_locator(
        record,
        "How much does 500xCompressor outperform ICAE on the NaturalQ "
        "benchmark under the 500-to-1 setting?",
    )

    assert coarse_locator(record) == {"page": 6, "table_id": "Table 3"}
    assert locator == {"page": 6, "table_id": "Table 4"}


def test_query_aware_prediction_locator_does_not_change_single_caption_record():
    record = _record("table", page=3, table_id="Table 2")
    record["text"] = "Table 2: NaturalQ results.\n| Method | F1 |\n| M | 42 |"

    assert query_aware_prediction_locator(record, "What is NaturalQ F1?") == {
        "page": 3,
        "table_id": "Table 2",
    }


def test_query_aware_prediction_locator_keeps_ambiguous_merged_metadata_id():
    record = _record("table", page=3, table_id="Table 2")
    record["text"] = """Table 2: Results for AlphaSet.
| Method | F1 | A | 40 |
Table 3: Results for BetaSet.
| Method | F1 | B | 41 |
"""

    assert query_aware_prediction_locator(
        record, "Compare AlphaSet and BetaSet."
    ) == {"page": 3, "table_id": "Table 2"}


def test_query_aware_prediction_locator_recovers_explicit_merged_figure_id():
    record = _record("figure", page=4, figure_id="Figure 2")
    record["text"] = """Figure 2: Training overview.
Figure 3: NaturalQ accuracy by model size.
"""

    assert query_aware_prediction_locator(
        record, "What trend is visible in Figure 3?"
    ) == {"page": 4, "figure_id": "Figure 3"}


def test_query_aware_prediction_locator_requires_metadata_to_match_a_caption():
    record = _record("table", page=3, table_id="Table 9")
    record["text"] = """Table 2: AlphaSet results.
Table 3: NaturalQ results.
"""

    assert query_aware_prediction_locator(record, "What is NaturalQ F1?") == {
        "page": 3,
        "table_id": "Table 9",
    }


def test_adjacent_split_captions_recover_previous_and_current_table_ids():
    previous = _record("table", page=17, table_id=None)
    previous["chunk_id"] = "p1#tab13-body"
    previous["text"] = """[NAACL 2025] Track-SQL
| Dataset | Total time(s) |
| --- | --- |
| SParC | 240.348±1.45 |
| CoSQL | 214.456±2.56 |
"""
    current = _record("table", page=17, table_id="Table 13")
    current["chunk_id"] = "p1#tab14-body"
    current["text"] = """[NAACL 2025] Track-SQL
Table 13: Inference time performance of the Track-SQL framework.
Table 14: Memory Costs of Training and Inference in the Track-SQL Framework.
| Metric | SESE(Inference) |
| --- | --- |
| Graphics Memory(GB) | 2.235 |
"""

    recovered = recover_split_caption_table_records([previous, current])

    assert split_caption_table_locator_overrides([previous, current]) == {
        "p1#tab13-body": "Table 13",
        "p1#tab14-body": "Table 14",
    }
    assert coarse_locator(recovered[0]) == {"page": 17, "table_id": "Table 13"}
    assert coarse_locator(recovered[1]) == {"page": 17, "table_id": "Table 14"}
    assert all(submission_evidence_eligible(record) for record in recovered)
    assert previous["metadata"]["table_id"] is None
    assert current["metadata"]["table_id"] == "Table 13"


def test_adjacent_split_caption_recovery_fails_closed_on_structural_mismatch():
    previous = _record("table", page=17, table_id=None)
    previous["chunk_id"] = "p1#previous"
    previous["text"] = "| Dataset | Value |\n| --- | --- |\n| A | 1 |"
    current = _record("table", page=18, table_id="Table 13")
    current["chunk_id"] = "p1#current"
    current["text"] = """Table 13: Prior table.
Table 14: Current table.
| Dataset | Value |
| --- | --- |
| B | 2 |
"""

    assert split_caption_table_locator_overrides([previous, current]) == {}

    current["metadata"]["page"] = 17
    current["metadata"]["table_id"] = "Table 12"
    assert split_caption_table_locator_overrides([previous, current]) == {}

    current["metadata"]["table_id"] = "Table 13"
    current["text"] = """Unrelated prose before the caption.
Table 13: Prior table.
Table 14: Current table.
| Dataset | Value |
| --- | --- |
| B | 2 |
"""
    assert split_caption_table_locator_overrides([previous, current]) == {}


def test_readable_image_path_requires_an_existing_table_or_figure_image(tmp_path):
    image = tmp_path / "table.png"
    image.write_bytes(VALID_PNG)

    assert readable_image_path(_record("table", page=1, image_path=str(image))) == str(
        image
    )
    assert readable_image_path(_record("text_span", page=1, image_path=str(image))) == ""
    assert readable_image_path(_record("figure", page=1, image_path="missing.png")) == ""


def test_readable_image_path_rejects_existing_corrupt_image(tmp_path):
    image = tmp_path / "corrupt.png"
    image.write_bytes(b"not-an-image")

    assert readable_image_path(_record("figure", page=1, image_path=str(image))) == ""


def test_readable_image_path_reuses_decode_until_file_stat_changes(
    tmp_path, monkeypatch
):
    image = tmp_path / "cached.png"
    image.write_bytes(VALID_PNG)
    record = _record("figure", page=1, image_path=str(image))
    decode_calls = 0
    original_decode = mineru_record._decode_and_validate_image

    def counting_decode(payload, *, source):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(payload, source=source)

    mineru_record._cached_image_is_valid.cache_clear()
    monkeypatch.setattr(
        mineru_record, "_decode_and_validate_image", counting_decode
    )

    assert readable_image_path(record) == str(image)
    assert readable_image_path(record) == str(image)
    assert decode_calls == 1

    stat = image.stat()
    os.utime(
        image,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )
    assert readable_image_path(record) == str(image)
    assert decode_calls == 2
    mineru_record._cached_image_is_valid.cache_clear()


def test_readable_image_path_revalidates_same_size_content_change(tmp_path):
    image = tmp_path / "replaced.png"
    image.write_bytes(VALID_PNG)
    record = _record("table", page=1, image_path=str(image))
    mineru_record._cached_image_is_valid.cache_clear()

    assert readable_image_path(record) == str(image)

    # Keep the byte count unchanged so invalidation cannot rely on size alone.
    image.write_bytes(b"x" * len(VALID_PNG))
    assert image.stat().st_size == len(VALID_PNG)
    assert readable_image_path(record) == ""
    mineru_record._cached_image_is_valid.cache_clear()


def test_image_validation_uses_content_not_extension(tmp_path):
    image = tmp_path / "actually-jpeg.jpg"
    image.write_bytes(VALID_PNG)

    assert validate_image_file(image) == "image/png"


def test_image_validation_rejects_header_only_png(tmp_path):
    image = tmp_path / "truncated.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nheader-only")

    try:
        validate_image_file(image)
    except ImageValidationError as exc:
        assert "unsupported/corrupt image" in str(exc)
    else:
        raise AssertionError("truncated PNG must be rejected")


def test_image_validation_rejects_corrupt_compressed_pixels(tmp_path):
    image = tmp_path / "bad-crc.png"
    corrupt = bytearray(VALID_PNG)
    corrupt[corrupt.index(b"IDAT") + 4] ^= 0x01
    image.write_bytes(corrupt)

    try:
        validate_image_file(image)
    except ImageValidationError as exc:
        assert "unsupported/corrupt image" in str(exc)
    else:
        raise AssertionError("PNG with corrupt image data must be rejected")
