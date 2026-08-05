from __future__ import annotations

import pytest

from littraceqa.validate_submission import CheckCounter, validate


def _input() -> dict:
    return {
        "query_id": "q1",
        "question": "question",
        "answer_types": ["multiple_choice", "table"],
        "table_schema": [
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "number", "is_row_key": False},
        ],
    }


def _prediction() -> dict:
    return {
        "query_id": "q1",
        "gold_papers": [{"paper_id": "p1"}],
        "evidence": [
            {
                "paper_id": "p1",
                "source_type": "table",
                "locator": {"page": 2, "table_id": "Table 1"},
            }
        ],
        "answer": {
            "multiple_choice": {"gold": "A"},
            "table": {"rows": [{"Method": "X", "Score": 91.2}]},
        },
    }


def test_strict_validator_accepts_official_shape():
    checks = CheckCounter()

    validate(
        [_input()],
        [_prediction()],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert not checks.failed


def test_strict_validator_rejects_silent_zero_score_shapes():
    prediction = _prediction()
    prediction["trace"] = []
    prediction["gold_papers"] = [{"paper_id": "p1"}, {"paper_id": "p1"}]
    prediction["evidence"][0] = {
        "paper_id": "p2",
        "source_type": "table",
        "locator": {"page": "2", "table_id": "Table 1", "row": "X"},
    }
    prediction["answer"]["multiple_choice"]["gold"] = "Z"
    prediction["answer"]["table"]["rows"] = [
        {"Method": "X", "Score": "91.2"},
        {"Method": "X", "Score": 92.0},
    ]
    checks = CheckCounter()

    validate(
        [_input()],
        [prediction],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert checks.failures["top-level keys exact"] == 1
    assert checks.failures["multiple_choice letter within valid keys"] == 1
    assert checks.failures["evidence locator has page"] == 1
    assert checks.failures["evidence locator is coarse official shape"] == 1
    assert checks.failures["evidence paper is submitted"] == 1
    assert checks.failures["paper entries duplicate-free"] == 1
    assert checks.failures["paper ids are canonical"] == 1
    assert checks.failures["table cell types match schema"] == 1
    assert checks.failures["table row keys duplicate-free"] == 1


def test_strict_validator_rejects_nested_oracle_and_alias_shapes():
    prediction = _prediction()
    prediction["gold_papers"] = ["p1"]
    prediction["evidence"][0]["evidence_text_or_value"] = "oracle excerpt"
    prediction["answer"]["multiple_choice"] = {
        "answer": "A",
        "options": {"A": "oracle option"},
    }
    prediction["answer"]["table"]["schema"] = [{"name": "oracle"}]
    checks = CheckCounter()

    validate(
        [_input()],
        [prediction],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert checks.failures["paper entry shape exact"] == 1
    assert checks.failures["evidence item keys exact"] == 1
    assert checks.failures["multiple_choice object exact"] == 1
    assert checks.failures["multiple_choice letter within valid keys"] == 1
    assert checks.failures["table schema matches input"] == 1


def test_strict_validator_accepts_optional_matching_table_schema():
    prediction = _prediction()
    prediction["answer"]["table"]["schema"] = _input()["table_schema"]
    checks = CheckCounter()

    validate(
        [_input()],
        [prediction],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert not checks.failed


def test_strict_validator_rejects_non_string_freeform_text():
    sample = {
        "query_id": "q1",
        "question": "question",
        "answer_types": ["freeform"],
        "table_schema": None,
    }
    prediction = _prediction()
    prediction["answer"] = {"freeform": {"text": 42}}
    checks = CheckCounter()

    validate(
        [sample],
        [prediction],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert checks.failures["freeform text is string"] == 1
    assert checks.failures["freeform text non-empty"] == 1


def test_compatible_validator_accepts_legacy_aliases_and_rich_locators():
    prediction = _prediction()
    prediction["trace"] = {"debug": True}
    prediction["papers"] = prediction.pop("gold_papers")
    prediction["papers"] = ["p1"]
    prediction["evidence"][0]["evidence_text_or_value"] = "diagnostic excerpt"
    prediction["evidence"][0]["locator"].update(
        {"row": "X", "column": "Score", "section": "Results"}
    )
    prediction["evidence"][0]["locator"]["page"] = "2"
    prediction["answer"]["multiple_choice"] = {"answer": "A"}
    checks = CheckCounter()

    validate([_input()], [prediction], {}, checks)

    assert not checks.failed


def test_strict_validator_rejects_extra_prediction_with_empty_query_id():
    extra = _prediction()
    extra["query_id"] = ""
    checks = CheckCounter()

    validate(
        [_input()],
        [_prediction(), extra],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert checks.failures["query_ids non-empty"] == 1
    assert checks.examples["query_ids non-empty"] == ["prediction x1"]


def test_strict_validator_rejects_duplicate_input_query_ids():
    checks = CheckCounter()

    validate(
        [_input(), _input()],
        [_prediction()],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert checks.failures["query_ids duplicate-free"] == 1
    assert checks.examples["query_ids duplicate-free"] == ["input q1 x2"]


@pytest.mark.parametrize("invalid_query_id", [1, " q1 ", "   "])
def test_strict_validator_rejects_noncanonical_query_id(invalid_query_id):
    sample = _input()
    prediction = _prediction()
    sample["query_id"] = invalid_query_id
    prediction["query_id"] = invalid_query_id
    checks = CheckCounter()

    validate(
        [sample],
        [prediction],
        {},
        checks,
        strict=True,
        canonical_papers={"p1"},
    )

    assert checks.failures["query_ids are canonical strings"] == 2
