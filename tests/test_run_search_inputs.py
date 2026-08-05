from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

RUN_SEARCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_search.py"
SPEC = importlib.util.spec_from_file_location("littraceqa_run_search", RUN_SEARCH_PATH)
assert SPEC is not None and SPEC.loader is not None
RUN_SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_SEARCH)
load_mc_options = RUN_SEARCH.load_mc_options
load_queries = RUN_SEARCH.load_queries
should_evaluate_against_validation = RUN_SEARCH.should_evaluate_against_validation
build_argument_parser = RUN_SEARCH.build_argument_parser


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _required_cli_args() -> list[str]:
    return [
        "--paths",
        "paths.yaml",
        "--process",
        "process.yaml",
        "--search",
        "search.yaml",
        "--agent",
        "agent.yaml",
        "--queries",
        "queries.jsonl",
        "--output",
        "predictions.jsonl",
    ]


def test_search_cli_projects_to_official_input_contract_by_default():
    args = build_argument_parser().parse_args(_required_cli_args())

    assert args.production_input is True


def test_search_cli_requires_explicit_opt_in_for_development_fields():
    args = build_argument_parser().parse_args(
        [*_required_cli_args(), "--include-development-fields"]
    )

    assert args.production_input is False


def test_legacy_production_input_flag_remains_accepted():
    args = build_argument_parser().parse_args(
        [*_required_cli_args(), "--production-input"]
    )

    assert args.production_input is True


def test_load_queries_preserves_current_official_options_and_drops_dev_fields(
    tmp_path,
):
    path = tmp_path / "test.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "ltqa_1234",
                "benchmark": "LitTraceQA",
                "question": "Choose the supported answer.",
                "answer_types": ["multiple_choice"],
                "multiple_choice_options": [
                    {"label": "A", "text": "Alpha"},
                    {"label": "E", "text": "Echo"},
                ],
                "task_family": "development-only",
                "primary_evidence_type": "development-only",
            }
        ],
    )

    query = load_queries(path, production_input=True)[0]

    assert query.options == {"A": "Alpha", "E": "Echo"}
    assert query.task_family is None
    assert query.primary_evidence_type is None


def test_legacy_options_file_backfills_public_text_without_reading_gold_label(
    tmp_path,
):
    queries = tmp_path / "queries.jsonl"
    options = tmp_path / "legacy.jsonl"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "benchmark": "LitTraceQA",
                "question": "Choose one.",
                "answer_types": ["multiple_choice"],
            }
        ],
    )
    _write_jsonl(
        options,
        [
            {
                "query_id": "q1",
                "answer": {
                    "multiple_choice": {
                        "options": {"A": "Alpha", "E": "Echo"},
                        "gold": "GOLD_SENTINEL",
                    }
                },
            }
        ],
    )

    assert load_mc_options(options) == {"q1": {"A": "Alpha", "E": "Echo"}}
    query = load_queries(
        queries,
        production_input=True,
        options_path=options,
    )[0]

    assert query.options == {"A": "Alpha", "E": "Echo"}
    assert "GOLD_SENTINEL" not in json.dumps(query.to_dict())


def test_production_search_loader_fails_closed_when_mc_options_are_missing(tmp_path):
    path = tmp_path / "bad.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "q1",
                "benchmark": "LitTraceQA",
                "question": "Choose one.",
                "answer_types": ["multiple_choice"],
            }
        ],
    )

    with pytest.raises(ValueError, match="requires at least two"):
        load_queries(path, production_input=True)


def test_validation_scoring_requires_exact_complete_query_id_set(tmp_path):
    gold = tmp_path / "validation.jsonl"
    _write_jsonl(gold, [{"query_id": "q1"}, {"query_id": "q2"}])
    complete = [
        RUN_SEARCH.Query("q2", "question", ["freeform"]),
        RUN_SEARCH.Query("q1", "question", ["freeform"]),
    ]
    partial = [RUN_SEARCH.Query("q1", "question", ["freeform"])]
    held_out = [
        RUN_SEARCH.Query("ltqa_0123456789abcdef", "question", ["freeform"])
    ]

    assert should_evaluate_against_validation(complete, gold) is True
    assert should_evaluate_against_validation(partial, gold) is False
    assert should_evaluate_against_validation(held_out, gold) is False


def test_validation_scoring_rejects_duplicate_query_ids(tmp_path):
    gold = tmp_path / "validation.jsonl"
    _write_jsonl(gold, [{"query_id": "q1"}, {"query_id": "q2"}])
    duplicated = [
        RUN_SEARCH.Query("q1", "question", ["freeform"]),
        RUN_SEARCH.Query("q1", "question", ["freeform"]),
        RUN_SEARCH.Query("q2", "question", ["freeform"]),
    ]

    assert should_evaluate_against_validation(duplicated, gold) is False
