from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from littraceqa.corpus_preflight import requires_visual_image
from littraceqa.di_pipeline.contracts import Query
from littraceqa.pairwise_prompts import (
    _query_tags,
    judgment_question_type,
    selected_answer_examples,
)
from littraceqa.query_requirements import explicit_table_row_items


_OFFICIAL_REVISION = "bd35dc14cf0483e0ffa51fa2a54d2689c13f9845"
_OFFICIAL_TEST = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "official_release"
    / _OFFICIAL_REVISION
    / "data"
    / "test.jsonl"
)

_EXPECTED_VISUAL_QUERY_IDS = {
    "ltqa_25546519dfb273c8",
    "ltqa_394b3fcabbc496d9",
    "ltqa_4900eb2c71c890c8",
    "ltqa_97a49a0392eeaee0",
    "ltqa_98ff929cb222a1b3",
    "ltqa_b61b62d2801d5f07",
    "ltqa_c0b2f8616b032d4b",
    "ltqa_cbad41e189930190",
    "ltqa_dc59b0be539a1b22",
    "ltqa_f6ae14ff5b8d177b",
}
_EXPECTED_CITATION_QUERY_IDS = {
    "ltqa_0a98ae6877634bf1",
    "ltqa_26a8f5550df35020",
    "ltqa_729fa13078b8135f",
    "ltqa_b18f17b22f0bfdbe",
    "ltqa_bde426d34c7e10bd",
    "ltqa_c4708554d734f4e1",
    "ltqa_d2b9a56db69fe43c",
    "ltqa_dd9546c035c87d9c",
}
_EXPECTED_COMPOUND_MC_QUERY_IDS = {
    "ltqa_03af2c583a696a04",
    "ltqa_0a98ae6877634bf1",
    "ltqa_16a585ec64d3fe52",
    "ltqa_22ff7b719c5625d4",
    "ltqa_2391a0f9afd48008",
    "ltqa_25546519dfb273c8",
    "ltqa_26a8f5550df35020",
    "ltqa_340528fbecedf89a",
    "ltqa_394b3fcabbc496d9",
    "ltqa_3bfb8111c92ba3d5",
    "ltqa_4900eb2c71c890c8",
    "ltqa_4de695d77c51ca01",
    "ltqa_571b8ccefde36062",
    "ltqa_5b08acb319329757",
    "ltqa_5e0dfcb644ec0a04",
    "ltqa_69178ae8aa769eda",
    "ltqa_729fa13078b8135f",
    "ltqa_98ff929cb222a1b3",
    "ltqa_9c93b4c3fdb98c15",
    "ltqa_a24c157315314b97",
    "ltqa_a2c8b9763a7ce26e",
    "ltqa_a467625518ca3ac4",
    "ltqa_b61b62d2801d5f07",
    "ltqa_ba83892a544258a6",
    "ltqa_c0b2f8616b032d4b",
    "ltqa_c95d638c01295a12",
    "ltqa_cbad41e189930190",
    "ltqa_cf29b3a6608039ea",
    "ltqa_cf4bd6505d859121",
    "ltqa_d2b9a56db69fe43c",
    "ltqa_d89df503ec813f2e",
    "ltqa_dada5a958af5068b",
    "ltqa_db8e3f7d548a7d24",
    "ltqa_dc59b0be539a1b22",
    "ltqa_dcaf2cccb716dcea",
    "ltqa_de53de0c4394d292",
    "ltqa_eb04a1b29408f7bb",
    "ltqa_f0de7fb4352ad29c",
    "ltqa_f399a24775b6f0b8",
    "ltqa_f6ae14ff5b8d177b",
    "ltqa_f7ead099a16ddbcd",
    "ltqa_fca1b7d3c8a697aa",
    "ltqa_ffb8499a6cbd3c4d",
}


def _load_released_queries() -> tuple[list[dict], list[Query]]:
    records = [
        json.loads(line)
        for line in _OFFICIAL_TEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, [Query.from_dict(record) for record in records]


def test_pinned_official_test_prompt_taxonomy_is_gold_free_and_stable() -> None:
    records, queries = _load_released_queries()

    multiple_choice_fields = {
        "query_id",
        "benchmark",
        "question",
        "answer_types",
        "multiple_choice_options",
    }
    table_fields = {
        "query_id",
        "benchmark",
        "question",
        "answer_types",
        "table_schema",
    }
    assert len(records) == 71
    assert Counter(tuple(query.answer_types) for query in queries) == {
        ("multiple_choice",): 50,
        ("table",): 21,
    }
    assert all(
        set(record)
        == (
            multiple_choice_fields
            if record["answer_types"] == ["multiple_choice"]
            else table_fields
        )
        for record in records
    )

    types_by_id = {
        query.query_id: judgment_question_type(query) for query in queries
    }
    assert Counter(types_by_id.values()) == {
        "other": 52,
        "visual": 10,
        "citation": 8,
        "calculation": 1,
    }
    assert {
        query_id for query_id, primary_type in types_by_id.items()
        if primary_type == "visual"
    } == _EXPECTED_VISUAL_QUERY_IDS
    assert {
        query_id for query_id, primary_type in types_by_id.items()
        if primary_type == "citation"
    } == _EXPECTED_CITATION_QUERY_IDS

    for query in queries:
        detector_requires_visual = requires_visual_image(query.question)
        assert ("visual" in _query_tags(query)) is detector_requires_visual
        assert (types_by_id[query.query_id] == "visual") is detector_requires_visual

    answer_examples_by_id = {
        query.query_id: {
            example.example_id for example in selected_answer_examples(query)
        }
        for query in queries
    }
    assert {
        query_id
        for query_id, example_ids in answer_examples_by_id.items()
        if "A19_compound_option_atomic_facts" in example_ids
    } == _EXPECTED_COMPOUND_MC_QUERY_IDS
    table_query_ids = {
        query.query_id for query in queries if query.answer_types == ["table"]
    }
    assert {
        query_id
        for query_id, example_ids in answer_examples_by_id.items()
        if "A33_table_requested_unit_checklist" in example_ids
    } == table_query_ids
    assert {
        query_id
        for query_id, example_ids in answer_examples_by_id.items()
        if "A29_exact_bibliography_titles_across_papers" in example_ids
    } == {"ltqa_c4708554d734f4e1"}
    percent_change_examples = answer_examples_by_id["ltqa_090478d0ddf8d27f"]
    assert "A34_relative_percent_change" in percent_change_examples
    assert "A23_operand_grounded_delta" not in percent_change_examples
    for query in queries:
        example_ids = answer_examples_by_id[query.query_id]
        if query.answer_types == ["table"]:
            assert "A11_variable_option_labels" not in example_ids
            assert "A15_atomic_text_fact" not in example_ids
            assert "A33_table_requested_unit_checklist" in example_ids
        if query.answer_types == ["multiple_choice"]:
            assert "A11_variable_option_labels" in example_ids
            assert "A33_table_requested_unit_checklist" not in example_ids
        if judgment_question_type(query) != "visual":
            assert "A12_missing_image" not in example_ids

    lcirc_erase = next(
        query for query in queries if query.query_id == "ltqa_ab60eb571239314b"
    )
    assert explicit_table_row_items(lcirc_erase) == ()
    assert "explicit_rows" not in _query_tags(lcirc_erase)
    assert (
        "A26_explicit_table_row_inventory"
        not in answer_examples_by_id[lcirc_erase.query_id]
    )
