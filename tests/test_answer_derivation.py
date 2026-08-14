from __future__ import annotations

from typing import Any

import pytest

from littraceqa.answer_derivation import (
    DerivationValidationError,
    is_aggregate_citation_count_query,
    is_axis_extent_lookup_query,
    is_mean_aggregation_query,
    requires_extremum_operation,
    validate_answer_semantics,
    validate_table_rows,
)
from littraceqa.di_pipeline.contracts import Query


def _fact(
    fact_id: str = "f1",
    value: Any = "value",
    *,
    name: str | None = None,
    value_kind: str = "reported",
) -> dict[str, Any]:
    return {
        "id": fact_id,
        "name": name or f"source fact {fact_id}",
        "value": value,
        "value_kind": value_kind,
        "paper_id": "p1",
        "chunk_ids": [f"p1#{fact_id}"],
    }


def _binding(
    answer_path: str,
    expected: Any,
    answer_fragment: str | None = None,
) -> dict[str, Any]:
    output = {"answer_path": answer_path, "expected": expected}
    if answer_fragment is not None:
        output["answer_fragment"] = answer_fragment
    return output


def _derivation(
    facts: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    final_answer: str,
    *,
    answer_bindings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if answer_bindings is None:
        if operations:
            answer_bindings = [
                {
                    "answer_path": operation["answer_binding"]["answer_path"],
                    "source_type": "operation",
                    "source_id": operation["id"],
                    **(
                        {
                            "answer_fragment": operation["answer_binding"][
                                "answer_fragment"
                            ]
                        }
                        if "answer_fragment" in operation["answer_binding"]
                        else {}
                    ),
                }
                for operation in operations
            ]
        else:
            answer_bindings = [
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": facts[0]["id"],
                    "answer_fragment": final_answer,
                }
            ]
    return {
        "facts": facts,
        "operations": operations,
        "answer_bindings": answer_bindings,
        "final_semantic_answer": final_answer,
    }


def test_compare_recomputes_yes_no_polarity() -> None:
    query = Query("q", "Does A have more than B?", ["freeform"])
    facts = [_fact("left", 30), _fact("right", 21)]
    operation = {
        "id": "compare_ab",
        "kind": "compare",
        "fact_ids": ["left", "right"],
        "left": 30,
        "operator": ">",
        "right": 21,
        "result": True,
        "answer_binding": _binding("answer.freeform.text", True, "No"),
    }

    with pytest.raises(DerivationValidationError, match="does not express expected"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "No"),
            answer={"freeform": {"text": "No"}},
        )

    operation["answer_binding"] = _binding("answer.freeform.text", True, "Yes")
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "Yes"),
        answer={"freeform": {"text": "Yes"}},
    )
    assert validated["operations"][0]["result"] is True


def test_compare_rejects_incorrect_boolean_before_answer_binding() -> None:
    query = Query("q", "Does A have more than B?", ["freeform"])
    operation = {
        "id": "compare_ab",
        "kind": "compare",
        "fact_ids": ["left", "right"],
        "left": 30,
        "operator": ">",
        "right": 21,
        "result": False,
        "answer_binding": _binding("answer.freeform.text", False, "No"),
    }
    with pytest.raises(DerivationValidationError, match="30 > 21 is True"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("left", 30), _fact("right", 21)],
                [operation],
                "No",
            ),
            answer={"freeform": {"text": "No"}},
        )


def _same_performance_derivation() -> tuple[Query, dict[str, Any], dict[str, Any]]:
    query = Query(
        "same_performance",
        "On which task do System Cedar and System Flint achieve the same performance?",
        ["freeform", "multiple_choice"],
        options={"A": "Task A", "B": "Task B", "C": "Task C"},
    )
    facts = [
        _fact("cedar_a", 61),
        _fact("flint_a", 58),
        _fact("cedar_b", 73),
        _fact("flint_b", 73),
        _fact("cedar_c", 80),
        _fact("flint_c", 76),
    ]
    operation = {
        "id": "same_task",
        "kind": "select_where",
        "fact_ids": [fact["id"] for fact in facts],
        "comparisons": [
            {
                "label": "Task A",
                "left_fact_id": "cedar_a",
                "right_fact_id": "flint_a",
                "left": 61,
                "right": 58,
            },
            {
                "label": "Task B",
                "left_fact_id": "cedar_b",
                "right_fact_id": "flint_b",
                "left": 73,
                "right": 73,
            },
            {
                "label": "Task C",
                "left_fact_id": "cedar_c",
                "right_fact_id": "flint_c",
                "left": 80,
                "right": 76,
            },
        ],
        "operator": "==",
        "result": "Task B",
        "answer_binding": _binding(
            "answer.multiple_choice", "Task B", "Task B"
        ),
    }
    derivation = _derivation(
        facts,
        [operation],
        "Task B",
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "operation",
                "source_id": "same_task",
                "answer_fragment": "Task B",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "operation",
                "source_id": "same_task",
                "answer_fragment": "Task B",
            },
        ],
    )
    answer = {
        "freeform": {"text": "Task B"},
        "multiple_choice": {"label": "B", "selected_option_text": "Task B"},
    }
    return query, derivation, answer


def test_select_where_binds_unique_matching_label_not_boolean_predicate() -> None:
    query, derivation, answer = _same_performance_derivation()

    validated = validate_answer_semantics(
        query,
        derivation=derivation,
        answer=answer,
    )

    operation = validated["operations"][0]
    assert operation["kind"] == "select_where"
    assert operation["result"] == "Task B"


def test_select_where_rejects_non_unique_matching_labels() -> None:
    query, derivation, answer = _same_performance_derivation()
    fact = next(item for item in derivation["facts"] if item["id"] == "flint_c")
    fact["value"] = 80
    comparison = derivation["operations"][0]["comparisons"][2]
    comparison["right"] = 80

    with pytest.raises(DerivationValidationError, match="exactly one matching label"):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_select_where_rejects_comparison_operand_not_copied_from_fact() -> None:
    query, derivation, answer = _same_performance_derivation()
    derivation["operations"][0]["comparisons"][1]["right"] = 72

    with pytest.raises(
        DerivationValidationError, match="do not match referenced fact values"
    ):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_labeled_equality_selection_requires_equality_operator() -> None:
    query, derivation, answer = _same_performance_derivation()
    operation = derivation["operations"][0]
    # <= still has exactly one deterministic winner (Task B), so operation
    # validation succeeds before the query-level equality contract rejects it.
    operation["operator"] = "<="

    with pytest.raises(
        DerivationValidationError, match="labeled equality selection"
    ):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_select_where_rejects_reused_operand_fact_across_rows() -> None:
    query, derivation, answer = _same_performance_derivation()
    comparison = derivation["operations"][0]["comparisons"][2]
    comparison["left_fact_id"] = "cedar_a"
    comparison["left"] = 61

    with pytest.raises(DerivationValidationError, match="exactly once"):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_operation_rejects_invented_operands_not_present_in_facts() -> None:
    query = Query("q", "What is the difference?", ["freeform"])
    operation = {
        "id": "difference",
        "kind": "subtract",
        "fact_ids": ["left", "right"],
        "operands": [10, 3],
        "result": 7,
        "answer_binding": _binding("answer.freeform.text", 7, "7"),
    }
    with pytest.raises(DerivationValidationError, match="referenced fact values"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("left", 10), _fact("right", 2)], [operation], "7"
            ),
            answer={"freeform": {"text": "7"}},
        )


def test_count_requires_distinct_fact_grounded_items_and_matching_result() -> None:
    query = Query("q", "How many panels?", ["freeform"])
    facts = [_fact("panels", ["(a)", "(b)", "(c)"], value_kind="visual")]
    operation = {
        "id": "panel_count",
        "kind": "count",
        "fact_ids": ["panels"],
        "items": ["(a)", "(b)", "(c)"],
        "result": 2,
        "answer_binding": _binding("answer.freeform.text", 2, "2"),
    }
    with pytest.raises(DerivationValidationError, match="contain 3 entries"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "2"),
            answer={"freeform": {"text": "2"}},
        )

    operation.update(
        {
            "items": ["(a)", "(b)", "(b)"],
            "result": 3,
            "answer_binding": _binding("answer.freeform.text", 3, "3"),
        }
    )
    with pytest.raises(DerivationValidationError, match="distinct items"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "3"),
            answer={"freeform": {"text": "3"}},
        )


def test_count_flattens_list_fact_values() -> None:
    query = Query("q", "How many panels?", ["freeform"])
    operation = {
        "id": "panel_count",
        "kind": "count",
        "fact_ids": ["row1", "row2"],
        "items": ["(a)", "(b)", "(c)", "(d)"],
        "result": 4,
        "answer_binding": _binding("answer.freeform.text", 4, "4"),
    }
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [
                _fact("row1", [["(a)", "(b)"]], value_kind="visual"),
                _fact("row2", ["(c)", "(d)"], value_kind="visual"),
            ],
            [operation],
            "4",
        ),
        answer={"freeform": {"text": "4"}},
    )
    assert validated["operations"][0]["result"] == 4


@pytest.mark.parametrize(
    "items",
    [
        ["(a)", "(b)"],
        ["Figure 4(a)", "Figure 4(b)"],
        ["subfigure (a)", "subfigure (b)"],
        ["Qwen row (a)", "Qwen row (b)"],
    ],
)
def test_visual_subfigure_count_rejects_bare_group_labels(
    items: list[str],
) -> None:
    query = Query(
        "q004",
        "How many subfigures are shown in Figure 4?",
        ["freeform"],
    )
    facts = [_fact("axes", items, value_kind="visual")]
    operation = {
        "id": "subfigure_count",
        "kind": "count",
        "fact_ids": ["axes"],
        "items": items,
        "result": 2,
        "answer_binding": _binding("answer.freeform.text", 2, "2"),
    }

    with pytest.raises(
        DerivationValidationError, match="visual subfigure count"
    ):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "2"),
            answer={"freeform": {"text": "2"}},
        )


def test_visual_subfigure_count_accepts_spatial_axes_inventory() -> None:
    query = Query(
        "q_two_panels",
        "How many subfigures are shown in Figure 2?",
        ["freeform"],
    )
    items = ["(a)-left axes", "(b)-right axes"]
    facts = [_fact("axes", items, value_kind="visual")]
    operation = {
        "id": "subfigure_count",
        "kind": "count",
        "fact_ids": ["axes"],
        "items": items,
        "result": 2,
        "answer_binding": _binding("answer.freeform.text", 2, "2"),
    }

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "2"),
        answer={"freeform": {"text": "2"}},
    )

    assert validated["operations"][0]["items"] == items


def test_visual_subfigure_count_rejects_text_inventory_with_unused_visual_fact() -> None:
    query = Query(
        "q_visual_bypass",
        "How many subfigures are shown in Figure 4?",
        ["freeform"],
    )
    items = ["left axes", "right axes"]
    facts = [
        _fact("text_labels", items, value_kind="text"),
        _fact("unused_visual", "attached Figure 4", value_kind="visual"),
    ]
    operation = {
        "id": "subfigure_count",
        "kind": "count",
        "fact_ids": ["text_labels"],
        "items": items,
        "result": 2,
        "answer_binding": _binding("answer.freeform.text", 2, "2"),
    }

    with pytest.raises(
        DerivationValidationError,
        match="operation must reference at least one visual fact",
    ):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "2"),
            answer={"freeform": {"text": "2"}},
        )


def _citation_count_case(
    *,
    question: str,
    items: list[str],
    result: int,
    label: str,
    options: dict[str, str],
) -> tuple[Query, dict[str, Any], dict[str, Any]]:
    query = Query(
        "q_citation_count",
        question,
        ["freeform", "multiple_choice"],
        options=options,
    )
    operation = {
        "id": "citation_count",
        "kind": "count",
        "fact_ids": ["citations"],
        "items": items,
        "result": result,
        "answer_binding": _binding(
            "answer.multiple_choice.selected_option_text",
            result,
            str(result),
        ),
    }
    derivation = _derivation(
        [_fact("citations", items, value_kind="text")],
        [operation],
        str(result),
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "operation",
                "source_id": "citation_count",
                "answer_fragment": str(result),
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "operation",
                "source_id": "citation_count",
                "answer_fragment": str(result),
            },
        ],
    )
    answer = {
        "freeform": {"text": str(result)},
        "multiple_choice": {
            "label": label,
            "selected_option_text": options[label],
        },
    }
    return query, derivation, answer


def test_aggregate_citation_count_rejects_method_names_and_url_in_thirteen_items():
    valid_items = [
        "Alder et al. (2009)",
        "Birch et al. (2017)",
        "Cedar (2010)",
        "Dove et al. (2017)",
        "Elm et al. (2017)",
        "Finch et al. (2019)",
        "Grove et al. (2022)",
        "Hazel et al. (2023)",
        "Iris et al. (2023)",
    ]
    query, derivation, answer = _citation_count_case(
        question="How many papers were cited in the Introduction?",
        items=[*valid_items, "FedRec", "SecAgg", "SecEmb", "https://example.test"],
        result=13,
        label="C",
        options={"A": "5", "B": "9", "C": "13", "D": "15"},
    )

    with pytest.raises(
        DerivationValidationError, match="not a stable citation identity"
    ):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_aggregate_citation_count_accepts_nine_identities_and_option_b():
    items = [
        "Alder et al. (2009)",
        "Birch et al. (2017)",
        "Cedar (2010)",
        "Dove et al. (2017)",
        "Elm et al. (2017)",
        "Finch et al. (2019)",
        "Grove et al. (2022)",
        "Hazel et al. (2023)",
        "Iris et al. (2023)",
    ]
    query, derivation, answer = _citation_count_case(
        question="How many papers were cited in the Introduction?",
        items=items,
        result=9,
        label="B",
        options={"A": "5", "B": "9", "C": "13", "D": "15"},
    )

    validated = validate_answer_semantics(
        query, derivation=derivation, answer=answer
    )

    assert is_aggregate_citation_count_query(query) is True
    assert validated["operations"][0]["items"] == items
    assert validated["operations"][0]["result"] == 9


def test_author_filtered_reference_count_accepts_three_author_year_identities():
    items = [
        "Bell et al. (2020)",
        "Bonawitz et al. (2017)",
        "Bonawitz et al. (2019)",
    ]
    query, derivation, answer = _citation_count_case(
        question="How many references include Bonawitz as an author?",
        items=items,
        result=3,
        label="B",
        options={"A": "2", "B": "3", "C": "4", "D": "8"},
    )

    validated = validate_answer_semantics(
        query, derivation=derivation, answer=answer
    )

    assert is_aggregate_citation_count_query(query) is True
    assert validated["operations"][0]["result"] == 3


def test_last_reference_index_is_not_an_aggregate_citation_count():
    lookup = Query(
        "q_last_reference",
        "What is the index of the last reference in JuniperMesh?",
        ["freeform"],
    )
    explicit_number_lookup = Query(
        "q_last_reference_number",
        "What is the number of the last reference in JuniperMesh?",
        ["freeform"],
    )
    present_tense_count = Query(
        "q_present_tense_count",
        "How many papers are cited in the Introduction?",
        ["freeform"],
    )
    unrelated_questions = [
        Query(
            "q_reference_parameters",
            "How many parameters does reference [5] have?",
            ["freeform"],
        ),
        Query(
            "q_reference_free",
            "How many methods are reference-free?",
            ["freeform"],
        ),
        Query(
            "q_cited_paper_parameters",
            "What is the number of parameters in the cited paper?",
            ["freeform"],
        ),
    ]

    assert is_aggregate_citation_count_query(lookup) is False
    assert is_aggregate_citation_count_query(explicit_number_lookup) is False
    assert is_aggregate_citation_count_query(present_tense_count) is True
    assert all(
        is_aggregate_citation_count_query(query) is False
        for query in unrelated_questions
    )


def test_count_rejects_eight_vs_seven_word_answer() -> None:
    query = Query("q", "How many panels?", ["freeform"])
    items = list("abcdefgh")
    operation = {
        "id": "panel_count",
        "kind": "count",
        "fact_ids": ["panels"],
        "items": items,
        "result": 8,
        "answer_binding": _binding("answer.freeform.text", 8, "seven"),
    }
    with pytest.raises(DerivationValidationError, match="does not express expected"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("panels", items, value_kind="visual")],
                [operation],
                "seven",
            ),
            answer={"freeform": {"text": "seven"}},
        )


def test_multi_operation_contradictory_multiple_choice_is_rejected() -> None:
    query = Query(
        "q",
        "Which count is correct?",
        ["multiple_choice"],
        options={"A": "Eight panels", "B": "Seven panels"},
    )
    eight_items = list("abcdefgh")
    seven_items = list("abcdefg")
    operations = [
        {
            "id": "first_count",
            "kind": "count",
            "fact_ids": ["eight_panels"],
            "items": eight_items,
            "result": 8,
            "answer_binding": _binding(
                "answer.multiple_choice.selected_option_text", 8, "Seven"
            ),
        },
        {
            "id": "second_count",
            "kind": "count",
            "fact_ids": ["seven_panels"],
            "items": seven_items,
            "result": 7,
            "answer_binding": _binding("answer.multiple_choice", 7, "Seven"),
        },
    ]
    with pytest.raises(DerivationValidationError, match="expected result"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [
                    _fact("eight_panels", eight_items, value_kind="visual"),
                    _fact("seven_panels", seven_items, value_kind="visual"),
                ],
                operations,
                "Seven panels",
            ),
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "Seven panels",
                }
            },
        )


def test_decimal_arithmetic_is_recomputed_exactly() -> None:
    query = Query("q", "What is the difference?", ["freeform"])
    facts = [_fact("left", "53.46"), _fact("right", "41.15")]
    operation = {
        "id": "difference",
        "kind": "subtract",
        "fact_ids": ["left", "right"],
        "operands": ["53.46", "41.15"],
        "result": "12.30",
        "answer_binding": _binding("answer.freeform.text", "12.30", "12.30"),
    }
    with pytest.raises(DerivationValidationError, match="gives 12.31"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "12.30"),
            answer={"freeform": {"text": "12.30"}},
        )

    # A directly reported lookup remains valid with no fake arithmetic.
    validated = validate_answer_semantics(
        query,
        derivation=_derivation([_fact("reported", "12.30")], [], "12.30"),
        answer={"freeform": {"text": "12.30"}},
    )
    assert validated["operations"] == []


def test_divide_requires_explicit_rounding_or_exact_contract() -> None:
    query = Query("q", "What is one third?", ["freeform"])
    facts = [_fact("numerator", 1), _fact("denominator", 3)]
    operation = {
        "id": "ratio",
        "kind": "divide",
        "fact_ids": ["numerator", "denominator"],
        "operands": [1, 3],
        "result": "0.333",
        "answer_binding": _binding("answer.freeform.text", "0.333", "0.333"),
    }
    with pytest.raises(DerivationValidationError, match="requires exact=true or a rounding"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "0.333"),
            answer={"freeform": {"text": "0.333"}},
        )

    operation["rounding"] = {"decimal_places": 3, "mode": "half_up"}
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "0.333"),
        answer={"freeform": {"text": "0.333"}},
    )
    assert validated["operations"][0]["result"] == "0.333"


def _mean_operation(
    *,
    fact_ids: list[str],
    operands: list[Any],
    result: Any,
    exact: bool = True,
) -> dict[str, Any]:
    operation: dict[str, Any] = {
        "id": "task_mean",
        "kind": "mean",
        "fact_ids": fact_ids,
        "operands": operands,
        "result": result,
        "answer_binding": _binding(
            "answer.multiple_choice", result, str(result)
        ),
    }
    if exact:
        operation["exact"] = True
    return operation


def test_mean_uses_distinct_source_facts_and_exact_terminating_result() -> None:
    query = Query(
        "mean_scores",
        "What is the arithmetic mean of Cedar's three reported task scores?",
        ["multiple_choice"],
        options={"A": "5.8", "B": "6.2", "C": "6.4"},
    )
    facts = [_fact("one", 4.2), _fact("two", 6.6), _fact("three", 8.4)]
    operation = _mean_operation(
        fact_ids=["one", "two", "three"],
        operands=[4.2, 6.6, 8.4],
        result=6.4,
    )

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "6.4"),
        answer={
            "multiple_choice": {
                "label": "C",
                "selected_option_text": "6.4",
            }
        },
    )

    assert validated["operations"][0]["kind"] == "mean"
    assert validated["operations"][0]["result"] == 6.4


def test_mean_requires_rounding_for_nonterminating_result() -> None:
    query = Query(
        "rounded_mean",
        "What is the average of the three reported values?",
        ["multiple_choice"],
        options={"A": "1.66", "B": "1.67"},
    )
    facts = [_fact("one", 1), _fact("two", 2), _fact("three", 2)]
    operation = _mean_operation(
        fact_ids=["one", "two", "three"],
        operands=[1, 2, 2],
        result="1.67",
    )

    with pytest.raises(DerivationValidationError, match="exact mean is non-terminating"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "1.67"),
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "1.67",
                }
            },
        )

    operation.pop("exact")
    operation["rounding"] = {"decimal_places": 2, "mode": "half_up"}
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "1.67"),
        answer={
            "multiple_choice": {
                "label": "B",
                "selected_option_text": "1.67",
            }
        },
    )
    assert validated["operations"][0]["result"] == "1.67"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"fact_ids": ["one"]}, "at least two distinct sourced numeric facts"),
        ({"operands": [4.2, 8.4, 6.6]}, "do not match referenced fact values"),
        ({"result": 6.3}, "deterministic mean gives 6.4"),
        ({"exact": True, "rounding": {"decimal_places": 1, "mode": "half_up"}},
         "either exact=true or rounding"),
    ],
)
def test_mean_rejects_incomplete_or_inconsistent_contract(
    mutation: dict[str, Any], error: str
) -> None:
    query = Query(
        "mean_scores",
        "What is the average of the three reported task scores?",
        ["multiple_choice"],
        options={"C": "6.4"},
    )
    facts = [_fact("one", 4.2), _fact("two", 6.6), _fact("three", 8.4)]
    operation = _mean_operation(
        fact_ids=["one", "two", "three"],
        operands=[4.2, 6.6, 8.4],
        result=6.4,
    )
    operation.update(mutation)

    with pytest.raises(DerivationValidationError, match=error):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "6.4"),
            answer={
                "multiple_choice": {
                    "label": "C",
                    "selected_option_text": "6.4",
                }
            },
        )


def test_mean_operation_is_query_gated_and_direct_reported_mean_is_allowed() -> None:
    non_mean_query = Query(
        "reported_score",
        "What score does Cedar report?",
        ["multiple_choice"],
        options={"C": "6.4"},
    )
    facts = [_fact("one", 4.2), _fact("two", 8.6)]
    operation = _mean_operation(
        fact_ids=["one", "two"], operands=[4.2, 8.6], result=6.4
    )
    with pytest.raises(DerivationValidationError, match="requires mean/average wording"):
        validate_answer_semantics(
            non_mean_query,
            derivation=_derivation(facts, [operation], "6.4"),
            answer={
                "multiple_choice": {
                    "label": "C",
                    "selected_option_text": "6.4",
                }
            },
        )

    mean_query = Query(
        "reported_mean",
        "What average does the paper directly report?",
        ["freeform"],
    )
    validated = validate_answer_semantics(
        mean_query,
        derivation=_derivation([_fact("reported_mean", 6.4)], [], "6.4"),
        answer={"freeform": {"text": "6.4"}},
    )
    assert validated["operations"] == []

    with pytest.raises(DerivationValidationError, match="arithmetic mean"):
        validate_answer_semantics(
            mean_query,
            derivation=_derivation(
                [_fact("invented_mean", 6.4, value_kind="text")], [], "6.4"
            ),
            answer={"freeform": {"text": "6.4"}},
        )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the average of the three values?", True),
        ("What is the mean value across the three tasks?", True),
        ("What mean squared error does Cedar report?", False),
        ("What average precision does Cedar report?", False),
    ],
)
def test_mean_aggregation_query_detection_avoids_named_metrics(
    question: str, expected: bool
) -> None:
    assert is_mean_aggregation_query(Query("mean_gate", question, ["freeform"])) is expected


def _percent_change_operation(
    *,
    old: Any,
    new: Any,
    result: Any,
    direction: str,
    exact: bool = True,
) -> dict[str, Any]:
    operation: dict[str, Any] = {
        "id": "relative_change",
        "kind": "percent_change",
        "fact_ids": ["old", "new"],
        "old_fact_id": "old",
        "new_fact_id": "new",
        "old": old,
        "new": new,
        "direction": direction,
        "scale": 100,
        "result": result,
        "answer_binding": _binding(
            "answer.freeform.text", result, f"{result}%"
        ),
    }
    if exact:
        operation["exact"] = True
    return operation


def test_percent_change_exact_decrease_uses_old_as_denominator() -> None:
    query = Query(
        "percent_decrease",
        "By what percentage did the error decrease from 120 to 90?",
        ["freeform"],
    )
    facts = [_fact("old", 120), _fact("new", 90)]
    operation = _percent_change_operation(
        old=120,
        new=90,
        result=25,
        direction="decrease",
    )

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "25%"),
        answer={"freeform": {"text": "25%"}},
    )

    assert validated["operations"][0]["result"] == 25


def test_percent_change_exact_increase_uses_explicit_fact_roles() -> None:
    query = Query(
        "percent_increase",
        "What is the relative increase from the baseline to the refined value?",
        ["freeform"],
    )
    facts = [_fact("old", 80), _fact("new", 100)]
    operation = _percent_change_operation(
        old=80,
        new=100,
        result=25,
        direction="increase",
    )
    # fact_ids are an unordered dependency inventory; the explicit role fields
    # determine which sourced value is the denominator.
    operation["fact_ids"] = ["new", "old"]
    operation["answer_binding"]["answer_fragment"] = "25 percent"

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "25 percent"),
        answer={"freeform": {"text": "25 percent"}},
    )

    assert validated["operations"][0]["direction"] == "increase"


def test_percent_change_requires_deterministic_rounding_when_nonterminating() -> None:
    query = Query(
        "rounded_percent_decrease",
        "What is the percentage decrease from 30 to 20?",
        ["freeform"],
    )
    facts = [_fact("old", 30), _fact("new", 20)]
    operation = _percent_change_operation(
        old=30,
        new=20,
        result="33.3",
        direction="decrease",
        exact=False,
    )

    with pytest.raises(DerivationValidationError, match="requires exact=true or a rounding"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "33.3%"),
            answer={"freeform": {"text": "33.3%"}},
        )

    operation["rounding"] = {"decimal_places": 1, "mode": "half_up"}
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "33.3%"),
        answer={"freeform": {"text": "33.3%"}},
    )
    assert validated["operations"][0]["result"] == "33.3"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"new": 89}, "old/new do not match"),
        ({"direction": "increase"}, "contradicts new < old"),
        ({"scale": 1}, "scale must equal 100"),
        ({"old_fact_id": "new"}, "must be distinct"),
    ],
)
def test_percent_change_rejects_ungrounded_or_ambiguous_formula(
    mutation: dict[str, Any], error: str
) -> None:
    query = Query(
        "percent_decrease",
        "By what percentage did the error decrease?",
        ["freeform"],
    )
    facts = [_fact("old", 120), _fact("new", 90)]
    operation = _percent_change_operation(
        old=120,
        new=90,
        result=25,
        direction="decrease",
    )
    operation.update(mutation)

    with pytest.raises(DerivationValidationError, match=error):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "25%"),
            answer={"freeform": {"text": "25%"}},
        )


def test_percent_change_rejects_zero_origin_denominator() -> None:
    query = Query(
        "percent_increase",
        "What is the percentage increase from zero to ten?",
        ["freeform"],
    )
    facts = [_fact("old", 0), _fact("new", 10)]
    operation = _percent_change_operation(
        old=0,
        new=10,
        result=100,
        direction="increase",
    )

    with pytest.raises(DerivationValidationError, match="denominator must be non-zero"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "100%"),
            answer={"freeform": {"text": "100%"}},
        )


def test_percentage_change_query_cannot_bypass_grounded_operation() -> None:
    query = Query(
        "percent_decrease",
        "What is the percentage decrease from the original to the refined score?",
        ["freeform"],
    )

    with pytest.raises(DerivationValidationError, match="percentage change"):
        validate_answer_semantics(
            query,
            derivation=_derivation([_fact("reported", 25)], [], "25%"),
            answer={"freeform": {"text": "25%"}},
        )


def test_divide_rounding_mode_and_exact_terminating_result() -> None:
    query = Query("q", "What is the quotient?", ["freeform"])
    facts = [_fact("numerator", 1), _fact("denominator", 8)]
    half_even = {
        "id": "ratio",
        "kind": "divide",
        "fact_ids": ["numerator", "denominator"],
        "operands": [1, 8],
        "result": "0.12",
        "rounding": {"decimal_places": 2, "mode": "half_even"},
        "answer_binding": _binding("answer.freeform.text", "0.12", "0.12"),
    }
    assert validate_answer_semantics(
        query,
        derivation=_derivation(facts, [half_even], "0.12"),
        answer={"freeform": {"text": "0.12"}},
    )

    exact = {
        **half_even,
        "result": "0.125",
        "exact": True,
        "answer_binding": _binding("answer.freeform.text", "0.125", "0.125"),
    }
    exact.pop("rounding")
    assert validate_answer_semantics(
        query,
        derivation=_derivation(facts, [exact], "0.125"),
        answer={"freeform": {"text": "0.125"}},
    )


def test_nonterminating_exact_divide_is_rejected() -> None:
    query = Query("q", "What is one third?", ["freeform"])
    operation = {
        "id": "ratio",
        "kind": "divide",
        "fact_ids": ["numerator", "denominator"],
        "operands": [1, 3],
        "result": "0.333",
        "exact": True,
        "answer_binding": _binding("answer.freeform.text", "0.333", "0.333"),
    }
    with pytest.raises(DerivationValidationError, match="non-terminating"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("numerator", 1), _fact("denominator", 3)],
                [operation],
                "0.333",
            ),
            answer={"freeform": {"text": "0.333"}},
        )


def test_variable_mc_label_and_exact_option_text_are_required() -> None:
    query = Query(
        "q",
        "Which option?",
        ["multiple_choice"],
        options={"A": "Alpha", "B": "Beta", "E": "Epsilon"},
    )
    derivation = _derivation(
        [_fact(value="Epsilon")],
        [],
        "Epsilon",
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "f1",
                "answer_fragment": "Epsilon",
            }
        ],
    )
    answer = {
        "multiple_choice": {
            "label": "E",
            "selected_option_text": "Epsilon",
        }
    }
    assert validate_answer_semantics(query, derivation=derivation, answer=answer)

    answer["multiple_choice"]["selected_option_text"] = "Epsilon "
    with pytest.raises(DerivationValidationError, match="exactly equal"):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_combined_freeform_and_mc_bind_independent_surface_forms() -> None:
    query = Query(
        "q",
        "By how much does the method improve?",
        ["freeform", "multiple_choice"],
        options={"A": "10.2", "D": "20.6"},
    )
    freeform = "The method improves Avg@64 from 11.7 to 32.3, or 20.6 points."
    derivation = _derivation(
        [_fact("improvement", "20.6")],
        [],
        freeform,
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "fact",
                "source_id": "improvement",
                "answer_fragment": "20.6",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "improvement",
                "answer_fragment": "20.6",
            },
        ],
    )
    answer = {
        "freeform": {"text": freeform},
        "multiple_choice": {"label": "D", "selected_option_text": "20.6"},
    }

    validated = validate_answer_semantics(
        query, derivation=derivation, answer=answer
    )
    assert validated["final_semantic_answer"] == freeform
    assert len(validated["answer_bindings"]) == 2


def test_combined_freeform_and_mc_cannot_bind_different_conclusions() -> None:
    query = Query(
        "q",
        "Which result is supported?",
        ["freeform", "multiple_choice"],
        options={"A": "Alpha", "B": "Beta"},
    )
    derivation = _derivation(
        [_fact("freeform_result", "Alpha"), _fact("mc_result", "Beta")],
        [],
        "Alpha",
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "fact",
                "source_id": "freeform_result",
                "answer_fragment": "Alpha",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "mc_result",
                "answer_fragment": "Beta",
            },
        ],
    )

    with pytest.raises(DerivationValidationError, match="must share at least one"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={
                "freeform": {"text": "Alpha"},
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "Beta",
                },
            },
        )


def test_mc_only_still_requires_final_to_equal_selected_option() -> None:
    query = Query(
        "q",
        "Which option?",
        ["multiple_choice"],
        options={"A": "Alpha", "B": "Beta"},
    )
    derivation = _derivation(
        [_fact("selected", "Beta")],
        [],
        "A longer answer about Beta",
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "selected",
                "answer_fragment": "Beta",
            }
        ],
    )
    with pytest.raises(
        DerivationValidationError,
        match="exactly equal selected_option_text",
    ):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "Beta",
                }
            },
        )


def test_lookup_binding_rejects_fact_final_answer_contradiction() -> None:
    query = Query("q", "What is reported?", ["freeform"])
    derivation = _derivation(
        [_fact("reported", "42")],
        [],
        "99",
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "fact",
                "source_id": "reported",
                "answer_fragment": "99",
            }
        ],
    )
    with pytest.raises(DerivationValidationError, match="sourced fact value"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={"freeform": {"text": "99"}},
        )


def test_lookup_binding_accepts_atomic_string_fact_phrase() -> None:
    query = Query("q", "What hardware was used?", ["freeform"])
    value = "single NVIDIA RTX 4090 GPU"
    answer_text = value

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [_fact("hardware", value)],
            [],
            answer_text,
            answer_bindings=[
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "hardware",
                    "answer_fragment": value,
                }
            ],
        ),
        answer={"freeform": {"text": answer_text}},
    )

    assert validated["facts"][0]["value"] == value


def test_atomic_freeform_rejects_verbose_q005_style_lookup() -> None:
    query = Query(
        "q005",
        "What is the index of the last reference in the paper?",
        ["freeform"],
    )
    fact = _fact("last_reference_index", "67")
    answer_text = "The last reference index is 67."

    with pytest.raises(
        DerivationValidationError, match="minimal atomic freeform"
    ):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [fact],
                [],
                answer_text,
                answer_bindings=[
                    {
                        "answer_path": "answer.freeform.text",
                        "source_type": "fact",
                        "source_id": "last_reference_index",
                        "answer_fragment": "67",
                    }
                ],
            ),
            answer={"freeform": {"text": answer_text}},
        )


def test_atomic_freeform_accepts_minimal_q005_style_lookup() -> None:
    query = Query(
        "q005",
        "What is the index of the last reference in the paper?",
        ["freeform"],
    )
    fact = _fact("last_reference_index", "67")

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [fact],
            [],
            "67",
            answer_bindings=[
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "last_reference_index",
                    "answer_fragment": "67",
                }
            ],
        ),
        answer={"freeform": {"text": "67"}},
    )

    assert validated["final_semantic_answer"] == "67"


@pytest.mark.parametrize(
    "answer_text",
    ["67.0", "sixty-seven", "sixty seven", "67."],
)
def test_atomic_integer_freeform_rejects_noncanonical_numeric_surfaces(
    answer_text: str,
) -> None:
    query = Query(
        "q005",
        "What is the index of the last reference in the paper?",
        ["freeform"],
    )
    fact = _fact("last_reference_index", 67)

    with pytest.raises(DerivationValidationError):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [fact],
                [],
                answer_text,
                answer_bindings=[
                    {
                        "answer_path": "answer.freeform.text",
                        "source_type": "fact",
                        "source_id": "last_reference_index",
                        "answer_fragment": answer_text,
                    }
                ],
            ),
            answer={"freeform": {"text": answer_text}},
        )


def test_atomic_integer_freeform_rejects_comma_form_but_accepts_outer_quotes() -> None:
    query = Query("q", "What is the reported index?", ["freeform"])

    with pytest.raises(
        DerivationValidationError, match="minimal atomic freeform"
    ):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("index", 1000)],
                [],
                "1,000",
                answer_bindings=[
                    {
                        "answer_path": "answer.freeform.text",
                        "source_type": "fact",
                        "source_id": "index",
                        "answer_fragment": "1,000",
                    }
                ],
            ),
            answer={"freeform": {"text": "1,000"}},
        )

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [_fact("index", 67)],
            [],
            '"67"',
            answer_bindings=[
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "index",
                    "answer_fragment": '"67"',
                }
            ],
        ),
        answer={"freeform": {"text": '"67"'}},
    )

    assert validated["final_semantic_answer"] == '"67"'


def test_atomic_boolean_freeform_rejects_true_when_canonical_surface_is_yes() -> None:
    query = Query("q", "Does A exceed B?", ["freeform"])
    facts = [_fact("left", 2), _fact("right", 1)]
    operation = {
        "id": "comparison",
        "kind": "compare",
        "fact_ids": ["left", "right"],
        "left": 2,
        "operator": ">",
        "right": 1,
        "result": True,
        "answer_binding": _binding("answer.freeform.text", True, "true"),
    }

    with pytest.raises(
        DerivationValidationError, match="minimal atomic freeform"
    ):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "true"),
            answer={"freeform": {"text": "true"}},
        )


def test_duplicate_answer_binding_cannot_bypass_minimal_freeform_check() -> None:
    query = Query(
        "q005",
        "What is the index of the last reference in the paper?",
        ["freeform"],
    )
    binding = {
        "answer_path": "answer.freeform.text",
        "source_type": "fact",
        "source_id": "last_reference_index",
        "answer_fragment": "67",
    }

    with pytest.raises(
        DerivationValidationError, match="duplicates an earlier"
    ):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("last_reference_index", "67")],
                [],
                "The last reference index is 67.",
                answer_bindings=[binding, dict(binding)],
            ),
            answer={
                "freeform": {"text": "The last reference index is 67."}
            },
        )


def test_distinct_sources_allow_a_genuinely_compound_freeform_answer() -> None:
    query = Query(
        "q_compound",
        "What hardware and runtime were used?",
        ["freeform"],
    )
    answer_text = "Helios X90; CUDA 14"

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [
                _fact("hardware", "Helios X90"),
                _fact("runtime", "CUDA 14"),
            ],
            [],
            answer_text,
            answer_bindings=[
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "hardware",
                    "answer_fragment": "Helios X90",
                },
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "runtime",
                    "answer_fragment": "CUDA 14",
                },
            ],
        ),
        answer={"freeform": {"text": answer_text}},
    )

    assert len(validated["answer_bindings"]) == 2


def test_atomic_freeform_allows_prose_when_question_requests_explanation() -> None:
    query = Query(
        "q_explain",
        "Explain what the last reference index is.",
        ["freeform"],
    )
    fact = _fact("last_reference_index", "67")
    answer_text = "The last reference index is 67."

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [fact],
            [],
            answer_text,
            answer_bindings=[
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "last_reference_index",
                    "answer_fragment": "67",
                }
            ],
        ),
        answer={"freeform": {"text": answer_text}},
    )

    assert validated["final_semantic_answer"] == answer_text


@pytest.mark.parametrize(
    "answer_text",
    [
        "The experiments did not use a single NVIDIA RTX 4090 GPU.",
        "A single NVIDIA RTX 4090 GPU was not used.",
        "A single NVIDIA RTX 4090 GPU could not be used.",
        "A single NVIDIA RTX 4090 GPU should not be used.",
        "The experiments used an A100 instead of a single NVIDIA RTX 4090 GPU.",
        "The experiments ran without a single NVIDIA RTX 4090 GPU.",
    ],
)
def test_lookup_binding_rejects_locally_negated_atomic_fact(
    answer_text: str,
) -> None:
    query = Query("q", "What hardware was used?", ["freeform"])
    value = "single NVIDIA RTX 4090 GPU"

    with pytest.raises(DerivationValidationError, match="locally negated"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("hardware", value)],
                [],
                answer_text,
                answer_bindings=[
                    {
                        "answer_path": "answer.freeform.text",
                        "source_type": "fact",
                        "source_id": "hardware",
                        "answer_fragment": value,
                    }
                ],
            ),
            answer={"freeform": {"text": answer_text}},
        )


@pytest.mark.parametrize(
    "answer_text",
    [
        "The experiments used a single NVIDIA RTX 4090 GPU, not multiple GPUs.",
        "Without modification, all experiments used a single NVIDIA RTX 4090 GPU.",
        "The method ran without fine-tuning on a single NVIDIA RTX 4090 GPU.",
    ],
)
def test_lookup_binding_does_not_overreject_unrelated_negation(
    answer_text: str,
) -> None:
    query = Query("q", "Explain what hardware was used.", ["freeform"])
    value = "single NVIDIA RTX 4090 GPU"

    assert validate_answer_semantics(
        query,
        derivation=_derivation(
            [_fact("hardware", value)],
            [],
            answer_text,
            answer_bindings=[
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "hardware",
                    "answer_fragment": value,
                }
            ],
        ),
        answer={"freeform": {"text": answer_text}},
    )


def test_lookup_binding_accepts_explicit_negative_source_fact() -> None:
    query = Query("q", "Explain whether an A100 was used.", ["freeform"])
    value = "did not use an A100 GPU"
    answer_text = f"The experiments {value}."

    assert validate_answer_semantics(
        query,
        derivation=_derivation(
            [_fact("hardware", value)],
            [],
            answer_text,
            answer_bindings=[
                {
                    "answer_path": "answer.freeform.text",
                    "source_type": "fact",
                    "source_id": "hardware",
                    "answer_fragment": value,
                }
            ],
        ),
        answer={"freeform": {"text": answer_text}},
    )


def test_lookup_binding_rejects_string_fact_embedded_inside_another_number() -> None:
    query = Query("q", "What is reported?", ["freeform"])
    derivation = _derivation(
        [_fact("reported", "42")],
        [],
        "142",
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "fact",
                "source_id": "reported",
                "answer_fragment": "142",
            }
        ],
    )

    with pytest.raises(DerivationValidationError, match="sourced fact value"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={"freeform": {"text": "142"}},
        )


def test_argmax_candidates_must_match_fact_label_values() -> None:
    query = Query("q", "Which is largest?", ["freeform"])
    facts = [
        _fact("a", {"label": "A", "value": 29}),
        _fact("b", {"label": "B", "value": 32}),
        _fact("c", {"label": "C", "value": 30}),
    ]
    operation = {
        "id": "largest",
        "kind": "argmax",
        "fact_ids": ["a", "b", "c"],
        "candidates": [
            {"label": "A", "value": 29},
            {"label": "B", "value": 31},
            {"label": "C", "value": 30},
        ],
        "result": "B",
        "answer_binding": _binding("answer.freeform.text", "B", "B"),
    }
    with pytest.raises(DerivationValidationError, match="referenced facts"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "B"),
            answer={"freeform": {"text": "B"}},
        )


def test_argmax_result_must_match_candidate_values_and_answer() -> None:
    query = Query("q", "Which is largest?", ["freeform"])
    facts = [
        _fact("a", {"label": "A", "value": 29}),
        _fact("b", {"label": "B", "value": 32}),
        _fact("c", {"label": "C", "value": 30}),
    ]
    operation = {
        "id": "largest",
        "kind": "argmax",
        "fact_ids": ["a", "b", "c"],
        "candidates": [
            {"label": "A", "value": 29},
            {"label": "B", "value": 32},
            {"label": "C", "value": 30},
        ],
        "result": "A",
        "answer_binding": _binding("answer.freeform.text", "A", "A"),
    }
    with pytest.raises(DerivationValidationError, match="winner"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "A"),
            answer={"freeform": {"text": "A"}},
        )

    operation.update(
        {
            "result": "B",
            "answer_binding": _binding("answer.freeform.text", "B", "B"),
        }
    )
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "B"),
        answer={"freeform": {"text": "B"}},
    )
    assert validated["operations"][0]["result"] == "B"


def test_argmax_binding_rejects_winner_label_inside_a_larger_token() -> None:
    query = Query("q", "Which is largest?", ["freeform"])
    facts = [
        _fact("cedar", {"label": "Cedar", "value": 17}),
        _fact("flint", {"label": "Flint", "value": 24}),
        _fact("quartz", {"label": "Quartz", "value": 19}),
    ]
    operation = {
        "id": "largest",
        "kind": "argmax",
        "fact_ids": ["cedar", "flint", "quartz"],
        "candidates": [
            {"label": "Cedar", "value": 17},
            {"label": "Flint", "value": 24},
            {"label": "Quartz", "value": 19},
        ],
        "result": "Flint",
        "answer_binding": _binding(
            "answer.freeform.text", "Flint", "SuperFlint"
        ),
    }

    with pytest.raises(DerivationValidationError, match="does not express expected"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "SuperFlint"),
            answer={"freeform": {"text": "SuperFlint"}},
        )


def test_fact_contract_requires_unique_ids_names_and_typed_value() -> None:
    query = Query("q", "What is reported?", ["freeform"])
    invalid = _fact()
    invalid.pop("value_kind")
    with pytest.raises(DerivationValidationError, match="missing required fields"):
        validate_answer_semantics(
            query,
            derivation=_derivation([invalid], [], "value"),
            answer={"freeform": {"text": "value"}},
        )

    with pytest.raises(DerivationValidationError, match="duplicate derivation fact id"):
        validate_answer_semantics(
            query,
            derivation=_derivation([_fact("same"), _fact("same")], [], "value"),
            answer={"freeform": {"text": "value"}},
        )

    with pytest.raises(DerivationValidationError, match="duplicate derivation fact name"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("a", name="Repeated"), _fact("b", name=" repeated ")],
                [],
                "value",
            ),
            answer={"freeform": {"text": "value"}},
        )

    computed = _fact(value="99", value_kind="computed")
    with pytest.raises(DerivationValidationError, match="value_kind must be one of"):
        validate_answer_semantics(
            query,
            derivation=_derivation([computed], [], "99"),
            answer={"freeform": {"text": "99"}},
        )


def test_operation_contract_requires_unique_id_fact_ids_and_binding() -> None:
    query = Query("q", "What is the sum?", ["freeform"])
    operation = {
        "id": "sum",
        "kind": "add",
        "fact_ids": ["a", "b"],
        "operands": [1, 2],
        "result": 3,
        "answer_binding": _binding("answer.freeform.text", 3, "3"),
    }
    duplicate = {**operation}
    with pytest.raises(DerivationValidationError, match="duplicate derivation operation id"):
        validate_answer_semantics(
            query,
            derivation=_derivation(
                [_fact("a", 1), _fact("b", 2)],
                [operation, duplicate],
                "3",
            ),
            answer={"freeform": {"text": "3"}},
        )

    operation["fact_ids"] = ["a", "invented"]
    with pytest.raises(DerivationValidationError, match="unknown facts"):
        validate_answer_semantics(
            query,
            derivation=_derivation([_fact("a", 1)], [operation], "3"),
            answer={"freeform": {"text": "3"}},
        )


def test_typed_table_answer_binding_requires_exact_value() -> None:
    query = Query(
        "q",
        "How many panels?",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Panel Count", "type": "number", "is_row_key": False},
        ],
    )
    items = ["a", "b", "c"]
    operation = {
        "id": "count",
        "kind": "count",
        "fact_ids": ["panels"],
        "items": items,
        "result": 3,
        "answer_binding": _binding("answer.table.rows[0].Panel Count", 3),
    }
    derivation = _derivation(
        [
            _fact("method", "Nova"),
            _fact("panels", items, value_kind="visual"),
        ],
        [operation],
        "3",
        answer_bindings=[
            {
                "answer_path": "answer.table.rows[0].Method",
                "source_type": "fact",
                "source_id": "method",
                "answer_fragment": "Nova",
            },
            {
                "answer_path": "answer.table.rows[0].Panel Count",
                "source_type": "operation",
                "source_id": "count",
            },
        ],
    )
    answer = {"table": {"rows": [{"Method": "Nova", "Panel Count": 3}]}}
    assert validate_answer_semantics(query, derivation=derivation, answer=answer)

    answer["table"]["rows"][0]["Panel Count"] = 2
    with pytest.raises(DerivationValidationError, match="does not exactly equal"):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_table_schema_requires_native_types_and_exact_columns() -> None:
    query = Query(
        "q",
        "Return rows.",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": True},
            {"name": "Score", "type": "number", "is_row_key": False},
            {"name": "Passed", "type": "boolean", "is_row_key": False},
        ],
    )
    rows = [{"Method": "Nova", "Score": 0.9, "Passed": True}]
    assert validate_table_rows(query, rows) == rows

    with pytest.raises(DerivationValidationError, match="must be a JSON number"):
        validate_table_rows(
            query, [{"Method": "Nova", "Score": ".9", "Passed": True}]
        )
    with pytest.raises(DerivationValidationError, match="contain exactly"):
        validate_table_rows(
            query,
            [{"Method": "Nova", "Score": 0.9, "Passed": True, "Extra": 1}],
        )
    with pytest.raises(DerivationValidationError, match="duplicate table row key"):
        validate_table_rows(query, [rows[0], {**rows[0], "Score": 1.0}])


def test_table_implicit_row_key_matches_official_normalization() -> None:
    query = Query(
        "q",
        "Return rows.",
        ["table"],
        table_schema=[
            {"name": "Method", "type": "string", "is_row_key": False},
            {"name": "Score", "type": "number", "is_row_key": False},
        ],
    )

    with pytest.raises(DerivationValidationError, match="duplicate table row key"):
        validate_table_rows(
            query,
            [
                {"Method": "Nova X", "Score": 1.0},
                {"Method": " ‘nova   x’ ", "Score": 2.0},
            ],
        )

    # The official scorer's implicit fallback permits one empty-string key;
    # only explicitly declared row-key columns have a non-empty requirement.
    assert validate_table_rows(query, [{"Method": "", "Score": 1.0}]) == [
        {"Method": "", "Score": 1.0}
    ]


def test_lookup_final_answer_must_equal_freeform() -> None:
    query = Query("q", "What is reported?", ["freeform"])
    with pytest.raises(DerivationValidationError, match="freeform.text"):
        validate_answer_semantics(
            query,
            derivation=_derivation([_fact("reported", "42")], [], "42"),
            answer={"freeform": {"text": "41"}},
        )


def test_delta_query_rejects_computed_value_disguised_as_text_fact() -> None:
    query = Query(
        "delta",
        "By how much does the method increase the score?",
        ["freeform", "multiple_choice"],
        options={"A": "10.2", "D": "20.6"},
    )
    derivation = _derivation(
        [_fact("claimed_delta", "20.6", value_kind="text")],
        [],
        "20.6",
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "fact",
                "source_id": "claimed_delta",
                "answer_fragment": "20.6",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "claimed_delta",
                "answer_fragment": "20.6",
            },
        ],
    )

    with pytest.raises(DerivationValidationError, match="numeric delta"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={
                "freeform": {"text": "20.6"},
                "multiple_choice": {
                    "label": "D",
                    "selected_option_text": "20.6",
                },
            },
        )


def test_delta_query_accepts_operand_grounded_subtraction_for_both_outputs() -> None:
    query = Query(
        "delta",
        "By how much does the method increase the score?",
        ["freeform", "multiple_choice"],
        options={"A": "10.2", "D": "20.6"},
    )
    facts = [
        _fact("after", 32.3, value_kind="visual"),
        _fact("before", 11.7, value_kind="visual"),
    ]
    operation = {
        "id": "increase",
        "kind": "subtract",
        "fact_ids": ["after", "before"],
        "operands": [32.3, 11.7],
        "result": 20.6,
        "answer_binding": _binding("answer.multiple_choice", 20.6, "20.6"),
    }
    derivation = _derivation(
        facts,
        [operation],
        "20.6",
        answer_bindings=[
            {
                "answer_path": "answer.freeform.text",
                "source_type": "operation",
                "source_id": "increase",
                "answer_fragment": "20.6",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "operation",
                "source_id": "increase",
                "answer_fragment": "20.6",
            },
        ],
    )

    validated = validate_answer_semantics(
        query,
        derivation=derivation,
        answer={
            "freeform": {"text": "20.6"},
            "multiple_choice": {"label": "D", "selected_option_text": "20.6"},
        },
    )

    assert validated["operations"][0]["result"] == 20.6


def _two_value_option_case(
    *, bind_first_value: bool
) -> tuple[Query, dict[str, Any], dict[str, Any]]:
    query = Query(
        "two_values",
        "What values are reported for the two weighting rules?",
        ["multiple_choice"],
        options={
            "A": "1/t^2+1: 5.51, 1/t^2+1/sigma^2: 190.80",
            "B": "1/t^2+1: 20.65, 1/t^2+1/sigma^2: 3.18",
            "D": "1/t^2+1: 190.80, 1/t^2+1/sigma^2: 5.51",
        },
    )
    selected_text = query.options["D"]
    bindings = [
        {
            "answer_path": "answer.multiple_choice",
            "source_type": "fact",
            "source_id": "second_value",
            "answer_fragment": "5.51",
        }
    ]
    if bind_first_value:
        bindings.append(
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "first_value",
                "answer_fragment": "190.80",
            }
        )
    derivation = _derivation(
        [_fact("first_value", 190.80), _fact("second_value", 5.51)],
        [],
        selected_text,
        answer_bindings=bindings,
    )
    answer = {
        "multiple_choice": {"label": "D", "selected_option_text": selected_text}
    }
    return query, derivation, answer


def test_compound_multiple_choice_rejects_binding_only_one_numeric_component() -> None:
    query, derivation, answer = _two_value_option_case(bind_first_value=False)

    with pytest.raises(
        DerivationValidationError,
        match="ungrounded distinguishing numeric component",
    ):
        validate_answer_semantics(query, derivation=derivation, answer=answer)


def test_compound_mc_accepts_all_values_and_ignores_formula_constants() -> None:
    query, derivation, answer = _two_value_option_case(bind_first_value=True)

    validated = validate_answer_semantics(
        query, derivation=derivation, answer=answer
    )

    assert len(validated["answer_bindings"]) == 2


def test_mc_numeric_grounding_ignores_numbers_inside_dataset_identifiers() -> None:
    query = Query(
        "dataset_suffix",
        "What k values are used for CIFAR-10 and ImageNet-256?",
        ["multiple_choice"],
        options={
            "A": "CIFAR-10: k=10, ImageNet-256: k=14",
            "B": "CIFAR-10: k=15, ImageNet-256: k=12",
            "C": "Both use k=12",
        },
    )
    selected_text = query.options["B"]
    derivation = _derivation(
        [_fact("cifar_k", 15), _fact("imagenet_k", 12)],
        [],
        selected_text,
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "cifar_k",
                "answer_fragment": "k=15",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "imagenet_k",
                "answer_fragment": "k=12",
            },
        ],
    )

    assert validate_answer_semantics(
        query,
        derivation=derivation,
        answer={
            "multiple_choice": {"label": "B", "selected_option_text": selected_text}
        },
    )


def test_match_query_requires_vector_comparison_not_a_status_text_fact() -> None:
    query = Query(
        "vector_match",
        "What channel means do the two systems use, and do they match?",
        ["multiple_choice"],
        options={
            "A": "They use the same values: [0.865, -0.278, 0.216, 0.374]",
            "C": (
                "First: [1.56, -0.695, 0.483, 0.729]; "
                "second: [0.865, -0.278, 0.216, 0.374] — different values"
            ),
        },
    )
    selected_text = query.options["A"]
    derivation = _derivation(
        [_fact("match_status", "same", value_kind="text")],
        [],
        selected_text,
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "match_status",
                "answer_fragment": "same",
            }
        ],
    )

    with pytest.raises(DerivationValidationError, match="equality comparison"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={
                "multiple_choice": {
                    "label": "A",
                    "selected_option_text": selected_text,
                }
            },
        )


def test_matching_loss_noun_phrase_is_not_misclassified_as_equality_question() -> None:
    query = Query(
        "matching_loss",
        (
            "What kernel form is used for the moment matching loss, and what "
            "is its width expression?"
        ),
        ["multiple_choice"],
        options={
            "A": "RBF with w(s,t) = 1/|c_skip(s,t)|",
            "B": "Laplace with w(s,t) = 1/|c_out(s,t)|",
            "C": "Laplace with w(s,t) = |c_out(s,t)|",
        },
    )
    selected_text = query.options["B"]
    derivation = _derivation(
        [_fact("kernel", "Laplace"), _fact("width", "1/|c_out(s,t)|")],
        [],
        selected_text,
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "kernel",
                "answer_fragment": "Laplace",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "width",
                "answer_fragment": "1/|c_out(s,t)|",
            },
        ],
    )

    assert validate_answer_semantics(
        query,
        derivation=derivation,
        answer={
            "multiple_choice": {"label": "B", "selected_option_text": selected_text}
        },
    )


def test_vector_comparison_and_rounded_compound_values_are_fully_grounded() -> None:
    query = Query(
        "vector_match",
        "What channel means do the two systems use, and do they match?",
        ["multiple_choice"],
        options={
            "A": "They use the same values: [0.865, -0.278, 0.216, 0.374]",
            "C": (
                "First: [1.56, -0.695, 0.483, 0.729]; "
                "second: [0.865, -0.278, 0.216, 0.374] — different values"
            ),
        },
    )
    selected_text = query.options["C"]
    first = [1.56, -0.695, 0.483, 0.729]
    second = [0.86488, -0.27787343, 0.21616915, 0.3738409]
    operation = {
        "id": "means_match",
        "kind": "compare",
        "fact_ids": ["first_means", "second_means"],
        "left": first,
        "operator": "==",
        "right": second,
        "result": False,
        "answer_binding": _binding(
            "answer.multiple_choice", False, "different values"
        ),
    }
    derivation = _derivation(
        [_fact("first_means", first), _fact("second_means", second)],
        [operation],
        selected_text,
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "first_means",
                "answer_fragment": "[1.56, -0.695, 0.483, 0.729]",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "second_means",
                "answer_fragment": "[0.865, -0.278, 0.216, 0.374]",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "operation",
                "source_id": "means_match",
                "answer_fragment": "different values",
            },
        ],
    )

    validated = validate_answer_semantics(
        query,
        derivation=derivation,
        answer={
            "multiple_choice": {"label": "C", "selected_option_text": selected_text}
        },
    )

    assert validated["operations"][0]["result"] is False

    operation["operator"] = "!="
    operation["result"] = True
    operation["answer_binding"] = _binding(
        "answer.multiple_choice", True, "different values"
    )
    validated_not_equal = validate_answer_semantics(
        query,
        derivation=derivation,
        answer={
            "multiple_choice": {"label": "C", "selected_option_text": selected_text}
        },
    )
    assert validated_not_equal["operations"][0]["result"] is True


def test_extreme_query_cannot_bypass_argmax_with_winner_fact() -> None:
    query = Query(
        "maximum",
        "Which system has the highest score?",
        ["multiple_choice"],
        options={"A": "Alpha", "B": "Beta"},
    )
    derivation = _derivation(
        [_fact("winner", "Beta", value_kind="text")],
        [],
        "Beta",
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "winner",
                "answer_fragment": "Beta",
            }
        ],
    )

    with pytest.raises(DerivationValidationError, match="argmax/argmin"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "Beta",
                }
            },
        )


def test_highest_visible_axis_extent_is_a_visual_lookup_not_argmax() -> None:
    selected_text = "population distance axis peaks near 70; SpeechSet error = 0.412"
    query = Query(
        "compound_axis_lookup",
        (
            "In the human-baseline color analysis, roughly what is the highest "
            "population distance value on the horizontal axis, and in the "
            "synthetic-audio study, what error does the multimodal model achieve "
            "on SpeechSet?"
        ),
        ["multiple_choice"],
        options={
            "A": "population distance axis peaks near 50; SpeechSet error = 0.517",
            "B": selected_text,
        },
    )
    facts = [
        _fact("axis_extent", 70, value_kind="visual"),
        _fact("table_metric", 0.412, value_kind="reported"),
    ]
    derivation = _derivation(
        facts,
        [],
        selected_text,
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "axis_extent",
                "answer_fragment": "70",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "table_metric",
                "answer_fragment": "0.412",
            },
        ],
    )

    assert is_axis_extent_lookup_query(query) is True
    assert requires_extremum_operation(query) is False
    validated = validate_answer_semantics(
        query,
        derivation=derivation,
        answer={
            "multiple_choice": {
                "label": "B",
                "selected_option_text": selected_text,
            }
        },
    )
    assert validated["operations"] == []
    assert [fact["value_kind"] for fact in validated["facts"]] == [
        "visual",
        "reported",
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Roughly what is the maximum labeled tick on the vertical axis?",
        "What is the highest x-axis value?",
        "Approximately what was the largest visible y axis limit?",
    ],
)
def test_axis_extent_detector_handles_generic_surface_forms(question: str) -> None:
    query = Query("axis_surface", question, ["freeform"])
    assert is_axis_extent_lookup_query(query) is True
    assert requires_extremum_operation(query) is False


def test_axis_extent_exception_does_not_waive_another_real_extremum_clause() -> None:
    query = Query(
        "axis_and_winner",
        (
            "What is the highest x-axis value, and which method has the best "
            "score across all systems?"
        ),
        ["multiple_choice"],
        options={"A": "axis 70; Cedar", "B": "axis 70; Flint"},
    )
    derivation = _derivation(
        [
            _fact("axis_extent", 70, value_kind="visual"),
            _fact("winner", "Flint", value_kind="text"),
        ],
        [],
        "axis 70; Flint",
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "axis_extent",
                "answer_fragment": "70",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "winner",
                "answer_fragment": "Flint",
            },
        ],
    )

    assert is_axis_extent_lookup_query(query) is True
    assert requires_extremum_operation(query) is True
    with pytest.raises(DerivationValidationError, match="argmax/argmin"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "axis 70; Flint",
                }
            },
        )


def test_explicitly_reported_compound_optimum_can_bind_without_argmax() -> None:
    selected_text = "gamma=0.98 optimal; gamma>1.0 harms performance"
    query = Query(
        "reported_optimum",
        (
            "In the Cedar paper, what optimal temporal decay factor gamma "
            "achieves the best performance across three base models, and what "
            "happens when gamma exceeds 1.0?"
        ),
        ["multiple_choice"],
        options={
            "A": "gamma=0.97 optimal; gamma>1.0 harms performance",
            "C": selected_text,
        },
    )
    facts = [
        _fact("optimum", "gamma=0.98", value_kind="reported"),
        _fact("effect", "harms performance", value_kind="text"),
    ]
    derivation = _derivation(
        facts,
        [],
        selected_text,
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "optimum",
                "answer_fragment": "gamma=0.98",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "effect",
                "answer_fragment": "harms performance",
            },
        ],
    )

    validated = validate_answer_semantics(
        query,
        derivation=derivation,
        answer={
            "multiple_choice": {
                "label": "C",
                "selected_option_text": selected_text,
            }
        },
    )

    assert validated["operations"] == []


def test_explicit_only_filter_allows_grounded_singleton_argmax() -> None:
    query = Query(
        "filtered_maximum",
        "Which system trained only on BaseSet has the highest score?",
        ["multiple_choice"],
        options={"A": "Cedar", "B": "Flint"},
    )
    facts = [_fact("flint", {"label": "Flint", "value": 74})]
    operation = {
        "id": "eligible_largest",
        "kind": "argmax",
        "fact_ids": ["flint"],
        "candidates": [{"label": "Flint", "value": 74}],
        "result": "Flint",
        "answer_binding": _binding(
            "answer.multiple_choice", "Flint", "Flint"
        ),
    }

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "Flint"),
        answer={
            "multiple_choice": {
                "label": "B",
                "selected_option_text": "Flint",
            }
        },
    )

    assert validated["operations"][0]["candidates"] == [
        {"label": "Flint", "value": 74}
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Which system has the highest score?",
        "Which system has the highest score? Return only the system name.",
        "Name the system with the highest score.",
    ],
)
def test_singleton_argmax_without_explicit_only_filter_is_rejected(
    question: str,
) -> None:
    query = Query(
        "unfiltered_maximum",
        question,
        ["multiple_choice"],
        options={"A": "Cedar", "B": "Flint"},
    )
    facts = [_fact("flint", {"label": "Flint", "value": 74})]
    operation = {
        "id": "largest",
        "kind": "argmax",
        "fact_ids": ["flint"],
        "candidates": [{"label": "Flint", "value": 74}],
        "result": "Flint",
        "answer_binding": _binding(
            "answer.multiple_choice", "Flint", "Flint"
        ),
    }

    with pytest.raises(DerivationValidationError, match="one-candidate argmax"):
        validate_answer_semantics(
            query,
            derivation=_derivation(facts, [operation], "Flint"),
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "Flint",
                }
            },
        )


def test_filtered_extremum_cannot_bypass_comparison_with_reported_name() -> None:
    query = Query(
        "filtered_maximum",
        "Which system trained only on BaseSet has the highest score?",
        ["multiple_choice"],
        options={"A": "Cedar", "B": "Flint"},
    )
    derivation = _derivation(
        [_fact("winner", "Flint", value_kind="reported")],
        [],
        "Flint",
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "winner",
                "answer_fragment": "Flint",
            }
        ],
    )

    with pytest.raises(DerivationValidationError, match="argmax/argmin"):
        validate_answer_semantics(
            query,
            derivation=derivation,
            answer={
                "multiple_choice": {
                    "label": "B",
                    "selected_option_text": "Flint",
                }
            },
        )


def test_numeric_fact_binding_accepts_equivalent_number_word_in_mc_option() -> None:
    query = Query(
        "count_words",
        "How many properties and neighbors are used?",
        ["multiple_choice"],
        options={
            "A": "Four properties and three neighbors",
            "B": "Five properties and five neighbors",
        },
    )
    answer = {
        "multiple_choice": {
            "label": "B",
            "selected_option_text": "Five properties and five neighbors",
        }
    }
    derivation = _derivation(
        [_fact("properties", 5), _fact("neighbors", 5)],
        [],
        "Five properties and five neighbors",
        answer_bindings=[
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "properties",
                "answer_fragment": "Five properties",
            },
            {
                "answer_path": "answer.multiple_choice",
                "source_type": "fact",
                "source_id": "neighbors",
                "answer_fragment": "five neighbors",
            },
        ],
    )

    validated = validate_answer_semantics(query, derivation=derivation, answer=answer)

    assert len(validated["answer_bindings"]) == 2


def test_inequality_binding_accepts_two_values_joined_by_contrast() -> None:
    query = Query(
        "citation_years",
        "Do the two papers use the same author-year citation?",
        ["multiple_choice"],
        options={
            "A": "Blei et al. (2003) in both papers",
            "B": "Blei et al. (2009) in paper A but Blei et al. (2003) in paper B",
        },
    )
    fragment = (
        "Blei et al. (2009) in paper A but Blei et al. (2003) in paper B"
    )
    operation = {
        "id": "different_years",
        "kind": "compare",
        "fact_ids": ["paper_a", "paper_b"],
        "left": 2009,
        "operator": "!=",
        "right": 2003,
        "result": True,
        "answer_binding": _binding("answer.multiple_choice", True, fragment),
    }
    answer = {
        "multiple_choice": {
            "label": "B",
            "selected_option_text": fragment,
        }
    }

    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [_fact("paper_a", 2009), _fact("paper_b", 2003)],
            [operation],
            fragment,
            answer_bindings=[
                {
                    "answer_path": "answer.multiple_choice",
                    "source_type": "fact",
                    "source_id": "paper_a",
                    "answer_fragment": "Blei et al. (2009)",
                },
                {
                    "answer_path": "answer.multiple_choice",
                    "source_type": "fact",
                    "source_id": "paper_b",
                    "answer_fragment": "Blei et al. (2003)",
                },
                {
                    "answer_path": "answer.multiple_choice",
                    "source_type": "operation",
                    "source_id": "different_years",
                    "answer_fragment": fragment,
                },
            ],
        ),
        answer=answer,
    )

    assert validated["operations"][0]["result"] is True
