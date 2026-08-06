from __future__ import annotations

from typing import Any

import pytest

from littraceqa.answer_derivation import (
    DerivationValidationError,
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
        "answer_binding": _binding("answer.freeform.text", 4, "four"),
    }
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(
            [
                _fact("row1", [["(a)", "(b)"]], value_kind="visual"),
                _fact("row2", ["(c)", "(d)"], value_kind="visual"),
            ],
            [operation],
            "four",
        ),
        answer={"freeform": {"text": "four"}},
    )
    assert validated["operations"][0]["result"] == 4


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
            "answer_binding": _binding("answer.freeform.text", "B", "Backbone-B"),
        }
    )
    validated = validate_answer_semantics(
        query,
        derivation=_derivation(facts, [operation], "Backbone-B"),
        answer={"freeform": {"text": "Backbone-B"}},
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
