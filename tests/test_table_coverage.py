from __future__ import annotations

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.paper_tables import PaperTable
from littraceqa.di_pipeline.select.selector import PaperSelection
from littraceqa.di_pipeline.select.table_coverage import (
    EvidenceCoverageRefiner,
    ExplicitTableAnchorRefiner,
    SingleTableCoverageRefiner,
)


class StubTableSource:
    def __init__(self, tables: dict[str, tuple[PaperTable, ...]]) -> None:
        self._tables = tables

    def tables(self, paper_id: str) -> tuple[PaperTable, ...]:
        return self._tables.get(paper_id, ())


def _query(
    question: str,
    *,
    row_key: str = "Benchmarks",
    value: str = "Kitchen",
) -> Query:
    return Query(
        query_id="q",
        question=question,
        answer_types=["table"],
        table_schema=[
            {"name": row_key, "type": "string", "is_row_key": True},
            {"name": value, "type": "string", "is_row_key": False},
        ],
    )


def _table(paper_id: str, rows: list[list[str]], caption: str = "") -> PaperTable:
    row_tuple = tuple(tuple(row) for row in rows)
    return PaperTable(
        paper_id=paper_id,
        table_id="Table 5",
        caption=caption,
        rows=row_tuple,
        text="\n".join([caption, *(" | ".join(row) for row in rows)]),
    )


def _selection(*paper_ids: str) -> PaperSelection:
    return PaperSelection(
        paper_ids=paper_ids,
        expected_count=len(paper_ids),
        reason="stated_in_question",
    )


QUESTION = (
    "What percentage belongs to the Kitchen category in SUN RGB-D, "
    "ARKitScenes, Hypersim, Objectron, KITTI, and nuScenes?"
)
ROWS = [
    ["Benchmark", "Kitchen"],
    ["SUN RGB-D", "6"],
    ["ARKitScenes", "10.3"],
    ["Hypersim", "0.9"],
    ["Objectron", "32.4"],
    ["KITTI", "-"],
    ["nuScenes", "-"],
]


def test_refiner_selects_the_unique_table_covering_every_requested_row():
    source = StubTableSource(
        {
            "wrong": (_table("wrong", [["Benchmark", "Depth"], *ROWS[1:]]),),
            "right": (_table("right", ROWS),),
        }
    )
    refiner = SingleTableCoverageRefiner(source)

    result = refiner.refine(
        _query(QUESTION), ["wrong", "right", "other"], _selection("wrong", "right")
    )

    assert result.paper_ids == ("right",)
    assert result.expected_count == 1
    assert result.reason.endswith("+single_table_coverage")


def test_refiner_falls_back_when_coverage_is_not_unique():
    table = _table("paper-a", ROWS)
    source = StubTableSource(
        {
            "paper-a": (table,),
            "paper-b": (_table("paper-b", ROWS),),
        }
    )
    original = _selection("paper-a", "paper-b")

    result = SingleTableCoverageRefiner(source).refine(
        _query(QUESTION), ["paper-a", "paper-b"], original
    )

    assert result is original


def test_refiner_requires_distinct_rows_and_the_value_column():
    one_row = [
        ["Benchmark", "Kitchen"],
        ["SUN RGB-D ARKitScenes Hypersim Objectron KITTI nuScenes", "1"],
    ]
    no_value_column = [["Benchmark", "Depth"], *ROWS[1:]]
    source = StubTableSource(
        {
            "one-row": (_table("one-row", one_row),),
            "no-column": (_table("no-column", no_value_column),),
        }
    )
    original = _selection("one-row", "no-column")

    result = SingleTableCoverageRefiner(source).refine(
        _query(QUESTION), ["one-row", "no-column"], original
    )

    assert result is original


def test_refiner_does_not_treat_a_dataset_variant_as_the_requested_row():
    variant_rows = [
        ["Benchmark", "Kitchen"],
        ["SUN RGB-D", "6"],
        ["ARKitScenes", "10.3"],
        ["Hypersim", "0.9"],
        ["Objectron", "32.4"],
        ["KITTI-360", "2"],
        ["nuScenes", "-"],
    ]
    source = StubTableSource({"variant": (_table("variant", variant_rows),)})
    original = _selection("variant", "other")

    result = SingleTableCoverageRefiner(source).refine(
        _query(QUESTION), ["variant", "other"], original
    )

    assert result is original


def test_refiner_does_not_treat_a_caption_mention_as_a_value_column():
    table = _table(
        "caption-only",
        [["Benchmark", "Depth"], *ROWS[1:]],
        caption="Kitchen results are discussed elsewhere",
    )
    source = StubTableSource({"caption-only": (table,)})
    original = _selection("caption-only", "other")

    result = SingleTableCoverageRefiner(source).refine(
        _query(QUESTION), ["caption-only", "other"], original
    )

    assert result is original


def test_refiner_does_not_collapse_method_or_multi_source_questions():
    source = StubTableSource({"paper": (_table("paper", ROWS),)})
    refiner = SingleTableCoverageRefiner(source)
    original = _selection("paper", "other")
    methods = _query(QUESTION, row_key="Methods")
    multiple_sources = _query(QUESTION.replace("What", "Across these papers, what"))

    assert refiner.refine(methods, ["paper", "other"], original) is original
    assert (
        refiner.refine(multiple_sources, ["paper", "other"], original) is original
    )


def test_refiner_falls_back_without_a_production_table_schema():
    query = Query(query_id="q", question=QUESTION, answer_types=["table"])
    original = _selection("paper", "other")
    source = StubTableSource({"paper": (_table("paper", ROWS),)})

    result = SingleTableCoverageRefiner(source).refine(
        query, ["paper", "other"], original
    )

    assert result is original


def test_explicit_table_refiner_replaces_rank_one_when_equations_match():
    question = (
        "In the ECM paper Table 3, what FID does w_bar(t) = 1/t^2 + 1 "
        "achieve, compared to w_bar(t) = 1/t^2 + 1/sigma_data^2?"
    )
    query = Query(query_id="q", question=question, answer_types=["multiple_choice"])
    wrong = _table("wrong", [["Method", "FID"], ["ECM", "2.0"]])
    right = PaperTable(
        paper_id="right",
        table_id="Table 3",
        caption="Table 3: Performance of ECMs",
        rows=(("w(t)", "FID"), ("1/t^2 + 1", "6.78"), ("1/t^2 + 1/sigma_data^2", "5.51")),
        text=(
            "Table 3: Performance of ECMs\nw(t) | FID\n"
            "1/t^2 + 1 | 6.78\n1/t^2 + 1/sigma_data^2 | 5.51"
        ),
    )
    source = StubTableSource({"wrong": (wrong,), "right": (right,)})

    result = ExplicitTableAnchorRefiner(source).refine(
        query, ["wrong", "right"], _selection("wrong")
    )

    assert result.paper_ids == ("right",)
    assert result.reason.endswith("+explicit_table_coverage")


def test_explicit_table_refiner_requires_a_unique_full_match():
    question = (
        "In the ECM paper Table 3, what FID does w(t) = 1/t^2 + 1 achieve, "
        "compared to w(t) = 1/t^2 + 1/sigma_data^2?"
    )
    query = Query(query_id="q", question=question, answer_types=["multiple_choice"])
    partial = PaperTable(
        paper_id="partial",
        table_id="Table 3",
        caption="Table 3: ECM",
        rows=(("1/t^2 + 1", "6.78"),),
        text="Table 3: ECM\n1/t^2 + 1 | 6.78",
    )
    original = _selection("partial")

    result = ExplicitTableAnchorRefiner(StubTableSource({"partial": (partial,)})).refine(
        query, ["partial"], original
    )

    assert result is original


def test_explicit_table_refiner_rejects_a_formula_prefix_match():
    question = (
        "In the ECM paper Table 3, what FID does w(t) = 1/t^2 + 1 achieve, "
        "compared to w(t) = 1/t^2 + 1/sigma_data^2?"
    )
    query = Query(query_id="q", question=question, answer_types=["multiple_choice"])
    longer_only = PaperTable(
        paper_id="partial",
        table_id="Table 3",
        caption="Table 3: ECM",
        rows=(("1/t^2 + 1/sigma_data^2", "5.51"),),
        text="Table 3: ECM\n1/t^2 + 1/sigma_data^2 | 5.51",
    )
    original = _selection("partial")

    result = ExplicitTableAnchorRefiner(
        StubTableSource({"partial": (longer_only,)})
    ).refine(query, ["partial"], original)

    assert result is original


def test_explicit_table_refiner_rejects_a_formula_with_a_different_sign():
    question = (
        "In the ECM paper Table 3, what FID does w(t) = 1/t^2 + 1 achieve, "
        "compared to w(t) = 1/t^2 + 1/sigma_data^2?"
    )
    query = Query(query_id="q", question=question, answer_types=["multiple_choice"])
    wrong_sign = PaperTable(
        paper_id="wrong-sign",
        table_id="Table 3",
        caption="Table 3: ECM",
        rows=(
            ("-1/t^2 + 1", "6.78"),
            ("1/t^2 + 1/sigma_data^2", "5.51"),
        ),
        text=(
            "Table 3: ECM\n-1/t^2 + 1 | 6.78\n"
            "1/t^2 + 1/sigma_data^2 | 5.51"
        ),
    )
    original = _selection("wrong-sign")

    result = ExplicitTableAnchorRefiner(
        StubTableSource({"wrong-sign": (wrong_sign,)})
    ).refine(query, ["wrong-sign"], original)

    assert result is original


def test_explicit_table_refiner_keeps_parenthesized_formulas_distinct():
    question = (
        "In the ECM paper Table 3, what FID does w(t) = 1/t^2 + 1 achieve, "
        "compared to w(t) = 1/t^2 + 1/sigma_data^2?"
    )
    query = Query(query_id="q", question=question, answer_types=["multiple_choice"])
    different_formula = PaperTable(
        paper_id="different",
        table_id="Table 3",
        caption="Table 3: ECM",
        rows=(
            ("1/(t^2 + 1)", "6.78"),
            ("1/t^2 + 1/sigma_data^2", "5.51"),
        ),
        text=(
            "Table 3: ECM\n1/(t^2 + 1) | 6.78\n"
            "1/t^2 + 1/sigma_data^2 | 5.51"
        ),
    )
    original = _selection("different")

    result = ExplicitTableAnchorRefiner(
        StubTableSource({"different": (different_formula,)})
    ).refine(query, ["different"], original)

    assert result is original


def test_explicit_table_refiner_does_not_match_an_anchor_inside_another_name():
    question = (
        "In the ECM paper Table 3, what FID does w(t) = 1/t^2 + 1 achieve, "
        "compared to w(t) = 1/t^2 + 1/sigma_data^2?"
    )
    query = Query(query_id="q", question=question, answer_types=["multiple_choice"])
    other_method = PaperTable(
        paper_id="decm",
        table_id="Table 3",
        caption="Table 3: DECM",
        rows=(
            ("1/t^2 + 1", "6.78"),
            ("1/t^2 + 1/sigma_data^2", "5.51"),
        ),
        text=(
            "Table 3: DECM\n1/t^2 + 1 | 6.78\n"
            "1/t^2 + 1/sigma_data^2 | 5.51"
        ),
    )
    original = _selection("decm")

    result = ExplicitTableAnchorRefiner(
        StubTableSource({"decm": (other_method,)})
    ).refine(query, ["decm"], original)

    assert result is original


def test_evidence_refiner_keeps_the_explicit_table_decision():
    question = (
        "In the ECM paper Table 3, what FID does w(t) = 1/t^2 + 1 achieve, "
        "compared to w(t) = 1/t^2 + 1/sigma_data^2?"
    )
    query = Query(query_id="q", question=question, answer_types=["multiple_choice"])
    table = PaperTable(
        paper_id="right",
        table_id="Table 3",
        caption="Table 3: ECM",
        rows=(("1/t^2 + 1", "6.78"), ("1/t^2 + 1/sigma_data^2", "5.51")),
        text="Table 3: ECM\n1/t^2 + 1 | 6.78\n1/t^2 + 1/sigma_data^2 | 5.51",
    )

    result = EvidenceCoverageRefiner(StubTableSource({"right": (table,)})).refine(
        query, ["wrong", "right"], _selection("wrong")
    )

    assert result.paper_ids == ("right",)
