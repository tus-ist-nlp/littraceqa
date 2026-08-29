"""Aggregating gold recall over the top k candidates, broken down by task_family.

paper_recall_macro measures recall against the submitted set, after the LLM has
narrowed it; this measures whether retrieval had the paper as a candidate at all.
Queries fall into single / multi / total by gold's task_family.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from littraceqa.search.contracts import MULTI, SINGLE
from evaluate import CANDIDATE_RECALL_KS, evaluate

# The headline k, matching ReadingConfig.max_candidates in pipeline.py. The curve
# is assumed to contain it.
CANDIDATE_RECALL_K = 20
assert CANDIDATE_RECALL_K in CANDIDATE_RECALL_KS


def _gold(query_id: str, task_family: str, paper_ids: list[str]) -> dict:
    return {
        "query_id": query_id,
        "task_family": task_family,
        "answer_types": [],
        "gold_papers": [{"paper_id": pid} for pid in paper_ids],
        "evidence": [],
        "answer": {},
    }


def _pred(query_id: str, candidate_papers: list[str] | None) -> dict:
    record = {"query_id": query_id, "gold_papers": [], "evidence": [], "answer": {}}
    if candidate_papers is not None:
        record["candidate_papers"] = candidate_papers
    return record


def test_old_predictions_without_field_yield_none() -> None:
    """An old prediction file without candidate_papers reports None, not a number.

    **0.0 would read as "retrieval found nothing at all"**, so "not measured" is
    said with None.
    """
    gold = [_gold("q1", SINGLE, ["p1"]), _gold("q2", MULTI, ["p1", "p2"])]
    pred = [_pred("q1", None), _pred("q2", None)]

    metrics = evaluate(gold, pred)["metrics"]

    for scenario in ("single", "multi", "total"):
        assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_{scenario}_macro"] is None


def test_recall_splits_by_task_family() -> None:
    """single, multi and total each aggregate correctly."""
    gold = [
        _gold("q1", SINGLE, ["p1"]),                    # hit at rank 1   -> 1.0
        _gold("q2", SINGLE, ["p9"]),                    # not a candidate -> 0.0
        _gold("q3", MULTI, ["p1", "p2", "p3", "p4"]),   # 2 of 4          -> 0.5
    ]
    pred = [
        _pred("q1", ["p1", "p2", "p3"]),
        _pred("q2", ["p1", "p2", "p3"]),
        _pred("q3", ["p1", "p2", "px", "py"]),
    ]

    metrics = evaluate(gold, pred)["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 0.5  # (1.0+0.0)/2
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_multi_macro"] == 0.5
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 0.5  # (1+0+0.5)/3


def test_only_top_k_counts() -> None:
    """Gold past rank k counts as not found — the ranking has to matter."""
    ranked = [f"noise{i}" for i in range(CANDIDATE_RECALL_K)] + ["p1"]  # p1 sits at k+1
    metrics = evaluate([_gold("q1", SINGLE, ["p1"])], [_pred("q1", ranked)])["metrics"]
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 0.0

    ranked_in = ["p1"] + [f"noise{i}" for i in range(CANDIDATE_RECALL_K)]  # p1 sits at 1
    metrics = evaluate([_gold("q1", SINGLE, ["p1"])], [_pred("q1", ranked_in)])["metrics"]
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 1.0


def test_missing_prediction_counts_as_zero() -> None:
    """A query with no prediction counts as recall 0, as prf() treats a missing submission."""
    gold = [_gold("q1", SINGLE, ["p1"]), _gold("q2", SINGLE, ["p2"])]
    pred = [_pred("q1", ["p1"])]  # nothing for q2

    metrics = evaluate(gold, pred)["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 0.5
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 0.5


def test_absent_task_family_group_is_none() -> None:
    """A task_family absent from gold makes only that key None.

    So that mean([]) == 0.0 is never read as "recall 0".
    """
    gold = [_gold("q1", MULTI, ["p1"])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] is None
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_multi_macro"] == 1.0


def test_unknown_task_family_counts_only_in_total() -> None:
    """An unknown or missing task_family counts towards total alone."""
    gold = [{"query_id": "q1", "answer_types": [], "gold_papers": [{"paper_id": "p1"}]}]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 1.0
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] is None
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_multi_macro"] is None


def test_empty_candidates_is_zero_not_none() -> None:
    """No candidates at all (retrieval returned nothing) is 0.0, distinct from None."""
    metrics = evaluate([_gold("q1", SINGLE, ["p1"])], [_pred("q1", [])])["metrics"]
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 0.0


def test_recall_curve_is_emitted_for_every_k_and_scenario() -> None:
    """The curve is reported for every k x scenario combination."""
    gold = [_gold("q1", SINGLE, ["p1"]), _gold("q2", MULTI, ["p2", "p3"])]
    pred = [_pred("q1", ["p1"]), _pred("q2", ["p2", "p3"])]

    metrics = evaluate(gold, pred)["metrics"]

    for k in CANDIDATE_RECALL_KS:
        for scenario in ("single", "multi", "total"):
            assert f"candidate_recall_at{k}_{scenario}_macro" in metrics


def test_recall_curve_is_monotonic_in_k() -> None:
    """Recall never falls as k grows: a larger k's top-k contains the smaller one's.

    A curve that is not monotonic means the ranking is being cut wrongly, so this
    catches implementation bugs.
    """
    # Gold at ranks 1, 4, 9, 18 and 40, so each k picks up one more.
    ranked = [f"noise{i}" for i in range(50)]
    for rank, pid in [(0, "g1"), (3, "g2"), (8, "g3"), (17, "g4"), (39, "g5")]:
        ranked[rank] = pid
    gold = [_gold("q1", MULTI, ["g1", "g2", "g3", "g4", "g5"])]

    metrics = evaluate(gold, [_pred("q1", ranked)])["metrics"]
    curve = [metrics[f"candidate_recall_at{k}_multi_macro"] for k in CANDIDATE_RECALL_KS]

    assert curve == sorted(curve), f"not monotonic in k: {curve}"
    # k=(1,5,10,20,50,70) finds 1,2,3,4,5,5 of them
    # (k=70 is there to observe the paper-to-paper expansion; this prediction has few
    #  candidates, so it equals @50)
    assert curve == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0]


def test_curve_distinguishes_rank1_from_rank20() -> None:
    """Equal recall@20, and recall@1 still separates good ranking from bad.

    This is why the curve exists: it shows whether the reranker or the fusion has
    anything left to give.
    """
    top = ["p1"] + [f"noise{i}" for i in range(19)]          # hit at rank 1
    bottom = [f"noise{i}" for i in range(19)] + ["p1"]        # scraped in at rank 20

    gold = [_gold("q1", SINGLE, ["p1"])]
    m_top = evaluate(gold, [_pred("q1", top)])["metrics"]
    m_bottom = evaluate(gold, [_pred("q1", bottom)])["metrics"]

    # recall@20 cannot tell them apart
    assert m_top["candidate_recall_at20_single_macro"] == 1.0
    assert m_bottom["candidate_recall_at20_single_macro"] == 1.0
    # recall@1 does
    assert m_top["candidate_recall_at1_single_macro"] == 1.0
    assert m_bottom["candidate_recall_at1_single_macro"] == 0.0


# ---------------------------------------------------------------------------
# evidence_candidate_recall: the same, with the denominator narrowed to the gold
# that has evidence attached.
#
# multi_paper gold includes papers named in gold_papers with no evidence at all —
# peer papers on the same topic that the question never names. No amount of
# searching with the question brings them near, so with all of gold as the
# denominator the ceiling sticks and improvements to an index or the reranker
# become unreadable. This measures the part that can actually move.
# ---------------------------------------------------------------------------


def _gold_with_evidence(
    query_id: str, task_family: str, paper_ids: list[str], backed: list[str]
) -> dict:
    record = _gold(query_id, task_family, paper_ids)
    record["evidence"] = [
        {"paper_id": pid, "source_type": "text_span", "locator": {"page": 1}}
        for pid in backed
    ]
    return record


def test_evidence_recall_excludes_gold_without_evidence() -> None:
    """Dropping unbacked gold from the denominator raises recall."""
    # Of the 4 gold papers only p1 is backed, and p1 is the only one retrieved.
    gold = [_gold_with_evidence("q1", MULTI, ["p1", "p2", "p3", "p4"], ["p1"])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics["candidate_recall_at20_multi_macro"] == 0.25  # 1 of 4
    assert metrics["evidence_candidate_recall_at20_multi_macro"] == 1.0  # 1 of 1


def test_evidence_recall_matches_plain_recall_when_all_gold_backed() -> None:
    """With every gold paper backed, the two metrics agree."""
    gold = [_gold_with_evidence("q1", MULTI, ["p1", "p2"], ["p1", "p2"])]
    metrics = evaluate(gold, [_pred("q1", ["p1", "noise"])])["metrics"]

    assert metrics["candidate_recall_at20_multi_macro"] == 0.5
    assert metrics["evidence_candidate_recall_at20_multi_macro"] == 0.5


def test_evidence_recall_ignores_evidence_for_non_gold_papers() -> None:
    """Evidence for a paper outside gold does not inflate the denominator."""
    gold = [_gold_with_evidence("q1", MULTI, ["p1", "p2"], ["p1", "outsider"])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    # The denominator is {p1}, one paper, not {p1, outsider}
    assert metrics["evidence_candidate_recall_at20_multi_macro"] == 1.0


def test_query_without_any_evidence_is_dropped_not_scored_as_one() -> None:
    """A query with no evidence-backed gold is left out of the aggregation.

    recall_at_k() returns 1.0 on empty gold (to match prf), so including it would
    pad the average with free full marks. The counts go into details as well, so the
    difference can be traced.
    """
    gold = [
        _gold_with_evidence("q1", MULTI, ["p1", "p2"], []),   # nothing backed -> excluded
        _gold_with_evidence("q2", MULTI, ["p3", "p4"], ["p3"]),  # 1/1 -> 1.0
    ]
    result = evaluate(gold, [_pred("q1", ["px"]), _pred("q2", ["p3"])])

    assert result["metrics"]["evidence_candidate_recall_at20_multi_macro"] == 1.0
    assert result["details"]["candidate_recall_counts"]["multi"] == 2
    assert result["details"]["evidence_candidate_recall_counts"]["multi"] == 1


def test_evidence_recall_is_none_when_no_query_qualifies() -> None:
    """With no qualifying query the value is None, not 0.0 — nothing was measured."""
    gold = [_gold_with_evidence("q1", MULTI, ["p1"], [])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics["evidence_candidate_recall_at20_multi_macro"] is None
    assert metrics["candidate_recall_at20_multi_macro"] == 1.0


def test_evidence_recall_curve_is_emitted_for_every_k_and_scenario() -> None:
    gold = [_gold_with_evidence("q1", SINGLE, ["p1"], ["p1"])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    for scenario in ("single", "multi", "total"):
        for k in CANDIDATE_RECALL_KS:
            assert f"evidence_candidate_recall_at{k}_{scenario}_macro" in metrics


SUBMISSION_METRIC_KEYS = (
    "paper_precision_macro",
    "paper_recall_macro",
    "paper_f1_macro",
    "evidence_precision_macro",
    "evidence_recall_macro",
    "evidence_f1_macro",
    "multiple_choice_accuracy",
    "freeform_exact_match",
    "table_row_f1_macro",
    "table_cell_accuracy_macro",
    "table_cell_accuracy_micro",
)


def test_submission_metrics_are_omitted_by_default():
    """By default the submission-side metrics are not reported.

    Choosing the papers to submit and generating the answers both belong to the
    reading team, so printed alongside, numbers we cannot move get read as noise —
    or worse, as improvement.
    """
    gold = [_gold("q1", SINGLE, ["p1"])]
    pred = [_pred("q1", ["p1", "p2"])]

    metrics = evaluate(gold, pred)["metrics"]
    assert not [key for key in SUBMISSION_METRIC_KEYS if key in metrics]
    # The retrieval side is reported as always
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 1.0


def test_include_submission_restores_the_old_metric_set():
    """`--metrics all` (include_submission=True) adds the submission side."""
    gold = [_gold("q1", SINGLE, ["p1"])]
    pred = [_pred("q1", ["p1", "p2"])]

    metrics = evaluate(gold, pred, include_submission=True)["metrics"]
    assert all(key in metrics for key in SUBMISSION_METRIC_KEYS)
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 1.0
