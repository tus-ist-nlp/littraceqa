"""Test candidate recall grouped by the actual number of gold papers.

Paper recall after LLM selection mixes retrieval and selection quality.
Candidate recall instead measures whether retrieval itself found each paper.
Grouping uses the gold-paper count rather than ``task_family``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from littraceqa.di_pipeline.agent.task_family import MULTI, SINGLE
from evaluate import CANDIDATE_RECALL_KS, evaluate

# 看板指標の k（reading.yaml の max_candidates と一致）。カーブにも含まれている前提。
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
    """candidate_papers を持たない古い予測ファイルでは、3指標とも None になる。

    0.0 を出してしまうと「検索が全く拾えていない」と誤読されるため、
    「測っていない」ことを None で表す。
    """
    gold = [_gold("q1", SINGLE, ["p1"]), _gold("q2", MULTI, ["p1", "p2"])]
    pred = [_pred("q1", None), _pred("q2", None)]

    metrics = evaluate(gold, pred)["metrics"]

    for scenario in ("single", "multi", "total"):
        assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_{scenario}_macro"] is None


def test_recall_splits_by_gold_paper_count_not_task_family() -> None:
    """Validation-only labels must not control the single/multi grouping."""
    gold = [
        _gold("q1", MULTI, ["p1"]),
        _gold("q2", MULTI, ["p9"]),
        _gold("q3", SINGLE, ["p1", "p2", "p3", "p4"]),
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
    assert metrics[f"candidate_all_gold_at{CANDIDATE_RECALL_K}_single_rate"] == 0.5
    assert metrics[f"candidate_all_gold_at{CANDIDATE_RECALL_K}_multi_rate"] == 0.0


def test_only_top_k_counts() -> None:
    """k位より後ろにある gold は拾えていない扱いになる（順位が意味を持つ）。"""
    ranked = [f"noise{i}" for i in range(CANDIDATE_RECALL_K)] + ["p1"]  # p1 は k+1 位
    metrics = evaluate([_gold("q1", SINGLE, ["p1"])], [_pred("q1", ranked)])["metrics"]
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 0.0

    ranked_in = ["p1"] + [f"noise{i}" for i in range(CANDIDATE_RECALL_K)]  # p1 は 1 位
    metrics = evaluate([_gold("q1", SINGLE, ["p1"])], [_pred("q1", ranked_in)])["metrics"]
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 1.0


def test_missing_prediction_counts_as_zero() -> None:
    """予測が欠けているクエリは recall 0 として総計に入る（prf() の未提出扱いと同じ）。"""
    gold = [_gold("q1", SINGLE, ["p1"]), _gold("q2", SINGLE, ["p2"])]
    pred = [_pred("q1", ["p1"])]  # q2 の予測が無い

    metrics = evaluate(gold, pred)["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 0.5
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 0.5


def test_absent_gold_count_group_is_none() -> None:
    """Return None when no query belongs to a gold-count group.

    mean([]) == 0.0 を「recall 0」と誤読しないため。
    """
    gold = [_gold("q1", SINGLE, ["p1", "p2"])]
    metrics = evaluate(gold, [_pred("q1", ["p1", "p2"])])["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] is None
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_multi_macro"] == 1.0


def test_missing_task_family_is_still_grouped_by_gold_count() -> None:
    """A missing task_family must not prevent single/multi evaluation."""
    gold = [{"query_id": "q1", "answer_types": [], "gold_papers": [{"paper_id": "p1"}]}]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 1.0
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 1.0
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_multi_macro"] is None


def test_empty_candidates_is_zero_not_none() -> None:
    """候補ゼロ（検索が何も返さなかった）は 0.0。未計測の None と区別する。"""
    metrics = evaluate([_gold("q1", SINGLE, ["p1"])], [_pred("q1", [])])["metrics"]
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] == 0.0


def test_recall_curve_is_emitted_for_every_k_and_scenario() -> None:
    """カーブは k × シナリオ の全組み合わせで出る。"""
    gold = [_gold("q1", SINGLE, ["p1"]), _gold("q2", MULTI, ["p2", "p3"])]
    pred = [_pred("q1", ["p1"]), _pred("q2", ["p2", "p3"])]

    metrics = evaluate(gold, pred)["metrics"]

    for k in CANDIDATE_RECALL_KS:
        for scenario in ("single", "multi", "total"):
            assert f"candidate_recall_at{k}_{scenario}_macro" in metrics
            assert f"candidate_all_gold_at{k}_{scenario}_rate" in metrics


def test_recall_curve_is_monotonic_in_k() -> None:
    """k を増やすと recall は下がらない（上位k件は k が大きいほど包含関係で広い）。

    カーブが単調でなければ順位の切り出し方がおかしいので、実装バグの検出になる。
    """
    # gold を 1位 / 4位 / 9位 / 18位 / 40位 に散らして、k ごとに段階的に拾えるようにする。
    ranked = [f"noise{i}" for i in range(50)]
    for rank, pid in [(0, "g1"), (3, "g2"), (8, "g3"), (17, "g4"), (39, "g5")]:
        ranked[rank] = pid
    gold = [_gold("q1", MULTI, ["g1", "g2", "g3", "g4", "g5"])]

    metrics = evaluate(gold, [_pred("q1", ranked)])["metrics"]
    curve = [metrics[f"candidate_recall_at{k}_multi_macro"] for k in CANDIDATE_RECALL_KS]

    assert curve == sorted(curve), f"kについて単調でない: {curve}"
    # k=(1,5,10,20,50) でそれぞれ 1,2,3,4,5 本拾える -> 0.2,0.4,0.6,0.8,1.0
    assert curve == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_curve_distinguishes_rank1_from_rank20() -> None:
    """recall@20 が同じでも、recall@1 で「順位付けの良さ」を区別できる。

    これがカーブを入れた理由（reranker/RRF重みの改善余地が見えるかどうか）。
    """
    top = ["p1"] + [f"noise{i}" for i in range(19)]          # 1位で当てる
    bottom = [f"noise{i}" for i in range(19)] + ["p1"]        # 20位でギリギリ

    gold = [_gold("q1", SINGLE, ["p1"])]
    m_top = evaluate(gold, [_pred("q1", top)])["metrics"]
    m_bottom = evaluate(gold, [_pred("q1", bottom)])["metrics"]

    # recall@20 では区別がつかない
    assert m_top["candidate_recall_at20_single_macro"] == 1.0
    assert m_bottom["candidate_recall_at20_single_macro"] == 1.0
    # recall@1 で差が出る
    assert m_top["candidate_recall_at1_single_macro"] == 1.0
    assert m_bottom["candidate_recall_at1_single_macro"] == 0.0
