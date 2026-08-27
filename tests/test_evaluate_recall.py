"""候補上位k本に対する gold 論文 recall（task_family別）の集計ロジックのテスト。

paper_recall_macro は LLM の絞り込み後の提出セットに対する recall なので、
「検索がそもそも候補に拾えていたか」を別途測るのがこの指標。gold の task_family で
single / multi / total に振り分ける。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from littraceqa.di_pipeline.contracts import MULTI, SINGLE
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


def test_recall_splits_by_task_family() -> None:
    """single / multi / total がそれぞれ正しく集計される。"""
    gold = [
        _gold("q1", SINGLE, ["p1"]),                    # 1位にヒット -> 1.0
        _gold("q2", SINGLE, ["p9"]),                    # 候補に無い    -> 0.0
        _gold("q3", MULTI, ["p1", "p2", "p3", "p4"]),   # 2/4 ヒット    -> 0.5
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


def test_absent_task_family_group_is_none() -> None:
    """gold にその task_family が1件も無ければ、そのキーだけ None。

    mean([]) == 0.0 を「recall 0」と誤読しないため。
    """
    gold = [_gold("q1", MULTI, ["p1"])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] is None
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_multi_macro"] == 1.0


def test_unknown_task_family_counts_only_in_total() -> None:
    """未知/欠落の task_family は total にだけ入れる。"""
    gold = [{"query_id": "q1", "answer_types": [], "gold_papers": [{"paper_id": "p1"}]}]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 1.0
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_single_macro"] is None
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
    # k=(1,5,10,20,50,70) でそれぞれ 1,2,3,4,5,5 本拾える
    # （k=70 は論文→論文展開ぶんの観測用。この予測は候補が少ないので50と同値）
    assert curve == [0.2, 0.4, 0.6, 0.8, 1.0, 1.0]


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


# ---------------------------------------------------------------------------
# evidence_candidate_recall: 分母を「evidence が紐づいた gold」に絞った版。
#
# multi_paper の gold には、gold_papers に名前だけあって evidence が1件も無い論文が
# 混ざる（質問文が名指ししていない同トピックのピア論文）。質問文をどう検索しても
# 近傍に来ないので、gold 全体を分母にすると天井が張り付いて索引や reranker の
# 改善が読めなくなる。そこを分けて測るための指標。
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
    """根拠の無い gold を分母から外すと recall が上がる。"""
    # gold 4本のうち根拠付きは p1 のみ。候補は p1 しか当てていない。
    gold = [_gold_with_evidence("q1", MULTI, ["p1", "p2", "p3", "p4"], ["p1"])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    assert metrics["candidate_recall_at20_multi_macro"] == 0.25  # 4本中1本
    assert metrics["evidence_candidate_recall_at20_multi_macro"] == 1.0  # 1本中1本


def test_evidence_recall_matches_plain_recall_when_all_gold_backed() -> None:
    """全 gold に根拠が付いていれば、両指標は一致する。"""
    gold = [_gold_with_evidence("q1", MULTI, ["p1", "p2"], ["p1", "p2"])]
    metrics = evaluate(gold, [_pred("q1", ["p1", "noise"])])["metrics"]

    assert metrics["candidate_recall_at20_multi_macro"] == 0.5
    assert metrics["evidence_candidate_recall_at20_multi_macro"] == 0.5


def test_evidence_recall_ignores_evidence_for_non_gold_papers() -> None:
    """gold に無い論文の evidence は分母を膨らませない（積は gold との共通部分）。"""
    gold = [_gold_with_evidence("q1", MULTI, ["p1", "p2"], ["p1", "outsider"])]
    metrics = evaluate(gold, [_pred("q1", ["p1"])])["metrics"]

    # 分母は {p1} の1本であって {p1, outsider} の2本ではない
    assert metrics["evidence_candidate_recall_at20_multi_macro"] == 1.0


def test_query_without_any_evidence_is_dropped_not_scored_as_one() -> None:
    """根拠付き gold が1本も無いクエリは集計から外す。

    recall_at_k() は gold が空だと 1.0 を返す（prf との整合）ため、そのまま
    足し込むと満点が水増しされる。件数も details に出して差分を追えるようにする。
    """
    gold = [
        _gold_with_evidence("q1", MULTI, ["p1", "p2"], []),   # 根拠ゼロ -> 除外
        _gold_with_evidence("q2", MULTI, ["p3", "p4"], ["p3"]),  # 1/1 -> 1.0
    ]
    result = evaluate(gold, [_pred("q1", ["px"]), _pred("q2", ["p3"])])

    assert result["metrics"]["evidence_candidate_recall_at20_multi_macro"] == 1.0
    assert result["details"]["candidate_recall_counts"]["multi"] == 2
    assert result["details"]["evidence_candidate_recall_counts"]["multi"] == 1


def test_evidence_recall_is_none_when_no_query_qualifies() -> None:
    """対象クエリが皆無なら 0.0 ではなく None（測っていないことを示す）。"""
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
    """既定では提出物側の指標を出さない（我々が上げるのは candidate_recall）。

    提出論文の選定も回答生成も読解チーム側の担当なので、並べておくと
    動かさない数字の上下をノイズとして読んでしまう。
    """
    gold = [_gold("q1", SINGLE, ["p1"])]
    pred = [_pred("q1", ["p1", "p2"])]

    metrics = evaluate(gold, pred)["metrics"]
    assert not [key for key in SUBMISSION_METRIC_KEYS if key in metrics]
    # 検索側は従来どおり出る
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 1.0


def test_include_submission_restores_the_old_metric_set():
    """`--metrics all`（include_submission=True）で提出物側も足せる。"""
    gold = [_gold("q1", SINGLE, ["p1"])]
    pred = [_pred("q1", ["p1", "p2"])]

    metrics = evaluate(gold, pred, include_submission=True)["metrics"]
    assert all(key in metrics for key in SUBMISSION_METRIC_KEYS)
    assert metrics[f"candidate_recall_at{CANDIDATE_RECALL_K}_total_macro"] == 1.0
