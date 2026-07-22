#!/usr/bin/env python3
"""Lightweight local evaluator for LitTraceQA development submissions.

This script is intended for participant-side validation on the public
development split. The official hidden-test evaluator may include additional
organizer-side checks, but follows the same high-level fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from littraceqa.di_pipeline.agent.task_family import MULTI, SINGLE

# 候補上位 k 本に gold 論文が入っていたか。paper_recall_macro は「LLM がどう絞ったか」と
# 「検索がそもそも拾えていたか」が混ざるが、こちらは絞り込み前の検索力だけを見る。
#
# 看板は recall@20（agent_style/reading.yaml の max_candidates=20 と一致＝LLM が実際に
# 見られる範囲＝実システムの天井）。k を並べてカーブで見るのは、recall@20 だけだと
# 「1位で当てた」と「20位でギリギリ入った」が区別できず、reranker や RRF 重みを
# いじる余地があるのかが見えないため:
#   recall@1 が高い          -> 順位付けまで的確
#   recall@1 低 / @20 高     -> 拾えているが順位が弱い（reranker/RRF重みが効く）
#   recall@20 も低い         -> そもそも索引に引っかかっていない（索引構成/分解の問題）
# 上限50は reading.py の CANDIDATE_PAPERS_LIMIT（予測に残す候補本数）に合わせている。
CANDIDATE_RECALL_KS = (1, 5, 10, 20, 50)
CANDIDATE_RECALL_SCENARIOS = ("single", "multi", "total")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.strip("\"'“”‘’`")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_id(value: Any) -> str:
    return str(value or "").strip()


def normalize_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def paper_id_set(record: dict[str, Any]) -> set[str]:
    papers = record.get("gold_papers") or record.get("papers") or []
    paper_ids: set[str] = set()
    if not isinstance(papers, list):
        return paper_ids
    for item in papers:
        if isinstance(item, str):
            paper_id = item
        elif isinstance(item, dict):
            paper_id = item.get("paper_id", "")
        else:
            paper_id = ""
        paper_id = normalize_id(paper_id)
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def normalize_visible_id(value: Any, prefix: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    match = re.fullmatch(rf"{prefix.lower()}\s*(\d+[a-z]?)", text)
    if match:
        return f"{prefix.lower()} {match.group(1)}"
    if re.fullmatch(r"\d+[a-z]?", text):
        return f"{prefix.lower()} {text}"
    return text


def coarse_evidence_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    paper_id = normalize_id(item.get("paper_id"))
    source_type = normalize_id(item.get("source_type"))
    if isinstance(item.get("locator"), dict):
        locator = item.get("locator")
    else:
        locator = {}
    page = str(locator.get("page", "")).strip()
    object_id = ""
    if source_type == "table":
        object_id = normalize_visible_id(locator.get("table_id"), "table")
    elif source_type == "figure":
        object_id = normalize_visible_id(locator.get("figure_id"), "figure")
    return (paper_id, source_type, page, object_id)


def evidence_set(record: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    evidence = record.get("evidence") or []
    if not isinstance(evidence, list):
        return set()
    keys = set()
    for item in evidence:
        if isinstance(item, dict):
            key = coarse_evidence_key(item)
            if key[0] and key[1] and key[2]:
                keys.add(key)
    return keys


def prf(gold: set[Any], pred: set[Any]) -> tuple[float, float, float]:
    if not gold and not pred:
        return (1.0, 1.0, 1.0)
    if not pred:
        if gold:
            return (0.0, 0.0, 0.0)
        return (0.0, 1.0, 0.0)
    correct = len(gold & pred)
    if pred:
        precision = correct / len(pred)
    else:
        precision = 0.0
    if gold:
        recall = correct / len(gold)
    else:
        recall = 1.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return (precision, recall, f1)


def candidate_paper_ids(record: dict[str, Any]) -> list[str] | None:
    """予測レコードの candidate_papers（関連度順）を正規化して返す。

    このフィールドを持たない古い予測ファイルとは区別したいので、「無い」場合は
    空リストではなく None を返す（呼び出し側で指標そのものを出さないため）。
    """
    papers = record.get("candidate_papers")
    if not isinstance(papers, list):
        return None
    paper_ids: list[str] = []
    for item in papers:
        if isinstance(item, str):
            paper_id = item
        elif isinstance(item, dict):
            paper_id = item.get("paper_id", "")
        else:
            paper_id = ""
        paper_id = normalize_id(paper_id)
        # 順位が意味を持つのでソートせず、重複だけ落として出現順を保つ。
        if paper_id and paper_id not in paper_ids:
            paper_ids.append(paper_id)
    return paper_ids


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    """関連度順 ranked の上位 k 本で gold をどれだけ拾えたか。

    gold が空のときに 1.0 を返すのは prf() の recall と挙動を揃えるため。
    """
    if not gold:
        return 1.0
    return len(gold & set(ranked[:k])) / len(gold)


def macro_or_none(values: list[float]) -> float | None:
    """該当クエリが1件も無ければ None を返す（mean() の 0.0 と区別する）。"""
    if not values:
        return None
    return mean(values)


def prediction_answer(record: dict[str, Any]) -> dict[str, Any]:
    answer = record.get("answer")
    if isinstance(answer, dict):
        return answer
    return {}


def multiple_choice_prediction(record: dict[str, Any]) -> str:
    answer = prediction_answer(record)
    mc = answer.get("multiple_choice")
    if isinstance(mc, dict):
        return normalize_id(mc.get("gold") or mc.get("answer") or mc.get("predicted_answer_id")).upper()
    return normalize_id(record.get("predicted_answer_id") or record.get("gold") or record.get("gold_answer")).upper()


def multiple_choice_gold(record: dict[str, Any]) -> str:
    mc = record.get("answer", {}).get("multiple_choice", {})
    if isinstance(mc, dict):
        return normalize_id(mc.get("gold")).upper()
    return ""


def freeform_prediction(record: dict[str, Any]) -> str:
    answer = prediction_answer(record)
    freeform = answer.get("freeform")
    if isinstance(freeform, dict):
        return normalize_text(freeform.get("text"))
    return normalize_text(record.get("answer_text") or record.get("freeform"))


def freeform_gold(record: dict[str, Any]) -> str:
    freeform = record.get("answer", {}).get("freeform", {})
    if isinstance(freeform, dict):
        return normalize_text(freeform.get("text"))
    return ""


def row_key_value(row: dict[str, Any], row_key_columns: list[str]) -> tuple[str, ...]:
    return tuple(normalize_text(row.get(column)) for column in row_key_columns)


def cell_equal(gold_value: Any, pred_value: Any, column_type: str) -> bool:
    if gold_value is None or pred_value is None:
        return gold_value is None and pred_value is None
    if column_type == "number":
        gold_number = normalize_number(gold_value)
        pred_number = normalize_number(pred_value)
        if gold_number is None or pred_number is None:
            return False
        return math.isclose(gold_number, pred_number, rel_tol=1e-6, abs_tol=1e-6)
    if column_type == "boolean":
        return bool(gold_value) == bool(pred_value)
    return normalize_text(gold_value) == normalize_text(pred_value)


def table_metrics(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, float | int]:
    gold_table = gold.get("answer", {}).get("table", {})
    pred_table = prediction_answer(pred).get("table", {})
    if not isinstance(gold_table, dict) or not isinstance(pred_table, dict):
        return {"row_precision": 0.0, "row_recall": 0.0, "row_f1": 0.0, "cell_accuracy": 0.0, "cell_correct": 0, "cell_total": 0}

    if isinstance(gold_table.get("schema"), list):
        schema = gold_table.get("schema")
    else:
        schema = []
    if isinstance(gold_table.get("rows"), list):
        gold_rows = gold_table.get("rows")
    else:
        gold_rows = []
    if isinstance(pred_table.get("rows"), list):
        pred_rows = pred_table.get("rows")
    else:
        pred_rows = []
    row_keys = [column["name"] for column in schema if isinstance(column, dict) and column.get("is_row_key")]
    if not row_keys and schema:
        row_keys = [str(schema[0].get("name", ""))]

    gold_by_key = {
        row_key_value(row, row_keys): row
        for row in gold_rows
        if isinstance(row, dict) and row_key_value(row, row_keys)
    }
    pred_by_key = {
        row_key_value(row, row_keys): row
        for row in pred_rows
        if isinstance(row, dict) and row_key_value(row, row_keys)
    }
    row_precision, row_recall, row_f1 = prf(set(gold_by_key), set(pred_by_key))

    cell_correct = 0
    cell_total = 0
    comparable_columns = [
        (str(column.get("name", "")), str(column.get("type", "string")))
        for column in schema
        if isinstance(column, dict) and str(column.get("name", "")) and not column.get("is_row_key")
    ]
    for key, gold_row in gold_by_key.items():
        pred_row = pred_by_key.get(key)
        if not isinstance(pred_row, dict):
            cell_total += len(comparable_columns)
            continue
        for column_name, column_type in comparable_columns:
            cell_total += 1
            if cell_equal(gold_row.get(column_name), pred_row.get(column_name), column_type):
                cell_correct += 1
    if cell_total:
        cell_accuracy = cell_correct / cell_total
    else:
        cell_accuracy = row_f1
    return {
        "row_precision": row_precision,
        "row_recall": row_recall,
        "row_f1": row_f1,
        "cell_accuracy": cell_accuracy,
        "cell_correct": cell_correct,
        "cell_total": cell_total,
    }


def mean(values: list[float]) -> float:
    if values:
        return sum(values) / len(values)
    return 0.0


def evaluate(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = {normalize_id(record.get("query_id")): record for record in gold_records}
    pred_by_id = {normalize_id(record.get("query_id")): record for record in pred_records}
    pred_by_id.pop("", None)

    missing_predictions = sorted(set(gold_by_id) - set(pred_by_id))
    extra_predictions = sorted(set(pred_by_id) - set(gold_by_id))

    # 1件でも候補列を持てば新形式の予測ファイルとみなす。ファイル全体に無ければ
    # （この変更より前に作った予測）指標を None にして、黙って 0.0 を出さない。
    has_candidate_papers = any("candidate_papers" in record for record in pred_records)

    paper_precision: list[float] = []
    paper_recall: list[float] = []
    paper_f1: list[float] = []
    evidence_precision: list[float] = []
    evidence_recall: list[float] = []
    evidence_f1: list[float] = []
    mc_correct = 0
    mc_total = 0
    freeform_exact_correct = 0
    freeform_total = 0
    table_row_f1: list[float] = []
    table_cell_accuracy: list[float] = []
    table_cell_correct = 0
    table_cell_total = 0
    # シナリオ -> k -> クエリごとの recall。single/multi は gold の task_family で振り分け、
    # total は全クエリ（＝single と multi の加重平均になる）。
    candidate_recall: dict[str, dict[int, list[float]]] = {
        scenario: {k: [] for k in CANDIDATE_RECALL_KS}
        for scenario in CANDIDATE_RECALL_SCENARIOS
    }

    for query_id, gold in gold_by_id.items():
        pred = pred_by_id.get(query_id, {})

        p, r, f = prf(paper_id_set(gold), paper_id_set(pred))
        paper_precision.append(p)
        paper_recall.append(r)
        paper_f1.append(f)

        if has_candidate_papers:
            # 打ち切り前の候補上位 k 本に gold 論文が入っていたか（＝検索力）。
            gold_paper_ids = paper_id_set(gold)
            ranked = candidate_paper_ids(pred) or []
            # task_family は本番入力から削られる（run_search.py --production-input）ため、
            # 予測側ではなく必ず gold 側から読む。未知/欠落なら total にだけ入れる。
            task_family = normalize_id(gold.get("task_family"))
            scenarios = ["total"]
            if task_family == SINGLE:
                scenarios.append("single")
            elif task_family == MULTI:
                scenarios.append("multi")
            for k in CANDIDATE_RECALL_KS:
                recall = recall_at_k(gold_paper_ids, ranked, k)
                for scenario in scenarios:
                    candidate_recall[scenario][k].append(recall)

        p, r, f = prf(evidence_set(gold), evidence_set(pred))
        evidence_precision.append(p)
        evidence_recall.append(r)
        evidence_f1.append(f)

        answer_types = set(gold.get("answer_types") or [])
        if "multiple_choice" in answer_types:
            mc_total += 1
            mc_correct += int(multiple_choice_prediction(pred) == multiple_choice_gold(gold))

        if "freeform" in answer_types:
            freeform_total += 1
            freeform_exact_correct += int(freeform_prediction(pred) == freeform_gold(gold))

        if "table" in answer_types:
            metrics = table_metrics(gold, pred)
            table_row_f1.append(float(metrics["row_f1"]))
            table_cell_accuracy.append(float(metrics["cell_accuracy"]))
            table_cell_correct += int(metrics["cell_correct"])
            table_cell_total += int(metrics["cell_total"])

    if mc_total:
        multiple_choice_accuracy = mc_correct / mc_total
    else:
        multiple_choice_accuracy = None
    if freeform_total:
        freeform_exact_match = freeform_exact_correct / freeform_total
    else:
        freeform_exact_match = None
    if table_cell_total:
        table_cell_accuracy_micro = table_cell_correct / table_cell_total
    else:
        table_cell_accuracy_micro = None

    return {
        "metrics": {
            "paper_precision_macro": mean(paper_precision),
            "paper_recall_macro": mean(paper_recall),
            "paper_f1_macro": mean(paper_f1),
            # 検索が候補として拾えていたか（LLM の絞り込み前）。gold の task_family 別に
            # k を並べてカーブで出す。シナリオごとに k 昇順で並べて表で読みやすくする。
            **{
                f"candidate_recall_at{k}_{scenario}_macro": macro_or_none(
                    candidate_recall[scenario][k]
                )
                for scenario in CANDIDATE_RECALL_SCENARIOS
                for k in CANDIDATE_RECALL_KS
            },
            "evidence_precision_macro": mean(evidence_precision),
            "evidence_recall_macro": mean(evidence_recall),
            "evidence_f1_macro": mean(evidence_f1),
            "multiple_choice_accuracy": multiple_choice_accuracy,
            "freeform_exact_match": freeform_exact_match,
            "table_row_f1_macro": mean(table_row_f1),
            "table_cell_accuracy_macro": mean(table_cell_accuracy),
            "table_cell_accuracy_micro": table_cell_accuracy_micro,
        },
        "details": {
            "total": len(gold_records),
            "missing_prediction_count": len(missing_predictions),
            "extra_prediction_count": len(extra_predictions),
            # 各シナリオが何件のクエリで測れたかの内訳（数字の妥当性確認用）。
            "candidate_recall_ks": list(CANDIDATE_RECALL_KS),
            "candidate_recall_counts": {
                scenario: len(candidate_recall[scenario][CANDIDATE_RECALL_KS[0]])
                for scenario in CANDIDATE_RECALL_SCENARIOS
            },
            "table_cell_correct": table_cell_correct,
            "table_cell_total": table_cell_total,
            "missing_predictions": missing_predictions,
            "extra_predictions": extra_predictions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LitTraceQA development predictions.")
    parser.add_argument("--gold", default="data/validation.jsonl", help="Gold validation JSONL file.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL file.")
    args = parser.parse_args()

    gold_records = read_jsonl(Path(args.gold))
    pred_records = read_jsonl(Path(args.pred))
    metrics = evaluate(gold_records, pred_records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
