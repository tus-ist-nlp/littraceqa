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


def paper_group_summary(
    values: list[tuple[float, float, float, bool]],
) -> dict[str, Any]:
    """Summarize paper retrieval for one gold-paper-count group."""
    return {
        "count": len(values),
        "paper_precision_macro": mean([value[0] for value in values]),
        "paper_recall_macro": mean([value[1] for value in values]),
        "paper_f1_macro": mean([value[2] for value in values]),
        "all_gold_count": sum(value[3] for value in values),
        "all_gold_rate": mean([float(value[3]) for value in values]),
    }


def ordered_paper_ids(record: dict[str, Any]) -> list[str]:
    """Return unique paper IDs while preserving their retrieval order.

    Ranking files use ``papers``. The other accepted field names make this
    helper usable with existing prediction and diagnostic artifacts without
    changing the legacy set-based evaluator.
    """
    papers: Any = []
    for field_name in ("papers", "ranked_papers", "results", "gold_papers"):
        candidate = record.get(field_name)
        if isinstance(candidate, list):
            papers = candidate
            break

    ordered: list[str] = []
    seen: set[str] = set()
    for item in papers:
        if isinstance(item, str):
            paper_id = item
        elif isinstance(item, dict):
            paper_id = item.get("paper_id", "")
        else:
            paper_id = ""
        normalized = normalize_id(paper_id)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _validated_cutoffs(cutoffs: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(cutoffs)))
    if not normalized or any(cutoff <= 0 for cutoff in normalized):
        raise ValueError("cutoffs must contain positive integers")
    return normalized


def ranked_query_metrics(
    gold_paper_ids: set[str],
    ranked_paper_ids: list[str],
    cutoffs: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float | int | bool]:
    """Compute ordered paper-retrieval metrics for one query.

    Precision@k uses ``k`` as its denominator even when a ranking is shorter.
    nDCG uses binary relevance, and duplicate retrieved IDs only count at their
    first occurrence.
    """
    cutoffs = _validated_cutoffs(cutoffs)
    unique_ranking: list[str] = []
    seen: set[str] = set()
    for paper_id in ranked_paper_ids:
        normalized = normalize_id(paper_id)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_ranking.append(normalized)

    metrics: dict[str, float | int | bool] = {
        "ranked_paper_count": len(unique_ranking),
    }
    for cutoff in cutoffs:
        retrieved = unique_ranking[:cutoff]
        relevant_count = sum(paper_id in gold_paper_ids for paper_id in retrieved)
        precision = relevant_count / cutoff
        recall = relevant_count / len(gold_paper_ids) if gold_paper_ids else 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics[f"paper_precision_at_{cutoff}"] = precision
        metrics[f"paper_recall_at_{cutoff}"] = recall
        metrics[f"paper_f1_at_{cutoff}"] = f1
        metrics[f"relevant_paper_count_at_{cutoff}"] = relevant_count
        metrics[f"all_gold_at_{cutoff}"] = bool(
            gold_paper_ids and gold_paper_ids.issubset(set(retrieved))
        )

    reciprocal_rank = 0.0
    for rank, paper_id in enumerate(unique_ranking, start=1):
        if paper_id in gold_paper_ids:
            reciprocal_rank = 1.0 / rank
            break
    metrics["reciprocal_rank"] = reciprocal_rank

    ndcg_cutoff = 10
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, paper_id in enumerate(unique_ranking[:ndcg_cutoff], start=1)
        if paper_id in gold_paper_ids
    )
    ideal_relevant = min(len(gold_paper_ids), ndcg_cutoff)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    metrics["ndcg_at_10"] = dcg / idcg if idcg else 0.0
    return metrics


def ranked_paper_summary(
    query_metrics: list[dict[str, float | int | bool]],
    cutoffs: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float | int]:
    """Aggregate ordered paper metrics across queries using macro averages."""
    cutoffs = _validated_cutoffs(cutoffs)
    summary: dict[str, float | int] = {"count": len(query_metrics)}
    ranking_lengths = [
        int(metrics["ranked_paper_count"]) for metrics in query_metrics
    ]
    summary["ranked_paper_count_mean"] = mean(
        [float(length) for length in ranking_lengths]
    )
    summary["ranked_paper_count_min"] = min(ranking_lengths, default=0)
    summary["ranked_paper_count_max"] = max(ranking_lengths, default=0)
    for cutoff in cutoffs:
        for metric_name in ("precision", "recall", "f1"):
            key = f"paper_{metric_name}_at_{cutoff}"
            summary[f"{key}_macro"] = mean(
                [float(metrics[key]) for metrics in query_metrics]
            )
        all_gold_key = f"all_gold_at_{cutoff}"
        all_gold_count = sum(bool(metrics[all_gold_key]) for metrics in query_metrics)
        summary[f"{all_gold_key}_count"] = all_gold_count
        summary[f"{all_gold_key}_rate"] = mean(
            [float(bool(metrics[all_gold_key])) for metrics in query_metrics]
        )
        relevant_count_key = f"relevant_paper_count_at_{cutoff}"
        summary[f"{relevant_count_key}_total"] = sum(
            int(metrics[relevant_count_key]) for metrics in query_metrics
        )
    summary["mrr"] = mean(
        [float(metrics["reciprocal_rank"]) for metrics in query_metrics]
    )
    summary["ndcg_at_10_macro"] = mean(
        [float(metrics["ndcg_at_10"]) for metrics in query_metrics]
    )
    return summary


def _records_by_query_id(
    records: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record_number, record in enumerate(records, start=1):
        query_id = normalize_id(record.get("query_id"))
        if not query_id:
            raise ValueError(f"{source_name} record {record_number} has no query_id")
        if query_id in indexed:
            raise ValueError(f"{source_name} contains duplicate query_id: {query_id}")
        indexed[query_id] = record
    return indexed


def evaluate_rankings(
    gold_records: list[dict[str, Any]],
    ranking_records: list[dict[str, Any]],
    cutoffs: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
    """Evaluate ordered paper rankings without using development-only labels."""
    cutoffs = _validated_cutoffs(cutoffs)
    gold_by_id = _records_by_query_id(gold_records, source_name="gold")
    ranking_by_id = _records_by_query_id(ranking_records, source_name="ranking")
    missing_rankings = sorted(set(gold_by_id) - set(ranking_by_id))
    extra_rankings = sorted(set(ranking_by_id) - set(gold_by_id))

    all_metrics: list[dict[str, float | int | bool]] = []
    grouped_metrics: dict[str, list[dict[str, float | int | bool]]] = {
        "single_gold_paper": [],
        "multiple_gold_papers": [],
    }
    query_details: list[dict[str, Any]] = []
    no_gold_paper_queries = 0

    for query_id, gold in gold_by_id.items():
        ranking_missing = query_id not in ranking_by_id
        ranking = ranking_by_id.get(query_id, {})
        gold_ids = paper_id_set(gold)
        ranked_ids = ordered_paper_ids(ranking)
        metrics = ranked_query_metrics(gold_ids, ranked_ids, cutoffs)

        if len(gold_ids) == 1:
            group = "single_gold_paper"
        elif len(gold_ids) > 1:
            group = "multiple_gold_papers"
        else:
            group = "no_gold_paper"
            no_gold_paper_queries += 1
        # Recall and all-gold recovery are undefined without a gold paper.
        # Keep such records visible for input auditing, but do not let them
        # dilute aggregate retrieval metrics or paired method comparisons.
        if gold_ids:
            all_metrics.append(metrics)
        if group in grouped_metrics:
            grouped_metrics[group].append(metrics)

        query_details.append(
            {
                "query_id": query_id,
                "gold_paper_ids": sorted(gold_ids),
                "gold_paper_count": len(gold_ids),
                "gold_count_group": group,
                "ranked_paper_ids": ranked_ids,
                "ranking_missing": ranking_missing,
                "metrics": metrics,
            }
        )

    return {
        "metrics": ranked_paper_summary(all_metrics, cutoffs),
        "paper_metrics_by_gold_count": {
            name: ranked_paper_summary(metrics, cutoffs)
            for name, metrics in grouped_metrics.items()
        },
        "details": {
            "total": len(gold_records),
            "paper_metric_query_count": len(all_metrics),
            "missing_ranking_count": len(missing_rankings),
            "extra_ranking_count": len(extra_rankings),
            "no_gold_paper_query_count": no_gold_paper_queries,
            "missing_rankings": missing_rankings,
            "extra_rankings": extra_rankings,
        },
        "queries": query_details,
    }


def evaluate(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = {normalize_id(record.get("query_id")): record for record in gold_records}
    pred_by_id = {normalize_id(record.get("query_id")): record for record in pred_records}
    pred_by_id.pop("", None)

    missing_predictions = sorted(set(gold_by_id) - set(pred_by_id))
    extra_predictions = sorted(set(pred_by_id) - set(gold_by_id))

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
    paper_groups: dict[str, list[tuple[float, float, float, bool]]] = {
        "single_gold_paper": [],
        "multiple_gold_papers": [],
    }
    no_gold_paper_queries = 0

    for query_id, gold in gold_by_id.items():
        pred = pred_by_id.get(query_id, {})

        gold_paper_ids = paper_id_set(gold)
        pred_paper_ids = paper_id_set(pred)
        p, r, f = prf(gold_paper_ids, pred_paper_ids)
        paper_precision.append(p)
        paper_recall.append(r)
        paper_f1.append(f)
        if len(gold_paper_ids) == 1:
            group = "single_gold_paper"
        elif len(gold_paper_ids) > 1:
            group = "multiple_gold_papers"
        else:
            group = None
            no_gold_paper_queries += 1
        if group is not None:
            paper_groups[group].append(
                (p, r, f, gold_paper_ids.issubset(pred_paper_ids))
            )

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
            "evidence_precision_macro": mean(evidence_precision),
            "evidence_recall_macro": mean(evidence_recall),
            "evidence_f1_macro": mean(evidence_f1),
            "multiple_choice_accuracy": multiple_choice_accuracy,
            "freeform_exact_match": freeform_exact_match,
            "table_row_f1_macro": mean(table_row_f1),
            "table_cell_accuracy_macro": mean(table_cell_accuracy),
            "table_cell_accuracy_micro": table_cell_accuracy_micro,
        },
        "paper_metrics_by_gold_count": {
            name: paper_group_summary(values)
            for name, values in paper_groups.items()
        },
        "details": {
            "total": len(gold_records),
            "missing_prediction_count": len(missing_predictions),
            "extra_prediction_count": len(extra_predictions),
            "table_cell_correct": table_cell_correct,
            "table_cell_total": table_cell_total,
            "no_gold_paper_query_count": no_gold_paper_queries,
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
