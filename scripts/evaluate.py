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

from littraceqa.di_pipeline.contracts import MULTI, SINGLE

# Whether the gold papers are among the top k candidates. paper_recall_macro mixes
# "how the LLM narrowed things down" with "whether retrieval had them at all"; this
# looks only at retrieval, before any narrowing.
#
# **The headline is recall@20** — it matches ReadingConfig.max_candidates=20 in
# pipeline.py, which is what the LLM actually gets to see, and therefore the ceiling
# of the real system. The curve over several k is there because recall@20 alone
# cannot tell "hit at rank 1" from "scraped in at rank 20", and so cannot say
# whether the reranker or the fusion has anything left to give:
#   recall@1 high         -> the ranking is right, not just the retrieval
#   recall@1 low, @20 high -> found, but ranked weakly (reranker/fusion can help)
#   recall@20 low too      -> not being retrieved at all (an index or decomposition
#                             problem)
# 50 matches CANDIDATE_PAPERS_LIMIT in reading.py, the number of candidates kept in
# a prediction.
#
# 70 is "the 50 recorded plus the at most 20 the paper-to-paper expansion adds" —
# the real length of the candidate list with expansion on. **Never put a k deeper
# than what is recorded.** recall_at_k only looks at ranked[:k], so k=100 against a
# prediction holding 50 candidates returns exactly the @50 value. That is
# arithmetically right and factually misleading: it is not "the result of looking
# 100 deep" but "nothing past 50 was recorded", and in a row of numbers it reads as
# a gain at @100 (which is exactly how it was misread). 100 used to sit here; no
# experiment ever had more than 70 candidates, so it was corrected to 70.
# CANDIDATE_PAPERS_LIMIT is **a recording limit, not retrieval's limit** —
# internally retrieval ranks far deeper (per_index_k / pool_k). Measuring deeper
# means raising the limit and running again. With 50-candidate lists, @70 is short
# of denominator and is a rough indication only.
CANDIDATE_RECALL_KS = (1, 5, 10, 20, 50, 70)
CANDIDATE_RECALL_SCENARIOS = ("single", "multi", "total")

# evidence_candidate_recall: candidate_recall with the denominator narrowed to the
# gold papers that actually have evidence attached.
#
# **Some multi_paper gold cannot be retrieved even in principle.** Those papers are
# named in gold_papers but carry no evidence at all — 29 of the 120 gold papers
# (24%) across the 55 validation queries. The two groups behave completely
# differently:
#
#   91 evidence-backed   recall@10 0.615 / @20 0.736 / @50 0.813
#   29 unbacked          recall@10 0.103 / @20 0.207 / @50 0.345
#
# The unbacked ones are peer papers on the same topic that the question never names,
# so no amount of searching with the question brings them near — q_036 asks "what
# batch size does TCM use?" and its gold lists IMM and sCT alongside. Measured over
# all gold, that unreachable group is always mixed in, the ceiling sticks, and the
# effect of changing an index or the reranker looks smaller than it is. Narrowing
# the denominator shows the movement in the part that can actually be moved.
#
# **paper_recall / paper_f1 keep all of gold as their denominator** — that is the
# scoring spec and does not change. This is a diagnostic for choosing what to work
# on, not a prediction of the submitted score.
#
# evidence_candidate_recall_by_backed: the same denominator as above, with **only
# the single/multi split** redone by the number of evidence-backed gold papers.
#
# The external team counts single/multi over gold reduced to the papers that back
# the answer, so the same 55 queries come out single 43 / multi 12 where ours
# (by task_family) are 26 / 29 — the classification is nearly reversed. Comparing
# the task_family numbers directly shows a difference that is only the difference in
# the single ratio. Reporting the realigned series as well makes single/multi
# comparable. total matches the existing series exactly: only the bucket changes,
# never the numerator or denominator.


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


def evidence_backed_paper_ids(record: dict[str, Any]) -> set[str]:
    """The gold papers with at least one piece of evidence attached.

    A paper listed in gold_papers with no evidence at all offers no handle to reach
    it from the question, so retrieval cannot get it in principle — and it cannot
    contribute to evidence_f1 either, having no evidence. It is therefore left out
    of the denominator when measuring how much headroom is left.
    """
    backed = {
        normalize_id(item.get("paper_id"))
        for item in (record.get("evidence") or [])
        if isinstance(item, dict)
    }
    backed.discard("")
    return paper_id_set(record) & backed


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
    """A prediction's candidate_papers, normalised, still in relevance order.

    **Absent returns None, not an empty list**, so an older prediction file without
    the field can be told apart from one where retrieval found nothing — the caller
    omits the metric entirely rather than reporting 0.0.
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
        # The order is the ranking, so nothing is sorted; only repeats are dropped.
        if paper_id and paper_id not in paper_ids:
            paper_ids.append(paper_id)
    return paper_ids


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    """How much of gold the top k of `ranked` covers.

    Empty gold returns 1.0, matching the recall in prf().
    """
    if not gold:
        return 1.0
    return len(gold & set(ranked[:k])) / len(gold)


def gold_ranks(gold_paper_ids: set[str], ranked: list[str]) -> list[int | None]:
    """Each gold paper's rank in candidate_papers (1-based; None if absent).

    **A macro average flattens "just missed, rank 21" and "never retrieved at all"
    into the same 0.** Which of the two it is decides what to do next — go deeper,
    or change the index — so the ranks are kept.
    """
    positions = {paper_id: i + 1 for i, paper_id in enumerate(ranked)}
    return [positions.get(paper_id) for paper_id in sorted(gold_paper_ids)]


def attribute_filter_of(record: dict[str, Any]) -> str:
    """What the attribute filter matched, from the trace reading.py leaves behind."""
    for step in record.get("trace") or []:
        af = step.get("attribute_filter")
        if af:
            return f"{af.get('venue') or ''} {af.get('year') or ''}".strip()
    return ""


def macro_or_none(values: list[float]) -> float | None:
    """None when no query fell into this bucket — distinct from mean()'s 0.0."""
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


def evaluate(
    gold_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    per_query: bool = False,
    include_submission: bool = False,
) -> dict[str, Any]:
    """Score the predictions. **By default only retrieval's metrics.**

    `include_submission=True` (`--metrics all` on the CLI) adds the submission-side
    metrics (paper_* / evidence_* / the answer ones). They are off by default
    because choosing which papers to submit and generating the answers both belong
    to the reading team, and what this side raises is candidate_recall — **printed
    side by side, the numbers we cannot move get read as improvement or regression.**
    """
    gold_by_id = {normalize_id(record.get("query_id")): record for record in gold_records}
    pred_by_id = {normalize_id(record.get("query_id")): record for record in pred_records}
    pred_by_id.pop("", None)

    missing_predictions = sorted(set(gold_by_id) - set(pred_by_id))
    extra_predictions = sorted(set(pred_by_id) - set(gold_by_id))

    # One record with a candidate list makes this a new-format prediction file. If
    # no record has one (a prediction made before the field existed) the metric is
    # None — never a silent 0.0.
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
    # scenario -> k -> per-query recall. single/multi are split by gold's
    # task_family; total is every query (so it is their weighted mean).
    candidate_recall: dict[str, dict[int, list[float]]] = {
        scenario: {k: [] for k in CANDIDATE_RECALL_KS}
        for scenario in CANDIDATE_RECALL_SCENARIOS
    }
    # The same shape, with the denominator narrowed to the evidence-backed gold.
    evidence_candidate_recall: dict[str, dict[int, list[float]]] = {
        scenario: {k: [] for k in CANDIDATE_RECALL_KS}
        for scenario in CANDIDATE_RECALL_SCENARIOS
    }
    # And the same again with the single/multi split redone by the number of
    # evidence-backed gold papers — a diagnostic series that exists only to line up
    # with the external team's numbers (see the comment at the top).
    evidence_candidate_recall_by_backed: dict[str, dict[int, list[float]]] = {
        scenario: {k: [] for k in CANDIDATE_RECALL_KS}
        for scenario in CANDIDATE_RECALL_SCENARIOS
    }

    per_query_rows: list[dict[str, Any]] = []

    for query_id, gold in gold_by_id.items():
        pred = pred_by_id.get(query_id, {})
        row: dict[str, Any] = {
            "query_id": query_id,
            "task_family": gold.get("task_family"),
            "answer_types": sorted(gold.get("answer_types") or []),
            "has_prediction": query_id in pred_by_id,
            "n_gold_papers": len(paper_id_set(gold)),
            "n_pred_papers": len(paper_id_set(pred)),
            "attribute_filter": attribute_filter_of(pred),
        }

        p, r, f = prf(paper_id_set(gold), paper_id_set(pred))
        paper_precision.append(p)
        paper_recall.append(r)
        paper_f1.append(f)
        row["paper_f1"] = f

        if has_candidate_papers:
            # Whether gold is in the top k candidates before any cut — retrieval alone.
            gold_paper_ids = paper_id_set(gold)
            ranked = candidate_paper_ids(pred) or []
            row["n_candidates"] = len(ranked)
            row["gold_ranks"] = gold_ranks(gold_paper_ids, ranked)
            # Production input has no task_family, and `Query` does not carry it
            # either, so it is **always read from gold, never from the prediction**.
            # Unknown or missing counts towards total only.
            task_family = normalize_id(gold.get("task_family"))
            scenarios = ["total"]
            if task_family == SINGLE:
                scenarios.append("single")
            elif task_family == MULTI:
                scenarios.append("multi")
            for k in CANDIDATE_RECALL_KS:
                recall = recall_at_k(gold_paper_ids, ranked, k)
                row[f"candidate_recall_at{k}"] = recall
                for scenario in scenarios:
                    candidate_recall[scenario][k].append(recall)

            # A query with no evidence-backed gold has an empty denominator, and
            # recall_at_k() returns 1.0 on empty gold (to match prf). **Including it
            # would pad the average with free full marks**, so it is left out and
            # macro_or_none reports the bucket as empty.
            backed_paper_ids = evidence_backed_paper_ids(gold)
            row["n_evidence_backed_papers"] = len(backed_paper_ids)
            if backed_paper_ids:
                # The series whose split is redone by evidence-backed count: a query
                # whose task_family is multi but which has one backed gold paper
                # lands on the single side.
                backed_scenarios = ["total"]
                backed_scenarios.append("single" if len(backed_paper_ids) == 1 else "multi")
                row["backed_scenario"] = backed_scenarios[1]
                for k in CANDIDATE_RECALL_KS:
                    recall = recall_at_k(backed_paper_ids, ranked, k)
                    row[f"evidence_candidate_recall_at{k}"] = recall
                    for scenario in scenarios:
                        evidence_candidate_recall[scenario][k].append(recall)
                    for scenario in backed_scenarios:
                        evidence_candidate_recall_by_backed[scenario][k].append(recall)

        p, r, f = prf(evidence_set(gold), evidence_set(pred))
        evidence_precision.append(p)
        evidence_recall.append(r)
        evidence_f1.append(f)
        row["evidence_f1"] = f

        answer_types = set(gold.get("answer_types") or [])
        if "multiple_choice" in answer_types:
            mc_total += 1
            correct = int(multiple_choice_prediction(pred) == multiple_choice_gold(gold))
            mc_correct += correct
            row["multiple_choice_correct"] = bool(correct)

        if "freeform" in answer_types:
            freeform_total += 1
            correct = int(freeform_prediction(pred) == freeform_gold(gold))
            freeform_exact_correct += correct
            row["freeform_correct"] = bool(correct)

        if "table" in answer_types:
            metrics = table_metrics(gold, pred)
            table_row_f1.append(float(metrics["row_f1"]))
            table_cell_accuracy.append(float(metrics["cell_accuracy"]))
            table_cell_correct += int(metrics["cell_correct"])
            table_cell_total += int(metrics["cell_total"])
            row["table_row_f1"] = float(metrics["row_f1"])
            row["table_cell_accuracy"] = float(metrics["cell_accuracy"])

        per_query_rows.append(row)

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

    # The submission-side metrics. **Off by default** (`--metrics all` adds them).
    # ReadingAgent no longer selects which papers to submit — it passes the candidate
    # ranking through — so paper_* is nothing but "what you get from submitting the
    # top 10 candidates", and the answer metrics (multiple_choice / freeform / table)
    # belong to the reading team.
    submission_metrics = {
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
    }

    return {
        "metrics": {
            # Whether retrieval had the paper as a candidate at all, before the LLM
            # narrowed anything. Broken down by gold's task_family, as a curve over
            # k, each scenario in ascending k so it reads as a table.
            **{
                f"candidate_recall_at{k}_{scenario}_macro": macro_or_none(
                    candidate_recall[scenario][k]
                )
                for scenario in CANDIDATE_RECALL_SCENARIOS
                for k in CANDIDATE_RECALL_KS
            },
            # The same candidate lists measured against the evidence-backed gold
            # only: retrieval's real strength, with the unreachable gold excluded.
            **{
                f"evidence_candidate_recall_at{k}_{scenario}_macro": macro_or_none(
                    evidence_candidate_recall[scenario][k]
                )
                for scenario in CANDIDATE_RECALL_SCENARIOS
                for k in CANDIDATE_RECALL_KS
            },
            # The same numbers again with only the single/multi split redone by
            # evidence-backed count. total matches the series above exactly — same
            # numerator, same denominator, only a different bucket.
            **{
                f"evidence_candidate_recall_by_backed_at{k}_{scenario}_macro": macro_or_none(
                    evidence_candidate_recall_by_backed[scenario][k]
                )
                for scenario in CANDIDATE_RECALL_SCENARIOS
                for k in CANDIDATE_RECALL_KS
            },
            **(submission_metrics if include_submission else {}),
        },
        "details": {
            "total": len(gold_records),
            "missing_prediction_count": len(missing_predictions),
            "extra_prediction_count": len(extra_predictions),
            # How many queries each scenario was measured over, to sanity-check the numbers.
            "candidate_recall_ks": list(CANDIDATE_RECALL_KS),
            "candidate_recall_counts": {
                scenario: len(candidate_recall[scenario][CANDIDATE_RECALL_KS[0]])
                for scenario in CANDIDATE_RECALL_SCENARIOS
            },
            # Only queries with evidence-backed gold count here; the shortfall
            # against candidate_recall_counts is how many queries have no backed gold.
            "evidence_candidate_recall_counts": {
                scenario: len(evidence_candidate_recall[scenario][CANDIDATE_RECALL_KS[0]])
                for scenario in CANDIDATE_RECALL_SCENARIOS
            },
            # The counts once the split is redone by backed count — how far it
            # diverges from the task_family split, i.e. from the external team's
            # single/multi ratio.
            "evidence_candidate_recall_by_backed_counts": {
                scenario: len(
                    evidence_candidate_recall_by_backed[scenario][CANDIDATE_RECALL_KS[0]]
                )
                for scenario in CANDIDATE_RECALL_SCENARIOS
            },
            "table_cell_correct": table_cell_correct,
            "table_cell_total": table_cell_total,
            "missing_predictions": missing_predictions,
            "extra_predictions": extra_predictions,
        },
        **({"per_query": per_query_rows} if per_query else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LitTraceQA development predictions.")
    parser.add_argument("--gold", default="data/validation.jsonl", help="Gold validation JSONL file.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL file.")
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="add a per-query diagnostic (gold's ranks, paper_f1, evidence_f1, "
        "answer correctness) to the output under per_query",
    )
    parser.add_argument(
        "--metrics",
        choices=("retrieval", "all"),
        default="retrieval",
        help="retrieval (the default) reports the candidate_recall series only; all "
        "adds the submission-side metrics (paper_* / evidence_* / the answer ones). "
        "They are off by default because choosing the papers to submit and "
        "generating the answers both belong to the reading team",
    )
    args = parser.parse_args()

    gold_records = read_jsonl(Path(args.gold))
    pred_records = read_jsonl(Path(args.pred))
    metrics = evaluate(
        gold_records,
        pred_records,
        per_query=args.per_query,
        include_submission=args.metrics == "all",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
