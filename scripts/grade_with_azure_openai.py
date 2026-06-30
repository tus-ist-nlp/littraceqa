#!/usr/bin/env python3
"""Grade LitTraceQA predictions against gold answers with Azure OpenAI.

This is a semantic judge, complementary to ``scripts/evaluate.py``. It compares
the generated answer with the gold answer and also asks the model to assess
paper/evidence quality from the supplied gold and predicted records.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from azure_config import OpenAISettings, build_openai_client, load_environment
from littrace_common import (
    ROOT,
    Record,
    compact_text,
    parse_json_object,
    read_jsonl,
    retry_chat_completion,
)


SYSTEM_PROMPT = """You are a strict but fair evaluator for LitTraceQA.
Compare the prediction to the gold record using only the JSON shown to you.
Do not use outside knowledge. Award semantic credit for equivalent freeform
answers, reasonable numeric formatting differences, and table rows with the
same meaning. Multiple-choice answers must match the gold option exactly.
Return JSON only."""


def json_compact(value: Any, *, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return compact_text(text, max_chars=max_chars)


def short_evidence(record: Record, *, max_items: int, max_text_chars: int) -> list[Record]:
    evidence = record.get("evidence") if isinstance(record.get("evidence"), list) else []
    shortened: list[Record] = []
    for item in evidence[:max_items]:
        if not isinstance(item, dict):
            continue
        shortened.append(
            {
                "evidence_id": item.get("evidence_id"),
                "paper_id": item.get("paper_id"),
                "source_type": item.get("source_type"),
                "evidence_text_or_value": compact_text(
                    item.get("evidence_text_or_value"), max_chars=max_text_chars
                ),
                "locator": item.get("locator"),
            }
        )
    return shortened


def grading_prompt(
    gold: Record,
    pred: Record,
    *,
    max_evidence_items: int,
    max_evidence_text_chars: int,
) -> str:
    payload = {
        "query_id": gold.get("query_id"),
        "question": gold.get("question"),
        "answer_types": gold.get("answer_types"),
        "task_family": gold.get("task_family"),
        "primary_evidence_type": gold.get("primary_evidence_type"),
        "gold": {
            "gold_papers": gold.get("gold_papers"),
            "evidence": short_evidence(
                gold,
                max_items=max_evidence_items,
                max_text_chars=max_evidence_text_chars,
            ),
            "answer": gold.get("answer"),
        },
        "prediction": {
            "gold_papers": pred.get("gold_papers"),
            "evidence": short_evidence(
                pred,
                max_items=max_evidence_items,
                max_text_chars=max_evidence_text_chars,
            ),
            "answer": pred.get("answer"),
        },
    }
    return f"""Grade this prediction against the gold record.

Return exactly this JSON shape:
{{
  "query_id": "...",
  "answer_score": number from 0.0 to 1.0,
  "paper_score": number from 0.0 to 1.0,
  "evidence_score": number from 0.0 to 1.0,
  "overall_score": number from 0.0 to 1.0,
  "verdict": "correct" | "partial" | "incorrect",
  "reason_ja": "short plain Japanese explanation in valid UTF-8"
}}

Scoring guidance:
- answer_score is the main score. Give 1.0 only if the predicted final answer is fully correct.
- paper_score measures whether the predicted papers include the necessary gold papers.
- evidence_score measures whether the predicted evidence points to the same supporting facts.
- overall_score should weight answer most, then papers, then evidence.
- If the prediction says the answer cannot be determined but the gold answer is provided, answer_score should be 0.

Record JSON:
{json_compact(payload, max_chars=24000)}
"""


def chat_json(
    client: Any,
    settings: OpenAISettings,
    prompt: str,
    *,
    max_tokens: int,
) -> Record:
    kwargs: Record = {
        "model": settings.chat_deployment,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = retry_chat_completion(client, kwargs)
    return parse_json_object(response.choices[0].message.content or "{}")


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def normalize_grade(raw: Record, query_id: str) -> Record:
    answer_score = clamp_score(raw.get("answer_score"))
    paper_score = clamp_score(raw.get("paper_score"))
    evidence_score = clamp_score(raw.get("evidence_score"))
    overall_score = clamp_score(
        raw.get("overall_score", 0.6 * answer_score + 0.25 * paper_score + 0.15 * evidence_score)
    )
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in {"correct", "partial", "incorrect"}:
        verdict = "correct" if overall_score >= 0.85 else "partial" if overall_score >= 0.35 else "incorrect"
    return {
        "query_id": query_id,
        "answer_score": answer_score,
        "paper_score": paper_score,
        "evidence_score": evidence_score,
        "overall_score": overall_score,
        "verdict": verdict,
        "reason_ja": compact_text(raw.get("reason_ja") or raw.get("reason"), max_chars=600),
    }


def load_existing(path: Path) -> dict[str, Record]:
    if not path.exists():
        return {}
    return {
        str(record.get("query_id")): record
        for record in read_jsonl(path)
        if record.get("query_id")
    }


def mean(records: list[Record], field: str) -> float:
    return sum(float(record[field]) for record in records) / len(records) if records else 0.0


def write_markdown_report(
    path: Path,
    gold_by_id: dict[str, Record],
    pred_by_id: dict[str, Record],
    grades: list[Record],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verdict_counts: dict[str, int] = {}
    for grade in grades:
        verdict_counts[grade["verdict"]] = verdict_counts.get(grade["verdict"], 0) + 1

    lines = [
        "# LitTraceQA Azure OpenAI Grading Report",
        "",
        "## Summary",
        "",
        f"- total: {len(grades)}",
        f"- answer_score_mean: {mean(grades, 'answer_score'):.3f}",
        f"- paper_score_mean: {mean(grades, 'paper_score'):.3f}",
        f"- evidence_score_mean: {mean(grades, 'evidence_score'):.3f}",
        f"- overall_score_mean: {mean(grades, 'overall_score'):.3f}",
        f"- verdict_counts: {json.dumps(verdict_counts, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Per Question",
        "",
    ]
    for grade in grades:
        query_id = str(grade["query_id"])
        gold = gold_by_id.get(query_id, {})
        pred = pred_by_id.get(query_id, {})
        lines.extend(
            [
                f"### {query_id}",
                "",
                f"Question: {gold.get('question', '')}",
                "",
                f"- verdict: {grade['verdict']}",
                f"- answer_score: {grade['answer_score']:.2f}",
                f"- paper_score: {grade['paper_score']:.2f}",
                f"- evidence_score: {grade['evidence_score']:.2f}",
                f"- overall_score: {grade['overall_score']:.2f}",
                f"- reason: {grade.get('reason_ja', '')}",
                "",
                "Gold answer:",
                "",
                "```json",
                json.dumps(gold.get("answer"), ensure_ascii=False, indent=2),
                "```",
                "",
                "Predicted answer:",
                "",
                "```json",
                json.dumps(pred.get("answer"), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade predictions with Azure OpenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--gold", type=Path, default=ROOT / "data" / "validation.jsonl")
    parser.add_argument(
        "--pred",
        type=Path,
        default=ROOT / "runs" / "validation_submission_metadata_plus_partial_pdf.jsonl",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "azure_openai_grades.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "runs" / "azure_openai_grading_report.md")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-answer-tokens", type=int, default=900)
    parser.add_argument("--max-evidence-items", type=int, default=6)
    parser.add_argument("--max-evidence-text-chars", type=int, default=700)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_environment(args.env_file)
    settings = OpenAISettings.from_env()
    client = build_openai_client(settings)

    gold_records = read_jsonl(args.gold)
    pred_records = read_jsonl(args.pred)
    gold_by_id = {str(record.get("query_id")): record for record in gold_records}
    pred_by_id = {str(record.get("query_id")): record for record in pred_records}
    query_ids = [str(record.get("query_id")) for record in gold_records if record.get("query_id")]
    if args.limit:
        query_ids = query_ids[: args.limit]

    existing = load_existing(args.output) if args.resume else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    grades: list[Record] = []
    if args.resume:
        grades.extend(existing[query_id] for query_id in query_ids if query_id in existing)

    failed = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, query_id in enumerate(query_ids, start=1):
            if query_id in existing:
                print(f"[{index}/{len(query_ids)}] {query_id} cached", file=sys.stderr)
                continue
            gold = gold_by_id[query_id]
            pred = pred_by_id.get(query_id, {"query_id": query_id})
            print(f"[{index}/{len(query_ids)}] {query_id}", file=sys.stderr)
            try:
                prompt = grading_prompt(
                    gold,
                    pred,
                    max_evidence_items=args.max_evidence_items,
                    max_evidence_text_chars=args.max_evidence_text_chars,
                )
                raw = chat_json(
                    client,
                    settings,
                    prompt,
                    max_tokens=args.max_answer_tokens,
                )
                grade = normalize_grade(raw, query_id)
            except Exception as exc:  # noqa: BLE001 - keep batch resumable.
                failed += 1
                grade = {
                    "query_id": query_id,
                    "answer_score": 0.0,
                    "paper_score": 0.0,
                    "evidence_score": 0.0,
                    "overall_score": 0.0,
                    "verdict": "incorrect",
                    "reason_ja": f"judge failed: {exc.__class__.__name__}: {exc}",
                }
                print(f"FAIL {query_id}: {exc}", file=sys.stderr)
            grades.append(grade)
            handle.write(json.dumps(grade, ensure_ascii=False) + "\n")
            handle.flush()
            if args.sleep:
                time.sleep(args.sleep)

    grades_by_id = {str(grade["query_id"]): grade for grade in grades}
    ordered_grades = [grades_by_id[query_id] for query_id in query_ids if query_id in grades_by_id]
    write_markdown_report(args.report, gold_by_id, pred_by_id, ordered_grades)

    summary = {
        "total": len(ordered_grades),
        "failed": failed,
        "answer_score_mean": mean(ordered_grades, "answer_score"),
        "paper_score_mean": mean(ordered_grades, "paper_score"),
        "evidence_score_mean": mean(ordered_grades, "evidence_score"),
        "overall_score_mean": mean(ordered_grades, "overall_score"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"grades: {args.output}", file=sys.stderr)
    print(f"report: {args.report}", file=sys.stderr)
    return 1 if failed == len(query_ids) and query_ids else 0


if __name__ == "__main__":
    sys.exit(main())
