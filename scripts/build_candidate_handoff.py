#!/usr/bin/env python3
"""Glue the retrieved candidate papers onto the questions, in one file for the
reading agent.

Building a reading agent separately needs two things — the question and the papers
retrieval found — and right now they live apart: the questions in
data/validation_inputs.jsonl, the candidates in `candidate_papers` inside
predictions_*.jsonl. This joins them into one JSONL, one query per line, so that
does not have to be redone every time.

The shape (production's five fields, plus candidates, meta and gold):

    {
      "query_id": "q_001",
      "question": "...",
      "answer_types": ["freeform", "multiple_choice"],
      "multiple_choice_options": null,   # filled in production; always null from validation
      "table_schema": null,
      "candidate_papers": [
        {"rank": 1, "paper_id": "acl2025_00005", "title": "...", "venue": "ACL", "year": 2025},
        ...
      ],
      "_meta": {"source_predictions": "...", "n_candidates": 50},
      "_gold": {"task_family": ..., "primary_evidence_type": ..., "gold_papers": [...],
                "evidence": [...], "answer": {...}}
    }

**Gold is quarantined under `_gold` to prevent accidents.** At the same level it
would be read in all innocence: `gold_papers[].title` is the answer to "which
paper" itself; `answer.multiple_choice.options` is oracle-only **in the validation
data** (production gives the options as top-level `multiple_choice_options`, where
they are not gold); `task_family` and `primary_evidence_type` do not exist in
production input at all. Only the five top-level fields are the same input
production gives; `_gold` may be touched at scoring time and nowhere else.

The title/venue/year on each candidate are not gold and are safe — they are there
so the candidate list can be read before pulling any full text. For the text and
the figure images, hand the paper_id to littraceqa.chunk_store.ChunkStore.

Usage:
    uv run python scripts/build_candidate_handoff.py \\
      --predictions predictions_8b_chunk_b_merged.jsonl \\
      --output data/validation_with_candidates.jsonl

    # imitate production input by dropping gold (for handing out)
    uv run python scripts/build_candidate_handoff.py \\
      --predictions predictions_8b_chunk_b_merged.jsonl --no-gold \\
      --output data/validation_with_candidates_blind.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

Record = dict[str, Any]

# The fields production input really has. **Adding one here makes the evaluation
# drift from production.**
#
# `multiple_choice_options` **is** in production input: 50 of the 71 queries in
# data/test_inputs.jsonl (pulled from HF on 2026-08-13) carry it. It was left out
# for a long time because in the validation data the options exist only on the gold
# side (`answer.multiple_choice.options`), which is oracle-only — but in production
# the input itself supplies them, so they are not gold. With 50 of 71 queries
# multiple_choice, dropping it would leave the reading team unable to answer from
# this file alone.
#
# WARNING: built from validation data it is **always None**, because
# validation_inputs.jsonl has no such field. Read that as "the options are under
# `_gold`", not as "this question has no options".
PRODUCTION_FIELDS = (
    "query_id",
    "question",
    "answer_types",
    "multiple_choice_options",
    "table_schema",
)

# Fields that may be used for scoring and nothing else; moved from the top level
# into _gold.
GOLD_FIELDS = (
    "task_family",
    "primary_evidence_type",
    "gold_papers",
    "evidence",
    "answer",
)


def read_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_paper_titles(path: Path) -> dict[str, Record]:
    """paper_id -> {title, venue, year}, used only as a heading in the candidate list."""
    titles: dict[str, Record] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            titles[record["paper_id"]] = {
                "title": record.get("title"),
                "venue": record.get("venue"),
                "year": record.get("year"),
            }
    return titles


def build_candidates(
    paper_ids: list[str], titles: dict[str, Record], limit: int
) -> list[Record]:
    ranked = paper_ids if limit <= 0 else paper_ids[:limit]
    candidates: list[Record] = []
    for rank, paper_id in enumerate(ranked, start=1):
        info = titles.get(paper_id, {})
        candidates.append(
            {
                "rank": rank,
                "paper_id": paper_id,
                "title": info.get("title"),
                "venue": info.get("venue"),
                "year": info.get("year"),
            }
        )
    return candidates


def build_rows(
    inputs: list[Record],
    gold: dict[str, Record],
    predictions: dict[str, Record],
    titles: dict[str, Record],
    meta_base: Record,
    limit: int,
    include_gold: bool,
) -> tuple[list[Record], list[str]]:
    rows: list[Record] = []
    missing: list[str] = []
    for sample in inputs:
        query_id = sample["query_id"]
        prediction = predictions.get(query_id)
        if prediction is None:
            missing.append(query_id)
            continue
        row: Record = {field: sample.get(field) for field in PRODUCTION_FIELDS}
        candidates = build_candidates(
            prediction.get("candidate_papers") or [], titles, limit
        )
        row["candidate_papers"] = candidates
        row["_meta"] = {**meta_base, "n_candidates": len(candidates)}
        if include_gold:
            source = gold.get(query_id) or sample
            row["_gold"] = {
                field: source.get(field)
                for field in GOLD_FIELDS
                if source.get(field) is not None
            }
        rows.append(row)
    return rows, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predictions", type=Path, required=True, help="the predictions_*.jsonl the candidates come from"
    )
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/validation_inputs.jsonl")
    parser.add_argument("--gold", type=Path, default=ROOT / "data/validation.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/paper_metadata.jsonl")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/validation_with_candidates.jsonl"
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="cap on candidates per query (0 takes every one in the predictions; the "
        "predictions themselves hold at most 50)",
    )
    parser.add_argument(
        "--no-gold", action="store_true", help="omit _gold (the blind version, for handing out)"
    )
    args = parser.parse_args(argv)

    inputs = read_jsonl(args.inputs)
    predictions = {record["query_id"]: record for record in read_jsonl(args.predictions)}
    gold = (
        {record["query_id"]: record for record in read_jsonl(args.gold)}
        if args.gold.exists()
        else {}
    )
    titles = load_paper_titles(args.metadata)

    # There is one configuration, pipeline.py, so the prediction file's name is
    # enough to say which run produced these candidates. (This used to look the run
    # up in results/experiments.jsonl and record the search/agent configuration
    # names; nothing writes that file in the final system.)
    meta_base: Record = {"source_predictions": args.predictions.name}

    rows, missing = build_rows(
        inputs, gold, predictions, titles, meta_base, args.max_candidates, not args.no_gold
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = [row["_meta"]["n_candidates"] for row in rows]
    no_title = sum(
        1 for row in rows for c in row["candidate_papers"] if c["title"] is None
    )
    print(f"{args.output}: {len(rows)} rows (from {len(inputs)} inputs)")
    print(
        f"  candidate papers: {sum(counts)} in total / per query "
        f"min {min(counts, default=0)}, max {max(counts, default=0)}"
    )
    if no_title:
        print(f"  warning: {no_title} candidate(s) not in the metadata, so no title")
    if missing:
        print(
            f"  warning: {len(missing)} query/queries absent from the predictions, "
            f"so they got no candidates: {', '.join(missing[:10])}",
            file=sys.stderr,
        )
    if not args.no_gold:
        without_gold = sum(1 for row in rows if not row.get("_gold"))
        if without_gold:
            print(f"  warning: no gold found for {without_gold} query/queries", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
