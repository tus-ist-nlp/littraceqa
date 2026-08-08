#!/usr/bin/env python3
"""Write a submission JSONL from a finished retrieval run and a select_style.

Only the paper set needs the retrieval pipeline. ``evidence`` and ``answer``
come from the reading agent, which needs an LLM; this script leaves them empty
so the gold-paper part of a submission can be produced without one.

Empty answers cost nothing on the paper metrics. ``scripts/evaluate.py`` scores
papers from ``gold_papers`` alone, and evidence and answers are accumulated
into separate macro averages. What does cost is a missing record: the scoring
loop walks the gold file, so a query with no prediction scores zero on papers
as well. Every query in the input therefore gets a record.

The answer skeleton mirrors ``answer_types`` so the file has the same shape as
``data/sample_submission.jsonl`` for each question.

Example:
    uv run python scripts/build_submission.py \\
      --retrieval test_retrieval_4b.json \\
      --queries data/test.jsonl \\
      --select configs/select_style/f1_balanced.yaml \\
      --output submissions/test_f1_balanced.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from littraceqa.di_pipeline.evaluation.output import validate_output_path
from littraceqa.di_pipeline.select import build_paper_selector

_ANSWER_SKELETONS: dict[str, dict[str, Any]] = {
    "freeform": {"text": ""},
    "multiple_choice": {"gold": ""},
    "table": {"rows": []},
}


def load_rankings(path: Path) -> dict[str, list[str]]:
    """Read ``query_id -> ranked paper ids`` from an eval_retrieval output."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries") or []
    if not queries:
        raise ValueError(f"{path} contains no queries")
    return {
        str(entry["query_id"]): list(entry.get("ranked_papers") or [])
        for entry in queries
    }


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Read the input questions in file order."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} contains no queries")
    return records


def empty_answer(answer_types: Any) -> dict[str, Any]:
    """Build the empty answer shape the sample submission uses."""

    if not isinstance(answer_types, list):
        return {}
    # Deep copy: the table skeleton holds a list, and a shallow copy would
    # hand every record the same one.
    return {
        name: copy.deepcopy(skeleton)
        for name in answer_types
        if (skeleton := _ANSWER_SKELETONS.get(name)) is not None
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--select", type=Path, required=True, help="configs/select_style/*.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--read-only-root",
        type=Path,
        default=Path("/data2/iseakira"),
        help="Shared input root that must never receive output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = validate_output_path(args.output, args.read_only_root)
    except ValueError as exc:
        parser.error(str(exc))

    rankings = load_rankings(args.retrieval)
    queries = load_queries(args.queries)
    selector = build_paper_selector(
        yaml.safe_load(args.select.read_text(encoding="utf-8"))
    )

    missing = [
        str(record["query_id"])
        for record in queries
        if str(record["query_id"]) not in rankings
    ]
    if missing:
        parser.error(
            f"{len(missing)} queries have no retrieval result, first: {missing[0]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    paper_total = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in queries:
            query_id = str(record["query_id"])
            selection = selector.select(
                record.get("question", ""), rankings[query_id]
            )
            paper_total += len(selection.paper_ids)
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "gold_papers": [
                            {"paper_id": paper_id}
                            for paper_id in selection.paper_ids
                        ],
                        "evidence": [],
                        "answer": empty_answer(record.get("answer_types")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"{len(queries)} records written to {output} "
        f"({paper_total} papers, {paper_total / len(queries):.2f} per query)"
    )
    print("evidence and answer are empty; those metrics will score zero")


if __name__ == "__main__":
    main()
