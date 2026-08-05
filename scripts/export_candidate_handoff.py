"""Remove all gold/development fields from a retrieval handoff JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

from littraceqa.candidate_handoff import candidate_papers_from_record


def sanitize_record(record: dict) -> dict:
    """Return the exact sidecar schema accepted by the corpus QA runner."""

    safe_input = {
        "query_id": record.get("query_id"),
        "candidate_papers": record.get("candidate_papers"),
    }
    papers = candidate_papers_from_record(safe_input)
    return {
        "query_id": str(record.get("query_id") or ""),
        "candidate_papers": [
            {
                "rank": paper.rank,
                "paper_id": paper.paper_id,
                "title": paper.title,
                "venue": paper.venue,
                "year": paper.year,
            }
            for paper in papers
        ],
    }


def sanitize_stream(source: TextIO, destination: TextIO) -> int:
    seen: set[str] = set()
    count = 0
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise TypeError(f"line {line_number} is not a JSON object")
        sanitized = sanitize_record(record)
        query_id = sanitized["query_id"]
        if not query_id:
            raise ValueError(f"line {line_number} has no query_id")
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        destination.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project a PR/search-result JSONL to query_id + candidate_papers only."
    )
    parser.add_argument("--input", required=True, help="Input JSONL, or - for stdin")
    parser.add_argument("--output", required=True, help="New sidecar JSONL path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with ExitStack() as stack:
            source = (
                sys.stdin
                if args.input == "-"
                else stack.enter_context(Path(args.input).open(encoding="utf-8"))
            )
            output = stack.enter_context(output_path.open("x", encoding="utf-8"))
            count = sanitize_stream(source, output)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    print(f"wrote {count} candidate handoffs to {output_path}")


if __name__ == "__main__":
    main()
