"""Generate gold-only, per-query error reports for an AOAI reading run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from littraceqa.reading_error_analysis import (
    analyze_reading_run,
    read_jsonl,
    write_analysis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a completed two-stage reading trace. Gold is loaded only "
            "by this post-inference command."
        )
    )
    parser.add_argument(
        "--gold",
        default="data/validation.jsonl",
        help="Validation gold JSONL (never supplied to the inference command).",
    )
    parser.add_argument(
        "--candidates",
        default="data/validation_candidates.jsonl",
        help="Sanitized per-query candidate-paper sidecar JSONL.",
    )
    parser.add_argument(
        "--traces",
        required=True,
        help="Two-stage AOAI reader trace JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary.json/.md and queries/<query_id>.json/.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis = analyze_reading_run(
        read_jsonl(args.gold),
        read_jsonl(args.candidates),
        read_jsonl(args.traces),
    )
    write_analysis_outputs(analysis, output_dir)
    print(
        json.dumps(
            {
                "summary_json": str(output_dir / "summary.json"),
                "summary_markdown": str(output_dir / "summary.md"),
                **analysis["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
