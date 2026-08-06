"""Verify a real MinerU corpus before spending pairwise-reader LLM calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from littraceqa.candidate_handoff import load_candidate_handoffs, read_jsonl
from littraceqa.chunk_store import ChunkStore
from littraceqa.corpus_preflight import inspect_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--paper-metadata", default="data/paper_metadata.jsonl")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--chunk-index", default=None)
    parser.add_argument(
        "--image-root",
        default=None,
        help="MinerU directory containing paper_id/auto/images.",
    )
    parser.add_argument(
        "--allow-missing-required-visual-images",
        "--allow-missing-figure-images",
        dest="allow_missing_figure_images",
        action="store_true",
        help=(
            "Warn for isolated explicit visual queries without an image. A global "
            "image-root failure remains fatal."
        ),
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    handoffs = load_candidate_handoffs(
        args.queries, args.candidates, args.paper_metadata
    )
    canonical_paper_ids = {
        str(record.get("paper_id") or "")
        for record in read_jsonl(args.paper_metadata)
        if record.get("paper_id")
    }
    if not canonical_paper_ids:
        raise SystemExit("paper metadata has no canonical paper IDs")
    image_root = None
    if args.image_root is not None:
        image_root_path = Path(args.image_root).expanduser().resolve()
        if not image_root_path.is_dir():
            raise SystemExit(
                "--image-root is not a directory: "
                f"{image_root_path}. Point it at the MinerU directory containing "
                "paper_id/auto/images."
            )
        image_root = str(image_root_path)
    store = ChunkStore(
        args.chunks,
        index_path=args.chunk_index,
        image_root=image_root,
    )
    report, errors = inspect_corpus(
        handoffs,
        store,
        canonical_paper_ids,
        allow_missing_figure_images=args.allow_missing_figure_images,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        if output_path.exists():
            raise SystemExit(f"refusing to overwrite existing report: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
