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
    parser.add_argument("--image-root", default=None)
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
    store = ChunkStore(
        args.chunks,
        index_path=args.chunk_index,
        image_root=args.image_root,
    )
    report, errors = inspect_corpus(handoffs, store, canonical_paper_ids)
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
