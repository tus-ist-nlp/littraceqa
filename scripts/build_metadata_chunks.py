#!/usr/bin/env python3
"""Build metadata-only search chunks for every candidate paper.

These chunks are cheap to produce and useful before every PDF has been parsed:
they let Azure Search retrieve candidate papers by title, abstract, venue, and
authors even when full-text Document Intelligence chunks are not available yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from littrace_common import (
    DEFAULT_METADATA,
    ROOT,
    Record,
    azure_document_key,
    clean_text,
    load_metadata,
    write_jsonl,
)


def metadata_content(record: Record) -> str:
    authors = ", ".join(str(author) for author in record.get("authors") or [])
    lines = [
        f"Title: {record.get('title') or ''}",
        f"Authors: {authors}",
        f"Venue: {record.get('venue') or ''}",
        f"Year: {record.get('year') or ''}",
        f"Track: {record.get('track') or ''}",
        f"Award: {record.get('award') or ''}",
        f"Anthology ID: {record.get('anthology_id') or ''}",
        f"arXiv ID: {record.get('arxiv_id') or ''}",
        f"DOI: {record.get('doi') or ''}",
        f"Abstract: {record.get('abstract') or ''}",
    ]
    return "\n".join(clean_text(line) for line in lines if clean_text(line))


def metadata_chunk(record: Record) -> Record:
    authors = ", ".join(str(author) for author in record.get("authors") or [])
    return {
        "id": azure_document_key(record["paper_id"], 0),
        "paper_id": record["paper_id"],
        "chunk_id": 0,
        "source_type": "metadata",
        "title": record.get("title") or "",
        "authors": authors,
        "abstract": record.get("abstract") or "",
        "venue": record.get("venue") or "",
        "year": record.get("year"),
        "track": record.get("track") or "",
        "award": record.get("award"),
        "source_url": record.get("source_url") or "",
        "pdf_url": record.get("pdf_url") or "",
        "anthology_id": record.get("anthology_id") or "",
        "section": "",
        "page_numbers": [],
        "locator_json": json.dumps({"source": "paper_metadata"}),
        "content": metadata_content(record),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build metadata-only chunks for all papers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "metadata" / "chunks.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_metadata(args.metadata)
    if args.limit:
        records = records[: args.limit]
    chunks = [metadata_chunk(record) for record in records if record.get("paper_id")]
    write_jsonl(args.output, chunks)
    print(f"wrote {len(chunks)} metadata chunks -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
