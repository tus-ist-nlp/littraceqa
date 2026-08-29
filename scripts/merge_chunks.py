#!/usr/bin/env python3
"""Join several chunks.jsonl files into one.

For combining the output of preprocessors that each cover part of a paper, so that
one index can be built over the result. Chunks with a chunk_id already seen are
dropped, keeping the first.

Usage:
    uv run python scripts/merge_chunks.py \\
      --inputs body_chunks.jsonl figure_chunks.jsonl \\
      --output chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from littraceqa.di_pipeline.contracts import Chunk


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def merge_chunks(chunk_lists: list[list[Chunk]], strict: bool = False) -> list[Chunk]:
    """Join several Chunk lists into one, dropping repeated chunk_ids."""
    seen: set[str] = set()
    merged: list[Chunk] = []
    duplicate_ids: list[str] = []

    for chunks in chunk_lists:
        for chunk in chunks:
            if chunk.chunk_id in seen:
                duplicate_ids.append(chunk.chunk_id)
                continue
            seen.add(chunk.chunk_id)
            merged.append(chunk)

    if duplicate_ids:
        message = (
            f"重複した chunk_id が {len(duplicate_ids)} 件見つかりました: "
            f"{duplicate_ids[:5]}"
        )
        if strict:
            raise ValueError(message)
        print(f"警告: {message}", file=sys.stderr)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="chunks.jsonl files to join (several allowed)"
    )
    parser.add_argument("--output", required=True, help="where to write the joined jsonl")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop with an error on a repeated chunk_id instead of warning",
    )
    args = parser.parse_args()

    chunk_lists = [load_chunks(Path(p)) for p in args.inputs]
    merged = merge_chunks(chunk_lists, strict=args.strict)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in merged:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print(
        f"{len(merged)} チャンクを {output_path} に書き出しました"
        f"（入力 {len(args.inputs)} ファイル）"
    )


if __name__ == "__main__":
    main()
