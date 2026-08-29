#!/usr/bin/env python3
"""Join the faiss_qwen3 index slices built on several machines into one index.

scripts/build_faiss_qwen3_shard.py leaves, on each machine,
  {index_dir}__shard{i}of{N}/index.faiss   (IndexFlatIP, that slice's vectors)
  {index_dir}__shard{i}of{N}/chunks.jsonl  (the chunks, in the same order)
and this concatenates them in shard order 0,1,...,N-1 into the shape run_search.py
expects:
  {output}/index.faiss
  {output}/chunks.jsonl

**Concatenating in order is all it takes.** faiss_qwen3.py's build() adds to the
IndexFlatIP so that row i of the memmap is line i of chunks.jsonl, so vector i of
the index is line i of the chunks; and an IndexFlatIP holds nothing but the flat
list of vectors, no internal state beyond their order. The result is therefore
identical to building the whole corpus on one machine.

The vectors were already L2-normalised in build(), and reconstruct preserves that,
so they are not normalised again.

To stay inside memory, the 8B index (41.6GB in total) is never loaded at once: rows
are reconstructed and added _ADD_ROWS at a time.

Usage (two shards):
    uv run python scripts/merge_faiss_qwen3.py \
      --parts /data2/iseakira/pdfs/index/mineru/faiss_qwen3_8b__shard0of2 \
              /data2/iseakira/pdfs/index/mineru/faiss_qwen3_8b__shard1of2 \
      --output /data2/iseakira/pdfs/index/mineru/faiss_qwen3_8b

**--parts must be in shard order 0,1,...,N-1** — the order is the whole contract.
Slices built on another machine have to be collected here (scp/rsync) first.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import faiss
import numpy as np

_CHUNKS_FILENAME = "chunks.jsonl"
_INDEX_FILENAME = "index.faiss"
_ADD_ROWS = 100_000  # keep in step with faiss_qwen3.py


def count_lines(path: Path) -> int:
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts",
        nargs="+",
        required=True,
        help="the slice index directories, in shard order 0,1,...,N-1",
    )
    parser.add_argument("--output", required=True, help="directory for the joined index")
    args = parser.parse_args()

    part_dirs = [Path(p) for p in args.parts]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check every slice first (index count == chunks line count). Noticing a
    # mismatch halfway through leaves the index misaligned and the whole merge
    # wasted, so nothing is written until all of them have been verified.
    part_indexes = []
    dim = None
    total_vectors = 0
    total_chunks = 0
    for part in part_dirs:
        index_path = part / _INDEX_FILENAME
        chunks_path = part / _CHUNKS_FILENAME
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"{part} has no index.faiss or no chunks.jsonl")
        index = faiss.read_index(str(index_path))
        n_chunks = count_lines(chunks_path)
        if index.ntotal != n_chunks:
            raise ValueError(
                f"{part}: the index holds {index.ntotal:,} vectors but chunks.jsonl "
                f"has {n_chunks:,} lines; merging these would misalign the index"
            )
        if dim is None:
            dim = index.d
        elif index.d != dim:
            raise ValueError(
                f"{part}: dimension {index.d} differs from the other slices ({dim})"
            )
        part_indexes.append(index)
        total_vectors += index.ntotal
        total_chunks += n_chunks
        print(f"  {part.name}: {index.ntotal:,} vectors / {n_chunks:,} chunks OK")

    print(
        f"merging {total_vectors:,} vectors ({dim} dims) into {output_dir}"
    )

    # Concatenate the vectors in shard order into a fresh IndexFlatIP.
    merged = faiss.IndexFlatIP(dim)
    for part, index in zip(part_dirs, part_indexes):
        n = index.ntotal
        for start in range(0, n, _ADD_ROWS):
            count = min(_ADD_ROWS, n - start)
            vectors = index.reconstruct_n(start, count)
            merged.add(np.ascontiguousarray(vectors))
        print(f"  {part.name}: vectors added (running total {merged.ntotal:,})")

    faiss.write_index(merged, str(output_dir / _INDEX_FILENAME))
    print(f"wrote index.faiss ({merged.ntotal:,} vectors)")

    # Concatenate the chunks.jsonl files in the same shard order.
    out_chunks = output_dir / _CHUNKS_FILENAME
    with out_chunks.open("w", encoding="utf-8") as out:
        for part in part_dirs:
            with (part / _CHUNKS_FILENAME).open(encoding="utf-8") as f:
                shutil.copyfileobj(f, out)
    written = count_lines(out_chunks)
    if written != merged.ntotal:
        raise ValueError(
            f"the merged chunks.jsonl has {written:,} lines but the index has "
            f"{merged.ntotal:,} vectors; the merge is broken"
        )
    print(f"wrote chunks.jsonl ({written:,} lines)")
    print("done; run_search.py can now read this as an ordinary index")


if __name__ == "__main__":
    main()
