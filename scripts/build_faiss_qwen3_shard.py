#!/usr/bin/env python3
"""Build one machine's slice of the faiss_qwen3 index (the distributed build).

Building the Qwen3-Embedding-8B index takes about 31 hours even on four GPUs.
Splitting it across machines (nlp01 + nlp02, 8 GPUs) so each embeds only its own
slice, then joining the slices with scripts/merge_faiss_qwen3.py, halves that.

**This builds the faiss_qwen3 index and nothing else.** bm25s has to walk the whole
corpus in document order and does not split usefully, so it is left to the ordinary
`run_search.py --build` on one machine. faiss_qwen3.py itself is untouched.

The parameters (model / batch_size / max_tokens ...) come from PRODUCTION_PARAMS in
faiss_qwen3.py, so **the build and the search can never drift apart** through
a value retyped in two places. `devices` is the exception — which GPUs are free
differs per machine, so it comes from the command line. The output path is derived
by the same rule search uses, `{index_dir}/{process}/{INDEX_NAME}`.

Slices are cut by the same integer split as _embed_shard in faiss_qwen3.py:
    start = n * shard_index // num_shards
    end   = n * (shard_index + 1) // num_shards
This assumes **every machine sees a byte-identical chunks.jsonl**, line order
included; given that, keeping num_shards the same and varying only shard_index
covers all the chunks exactly once.

Usage (two shards: nlp01 takes the first half, nlp02 the second):
    # on nlp01
    uv run python scripts/build_faiss_qwen3_shard.py \
      --paths configs/paths/default.yaml \
      --devices cuda:0,cuda:1,cuda:2,cuda:3 \
      --shard-index 0 --num-shards 2

    # on nlp02 (chunks.jsonl already transferred, using nlp02's paths)
    uv run python scripts/build_faiss_qwen3_shard.py \
      --paths configs/paths/nlp02.yaml \
      --devices cuda:0,cuda:1,cuda:2,cuda:3 \
      --shard-index 1 --num-shards 2

OOM is guarded three ways (the details are on each parameter in
faiss_qwen3.py):

1. **max_batch_tokens**: budget "rows x longest row in the batch" rather than a
   fixed row count. Chunks are sorted by length before batching, so a fixed count
   collects the long outliers into the last batches, at batch_size x max_tokens
   (8x8192 = 65,536). The measured token distribution is median 265, p99 738, with
   0.002% reaching 8192 — so under a budget the long outliers end up alone while
   most batches grow *larger*, which is also faster.
2. **oom_retries**: if it still OOMs, halve the batch and retry.
3. **resume**: a crash resumes from the completion flags in _embeddings.done — just
   run the same command again, rather than redoing tens of hours.

Setting
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
on the command line eases fragmentation further.

Which GPUs are free differs per machine, hence --devices. **Do not name a GPU with
less than 20GB free**: the 8B model peaks at 19.3GB even in fp16.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from littraceqa.di_pipeline.contracts import Chunk

# Importing di_pipeline.pipeline drags in the reranker (torch), the expanders
# (faiss/bm25s) and the LLM client, which fails on a machine that has none of them
# installed — nlp02, where this build runs. Only the embedding index is needed, so
# that module is imported directly. **The model settings are still the same shared
# constant**, never a second copy.
from littraceqa.di_pipeline.faiss_qwen3 import (
    INDEX_NAME,
    PRODUCTION_PARAMS,
    Qwen3FAISSIndex,
)


def load_paths(path: str) -> dict:
    """Read configs/paths/*.yaml as a dict, straight through yaml (no heavy imports)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def shard_bounds(n: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    """This shard's range [start, end), by the same integer split as _embed_shard."""
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index={shard_index} is out of range; expected 0..{num_shards - 1}"
        )
    start = n * shard_index // num_shards
    end = n * (shard_index + 1) // num_shards
    return start, end


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process-name", default="mineru", help="prefix of the chunks file name")
    parser.add_argument("--shard-index", type=int, required=True, help="this shard's number (from 0)")
    parser.add_argument("--num-shards", type=int, required=True, help="shards across all machines")
    parser.add_argument("--chunks", default=None, help="path to chunks.jsonl (default: derived from paths)")
    parser.add_argument("--index-dir", default=None, help="output directory (default: derived from paths)")
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "GPUs to use (e.g. 'cuda:0,cuda:1'). Which ones are free differs per "
            "machine, so they are named here for the build only (search uses "
            "devices[0] alone). The 8B model peaks at 19.3GB even in fp16, so do "
            "not name a GPU with less than 20GB free"
        ),
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help=(
            "cap on a batch's padded token count (the OOM guard), overriding the "
            "default. Lower it on a machine with less VRAM to spare"
        ),
    )
    args = parser.parse_args()

    paths = load_paths(args.paths)
    params = dict(PRODUCTION_PARAMS)
    # The one setting that comes from the command line: free GPUs and VRAM differ
    # per machine.
    if args.devices:
        params["devices"] = args.devices
    if args.max_batch_tokens:
        params["max_batch_tokens"] = args.max_batch_tokens

    chunks_path = (
        Path(args.chunks)
        if args.chunks
        else Path(paths["chunks_dir"]) / f"{args.process_name}_chunks.jsonl"
    )
    index_dir = (
        args.index_dir
        if args.index_dir
        else f"{paths['index_dir']}/{args.process_name}/{INDEX_NAME}"
    )
    # Each slice goes to its own directory, to be merged later. **The shard is part
    # of the name** so that a slice can never overwrite the real index by mistake.
    index_dir = f"{index_dir}__shard{args.shard_index}of{args.num_shards}"

    print(f"loading chunks: {chunks_path}")
    chunks = load_chunks(chunks_path)
    n = len(chunks)
    start, end = shard_bounds(n, args.shard_index, args.num_shards)
    shard = chunks[start:end]
    print(
        f"of {n:,} chunks, shard {args.shard_index}/{args.num_shards} = "
        f"[{start:,}, {end:,}), {len(shard):,} chunks, building into {index_dir}"
    )

    indexer = Qwen3FAISSIndex(index_dir=index_dir, **params)
    indexer.build(shard)
    print(f"done: {index_dir}")
    print(
        "once every machine's slice is here, merge shard0.."
        f"{args.num_shards - 1} in order with scripts/merge_faiss_qwen3.py"
    )


if __name__ == "__main__":
    main()
