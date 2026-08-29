#!/usr/bin/env python3
"""Convert PDFs to Markdown / content_list.json with MinerU (pipeline backend).

**Runs in an isolated venv (.venv-mineru)**, because MinerU's dependencies cannot
coexist with this package's; see requirements-mineru.txt. The content_list.json
written here is what MinerUChunker reads on the main side.

To get through 27,000 papers in a realistic time, one process is started per GPU
and the papers are sharded by paper_id. **Papers that already have a
content_list.json are skipped, so an interrupted run resumes with the same
command.**

Setup:
    bash scripts/setup_mineru_env.sh

Run (all four GPUs):
    .venv-mineru/bin/python scripts/run_mineru.py \\
      --paths configs/paths/default.yaml --gpus 0,1,2,3

Output goes to `mineru/`, a sibling of paths.yaml's pdf_dir, by default — paths are
derived rather than configured. --mineru-dir overrides it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Must be set before MinerU is imported.
os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")
os.environ.setdefault("MINERU_MODEL_SOURCE", "huggingface")


def default_mineru_dir(pdf_dir: Path) -> Path:
    """The default output location: a sibling of pdf_dir, so the two cannot collide."""
    return pdf_dir.parent / "mineru"


def content_list_path(mineru_dir: Path, paper_id: str) -> Path:
    """Where do_parse writes a paper's content_list.json."""
    return mineru_dir / paper_id / "auto" / f"{paper_id}_content_list.json"


def select_pdfs(pdf_dir: Path, mineru_dir: Path, shard: int, num_shards: int, overwrite: bool) -> list[Path]:
    """This shard's PDFs that have not been converted yet."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    mine = [p for i, p in enumerate(pdfs) if i % num_shards == shard]
    if overwrite:
        return mine
    return [p for p in mine if not content_list_path(mineru_dir, p.stem).exists()]


def parse_batch(pdfs: list[Path], mineru_dir: Path) -> None:
    """Hand do_parse a batch of PDFs; the larger the batch, the better the model calls amortise."""
    from mineru.cli.common import do_parse

    do_parse(
        str(mineru_dir),
        [p.stem for p in pdfs],
        [p.read_bytes() for p in pdfs],
        ["en"] * len(pdfs),
        backend="pipeline",
        # Only the Markdown and content_list.json are ever used. The bbox-annotated
        # PDFs and the raw model output are pure disk cost, so they are turned off.
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_middle_json=False,
        f_dump_md=True,
        f_dump_content_list=True,
    )


def run_shard(args: argparse.Namespace, pdf_dir: Path, mineru_dir: Path) -> int:
    """Process one shard. Returns the number of failures."""
    pdfs = select_pdfs(pdf_dir, mineru_dir, args.shard, args.num_shards, args.overwrite)
    if args.limit:
        pdfs = pdfs[: args.limit]

    log_dir = mineru_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failure_path = log_dir / f"failures_shard{args.shard}.jsonl"

    print(f"[shard {args.shard}] {len(pdfs)} PDFs left to convert", flush=True)
    failures = 0
    done = 0
    start = time.time()

    for i in range(0, len(pdfs), args.batch_size):
        batch = pdfs[i : i + args.batch_size]
        try:
            parse_batch(batch, mineru_dir)
        except Exception as exc:
            # One broken PDF must not take its batch down with it: retry one by one.
            print(f"[shard {args.shard}] batch failed, retrying one by one: {exc}", file=sys.stderr, flush=True)
            for pdf in batch:
                try:
                    parse_batch([pdf], mineru_dir)
                except Exception as inner:
                    failures += 1
                    with failure_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"paper_id": pdf.stem, "error": str(inner)}, ensure_ascii=False) + "\n")
                    print(f"[shard {args.shard}] failed {pdf.stem}: {inner}", file=sys.stderr, flush=True)

        done += len(batch)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed else 0.0
        remaining = (len(pdfs) - done) / rate / 3600 if rate else float("nan")
        print(
            f"[shard {args.shard}] {done}/{len(pdfs)} done "
            f"({rate * 3600:.0f}/hour, {remaining:.1f} hours left, {failures} failed)",
            flush=True,
        )

    print(f"[shard {args.shard}] finished: {done - failures} converted / {failures} failed", flush=True)
    return failures


def orchestrate(args: argparse.Namespace, pdf_dir: Path, mineru_dir: Path) -> int:
    """Launch this script again, once per GPU, with --shard.

    **The parent never initialises CUDA** — each child picks up its GPU through
    CUDA_VISIBLE_DEVICES.
    """
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    todo = len(select_pdfs(pdf_dir, mineru_dir, 0, 1, args.overwrite))
    print(f"converting {todo} PDFs across {len(gpus)} shards on GPU {gpus}", flush=True)

    log_dir = mineru_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[str, subprocess.Popen]] = []
    for shard, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--paths", args.paths,
            "--shard", str(shard),
            "--num-shards", str(len(gpus)),
            "--batch-size", str(args.batch_size),
            "--mineru-dir", str(mineru_dir),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.limit:
            cmd += ["--limit", str(args.limit)]

        log_path = log_dir / f"shard{shard}.log"
        log_file = log_path.open("w", encoding="utf-8")
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)))
        print(f"  GPU {gpu} -> shard {shard} (log: {log_path})", flush=True)

    exit_codes = [(gpu, proc.wait()) for gpu, proc in procs]
    failed = [gpu for gpu, code in exit_codes if code != 0]
    if failed:
        print(f"error: the shard on GPU {failed} died", file=sys.stderr)
        return 1

    remaining = len(select_pdfs(pdf_dir, mineru_dir, 0, 1, overwrite=False))
    print(f"every shard finished; {remaining} PDFs are still unconverted")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--gpus", default="0", help="GPU numbers to use, comma-separated (e.g. 0,1,2,3)")
    parser.add_argument("--batch-size", type=int, default=8, help="PDFs per do_parse call")
    parser.add_argument("--mineru-dir", help="output directory (default: mineru/ beside pdf_dir)")
    parser.add_argument("--limit", type=int, help="cap the papers per shard (for a smoke run)")
    parser.add_argument("--overwrite", action="store_true", help="convert papers that are already done")
    # The next two are what orchestrate() passes to its children; usable by hand too.
    parser.add_argument("--shard", type=int, help="the shard number this process handles")
    parser.add_argument("--num-shards", type=int, default=1, help="how many shards in total")
    args = parser.parse_args()

    with open(args.paths, encoding="utf-8") as f:
        paths = yaml.safe_load(f)
    pdf_dir = Path(paths["pdf_dir"])
    mineru_dir = Path(args.mineru_dir) if args.mineru_dir else default_mineru_dir(pdf_dir)
    mineru_dir.mkdir(parents=True, exist_ok=True)

    if args.shard is None:
        return orchestrate(args, pdf_dir, mineru_dir)
    return 1 if run_shard(args, pdf_dir, mineru_dir) else 0


# MinerU starts a ProcessPoolExecutor for PDF rendering with spawn, and a spawned
# child re-imports __main__. **Without this guard the conversion runs again inside
# each child** and the whole thing dies with BrokenProcessPool.
if __name__ == "__main__":
    sys.exit(main())
