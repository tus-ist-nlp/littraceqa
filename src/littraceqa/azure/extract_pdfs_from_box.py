#!/usr/bin/env python3
"""Stage selected papers' PDFs from the Box zip archives into a local cache.

OpenReview and several publisher sites block direct PDF downloads, so the
Box shared folder (the same source the Document Intelligence pipeline used)
is the reliable way to obtain paper PDFs on test day. This module:

1. collects target paper_ids from ``--paper-id`` / ``--paper-id-file`` /
   ``--from-predictions`` (a run_rag predictions JSONL: every paper_id in
   ``gold_papers`` and ``evidence`` is collected),
2. looks up which zip archive holds each paper via
   ``artifacts/docint/_box_archive_manifest.jsonl``,
3. downloads each needed archive once into ``artifacts/box_tmp`` (resumable),
   extracts only the matching members atomically into ``--pdf-cache-dir``
   (default ``artifacts/pdf_cache``, already-cached PDFs are skipped), and
   deletes the zip before moving to the next archive,
4. prints a per-archive and overall report; exit code 1 if any target is
   still missing.

No Document Intelligence calls are made. Reuses the Box download helper from
``process_box_archives_document_intelligence`` and the alias helpers from
``littraceqa.extract_pdf_archives``.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

import requests

from .process_box_archives_document_intelligence import (
    DEFAULT_BOX_SHARED_URL,
    PDF_MAGIC,
    BoxArchive,
    download_archive,
    fetch_box_archives,
    parse_shared_name,
)
from ..common import DEFAULT_METADATA, ROOT, load_metadata, read_jsonl
from ..extract_pdf_archives import build_alias_map, infer_paper_id

DEFAULT_MANIFEST = ROOT / "artifacts" / "docint" / "_box_archive_manifest.jsonl"
DEFAULT_PDF_CACHE = ROOT / "artifacts" / "pdf_cache"
DEFAULT_TEMP_DIR = ROOT / "artifacts" / "box_tmp"


def log(message: str) -> None:
    """Report to stdout on purpose: the staging report IS this module's
    output (unlike the DI pipelines, which log progress to stderr)."""
    print(message, flush=True)


def is_valid_pdf(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 5:
            return False
        with path.open("rb") as handle:
            return handle.read(5) == PDF_MAGIC
    except OSError:
        return False


def collect_target_ids(args: argparse.Namespace) -> list[str]:
    """Union of paper_ids from --paper-id, --paper-id-file, --from-predictions."""
    ordered: list[str] = []
    seen: set[str] = set()

    def add(paper_id: str) -> None:
        paper_id = paper_id.strip()
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            ordered.append(paper_id)

    for paper_id in args.paper_ids or []:
        add(paper_id)
    if args.paper_id_file is not None:
        for line in args.paper_id_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Tolerate TSV/CSV lists: the paper_id is the first field.
            add(line.split("\t", 1)[0].split(",", 1)[0])
    if args.from_predictions is not None:
        for row in read_jsonl(args.from_predictions):
            for paper in row.get("gold_papers") or []:
                if isinstance(paper, dict) and paper.get("paper_id"):
                    add(str(paper["paper_id"]))
            for item in row.get("evidence") or []:
                if isinstance(item, dict) and item.get("paper_id"):
                    add(str(item["paper_id"]))
    return ordered


def load_manifest_archives(path: Path) -> dict[str, str]:
    """paper_id -> archive zip name from the DI box-archive manifest.

    The manifest is append-only (retries re-log papers); the last entry with
    a non-empty archive name wins."""
    mapping: dict[str, str] = {}
    for record in read_jsonl(path):
        paper_id = str(record.get("paper_id") or "")
        archive = str(record.get("archive") or "")
        if paper_id and archive:
            mapping[paper_id] = archive
    return mapping


def extract_member_atomic(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_written = 0
    with archive.open(info, "r") as source, tmp.open("wb") as target:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            target.write(chunk)
            bytes_written += len(chunk)
    if not is_valid_pdf(tmp):
        tmp.unlink(missing_ok=True)
        raise ValueError("extracted member does not start with %PDF-")
    tmp.replace(dest)
    return bytes_written


def download_with_retries(
    session: requests.Session,
    *,
    shared_name: str,
    archive: BoxArchive,
    temp_dir: Path,
    reserve_gb: float,
    attempts: int,
) -> Path:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return download_archive(
                session,
                shared_name=shared_name,
                archive=archive,
                temp_dir=temp_dir,
                reserve_gb=reserve_gb,
                force=False,
            )
        except Exception as exc:  # noqa: BLE001 - resume via .part on retry.
            last_error = exc
            delay = min(60.0, 5.0 * attempt)
            log(
                f"  download attempt {attempt}/{attempts} failed: "
                f"{exc.__class__.__name__}: {exc}; retrying in {delay:.0f}s"
            )
            time.sleep(delay)
    raise RuntimeError(f"download failed after {attempts} attempts: {last_error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract selected papers' PDFs from the Box zip archives into a "
            "local cache (one archive downloaded at a time, deleted after use)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        default=None,
        metavar="PAPER_ID",
        help="Target paper_id (repeatable).",
    )
    parser.add_argument(
        "--paper-id-file",
        type=Path,
        default=None,
        help="Text file with one paper_id per line (first TSV/CSV field; # comments ok).",
    )
    parser.add_argument(
        "--from-predictions",
        type=Path,
        default=None,
        metavar="PREDICTIONS_JSONL",
        help=(
            "run_rag predictions JSONL: collect every paper_id referenced in "
            "gold_papers and evidence."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="DI box-archive manifest used to locate each paper's zip archive.",
    )
    parser.add_argument(
        "--pdf-cache-dir",
        type=Path,
        default=DEFAULT_PDF_CACHE,
        help="Where extracted PDFs land as {paper_id}.pdf (atomic writes, cached skipped).",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=DEFAULT_TEMP_DIR,
        help="Where zip archives are downloaded temporarily (deleted after extraction).",
    )
    parser.add_argument("--box-url", default=DEFAULT_BOX_SHARED_URL, help="Box shared folder URL.")
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--reserve-gb",
        type=float,
        default=20.0,
        help="Free disk space to keep in reserve when downloading an archive.",
    )
    parser.add_argument(
        "--download-attempts",
        type=int,
        default=5,
        help="Attempts per archive download (resumes the .part file between tries).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report targets, cache state and archive plan; no downloads, no writes.",
    )
    return parser


def _plan_archives(
    args: argparse.Namespace, target_set: set[str]
) -> tuple[set[str], dict[str, list[str]], list[str]]:
    """Report cache state and map each still-needed paper to its zip archive.

    Returns (remaining paper_ids, archive name -> planned paper_ids,
    paper_ids absent from the manifest)."""
    manifest_map = load_manifest_archives(args.manifest)
    log(f"manifest: {len(manifest_map)} papers located in {args.manifest}")

    already = {pid for pid in target_set if is_valid_pdf(args.pdf_cache_dir / f"{pid}.pdf")}
    remaining = target_set - already
    log(f"already cached with %PDF magic: {len(already)}; remaining: {len(remaining)}")

    unlocated = sorted(pid for pid in remaining if pid not in manifest_map)
    plan: dict[str, list[str]] = {}
    for pid in sorted(remaining - set(unlocated)):
        plan.setdefault(manifest_map[pid], []).append(pid)

    for name in sorted(plan):
        log(f"[{name}] needed for {len(plan[name])} papers: {', '.join(plan[name][:8])}"
            + (" ..." if len(plan[name]) > 8 else ""))
    if unlocated:
        log(f"NOT in manifest ({len(unlocated)}): {', '.join(unlocated)}")
    return remaining, plan, unlocated


def _extract_from_archive(
    zf: zipfile.ZipFile,
    *,
    name: str,
    alias_map: dict[str, str],
    target_set: set[str],
    remaining: set[str],
    pdf_cache_dir: Path,
    stats: dict[str, int],
    failures: list[str],
) -> None:
    """Extract every still-needed PDF member of one open zip archive.

    Mutates ``remaining``, ``stats`` and ``failures`` in place."""
    for info in zf.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".pdf"):
            continue
        paper_id = infer_paper_id(info.filename, alias_map)
        # Extract anything we still need that happens to be in
        # this zip, not just the manifest-planned members.
        if not paper_id or paper_id not in target_set:
            continue
        stats["matched"] += 1
        dest = pdf_cache_dir / f"{paper_id}.pdf"
        if is_valid_pdf(dest):
            stats["skipped"] += 1
            remaining.discard(paper_id)
            continue
        member_name = PurePosixPath(info.filename).name
        try:
            size = extract_member_atomic(zf, info, dest)
            stats["extracted"] += 1
            remaining.discard(paper_id)
            log(f"  ok {paper_id} <- {member_name} ({size} bytes)")
        except Exception as exc:  # noqa: BLE001 - keep going.
            stats["failed"] += 1
            failures.append(f"{paper_id} ({name}/{member_name}): {exc}")
            log(f"  FAIL {paper_id} <- {member_name}: {exc}")


def _print_summary(
    args: argparse.Namespace,
    *,
    target_set: set[str],
    per_archive: dict[str, dict[str, int]],
    failures: list[str],
    started: float,
) -> int:
    """Final per-archive and overall report; exit code 1 when any target PDF
    is still missing from the cache."""
    valid = sorted(pid for pid in target_set if is_valid_pdf(args.pdf_cache_dir / f"{pid}.pdf"))
    missing = sorted(target_set - set(valid))
    log("")
    log("==== SUMMARY ====")
    for name in sorted(per_archive):
        s = per_archive[name]
        log(
            f"{name}: matched={s['matched']} extracted={s['extracted']} "
            f"skipped={s['skipped']} failed={s['failed']}"
        )
    log(f"valid PDFs in {args.pdf_cache_dir}: {len(valid)}/{len(target_set)}")
    if failures:
        log("extraction failures:")
        for line in failures:
            log(f"  {line}")
    if missing:
        log("missing paper_ids:")
        for pid in missing:
            log(f"  {pid}")
    log(f"elapsed: {time.time() - started:.0f}s")
    return 1 if missing else 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()

    targets = collect_target_ids(args)
    if not targets:
        log("no target paper_ids: pass --paper-id / --paper-id-file / --from-predictions")
        return 2
    target_set = set(targets)
    log(f"targets: {len(target_set)} paper_ids")

    remaining, plan, unlocated = _plan_archives(args, target_set)
    cached = len(target_set) - len(remaining)

    if args.dry_run:
        log(
            f"dry-run: {cached} cached, {len(remaining) - len(unlocated)} to extract "
            f"from {len(plan)} archives, {len(unlocated)} unlocated; nothing downloaded"
        )
        return 0 if not unlocated else 1

    if not remaining:
        log("nothing to do: all targets already cached")
        return 0

    metadata = load_metadata(args.metadata_file)
    alias_map = build_alias_map(metadata)
    log(f"metadata: {len(metadata)} records, alias_map: {len(alias_map)} aliases")

    args.pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "LitTraceQA-Box-PDF-Stager/1.0"})
    shared_name = parse_shared_name(args.box_url)
    all_archives = {a.name: a for a in fetch_box_archives(args.box_url, session)}
    log(f"box listing: {len(all_archives)} zip archives found")

    per_archive: dict[str, dict[str, int]] = {}
    failures: list[str] = []

    for name in sorted(plan):
        stats = {"extracted": 0, "skipped": 0, "matched": 0, "failed": 0}
        per_archive[name] = stats
        still_needed = [pid for pid in plan[name] if pid in remaining]
        if not still_needed:
            log(f"[{name}] nothing remaining; skipping download")
            continue
        archive_meta = all_archives.get(name)
        if archive_meta is None:
            log(f"[{name}] NOT FOUND in Box listing; skipping ({len(still_needed)} papers stranded)")
            continue
        log(
            f"[{name}] downloading ({archive_meta.size / (1 << 30):.2f} GiB) "
            f"for {len(still_needed)} papers"
        )
        t0 = time.time()
        archive_path = download_with_retries(
            session,
            shared_name=shared_name,
            archive=archive_meta,
            temp_dir=args.temp_dir,
            reserve_gb=args.reserve_gb,
            attempts=args.download_attempts,
        )
        log(f"[{name}] downloaded in {time.time() - t0:.0f}s -> {archive_path}")
        try:
            with zipfile.ZipFile(archive_path) as zf:
                _extract_from_archive(
                    zf,
                    name=name,
                    alias_map=alias_map,
                    target_set=target_set,
                    remaining=remaining,
                    pdf_cache_dir=args.pdf_cache_dir,
                    stats=stats,
                    failures=failures,
                )
        finally:
            archive_path.unlink(missing_ok=True)
            log(f"[{name}] deleted zip")
        log(
            f"[{name}] done: matched={stats['matched']} extracted={stats['extracted']} "
            f"skipped={stats['skipped']} failed={stats['failed']}"
        )

    return _print_summary(
        args,
        target_set=target_set,
        per_archive=per_archive,
        failures=failures,
        started=started,
    )


if __name__ == "__main__":
    sys.exit(main())
