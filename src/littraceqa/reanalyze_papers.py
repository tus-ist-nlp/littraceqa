#!/usr/bin/env python3
"""Re-analyze specific papers with Document Intelligence via direct PDF downloads.

Unlike ``process_box_archives_document_intelligence`` (which streams whole Box
zip archives), this script targets a small, explicit set of papers: it downloads
each paper's PDF directly from ``pdf_url`` in ``data/paper_metadata.jsonl``,
runs Azure AI Document Intelligence, and rebuilds the caption-based chunks.

Intended uses
-------------
* The ~109 papers that never chunked successfully during the bulk Box run
  (``--failed-from-manifest`` / ``--missing-chunks``).
* Validation gold papers that need clean caption-based table/figure ids
  (``--paper-id`` / ``--paper-id-file`` with ``--overwrite``).
* Future test-time targeted refresh of individual papers.

Outputs
-------
``<--raw-dir>/{paper_id}.json``
    The full Document Intelligence result plus LitTraceQA metadata (always saved).

``<--output-chunks-dir>/{paper_id}.jsonl``
    Search-ready chunks for one paper (written atomically).

``<--output-chunks-dir>/../_reanalyze_manifest.jsonl``
    Append-only per-paper outcome log (override with ``--manifest``).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from .azure_config import (
    DocumentIntelligenceSettings,
    build_document_intelligence_client,
    load_environment,
)
from .common import (
    DEFAULT_METADATA,
    ROOT,
    Record,
    append_jsonl,
    load_metadata,
    read_jsonl,
    relative_or_absolute,
    write_json,
    write_jsonl,
)
from .process_box_archives_document_intelligence import (
    analyze_pdf_bytes_with_retries,
)
from .process_document_intelligence import build_chunks


DEFAULT_BOX_MANIFEST = ROOT / "artifacts" / "docint" / "_box_archive_manifest.jsonl"
MANIFEST_NAME = "_reanalyze_manifest.jsonl"

PDF_MAGIC = b"%PDF-"
RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}

DEFAULT_USER_AGENT = (
    "LitTraceQA-PDF-Fetcher/1.0 "
    "(+https://github.com/; academic dataset assembly; contact: maintainer)"
)


class DownloadPacer:
    """Space PDF downloads politely across worker threads."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = max(0.0, delay_seconds)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._delay <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._delay
        if wait > 0:
            time.sleep(wait)


def parse_paper_id_file(path: Path) -> list[str]:
    """Read paper ids, one per line; blank lines and ``#`` comments are ignored."""
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            token = line.split("#", 1)[0].strip()
            if token:
                ids.append(token)
    return ids


def chunk_file_exists(chunks_dir: Path, paper_id: str) -> bool:
    return (chunks_dir / f"{paper_id}.jsonl").exists()


def failed_from_box_manifest(manifest_path: Path, chunks_dir: Path) -> set[str]:
    """Paper ids whose LAST box-archive manifest status is failed and that still
    have no chunk file."""
    last_status: dict[str, str] = {}
    for record in read_jsonl(manifest_path):
        paper_id = str(record.get("paper_id") or "")
        if paper_id:
            last_status[paper_id] = str(record.get("status") or "")
    return {
        paper_id
        for paper_id, status in last_status.items()
        if status == "failed" and not chunk_file_exists(chunks_dir, paper_id)
    }


def missing_chunk_ids(records: list[Record], chunks_dir: Path) -> set[str]:
    return {
        str(record["paper_id"])
        for record in records
        if record.get("paper_id")
        and not chunk_file_exists(chunks_dir, str(record["paper_id"]))
    }


def select_records(records: list[Record], args: argparse.Namespace) -> list[Record]:
    """Resolve the requested paper ids into metadata records, in metadata order."""
    wanted: set[str] = set(args.paper_id)
    if args.paper_id_file:
        wanted.update(parse_paper_id_file(args.paper_id_file))
    if args.failed_from_manifest:
        wanted.update(
            failed_from_box_manifest(args.box_manifest, args.output_chunks_dir)
        )
    if args.missing_chunks:
        wanted.update(missing_chunk_ids(records, args.output_chunks_dir))

    known_ids = {str(record["paper_id"]) for record in records if record.get("paper_id")}
    for unknown in sorted(wanted - known_ids):
        print(f"WARN unknown paper_id (not in metadata): {unknown}", file=sys.stderr)

    selected = [
        record
        for record in records
        if str(record.get("paper_id") or "") in wanted
    ]

    skipped_existing = 0
    if not args.overwrite:
        remaining: list[Record] = []
        for record in selected:
            if chunk_file_exists(args.output_chunks_dir, str(record["paper_id"])):
                skipped_existing += 1
            else:
                remaining.append(record)
        selected = remaining
    if skipped_existing:
        print(
            f"skipping {skipped_existing} papers with existing chunk files "
            "(use --overwrite to reprocess)",
            file=sys.stderr,
        )

    if args.limit:
        selected = selected[: args.limit]
    return selected


def _retry_after_seconds(response: requests.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 120.0))
    except ValueError:
        return None  # HTTP-date form: fall back to normal backoff.


def download_pdf(
    record: Record,
    *,
    session: requests.Session,
    pacer: DownloadPacer,
    cache_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Download a paper's PDF into the cache directory (reused when present)."""
    paper_id = str(record["paper_id"])
    url = str(record.get("pdf_url") or "")
    if not url:
        raise ValueError("record has no pdf_url")

    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{paper_id}.pdf"
    max_bytes = int(args.max_pdf_mb * (1 << 20))
    if dest.exists() and dest.stat().st_size > 0:
        with dest.open("rb") as handle:
            if handle.read(len(PDF_MAGIC)) == PDF_MAGIC:
                return dest
        dest.unlink(missing_ok=True)

    tmp = dest.with_suffix(".pdf.part")
    attempts = args.download_retries + 1
    last_error = "download was not attempted"
    for attempt in range(1, attempts + 1):
        retry_after: Optional[float] = None
        try:
            pacer.wait()
            with session.get(
                url,
                stream=True,
                timeout=(args.connect_timeout, args.read_timeout),
                allow_redirects=True,
            ) as response:
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code} from {urlparse(url).netloc}"
                    if response.status_code not in RETRYABLE_HTTP_STATUS:
                        raise IOError(last_error)  # 404/403/...: dead link, give up.
                    retry_after = _retry_after_seconds(response)
                else:
                    declared = response.headers.get("Content-Length")
                    if declared and declared.isdigit() and int(declared) > max_bytes:
                        raise IOError(
                            f"PDF too large: Content-Length {declared} bytes "
                            f"> --max-pdf-mb {args.max_pdf_mb}"
                        )
                    downloaded = 0
                    with tmp.open("wb") as out:
                        for chunk in response.iter_content(chunk_size=1 << 16):
                            if not chunk:
                                continue
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise IOError(
                                    f"PDF too large: exceeded --max-pdf-mb "
                                    f"{args.max_pdf_mb} while streaming"
                                )
                            out.write(chunk)
                    with tmp.open("rb") as handle:
                        head = handle.read(1024)
                    if PDF_MAGIC not in head:
                        # Often a transient HTML error page served with 200.
                        last_error = "response is not a PDF (missing %PDF- header)"
                        tmp.unlink(missing_ok=True)
                    else:
                        tmp.replace(dest)
                        return dest
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            last_error = f"network error: {exc.__class__.__name__}: {exc}"
        except IOError:
            tmp.unlink(missing_ok=True)
            raise

        if attempt < attempts:
            if retry_after is None:
                retry_after = args.download_backoff_seconds * (2 ** (attempt - 1))
            retry_after += random.uniform(0.0, 0.5)
            print(
                f"  {paper_id}: download attempt {attempt}/{attempts} failed "
                f"({last_error}); retrying in {retry_after:.1f}s",
                file=sys.stderr,
            )
            time.sleep(retry_after)

    tmp.unlink(missing_ok=True)
    raise IOError(f"download failed after {attempts} attempts: {last_error}")


def process_paper(
    record: Record,
    *,
    client: Any,
    settings: DocumentIntelligenceSettings,
    session: requests.Session,
    pacer: DownloadPacer,
    manifest_path: Path,
    manifest_lock: threading.Lock,
    args: argparse.Namespace,
) -> bool:
    paper_id = str(record["paper_id"])
    raw_path = args.raw_dir / f"{paper_id}.json"
    chunks_path = args.output_chunks_dir / f"{paper_id}.jsonl"
    started = time.time()
    manifest: Record = {
        "paper_id": paper_id,
        "pdf_url": record.get("pdf_url") or "",
        "raw_path": relative_or_absolute(raw_path),
        "chunks_path": relative_or_absolute(chunks_path),
    }
    try:
        pdf_path = download_pdf(
            record,
            session=session,
            pacer=pacer,
            cache_dir=args.pdf_cache_dir,
            args=args,
        )
        manifest["pdf_path"] = relative_or_absolute(pdf_path)
        pdf_bytes = pdf_path.read_bytes()

        result = analyze_pdf_bytes_with_retries(
            client,
            pdf_bytes,
            settings=settings,
            features=args.feature,
            content_format=args.content_format,
            attempts=args.di_retries,
            base_delay_seconds=args.retry_base_seconds,
            paper_id=paper_id,
        )
        write_json(
            raw_path,
            {
                "paper": record,
                "document_intelligence": result,
                "processed_at": time.time(),
                "pdf_path": relative_or_absolute(pdf_path),
                "pdf_url": record.get("pdf_url") or "",
            },
        )
        chunks = build_chunks(
            record,
            result,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        write_jsonl(chunks_path, chunks)
        manifest.update(
            {
                "status": "ok",
                "pdf_bytes": len(pdf_bytes),
                "chunks": len(chunks),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        with manifest_lock:
            append_jsonl(manifest_path, manifest)
        print(f"  {paper_id}: ok chunks={len(chunks)}", file=sys.stderr)
        return True
    except Exception as exc:  # noqa: BLE001 - keep the batch resumable.
        manifest.update(
            {
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        with manifest_lock:
            append_jsonl(manifest_path, manifest)
        print(f"  FAIL {paper_id}: {exc}", file=sys.stderr)
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-analyze specific papers with Document Intelligence by "
            "downloading their PDFs directly from pdf_url."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Paper id to process, repeatable.",
    )
    parser.add_argument(
        "--paper-id-file",
        type=Path,
        default=None,
        help="File with one paper id per line (# comments allowed).",
    )
    parser.add_argument(
        "--failed-from-manifest",
        action="store_true",
        help="Add papers whose last box-archive manifest status is failed "
        "and that have no chunk file.",
    )
    parser.add_argument(
        "--missing-chunks",
        action="store_true",
        help="Add papers from the metadata that have no chunk file.",
    )
    parser.add_argument(
        "--box-manifest",
        type=Path,
        default=DEFAULT_BOX_MANIFEST,
        help="Box archive manifest consulted by --failed-from-manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess papers even when a chunk file already exists.",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--pdf-cache-dir", type=Path, default=ROOT / "artifacts" / "pdf_cache"
    )
    parser.add_argument(
        "--output-chunks-dir",
        type=Path,
        default=ROOT / "artifacts" / "docint" / "chunks",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "artifacts" / "docint" / "raw"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Defaults to <--output-chunks-dir>/../{MANIFEST_NAME}.",
    )
    parser.add_argument(
        "--max-pdf-mb",
        type=float,
        default=80.0,
        help="Skip PDFs larger than this many MiB.",
    )
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--content-format", default="markdown")
    parser.add_argument("--di-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=8.0)
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--download-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--download-delay-seconds", type=float, default=0.5)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--limit", type=int, default=0, help="Only process the first N papers."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the papers that would be processed, then exit.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (
        args.paper_id
        or args.paper_id_file
        or args.failed_from_manifest
        or args.missing_chunks
    ):
        parser.error(
            "select papers with --paper-id/--paper-id-file/"
            "--failed-from-manifest/--missing-chunks"
        )
    if args.failed_from_manifest and not args.box_manifest.exists():
        parser.error(f"box manifest not found: {args.box_manifest}")

    records = select_records(load_metadata(args.metadata), args)
    if not records:
        print("no papers selected", file=sys.stderr)
        return 0

    print(f"selected {len(records)} papers", file=sys.stderr)
    if args.dry_run:
        for record in records:
            print(record["paper_id"])
        return 0

    manifest_path = args.manifest or (
        args.output_chunks_dir.parent / MANIFEST_NAME
    )
    manifest_lock = threading.Lock()
    pacer = DownloadPacer(args.download_delay_seconds)

    load_environment(args.env_file)
    settings = DocumentIntelligenceSettings.from_env()
    client = build_document_intelligence_client(settings)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": args.user_agent, "Accept": "application/pdf,*/*"}
    )

    def run_job(index: int, record: Record) -> bool:
        print(
            f"[{index}/{len(records)}] {record['paper_id']} {record.get('title', '')}",
            file=sys.stderr,
        )
        return process_paper(
            record,
            client=client,
            settings=settings,
            session=session,
            pacer=pacer,
            manifest_path=manifest_path,
            manifest_lock=manifest_lock,
            args=args,
        )

    ok = 0
    failed = 0
    if args.workers <= 1:
        for index, record in enumerate(records, start=1):
            if run_job(index, record):
                ok += 1
            else:
                failed += 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_job, index, record)
                for index, record in enumerate(records, start=1)
            ]
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    ok += 1
                else:
                    failed += 1

    print(
        f"finished: ok={ok} failed={failed} manifest={manifest_path}",
        file=sys.stderr,
    )
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
