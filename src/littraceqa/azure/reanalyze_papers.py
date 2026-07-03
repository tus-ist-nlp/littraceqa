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
import io
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from pypdf import PdfReader

from .azure_config import (
    DocumentIntelligenceSettings,
    build_document_intelligence_client,
    load_environment,
)
from ..common import (
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
    is_transient_document_intelligence_error,
)
from .process_document_intelligence import analyze_pdf_bytes, build_chunks, value


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


# ---------------------------------------------------------------------------
# Page-window splitting for giant PDFs.
#
# Some long, media-heavy PDFs make a whole-document Document Intelligence
# analysis fail with (Timeout)/(InternalServerError). The layout model accepts
# a ``pages="<start>-<end>"`` restriction, so with ``--page-window N`` we count
# pages locally (pypdf), analyze the document in N-page windows, and stitch the
# per-window results into one dict compatible with ``build_chunks``.
#
# Empirical note (verified 2026-07 against the deployed layout model with
# ``pages="2-3"``): the returned ``pageNumber`` values in ``pages[]`` and in
# every ``boundingRegions`` entry are ABSOLUTE document page numbers (2, 3),
# NOT window-relative. ``window_page_offset`` therefore normally returns 0 and
# exists only as a guard should a future API version switch to relative
# numbering.
# ---------------------------------------------------------------------------

STITCH_LIST_KEYS = ("pages", "paragraphs", "tables", "figures")
STITCH_SCALAR_KEYS = ("apiVersion", "modelId", "contentFormat", "stringIndexType")


def count_pdf_pages(pdf_bytes: bytes) -> Optional[int]:
    """Local page count via pypdf; None when the PDF cannot be parsed."""
    try:
        return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception as exc:  # noqa: BLE001 - malformed PDFs raise many types.
        print(f"WARN pypdf could not count pages: {exc}", file=sys.stderr)
        return None


def plan_page_windows(
    page_count: int, window: int, max_pages: int
) -> list[tuple[int, int]]:
    """Inclusive 1-based (start, end) windows; --max-pages caps total coverage."""
    last = page_count if max_pages <= 0 else min(page_count, max_pages)
    return [
        (start, min(start + window - 1, last))
        for start in range(1, last + 1, window)
    ]


def window_page_offset(result: Record, window_start: int) -> int:
    """0 when page numbers are absolute (the verified behavior); window_start-1
    when the result numbers pages relative to the analyzed window."""
    numbers: list[int] = []
    for page in result.get("pages") or []:
        try:
            numbers.append(int(value(page, "pageNumber", "page_number")))
        except (TypeError, ValueError):
            continue
    if window_start > 1 and numbers and min(numbers) == 1:
        return window_start - 1
    return 0


def apply_page_offset(node: Any, offset: int) -> None:
    """Recursively shift every pageNumber (pages[] entries and boundingRegions)
    by ``offset`` in place. Only used when a window came back relative."""
    if offset == 0:
        return
    if isinstance(node, dict):
        for key, val in node.items():
            if key in ("pageNumber", "page_number"):
                try:
                    node[key] = int(val) + offset
                except (TypeError, ValueError):
                    pass
            else:
                apply_page_offset(val, offset)
    elif isinstance(node, list):
        for item in node:
            apply_page_offset(item, offset)


def stitch_window_results(window_results: list[Record]) -> Record:
    """Merge per-window DI results (in window order) into one build_chunks-
    compatible dict: ``content`` strings are joined and the ``pages`` /
    ``paragraphs`` / ``tables`` / ``figures`` arrays are concatenated after any
    page-offset fix. Span offsets are NOT rewritten (build_chunks ignores them).
    """
    stitched: Record = {key: [] for key in STITCH_LIST_KEYS}
    contents: list[str] = []
    for entry in window_results:
        result = entry["result"]
        offset = int(entry.get("page_offset") or 0)
        if offset:
            apply_page_offset(result, offset)
        for key in STITCH_SCALAR_KEYS:
            if key not in stitched and result.get(key) is not None:
                stitched[key] = result[key]
        content = str(result.get("content") or "")
        if content.strip():
            contents.append(content)
        for key in STITCH_LIST_KEYS:
            stitched[key].extend(result.get(key) or [])
    stitched["content"] = "\n\n".join(contents)
    return stitched


def analyze_pdf_windowed(
    client: Any,
    pdf_bytes: bytes,
    *,
    settings: DocumentIntelligenceSettings,
    args: argparse.Namespace,
    paper_id: str,
    page_count: int,
) -> tuple[Record, list[Record]]:
    """Analyze the PDF one page window at a time and stitch the results.

    A window that still fails after retries is logged and skipped (partial
    coverage beats nothing). Raises only when EVERY window failed.
    Returns (stitched_result, window_meta).
    """
    windows = plan_page_windows(page_count, args.page_window, args.max_pages)
    print(
        f"  {paper_id}: {page_count} pages -> {len(windows)} windows of "
        f"{args.page_window} pages"
        + (f" (capped at --max-pages {args.max_pages})" if args.max_pages else ""),
        file=sys.stderr,
    )
    window_results: list[Record] = []
    window_meta: list[Record] = []
    for start, end in windows:
        pages = f"{start}-{end}"
        try:
            result = analyze_window_with_retries(
                client,
                pdf_bytes,
                pages=pages,
                settings=settings,
                features=args.feature,
                content_format=args.content_format,
                attempts=args.di_retries,
                base_delay_seconds=args.retry_base_seconds,
                paper_id=paper_id,
            )
        except Exception as exc:  # noqa: BLE001 - skip the window, keep going.
            print(
                f"  {paper_id}: window {pages} failed after retries "
                f"({exc.__class__.__name__}: {exc}); skipping window",
                file=sys.stderr,
            )
            window_meta.append(
                {
                    "pages": pages,
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue
        offset = window_page_offset(result, start)
        window_results.append(
            {"start": start, "pages": pages, "result": result, "page_offset": offset}
        )
        window_meta.append(
            {
                "pages": pages,
                "status": "ok",
                "page_offset": offset,
                "paragraphs": len(result.get("paragraphs") or []),
                "tables": len(result.get("tables") or []),
                "figures": len(result.get("figures") or []),
            }
        )
        print(
            f"  {paper_id}: window {pages} ok "
            f"(paragraphs={len(result.get('paragraphs') or [])}, "
            f"offset={offset})",
            file=sys.stderr,
        )
    if not window_results:
        raise RuntimeError(
            f"all {len(windows)} page windows failed for {paper_id}"
        )
    stitched = stitch_window_results(window_results)
    stitched["page_windows"] = window_meta
    return stitched, window_meta


def is_transient_window_error(exc: Exception) -> bool:
    """Extend the shared transient check: DI long-running-operation failures
    surface as e.g. "(InternalServerError) An unexpected error occurred." with
    no space in the code and no HTTP 5xx status on the polling response, so the
    box-run marker list misses them."""
    if is_transient_document_intelligence_error(exc):
        return True
    compact = str(exc).lower().replace(" ", "")
    return "internalservererror" in compact or "timeout" in compact


def analyze_window_with_retries(
    client: Any,
    pdf_bytes: bytes,
    *,
    pages: str,
    settings: DocumentIntelligenceSettings,
    features: list[str],
    content_format: str,
    attempts: int,
    base_delay_seconds: float,
    paper_id: str,
) -> Record:
    """Same retry policy as ``analyze_pdf_bytes_with_retries`` but restricted
    to a page range (that helper does not take ``pages``)."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return analyze_pdf_bytes(
                client,
                pdf_bytes,
                model=settings.model,
                features=features,
                content_format=content_format,
                pages=pages,
            )
        except Exception as exc:  # noqa: BLE001 - SDK error classes vary.
            last_error = exc
            if attempt >= attempts or not is_transient_window_error(exc):
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0, max(base_delay_seconds, 0.1))
            print(
                f"  {paper_id}: transient DI error on window {pages} attempt "
                f"{attempt}/{attempts}: {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Document Intelligence analysis was not attempted")


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

        page_count: Optional[int] = None
        if args.page_window > 0:
            page_count = count_pdf_pages(pdf_bytes)
            if page_count is not None:
                manifest["page_count"] = page_count

        use_windows = (
            args.page_window > 0
            and page_count is not None
            and (
                page_count > args.page_window
                or (args.max_pages > 0 and page_count > args.max_pages)
            )
        )
        window_meta: list[Record] = []
        if use_windows:
            result, window_meta = analyze_pdf_windowed(
                client,
                pdf_bytes,
                settings=settings,
                args=args,
                paper_id=paper_id,
                page_count=page_count,
            )
        else:
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
        raw_payload: Record = {
            "paper": record,
            "document_intelligence": result,
            "processed_at": time.time(),
            "pdf_path": relative_or_absolute(pdf_path),
            "pdf_url": record.get("pdf_url") or "",
        }
        if window_meta:
            raw_payload["page_windows"] = window_meta
        write_json(raw_path, raw_payload)
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
        if window_meta:
            manifest["page_windows"] = [
                entry["pages"] for entry in window_meta if entry["status"] == "ok"
            ]
            skipped = [
                entry["pages"] for entry in window_meta if entry["status"] != "ok"
            ]
            if skipped:
                manifest["skipped_windows"] = skipped
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
    parser.add_argument(
        "--page-window",
        type=int,
        default=0,
        help="Analyze giant PDFs in windows of this many pages (0 = disabled). "
        "Triggers when the local pypdf page count exceeds the window; failed "
        "windows are skipped so partial coverage still yields chunks.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="With --page-window, stop after this many pages (0 = no cap).",
    )
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument(
        "--content-format",
        default="",
        help="Passed as output_content_format when non-empty (markdown mode broke service-side 2026-07-03; the chunker does not need it).",
    )
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
