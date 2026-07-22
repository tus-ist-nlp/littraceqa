#!/usr/bin/env python3
"""Process Box-hosted PDF zip archives with Document Intelligence.

This script avoids keeping individual PDFs on disk. It downloads one Box zip
archive at a time into a temporary directory, reads each PDF member into memory,
sends it to Azure AI Document Intelligence, writes JSONL chunks, then removes
the temporary archive before moving to the next one.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import random
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import requests

from .azure_config import (
    DocumentIntelligenceSettings,
    build_document_intelligence_client,
    load_environment,
)
from ..extract_pdf_archives import build_alias_map, infer_paper_id
from ..common import (
    DEFAULT_METADATA,
    ROOT,
    Record,
    append_jsonl,
    load_metadata,
    relative_or_absolute,
    write_json,
    write_jsonl,
)
from .process_document_intelligence import (
    analyze_pdf_bytes,
    build_chunks,
    merge_chunk_files,
)


DEFAULT_BOX_SHARED_URL = "https://tus.box.com/s/uelrgaeuiof6392jrkneanthiflr4z5e"

# Shared by reanalyze_papers / extract_pdfs_from_box; the exact sniffing rules
# differ per call site (strict prefix vs anywhere in the first 1 KiB).
PDF_MAGIC = b"%PDF-"


@dataclasses.dataclass(frozen=True)
class BoxArchive:
    file_id: str
    name: str
    size: int

    @property
    def typed_file_id(self) -> str:
        return f"f_{self.file_id}"


def parse_shared_name(shared_url: str) -> str:
    path = urlparse(shared_url).path.rstrip("/")
    shared_name = path.rsplit("/", maxsplit=1)[-1]
    if not shared_name:
        raise ValueError(f"could not parse Box shared name from {shared_url}")
    return shared_name


def fetch_box_archives(shared_url: str, session: requests.Session) -> list[BoxArchive]:
    response = session.get(shared_url, timeout=60)
    response.raise_for_status()
    match = re.search(r"Box\.postStreamData\s*=\s*(\{.*?\});</script>", response.text)
    if not match:
        raise RuntimeError("could not find Box.postStreamData in shared page")
    payload = json.loads(match.group(1))
    folder = payload.get("/app-api/enduserapp/shared-folder") or {}
    items = folder.get("items") or []
    archives: list[BoxArchive] = []
    for item in items:
        if item.get("type") != "file" or item.get("extension") != "zip":
            continue
        archives.append(
            BoxArchive(
                file_id=str(item["id"]),
                name=str(item["name"]),
                size=int(item.get("itemSize") or 0),
            )
        )
    return sorted(archives, key=lambda archive: archive.name)


def download_url(shared_name: str, archive: BoxArchive) -> str:
    return (
        "https://tus.app.box.com/index.php"
        f"?rm=box_download_shared_file&shared_name={shared_name}"
        f"&file_id={archive.typed_file_id}"
    )


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def ensure_space(path: Path, needed_bytes: int, reserve_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    reserve_bytes = int(reserve_gb * (1 << 30))
    available = free_bytes(path)
    if available - reserve_bytes < needed_bytes:
        raise RuntimeError(
            "not enough free disk space for temporary archive: "
            f"need {needed_bytes / (1 << 30):.1f} GiB plus reserve "
            f"{reserve_gb:.1f} GiB, available {available / (1 << 30):.1f} GiB"
        )


def download_archive(
    session: requests.Session,
    *,
    shared_name: str,
    archive: BoxArchive,
    temp_dir: Path,
    reserve_gb: float,
    force: bool,
) -> Path:
    ensure_space(temp_dir, archive.size, reserve_gb)
    final_path = temp_dir / archive.name
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists() and final_path.stat().st_size == archive.size and not force:
        return final_path
    if force:
        final_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    url = download_url(shared_name, archive)
    with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            existing = 0
            mode = "wb"
        downloaded = existing
        with part_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded and downloaded % (1 << 30) < (1 << 20):
                    print(
                        f"  downloading {archive.name}: {downloaded / (1 << 30):.1f} GiB",
                        file=sys.stderr,
                    )
    actual_size = part_path.stat().st_size
    if archive.size and actual_size != archive.size:
        raise IOError(
            f"downloaded size mismatch for {archive.name}: "
            f"{actual_size} != {archive.size}"
        )
    part_path.replace(final_path)
    return final_path


def should_process_member(
    *,
    paper_id: Optional[str],
    chunks_dir: Path,
    force: bool,
) -> bool:
    if not paper_id:
        return False
    if force:
        return True
    return not (chunks_dir / f"{paper_id}.jsonl").exists()


def load_pdf_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, max_pdf_bytes: int) -> bytes:
    if max_pdf_bytes and info.file_size > max_pdf_bytes:
        raise ValueError(
            f"PDF too large: {info.file_size} bytes > --max-pdf-bytes {max_pdf_bytes}"
        )
    with archive.open(info, "r") as handle:
        data = handle.read()
    if not data.startswith(PDF_MAGIC):
        raise ValueError("zip member does not start with %PDF-")
    return data


def is_transient_document_intelligence_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    transient_markers = [
        "timeout",
        "timed out",
        "too many requests",
        "rate limit",
        "throttl",
        "failed to resolve",
        "getaddrinfo failed",
        "name resolution",
        "temporary failure in name resolution",
        "connection error",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
    ]
    return any(marker in message for marker in transient_markers)


def analyze_pdf_bytes_with_retries(
    client: Any,
    pdf_bytes: bytes,
    *,
    settings: DocumentIntelligenceSettings,
    features: list[str],
    content_format: str,
    attempts: int,
    base_delay_seconds: float,
    paper_id: str,
    pages: Optional[str] = None,
    is_transient: Callable[[Exception], bool] = is_transient_document_intelligence_error,
) -> Record:
    """Analyze PDF bytes, retrying transient DI errors with exponential
    backoff plus jitter.

    ``pages`` restricts analysis to a 1-based inclusive range (e.g. "1-40");
    ``is_transient`` lets callers widen the retryable-error test (see
    ``reanalyze_papers.is_transient_window_error``).
    """
    window_label = f"window {pages} " if pages else ""
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
        except Exception as exc:  # noqa: BLE001 - SDK error classes vary by version.
            last_error = exc
            if attempt >= attempts or not is_transient(exc):
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            delay += random.uniform(0, max(base_delay_seconds, 0.1))
            print(
                f"  {paper_id}: transient DI error on {window_label}attempt "
                f"{attempt}/{attempts}: {exc}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Document Intelligence analysis was not attempted")


def process_member(
    *,
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    paper: Record,
    client: Any,
    settings: DocumentIntelligenceSettings,
    raw_dir: Path,
    chunks_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    archive_name: str,
    manifest_lock: threading.Lock | None = None,
) -> bool:
    paper_id = str(paper["paper_id"])
    raw_path = raw_dir / f"{paper_id}.json"
    chunks_path = chunks_dir / f"{paper_id}.jsonl"
    started = time.time()
    manifest: Record = {
        "paper_id": paper_id,
        "archive": archive_name,
        "member": info.filename,
        "raw_path": relative_or_absolute(raw_path),
        "chunks_path": relative_or_absolute(chunks_path),
    }
    try:
        pdf_bytes = load_pdf_member(archive, info, args.max_pdf_bytes)
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
        chunks = build_chunks(
            paper,
            result,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
        write_jsonl(chunks_path, chunks)
        if args.save_raw:
            write_json(
                raw_path,
                {
                    "paper": paper,
                    "document_intelligence": result,
                    "processed_at": time.time(),
                    "archive": archive_name,
                    "member": info.filename,
                },
            )
        manifest.update(
            {
                "status": "ok",
                "pdf_bytes": len(pdf_bytes),
                "chunks": len(chunks),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        if manifest_lock:
            with manifest_lock:
                append_jsonl(manifest_path, manifest)
        else:
            append_jsonl(manifest_path, manifest)
        print(f"  {paper_id}: ok chunks={len(chunks)}", file=sys.stderr)
        return True
    except Exception as exc:  # noqa: BLE001 - keep long archive jobs resumable.
        manifest.update(
            {
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        if manifest_lock:
            with manifest_lock:
                append_jsonl(manifest_path, manifest)
        else:
            append_jsonl(manifest_path, manifest)
        print(f"  FAIL {paper_id}: {exc}", file=sys.stderr)
        return False


def planned_members(
    archive_path: Path,
    *,
    alias_map: dict[str, str],
    metadata_by_id: dict[str, Record],
    chunks_dir: Path,
    force: bool,
    log_unknown: bool,
) -> tuple[list[tuple[str, str]], int, int]:
    jobs: list[tuple[str, str]] = []
    skipped = 0
    unknown = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            info
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".pdf")
        ]
        for info in members:
            paper_id = infer_paper_id(info.filename, alias_map)
            if not paper_id or paper_id not in metadata_by_id:
                unknown += 1
                if log_unknown:
                    print(f"  UNKNOWN {info.filename}", file=sys.stderr)
                continue
            if not should_process_member(
                paper_id=paper_id,
                chunks_dir=chunks_dir,
                force=force,
            ):
                skipped += 1
                continue
            jobs.append((info.filename, paper_id))
    return jobs, skipped, unknown


def process_archive(
    archive_path: Path,
    *,
    alias_map: dict[str, str],
    metadata_by_id: dict[str, Record],
    client: Any,
    settings: DocumentIntelligenceSettings,
    raw_dir: Path,
    chunks_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    ok = 0
    jobs, skipped, unknown = planned_members(
        archive_path,
        alias_map=alias_map,
        metadata_by_id=metadata_by_id,
        chunks_dir=chunks_dir,
        force=args.force_analyze,
        log_unknown=args.log_unknown,
    )
    if args.limit:
        jobs = jobs[: args.limit]
    print(
        f"  planned: process={len(jobs)} skipped={skipped} unknown={unknown}",
        file=sys.stderr,
    )
    if not jobs:
        return ok, 0, skipped, unknown

    manifest_lock = threading.Lock()

    def run_job(
        job_index: int,
        total_jobs: int,
        filename: str,
        paper_id: str,
        pass_label: str,
    ) -> tuple[bool, tuple[str, str]]:
        paper = metadata_by_id[paper_id]
        print(
            f"  [{pass_label} {job_index}/{total_jobs}] "
            f"{paper_id} {PurePosixPath(filename).name}",
            file=sys.stderr,
        )
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(filename)
            ok = process_member(
                archive=archive,
                info=info,
                paper=paper,
                client=client,
                settings=settings,
                raw_dir=raw_dir,
                chunks_dir=chunks_dir,
                manifest_path=manifest_path,
                args=args,
                archive_name=archive_path.name,
                manifest_lock=manifest_lock,
            )
            return ok, (filename, paper_id)

    def run_pass(
        pass_jobs: list[tuple[str, str]],
        *,
        pass_label: str,
    ) -> tuple[int, list[tuple[str, str]]]:
        pass_ok = 0
        failed_jobs: list[tuple[str, str]] = []
        total_jobs = len(pass_jobs)
        if args.workers <= 1:
            for index, (filename, paper_id) in enumerate(pass_jobs, start=1):
                success, job = run_job(index, total_jobs, filename, paper_id, pass_label)
                if success:
                    pass_ok += 1
                else:
                    failed_jobs.append(job)
            return pass_ok, failed_jobs

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_job, index, total_jobs, filename, paper_id, pass_label)
                for index, (filename, paper_id) in enumerate(pass_jobs, start=1)
            ]
            for future in concurrent.futures.as_completed(futures):
                success, job = future.result()
                if success:
                    pass_ok += 1
                else:
                    failed_jobs.append(job)
        return pass_ok, failed_jobs

    remaining_jobs = jobs
    for pass_number in range(args.failed_job_retries + 1):
        if pass_number:
            if not remaining_jobs:
                break
            print(
                f"  retry pass {pass_number}/{args.failed_job_retries}: "
                f"{len(remaining_jobs)} failed PDFs",
                file=sys.stderr,
            )
            if args.failed_retry_delay_seconds:
                time.sleep(args.failed_retry_delay_seconds)
        pass_label = "main" if pass_number == 0 else f"retry-{pass_number}"
        pass_ok, remaining_jobs = run_pass(remaining_jobs, pass_label=pass_label)
        ok += pass_ok

    return ok, len(remaining_jobs), skipped, unknown


def select_archives(archives: list[BoxArchive], names: set[str]) -> list[BoxArchive]:
    if not names:
        return archives
    return [archive for archive in archives if archive.name in names]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process Box zip archives with Document Intelligence without saving PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--box-url", default=DEFAULT_BOX_SHARED_URL)
    parser.add_argument("--archive-name", action="append", default=[])
    parser.add_argument("--temp-dir", type=Path, default=ROOT / "artifacts" / "box_tmp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "docint")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reserve-gb", type=float, default=20.0)
    parser.add_argument("--delete-archive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-analyze", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--max-pdf-bytes", type=int, default=300 * 1024 * 1024)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument(
        "--content-format",
        default="",
        help="Passed as output_content_format when non-empty (markdown mode broke service-side 2026-07-03; the chunker does not need it).",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--di-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=8.0)
    parser.add_argument("--failed-job-retries", type=int, default=2)
    parser.add_argument("--failed-retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit processed PDFs per archive.")
    parser.add_argument("--list-archives", action="store_true")
    parser.add_argument("--log-unknown", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    chunks_dir = output_dir / "chunks"
    merged_chunks = output_dir / "chunks.jsonl"
    manifest_path = args.manifest or (output_dir / "_box_archive_manifest.jsonl")

    metadata = load_metadata(args.metadata)
    metadata_by_id = {
        str(record["paper_id"]): record
        for record in metadata
        if record.get("paper_id")
    }
    alias_map = build_alias_map(metadata)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "LitTraceQA-Box-DI-Pipeline/1.0"}
    )
    archives = select_archives(
        fetch_box_archives(args.box_url, session),
        set(args.archive_name),
    )
    if args.list_archives:
        for archive in archives:
            print(
                f"{archive.name}\t{archive.file_id}\t{archive.size / (1 << 30):.2f} GiB"
            )
        return 0
    if not archives:
        print("no archives selected", file=sys.stderr)
        return 0

    load_environment(args.env_file)
    settings = DocumentIntelligenceSettings.from_env()
    client = build_document_intelligence_client(settings)
    shared_name = parse_shared_name(args.box_url)

    total_ok = 0
    total_failed = 0
    total_skipped = 0
    total_unknown = 0
    for archive_index, archive in enumerate(archives, start=1):
        print(
            f"[archive {archive_index}/{len(archives)}] {archive.name} "
            f"({archive.size / (1 << 30):.2f} GiB)",
            file=sys.stderr,
        )
        archive_path = download_archive(
            session,
            shared_name=shared_name,
            archive=archive,
            temp_dir=args.temp_dir,
            reserve_gb=args.reserve_gb,
            force=args.force_download,
        )
        try:
            ok, failed, skipped, unknown = process_archive(
                archive_path,
                alias_map=alias_map,
                metadata_by_id=metadata_by_id,
                client=client,
                settings=settings,
                raw_dir=raw_dir,
                chunks_dir=chunks_dir,
                manifest_path=manifest_path,
                args=args,
            )
        finally:
            if args.delete_archive:
                archive_path.unlink(missing_ok=True)
        total_ok += ok
        total_failed += failed
        total_skipped += skipped
        total_unknown += unknown
        merged = merge_chunk_files(chunks_dir, merged_chunks)
        print(
            f"archive done: ok={ok} failed={failed} skipped={skipped} "
            f"unknown={unknown} merged_chunks={merged}",
            file=sys.stderr,
        )

    print(
        "finished: "
        f"ok={total_ok} failed={total_failed} skipped={total_skipped} "
        f"unknown={total_unknown} chunks={merged_chunks}",
        file=sys.stderr,
    )
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
