#!/usr/bin/env python3
"""Download and validate the paper PDFs listed in ``data/paper_metadata.jsonl``.

For every record this script reads ``pdf_url``, downloads the PDF with per-host
rate limiting and automatic retries, validates that the file is a genuine,
non-empty PDF, and records the outcome in a resumable manifest.

Design
------
* **Per-host parallelism** -- each host is processed by its own worker thread,
  so different hosts download concurrently. Within a host, requests are issued
  sequentially and spaced by a minimum interval (with jitter), so we never
  hammer a single server.
* **Automatic retries** -- transient failures (timeouts, connection resets,
  HTTP 429/5xx) are retried with exponential backoff. ``Retry-After`` headers
  are honoured. Permanent failures (404, 403, ...) are not retried.
* **Resumable** -- already-downloaded-and-validated PDFs are skipped on re-run,
  so the job can be interrupted (Ctrl-C) and restarted at any time.
* **Validated** -- a download only counts as success once the bytes pass PDF
  structural checks (magic header, EOF trailer, minimum size), parse with
  ``pypdf``, and yield at least ``--min-text-chars`` of extractable text -- a
  cheap way to reject login/paywall/scanned stubs that are technically PDFs.

Usage
-----
    python scripts/download_pdfs.py                 # download everything
    python scripts/download_pdfs.py --limit 8       # smoke test (first 8)
    python scripts/download_pdfs.py --filter-host openreview.net
    python scripts/download_pdfs.py --force         # re-download everything
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import random
import signal
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import pypdf
import requests


# A metadata/manifest record. Aliased to keep the many signatures readable.
Record = dict[str, Any]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = ROOT / "data" / "paper_metadata.jsonl"
DEFAULT_OUTPUT = ROOT / "pdfs"

# Conservative, host-specific minimum seconds between requests. openreview.net
# is a single application host serving ~12k of our PDFs, so it gets the most
# breathing room; the static mirrors tolerate a slightly tighter cadence.
DEFAULT_HOST_INTERVALS: dict[str, float] = {
    "openreview.net": 5.0,
    "aclanthology.org": 3.0,
    "openaccess.thecvf.com": 3.0,
    "www.ecva.net": 3.0,
}
FALLBACK_INTERVAL = 5.0

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
PDF_MAGIC = b"%PDF-"

DEFAULT_USER_AGENT = (
    "LitTraceQA-PDF-Fetcher/1.0 "
    "(+https://github.com/; academic dataset assembly; contact: maintainer)"
)


def resolve_interval(
    host: str, overrides: dict[str, float], default_interval: float
) -> float:
    """Most specific configured minimum interval for ``host``.

    Precedence: an explicit ``--interval`` override, then the built-in per-host
    default, then the catch-all ``default_interval``.
    """
    return overrides.get(host, DEFAULT_HOST_INTERVALS.get(host, default_interval))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Validation:
    ok: bool
    reason: Optional[str] = None
    pages: Optional[int] = None
    text_chars: Optional[int] = None


def _extracted_text_length(reader: pypdf.PdfReader, *, max_pages: int) -> int:
    """Total non-whitespace characters extractable from the first pages."""
    total = 0
    for page in reader.pages[:max_pages]:
        try:
            total += len((page.extract_text() or "").strip())
        except Exception:  # noqa: BLE001 - text layer can be absent/corrupt.
            continue
    return total


def validate_pdf(path: Path, *, min_bytes: int, min_text_chars: int) -> Validation:
    """Check that ``path`` holds a genuine, readable PDF with real text.

    Structural checks (header, trailer, size) catch HTML error pages; parsing
    with ``pypdf`` and requiring ``min_text_chars`` of extractable text rejects
    login/paywall/scanned stubs that happen to be valid PDF containers.
    """
    size = path.stat().st_size
    if size < min_bytes:
        return Validation(False, reason=f"file too small ({size} bytes)")

    with path.open("rb") as handle:
        head = handle.read(1024)
        handle.seek(max(0, size - 2048))
        tail = handle.read(2048)

    if PDF_MAGIC not in head:
        # An HTML error / login / paywall page will not contain "%PDF-".
        return Validation(False, reason="missing %PDF- header (not a PDF?)")
    if b"%%EOF" not in tail:
        return Validation(False, reason="missing %%EOF trailer (truncated?)")

    try:
        reader = pypdf.PdfReader(str(path))
        pages = len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a reject.
        return Validation(False, reason=f"pypdf could not parse: {exc}")
    if pages == 0:
        return Validation(False, reason="PDF has zero pages")

    text_chars = _extracted_text_length(reader, max_pages=2)
    if text_chars < min_text_chars:
        return Validation(
            False, reason=f"too little text ({text_chars} chars < {min_text_chars})"
        )

    return Validation(True, pages=pages, text_chars=text_chars)


# --------------------------------------------------------------------------- #
# Manifest (resumable progress log)
# --------------------------------------------------------------------------- #
class Manifest:
    """Append-only JSONL log of per-paper outcomes, safe across threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.records: dict[str, Record] = {}
        self._handle = None

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = record.get("paper_id")
                if pid:
                    self.records[pid] = record  # last write wins

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def write(self, record: Record) -> None:
        with self._lock:
            self.records[record["paper_id"]] = record
            if self._handle is not None:
                self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class FetchResult:
    """Outcome of one paper download; ``to_record`` serializes it for the manifest."""

    paper_id: str
    url: str
    host: str
    status: str = "failed"
    http_status: Optional[int] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    pages: Optional[int] = None
    text_chars: Optional[int] = None
    attempts: int = 0
    error: Optional[str] = None
    path: Optional[str] = None
    timestamp: Optional[float] = None

    def to_record(self) -> Record:
        return dataclasses.asdict(self)


def _retry_after_seconds(resp: requests.Response) -> Optional[float]:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)  # delta-seconds form
    except ValueError:
        return None  # HTTP-date form: fall back to normal backoff


def _stream_to_temp(
    resp: requests.Response,
    tmp: Path,
    *,
    max_bytes: int,
    digest: Any,
    stop_event: threading.Event,
) -> int:
    """Stream ``resp`` into ``tmp``, updating ``digest``; return byte count.

    Raises ``IOError`` if the body grows past ``max_bytes`` (0 = no limit) or if
    ``stop_event`` fires mid-stream, so Ctrl-C is honoured even during a large or
    trickle-fed download (otherwise the read-timeout resets on every chunk).
    """
    downloaded = 0
    with tmp.open("wb") as out:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if stop_event.is_set():
                raise IOError("aborted mid-download")
            if not chunk:
                continue
            out.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if max_bytes and downloaded > max_bytes:
                raise IOError(f"exceeded --max-bytes ({max_bytes})")
    return downloaded


def _compute_backoff_delay(
    attempt: int,
    retry_after: Optional[float],
    args: argparse.Namespace,
    interval: float,
) -> float:
    """Seconds to wait before the next attempt (capped, jittered, host-paced).

    Floored at the host ``interval`` so a paper's retries never undercut the
    host's minimum spacing -- including when a server replies ``Retry-After: 0``.
    """
    if retry_after is not None:
        delay = min(retry_after, args.backoff_cap)
    else:
        base = args.backoff_base * (2 ** (attempt - 1))
        delay = min(base, args.backoff_cap)
    delay *= 1.0 + random.uniform(0.0, args.jitter)
    return max(delay, interval)


def fetch_one(
    record: Record,
    *,
    session: requests.Session,
    output_dir: Path,
    args: argparse.Namespace,
    interval: float,
    stop_event: threading.Event,
) -> FetchResult:
    """Download + validate a single paper, retrying transient failures."""
    paper_id = record["paper_id"]
    url = record["pdf_url"]
    dest = output_dir / f"{paper_id}.pdf"
    tmp = dest.with_suffix(".pdf.part")

    result = FetchResult(paper_id=paper_id, url=url, host=urlparse(url).netloc)

    max_attempts = args.max_retries + 1
    for attempt in range(1, max_attempts + 1):
        if stop_event.is_set():
            result.error = "aborted before request"
            break

        result.attempts = attempt
        retry_after: Optional[float] = None

        try:
            with session.get(
                url,
                stream=True,
                timeout=(args.connect_timeout, args.read_timeout),
                allow_redirects=True,
            ) as resp:
                result.http_status = resp.status_code

                if resp.status_code != 200:
                    result.error = f"HTTP {resp.status_code}"
                    if resp.status_code in RETRYABLE_STATUS:
                        retry_after = _retry_after_seconds(resp)
                    else:
                        break  # permanent (404, 403, 401, ...): give up now
                else:
                    digest = hashlib.sha256()
                    downloaded = _stream_to_temp(
                        resp,
                        tmp,
                        max_bytes=args.max_bytes,
                        digest=digest,
                        stop_event=stop_event,
                    )
                    validation = validate_pdf(
                        tmp,
                        min_bytes=args.min_bytes,
                        min_text_chars=args.min_text_chars,
                    )
                    if validation.ok:
                        tmp.replace(dest)
                        result.status = "ok"
                        result.bytes = downloaded
                        result.sha256 = digest.hexdigest()
                        result.pages = validation.pages
                        result.text_chars = validation.text_chars
                        result.error = None
                        result.path = (
                            str(dest.relative_to(ROOT))
                            if dest.is_relative_to(ROOT)
                            else str(dest)
                        )
                        result.timestamp = time.time()
                        return result

                    # 200 but not a valid PDF: could be a transient HTML error
                    # page, so allow a couple of retries within the budget.
                    tmp.unlink(missing_ok=True)
                    result.error = f"invalid PDF: {validation.reason}"

        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            result.error = f"network error: {exc.__class__.__name__}: {exc}"
        except IOError as exc:
            tmp.unlink(missing_ok=True)
            result.error = f"io error: {exc}"
        except Exception as exc:  # noqa: BLE001 - degrade any surprise to a failed record.
            tmp.unlink(missing_ok=True)
            result.error = f"unexpected error: {exc.__class__.__name__}: {exc}"

        # Decide whether to retry.
        if attempt < max_attempts and not stop_event.is_set():
            delay = _compute_backoff_delay(attempt, retry_after, args, interval)
            # Interruptible sleep so Ctrl-C is responsive during long backoffs.
            stop_event.wait(timeout=delay)

    result.timestamp = time.time()
    tmp.unlink(missing_ok=True)
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Progress:
    total: int
    ok: int = 0
    failed: int = 0
    done: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def record(self, status: str) -> None:
        with self._lock:
            self.done += 1
            if status == "ok":
                self.ok += 1
            else:
                self.failed += 1


def download_host(
    papers: list[Record],
    *,
    interval: float,
    session: requests.Session,
    output_dir: Path,
    args: argparse.Namespace,
    manifest: Manifest,
    progress: Progress,
    stop_event: threading.Event,
) -> None:
    """Download one host's papers sequentially, spaced by ``interval`` seconds."""
    next_allowed = 0.0
    for record in papers:
        if stop_event.is_set():
            return
        wait = next_allowed - time.monotonic()
        if wait > 0 and stop_event.wait(timeout=wait):
            return  # stop_event fired during the pacing sleep
        next_allowed = time.monotonic() + interval * (1.0 + random.uniform(0.0, args.jitter))

        outcome = fetch_one(
            record,
            session=session,
            output_dir=output_dir,
            args=args,
            interval=interval,
            stop_event=stop_event,
        )
        manifest.write(outcome.to_record())
        progress.record(outcome.status)


def load_metadata(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("pdf_url") and record.get("paper_id"):
                records.append(record)
    return records


def should_skip(
    record: Record,
    *,
    manifest: Manifest,
    output_dir: Path,
    args: argparse.Namespace,
) -> bool:
    """A paper is skipped only if it already succeeded and its file is present.

    Past *failures* are always re-attempted on the next run, which is exactly
    the "automatically retry" behaviour we want across restarts. ``--force``
    re-downloads everything regardless.
    """
    if args.force:
        return False
    prior = manifest.records.get(record["paper_id"])
    dest = output_dir / f"{record['paper_id']}.pdf"
    if prior and prior.get("status") == "ok" and dest.exists():
        return True
    if args.skip_failed and prior and prior.get("status") == "failed":
        return True
    return False


def build_session(args: argparse.Namespace) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent,
            "Accept": "application/pdf,*/*",
        }
    )
    return session


def parse_interval_overrides(values: Optional[list[str]]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--interval expects HOST=SECONDS, got {item!r}"
            )
        host, _, seconds = item.partition("=")
        overrides[host.strip()] = float(seconds)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and validate paper PDFs from paper_metadata.jsonl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <output>/_manifest.jsonl.",
    )
    parser.add_argument(
        "--interval",
        action="append",
        metavar="HOST=SECONDS",
        help="Override a host's minimum interval (repeatable).",
    )
    parser.add_argument(
        "--default-interval",
        type=float,
        default=FALLBACK_INTERVAL,
        help="Interval for hosts without a configured default.",
    )
    parser.add_argument("--jitter", type=float, default=0.3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--backoff-cap", type=float, default=120.0)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=100,
        help="Reject a PDF whose first pages yield less extractable text.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="Abort a download exceeding this many bytes (0 = no limit).",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--limit", type=int, default=0, help="Only process the first N papers."
    )
    parser.add_argument(
        "--filter-host",
        action="append",
        help="Only process papers on this host (repeatable).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already ok."
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Do not re-attempt papers that failed on a previous run "
        "(by default every re-run retries past failures).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be downloaded, then exit.",
    )
    return parser


def plan_work(
    records: list[Record],
    *,
    manifest: Manifest,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, list[Record]], int]:
    """Apply --filter-host/--limit, then split into per-host work vs. skipped."""
    if args.filter_host:
        wanted = set(args.filter_host)
        records = [r for r in records if urlparse(r["pdf_url"]).netloc in wanted]
    if args.limit:
        records = records[: args.limit]

    by_host: dict[str, list[Record]] = defaultdict(list)
    skipped = 0
    for record in records:
        if should_skip(record, manifest=manifest, output_dir=output_dir, args=args):
            skipped += 1
            continue
        by_host[urlparse(record["pdf_url"]).netloc].append(record)
    return by_host, skipped


def print_plan(
    by_host: dict[str, list[Record]],
    skipped: int,
    interval_overrides: dict[str, float],
    args: argparse.Namespace,
) -> int:
    """Report the planned work to stderr; return the number of papers to fetch."""
    pending = sum(len(v) for v in by_host.values())
    # Every selected paper is either skipped or queued, so this is len(records).
    total = skipped + pending
    print(
        f"papers: {total} | already ok (skip): {skipped} | to fetch: {pending}",
        file=sys.stderr,
    )
    for host in sorted(by_host):
        interval = resolve_interval(host, interval_overrides, args.default_interval)
        print(
            f"  {host:28s} {len(by_host[host]):6d} papers  @ >= {interval:.1f}s",
            file=sys.stderr,
        )
    return pending


def print_summary(
    progress: Progress, skipped: int, manifest_path: Path, elapsed: float
) -> None:
    print(
        f"\nfinished in {elapsed/60:.1f} min: "
        f"ok {progress.ok}, failed {progress.failed}, "
        f"skipped {skipped}, processed {progress.done}/{progress.total}",
        file=sys.stderr,
    )
    print(f"manifest: {manifest_path}", file=sys.stderr)
    if progress.failed:
        print(
            "re-run the same command to retry the failures "
            "(successful PDFs are skipped automatically).",
            file=sys.stderr,
        )


def run_downloads(
    by_host: dict[str, list[Record]],
    *,
    interval_overrides: dict[str, float],
    session: requests.Session,
    output_dir: Path,
    args: argparse.Namespace,
    manifest: Manifest,
    progress: Progress,
    stop_event: threading.Event,
) -> None:
    """Run one worker per host concurrently; block until all finish or stop.

    On a worker error or KeyboardInterrupt, signal every worker to stop and tear
    the pool down *without waiting*, so a single failed or stuck host cannot pin
    the whole run (workers poll ``stop_event`` and abort promptly).
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(by_host))
    futures = [
        executor.submit(
            download_host,
            papers,
            interval=resolve_interval(host, interval_overrides, args.default_interval),
            session=session,
            output_dir=output_dir,
            args=args,
            manifest=manifest,
            progress=progress,
            stop_event=stop_event,
        )
        for host, papers in by_host.items()
    ]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()  # surface unexpected worker exceptions
    except BaseException:
        stop_event.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.metadata.exists():
        parser.error(f"metadata not found: {args.metadata}")

    interval_overrides = parse_interval_overrides(args.interval)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (output_dir / "_manifest.jsonl")

    manifest = Manifest(manifest_path)
    manifest.load()

    records = load_metadata(args.metadata)
    by_host, skipped = plan_work(
        records, manifest=manifest, output_dir=output_dir, args=args
    )
    pending = print_plan(by_host, skipped, interval_overrides, args)

    if args.dry_run:
        return 0
    if pending == 0:
        print("nothing to do.", file=sys.stderr)
        return 0

    progress = Progress(total=pending)
    stop_event = threading.Event()

    def handle_sigint(signum, frame):  # noqa: ANN001 - signal handler signature.
        if not stop_event.is_set():
            print(
                "\ninterrupt received -- finishing in-flight downloads, "
                "then stopping (press Ctrl-C again to force).",
                file=sys.stderr,
            )
            stop_event.set()
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    manifest.open()
    session = build_session(args)
    start = time.monotonic()

    # Periodic progress reporter (daemon: dies with the process).
    def reporter() -> None:
        while not stop_event.wait(timeout=15.0):
            elapsed = time.monotonic() - start
            rate = progress.done / elapsed if elapsed else 0.0
            remaining = progress.total - progress.done
            eta = remaining / rate if rate else float("inf")
            print(
                f"[{elapsed:6.0f}s] done {progress.done}/{progress.total} "
                f"(ok {progress.ok}, fail {progress.failed}) "
                f"~{rate*60:.1f}/min eta ~{eta/60:.0f}min",
                file=sys.stderr,
            )
            if progress.done >= progress.total:
                break

    threading.Thread(target=reporter, daemon=True).start()

    try:
        run_downloads(
            by_host,
            interval_overrides=interval_overrides,
            session=session,
            output_dir=output_dir,
            args=args,
            manifest=manifest,
            progress=progress,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        manifest.close()

    print_summary(progress, skipped, manifest_path, time.monotonic() - start)
    return 1 if progress.failed and progress.ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
