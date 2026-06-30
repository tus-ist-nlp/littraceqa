#!/usr/bin/env python3
"""Extract PDF archives into the canonical ``pdfs/{paper_id}.pdf`` layout.

The Box bundle can be downloaded as multiple zip files. Put those files under
``archives/`` and run this script once; it will scan every PDF in every archive,
infer the LitTraceQA ``paper_id`` from the file name or metadata aliases, and
write a resumable manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional
from urllib.parse import urlparse

from littrace_common import (
    DEFAULT_METADATA,
    ROOT,
    Record,
    append_jsonl,
    load_metadata,
    normalize_alias,
    relative_or_absolute,
)


def metadata_aliases(record: Record) -> set[str]:
    aliases = {
        record.get("paper_id"),
        record.get("anthology_id"),
        record.get("arxiv_id"),
        record.get("doi"),
    }
    for url_field in ("pdf_url", "source_url"):
        url = str(record.get(url_field) or "")
        if url:
            parsed = urlparse(url)
            aliases.add(PurePosixPath(parsed.path).name)
            aliases.add(PurePosixPath(parsed.path).stem)
    return {normalize_alias(alias) for alias in aliases if normalize_alias(alias)}


def build_alias_map(metadata: list[Record]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for record in metadata:
        paper_id = str(record.get("paper_id") or "")
        if not paper_id:
            continue
        for alias in metadata_aliases(record):
            alias_map.setdefault(alias, paper_id)
    return alias_map


def iter_archives(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"archive path not found: {path}")
    archives = sorted(path.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"no .zip files found under {path}")
    return archives


def infer_paper_id(name: str, alias_map: dict[str, str]) -> Optional[str]:
    entry = PurePosixPath(name)
    candidates = [
        entry.name,
        entry.stem,
        str(entry),
        str(entry.with_suffix("")),
    ]
    for candidate in candidates:
        paper_id = alias_map.get(normalize_alias(candidate))
        if paper_id:
            return paper_id
    return None


def is_probable_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    dest: Path,
    *,
    force: bool,
) -> tuple[str, int, str]:
    if dest.exists() and not force and is_probable_pdf(dest):
        digest = sha256_file(dest)
        return ("skipped", dest.stat().st_size, digest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    bytes_written = 0
    with archive.open(info, "r") as source, tmp.open("wb") as target:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            target.write(chunk)
            digest.update(chunk)
            bytes_written += len(chunk)
    tmp.replace(dest)
    return ("ok", bytes_written, digest.hexdigest())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_pdf_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [
        info
        for info in archive.infolist()
        if not info.is_dir() and info.filename.lower().endswith(".pdf")
    ]


def process_archive(
    archive_path: Path,
    *,
    alias_map: dict[str, str],
    output_dir: Path,
    unknown_dir: Optional[Path],
    manifest_path: Path,
    force: bool,
    dry_run: bool,
    limit: int,
) -> tuple[int, int, int]:
    processed = 0
    unknown = 0
    failed = 0

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive_pdf_members(archive):
            if limit and processed >= limit:
                break

            paper_id = infer_paper_id(info.filename, alias_map)
            if paper_id:
                dest = output_dir / f"{paper_id}.pdf"
            elif unknown_dir:
                safe_name = PurePosixPath(info.filename).name
                dest = unknown_dir / archive_path.stem / safe_name
                unknown += 1
            else:
                unknown += 1
                print(
                    f"UNKNOWN {archive_path.name}: {info.filename}",
                    file=sys.stderr,
                )
                continue

            record: Record = {
                "archive": str(archive_path),
                "member": info.filename,
                "paper_id": paper_id,
                "path": relative_or_absolute(dest),
                "status": "dry_run" if dry_run else "pending",
            }

            if dry_run:
                print(json.dumps(record, ensure_ascii=False))
                processed += 1
                continue

            try:
                status, size, digest = extract_member(
                    archive,
                    info,
                    dest,
                    force=force,
                )
                if not is_probable_pdf(dest):
                    raise ValueError("extracted file does not start with %PDF-")
                record.update(
                    {
                        "status": status,
                        "bytes": size,
                        "sha256": digest,
                    }
                )
                processed += 1
            except Exception as exc:  # noqa: BLE001 - keep going across files.
                failed += 1
                record.update({"status": "failed", "error": str(exc)})
                print(
                    f"FAIL {archive_path.name}: {info.filename}: {exc}",
                    file=sys.stderr,
                )
            append_jsonl(manifest_path, record)

    return processed, unknown, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract zip archives into pdfs/{paper_id}.pdf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--archives", type=Path, default=ROOT / "archives")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=ROOT / "pdfs")
    parser.add_argument(
        "--unknown-output",
        type=Path,
        default=None,
        help="Optional directory for PDFs that cannot be mapped to paper_id.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to <output>/_archive_manifest.jsonl.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit PDFs per archive; useful for smoke tests.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = load_metadata(args.metadata)
    alias_map = build_alias_map(metadata)
    archives = iter_archives(args.archives)
    manifest_path = args.manifest or (args.output / "_archive_manifest.jsonl")

    total_processed = 0
    total_unknown = 0
    total_failed = 0
    print(f"archives: {len(archives)}", file=sys.stderr)
    for archive_path in archives:
        processed, unknown, failed = process_archive(
            archive_path,
            alias_map=alias_map,
            output_dir=args.output,
            unknown_dir=args.unknown_output,
            manifest_path=manifest_path,
            force=args.force,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        total_processed += processed
        total_unknown += unknown
        total_failed += failed
        print(
            f"{archive_path.name}: processed={processed} unknown={unknown} failed={failed}",
            file=sys.stderr,
        )

    print(
        f"done: processed={total_processed} unknown={total_unknown} failed={total_failed}",
        file=sys.stderr,
    )
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
