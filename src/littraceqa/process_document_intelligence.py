#!/usr/bin/env python3
"""Run Azure AI Document Intelligence over PDFs and build search chunks.

Outputs
-------
``artifacts/docint/raw/{paper_id}.json``
    The full Document Intelligence result plus LitTraceQA metadata.

``artifacts/docint/chunks/{paper_id}.jsonl``
    Search-ready chunks for one paper.

``artifacts/docint/chunks.jsonl``
    A merged JSONL file used by ``build_azure_search_index.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

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
    azure_document_key,
    clean_text,
    compact_text,
    load_metadata,
    normalize_alias,
    read_json,
    relative_or_absolute,
    write_json,
    write_jsonl,
)


TEXT_ROLES_AS_HEADINGS = {
    "title",
    "sectionHeading",
    "pageHeader",
}


@dataclasses.dataclass
class TextUnit:
    text: str
    pages: set[int]
    section: str = ""


def value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def to_plain_json(obj: Any) -> Any:
    if hasattr(obj, "as_dict"):
        return to_plain_json(obj.as_dict())
    if isinstance(obj, dict):
        return {str(key): to_plain_json(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_json(item) for item in obj]
    return obj


def analyze_pdf(
    client: Any,
    pdf_path: Path,
    *,
    model: str,
    features: list[str],
    content_format: str,
) -> Record:
    kwargs: Record = {}
    if features:
        kwargs["features"] = features
    if content_format:
        kwargs["output_content_format"] = content_format

    with pdf_path.open("rb") as handle:
        try:
            poller = client.begin_analyze_document(model, body=handle, **kwargs)
        except TypeError:
            kwargs.pop("output_content_format", None)
            handle.seek(0)
            poller = client.begin_analyze_document(model, body=handle, **kwargs)
    result = poller.result()
    return to_plain_json(result)


def analyze_pdf_bytes(
    client: Any,
    pdf_bytes: bytes,
    *,
    model: str,
    features: list[str],
    content_format: str,
) -> Record:
    kwargs: Record = {}
    if features:
        kwargs["features"] = features
    if content_format:
        kwargs["output_content_format"] = content_format

    body = io.BytesIO(pdf_bytes)
    try:
        poller = client.begin_analyze_document(model, body=body, **kwargs)
    except TypeError:
        kwargs.pop("output_content_format", None)
        body.seek(0)
        poller = client.begin_analyze_document(model, body=body, **kwargs)
    result = poller.result()
    return to_plain_json(result)


def get_pages(item: Any) -> set[int]:
    pages: set[int] = set()
    regions = value(item, "boundingRegions", "bounding_regions", default=[]) or []
    for region in regions:
        page = value(region, "pageNumber", "page_number")
        if isinstance(page, int):
            pages.add(page)
        else:
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                pass
    return pages


def first_page(pages: Iterable[int]) -> Optional[int]:
    ordered = sorted(set(pages))
    return ordered[0] if ordered else None


def paper_metadata_text(record: Record) -> str:
    authors = ", ".join(str(author) for author in record.get("authors") or [])
    lines = [
        f"Title: {record.get('title') or ''}",
        f"Authors: {authors}",
        f"Venue: {record.get('venue') or ''}",
        f"Year: {record.get('year') or ''}",
        f"Track: {record.get('track') or ''}",
        f"Anthology ID: {record.get('anthology_id') or ''}",
        f"Abstract: {record.get('abstract') or ''}",
    ]
    return "\n".join(clean_text(line) for line in lines if clean_text(line))


def base_chunk(record: Record, chunk_id: int, source_type: str, content: str) -> Record:
    authors = ", ".join(str(author) for author in record.get("authors") or [])
    return {
        "id": azure_document_key(record["paper_id"], chunk_id),
        "paper_id": record["paper_id"],
        "chunk_id": chunk_id,
        "source_type": source_type,
        "title": record.get("title") or "",
        "authors": authors,
        "abstract": record.get("abstract") or "",
        "venue": record.get("venue") or "",
        "year": record.get("year"),
        "track": record.get("track") or "",
        "award": record.get("award"),
        "source_url": record.get("source_url") or "",
        "pdf_url": record.get("pdf_url") or "",
        "anthology_id": record.get("anthology_id") or "",
        "section": "",
        "page_numbers": [],
        "locator_json": "{}",
        "content": content,
    }


def make_chunk(
    record: Record,
    chunk_id: int,
    source_type: str,
    content: str,
    *,
    pages: Iterable[int] = (),
    section: str = "",
    locator: Optional[Record] = None,
) -> Record:
    chunk = base_chunk(record, chunk_id, source_type, clean_text(content))
    ordered_pages = sorted(set(int(page) for page in pages if page))
    locator_payload = dict(locator or {})
    if ordered_pages and "page" not in locator_payload:
        locator_payload["page"] = ordered_pages[0]
    if section and "section" not in locator_payload:
        locator_payload["section"] = section
    chunk.update(
        {
            "section": section,
            "page_numbers": ordered_pages,
            "locator_json": json.dumps(locator_payload, ensure_ascii=False),
        }
    )
    return chunk


def paragraph_units(result: Record) -> list[TextUnit]:
    paragraphs = value(result, "paragraphs", default=[]) or []
    units: list[TextUnit] = []
    current_section = ""
    for paragraph in paragraphs:
        text = clean_text(value(paragraph, "content", default=""))
        if not text:
            continue
        role = value(paragraph, "role", default="") or ""
        if role in TEXT_ROLES_AS_HEADINGS:
            current_section = compact_text(text, max_chars=200)
        prefix = f"[{role}] " if role and role not in {"body", "footnote"} else ""
        units.append(
            TextUnit(
                text=f"{prefix}{text}",
                pages=get_pages(paragraph),
                section=current_section,
            )
        )
    return units


def fallback_units(result: Record) -> list[TextUnit]:
    content = clean_text(value(result, "content", default=""))
    if not content:
        return []
    units: list[TextUnit] = []
    for block in content.split("\n\n"):
        block = clean_text(block)
        if block:
            units.append(TextUnit(text=block, pages=set(), section=""))
    return units


def split_long_text(text: str, max_chars: int) -> list[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        pieces.append(text[start:end].strip())
        start = end
    return [piece for piece in pieces if piece]


def chunk_text_units(
    record: Record,
    units: list[TextUnit],
    *,
    start_chunk_id: int,
    max_chars: int,
    overlap_chars: int,
) -> tuple[list[Record], int]:
    chunks: list[Record] = []
    buffer: list[TextUnit] = []
    buffer_len = 0
    chunk_id = start_chunk_id

    def emit() -> None:
        nonlocal chunk_id, buffer, buffer_len
        if not buffer:
            return
        text = "\n\n".join(unit.text for unit in buffer)
        pages: set[int] = set()
        sections = [unit.section for unit in buffer if unit.section]
        for unit in buffer:
            pages.update(unit.pages)
        chunks.append(
            make_chunk(
                record,
                chunk_id,
                "text_span",
                text,
                pages=pages,
                section=sections[-1] if sections else "",
            )
        )
        chunk_id += 1

        if overlap_chars <= 0:
            buffer = []
            buffer_len = 0
            return
        tail: list[TextUnit] = []
        tail_len = 0
        for unit in reversed(buffer):
            unit_len = len(unit.text) + 2
            if tail and tail_len + unit_len > overlap_chars:
                break
            tail.insert(0, unit)
            tail_len += unit_len
        buffer = tail
        buffer_len = tail_len

    for unit in units:
        pieces = split_long_text(unit.text, max_chars)
        for piece in pieces:
            piece_unit = TextUnit(piece, unit.pages, unit.section)
            piece_len = len(piece) + 2
            if buffer and buffer_len + piece_len > max_chars:
                emit()
            buffer.append(piece_unit)
            buffer_len += piece_len
    emit()
    return chunks, chunk_id


def markdown_escape_cell(value_: Any) -> str:
    return compact_text(value_, max_chars=400).replace("|", "\\|")


def table_to_markdown(table: Any, table_number: int) -> str:
    row_count = int(value(table, "rowCount", "row_count", default=0) or 0)
    column_count = int(value(table, "columnCount", "column_count", default=0) or 0)
    if row_count <= 0 or column_count <= 0:
        return ""

    grid = [["" for _ in range(column_count)] for _ in range(row_count)]
    for cell in value(table, "cells", default=[]) or []:
        row = int(value(cell, "rowIndex", "row_index", default=0) or 0)
        col = int(value(cell, "columnIndex", "column_index", default=0) or 0)
        if 0 <= row < row_count and 0 <= col < column_count:
            grid[row][col] = markdown_escape_cell(value(cell, "content", default=""))

    header = grid[0]
    body = grid[1:] if len(grid) > 1 else []
    lines = [
        f"Table {table_number}",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_chunks(
    record: Record,
    result: Record,
    *,
    start_chunk_id: int,
) -> tuple[list[Record], int]:
    chunks: list[Record] = []
    chunk_id = start_chunk_id
    tables = value(result, "tables", default=[]) or []
    for index, table in enumerate(tables, start=1):
        content = table_to_markdown(table, index)
        if not content:
            continue
        pages = get_pages(table)
        locator: Record = {
            "table_id": f"Table {index}",
        }
        page = first_page(pages)
        if page is not None:
            locator["page"] = page
        chunks.append(
            make_chunk(
                record,
                chunk_id,
                "table",
                content,
                pages=pages,
                locator=locator,
            )
        )
        chunk_id += 1
    return chunks, chunk_id


def figure_chunks(
    record: Record,
    result: Record,
    *,
    start_chunk_id: int,
) -> tuple[list[Record], int]:
    chunks: list[Record] = []
    chunk_id = start_chunk_id
    figures = value(result, "figures", default=[]) or []
    for index, figure in enumerate(figures, start=1):
        caption = value(figure, "caption", default={}) or {}
        content = clean_text(value(caption, "content", default=""))
        if not content:
            continue
        pages = get_pages(figure) or get_pages(caption)
        locator: Record = {"figure_id": f"Figure {index}"}
        page = first_page(pages)
        if page is not None:
            locator["page"] = page
        chunks.append(
            make_chunk(
                record,
                chunk_id,
                "figure",
                f"Figure {index}: {content}",
                pages=pages,
                locator=locator,
            )
        )
        chunk_id += 1
    return chunks, chunk_id


def build_chunks(
    record: Record,
    result: Record,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[Record]:
    chunks: list[Record] = [
        make_chunk(
            record,
            0,
            "metadata",
            paper_metadata_text(record),
            locator={"source": "paper_metadata"},
        )
    ]
    chunk_id = 1

    units = paragraph_units(result) or fallback_units(result)
    text_chunks, chunk_id = chunk_text_units(
        record,
        units,
        start_chunk_id=chunk_id,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    chunks.extend(text_chunks)

    tables, chunk_id = table_chunks(record, result, start_chunk_id=chunk_id)
    chunks.extend(tables)

    figures, _ = figure_chunks(record, result, start_chunk_id=chunk_id)
    chunks.extend(figures)
    return [chunk for chunk in chunks if chunk.get("content")]


def candidate_pdf_paths(record: Record, pdf_dir: Path) -> list[Path]:
    candidates = [pdf_dir / f"{record['paper_id']}.pdf"]
    for field in ("anthology_id", "arxiv_id"):
        value_ = record.get(field)
        if value_:
            candidates.append(pdf_dir / f"{value_}.pdf")
    pdf_url = str(record.get("pdf_url") or "")
    if pdf_url:
        stem = Path(pdf_url.split("?")[0]).stem
        if stem:
            candidates.append(pdf_dir / f"{stem}.pdf")
    return candidates


def find_pdf(record: Record, pdf_dir: Path) -> Optional[Path]:
    for path in candidate_pdf_paths(record, pdf_dir):
        if path.exists():
            return path
    normalized_id = normalize_alias(record.get("paper_id"))
    for path in pdf_dir.glob("*.pdf"):
        if normalize_alias(path.stem) == normalized_id:
            return path
    return None


def select_records(records: list[Record], paper_ids: set[str], limit: int) -> list[Record]:
    if paper_ids:
        records = [record for record in records if record.get("paper_id") in paper_ids]
    if limit:
        records = records[:limit]
    return records


def process_one(
    record: Record,
    *,
    client: Any,
    settings: DocumentIntelligenceSettings,
    pdf_dir: Path,
    raw_dir: Path,
    chunks_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
) -> bool:
    paper_id = str(record["paper_id"])
    raw_path = raw_dir / f"{paper_id}.json"
    chunks_path = chunks_dir / f"{paper_id}.jsonl"
    pdf_path = find_pdf(record, pdf_dir)

    started = time.time()
    manifest: Record = {
        "paper_id": paper_id,
        "raw_path": relative_or_absolute(raw_path),
        "chunks_path": relative_or_absolute(chunks_path),
    }

    try:
        if not pdf_path:
            raise FileNotFoundError(f"missing PDF under {pdf_dir}")
        manifest["pdf_path"] = relative_or_absolute(pdf_path)

        if raw_path.exists() and not args.force_analyze:
            payload = read_json(raw_path)
            if not isinstance(payload, dict) or "document_intelligence" not in payload:
                raise ValueError(f"invalid raw cache: {raw_path}")
            result = payload["document_intelligence"]
            status = "cached"
        else:
            result = analyze_pdf(
                client,
                pdf_path,
                model=settings.model,
                features=args.feature,
                content_format=args.content_format,
            )
            payload = {
                "paper": record,
                "document_intelligence": result,
                "processed_at": time.time(),
                "pdf_path": relative_or_absolute(pdf_path),
            }
            write_json(raw_path, payload)
            status = "ok"

        if chunks_path.exists() and not args.force_chunks and status == "cached":
            with chunks_path.open(encoding="utf-8") as handle:
                chunk_count = sum(1 for _ in handle)
        else:
            chunks = build_chunks(
                record,
                result,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
            )
            write_jsonl(chunks_path, chunks)
            chunk_count = len(chunks)

        manifest.update(
            {
                "status": status,
                "chunks": chunk_count,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        append_jsonl(manifest_path, manifest)
        print(f"{paper_id}: {status}, chunks={chunk_count}", file=sys.stderr)
        return True
    except Exception as exc:  # noqa: BLE001 - keep a long batch resumable.
        manifest.update(
            {
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        append_jsonl(manifest_path, manifest)
        print(f"FAIL {paper_id}: {exc}", file=sys.stderr)
        return False


def merge_chunk_files(chunks_dir: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for path in sorted(chunks_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line)
                        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze PDFs with Azure AI Document Intelligence and chunk them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "pdfs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "docint")
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=250)
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        help="Optional Document Intelligence feature, repeatable.",
    )
    parser.add_argument(
        "--content-format",
        default="markdown",
        help="Passed as output_content_format when supported by the SDK/API.",
    )
    parser.add_argument("--force-analyze", action="store_true")
    parser.add_argument("--force-chunks", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip Azure calls and rebuild chunks.jsonl from existing per-paper files.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    chunks_dir = output_dir / "chunks"
    merged_chunks = output_dir / "chunks.jsonl"
    manifest_path = output_dir / "_manifest.jsonl"

    if args.merge_only:
        merged = merge_chunk_files(chunks_dir, merged_chunks)
        print(f"merged chunks: {merged} -> {merged_chunks}", file=sys.stderr)
        return 0

    load_environment(args.env_file)
    settings = DocumentIntelligenceSettings.from_env()
    client = build_document_intelligence_client(settings)

    records = select_records(load_metadata(args.metadata), set(args.paper_id), args.limit)
    if not records:
        print("no records selected", file=sys.stderr)
        return 0

    ok = 0
    failed = 0
    for index, record in enumerate(records, start=1):
        print(
            f"[{index}/{len(records)}] {record['paper_id']} {record.get('title', '')}",
            file=sys.stderr,
        )
        if process_one(
            record,
            client=client,
            settings=settings,
            pdf_dir=args.pdf_dir,
            raw_dir=raw_dir,
            chunks_dir=chunks_dir,
            manifest_path=manifest_path,
            args=args,
        ):
            ok += 1
        else:
            failed += 1

    merged = merge_chunk_files(chunks_dir, merged_chunks)
    print(
        f"finished: ok={ok} failed={failed} merged_chunks={merged} -> {merged_chunks}",
        file=sys.stderr,
    )
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
