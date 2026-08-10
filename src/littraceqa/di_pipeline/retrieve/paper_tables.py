"""Read paper tables without coupling selection logic to MinerU files."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol


_TABLE_ID_RE = re.compile(r"\btable\s+([A-Z]?\d+[A-Z]?|[IVXLCDM]+)\b", re.IGNORECASE)
_MAX_TABLE_SPAN = 128


@dataclass(frozen=True)
class PaperTable:
    paper_id: str
    table_id: str | None
    caption: str
    rows: tuple[tuple[str, ...], ...]
    text: str


@dataclass(frozen=True)
class PaperEvidenceDocument:
    """Text evidence read from one MinerU content list."""

    paper_id: str
    title: str
    text_blocks: tuple[str, ...]
    tables: tuple[PaperTable, ...]
    reference_entries: tuple[str, ...] = ()


class PaperTableSource(Protocol):
    def tables(self, paper_id: str) -> tuple[PaperTable, ...]: ...


class PaperDocumentSource(Protocol):
    def document(self, paper_id: str) -> PaperEvidenceDocument: ...


class PaperEvidenceSource(PaperTableSource, PaperDocumentSource, Protocol):
    pass


class MinerUPaperTableSource:
    """Read text and table evidence from existing MinerU content lists."""

    def __init__(self, mineru_dir: str | Path, cache_size: int = 32) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.mineru_dir = Path(mineru_dir)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, PaperEvidenceDocument] = OrderedDict()

    def tables(self, paper_id: str) -> tuple[PaperTable, ...]:
        return self.document(paper_id).tables

    def document(self, paper_id: str) -> PaperEvidenceDocument:
        _validate_paper_id(paper_id)
        cached = self._cache.get(paper_id)
        if cached is not None:
            self._cache.move_to_end(paper_id)
            return cached

        document = self._read_document(paper_id)
        if self.cache_size:
            self._cache[paper_id] = document
            self._cache.move_to_end(paper_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return document

    def _read_document(self, paper_id: str) -> PaperEvidenceDocument:
        path = self.mineru_dir / paper_id / "auto" / f"{paper_id}_content_list.json"
        try:
            with path.open(encoding="utf-8") as stream:
                blocks = json.load(stream)
        except (OSError, UnicodeError, ValueError):
            return _empty_document(paper_id)
        if not isinstance(blocks, list):
            return _empty_document(paper_id)

        title = ""
        text_blocks: list[str] = []
        tables: list[PaperTable] = []
        reference_entries: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = _join_text(block.get("text"))
                if text:
                    text_blocks.append(text)
                    if not title and block.get("text_level") == 1:
                        title = text
                continue
            if block.get("type") == "list" and block.get("sub_type") == "ref_text":
                items = block.get("list_items")
                if isinstance(items, list):
                    reference_entries.extend(
                        text for item in items if (text := _join_text(item))
                    )
                continue
            if block.get("type") != "table":
                continue
            caption = _join_text(block.get("table_caption"))
            footnote = _join_text(block.get("table_footnote"))
            body = block.get("table_body")
            body = body if isinstance(body, str) else ""
            rows, body_text = _parse_table_body(body)
            text = "\n".join(part for part in (caption, body_text, footnote) if part)
            tables.append(
                PaperTable(
                    paper_id=paper_id,
                    table_id=_table_id(caption),
                    caption=caption,
                    rows=rows,
                    text=text,
                )
            )
        return PaperEvidenceDocument(
            paper_id=paper_id,
            title=title,
            text_blocks=tuple(text_blocks),
            tables=tuple(tables),
            reference_entries=tuple(reference_entries),
        )


def _empty_document(paper_id: str) -> PaperEvidenceDocument:
    return PaperEvidenceDocument(paper_id=paper_id, title="", text_blocks=(), tables=())


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self.all_text: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._column = 0
        self._rowspan = 1
        self._colspan = 1
        self._spans: dict[int, tuple[str, int]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._finish_row()
            self._row = []
            self._column = 0
        elif tag in {"td", "th"}:
            if self._row is None:
                self._row = []
            self._finish_cell()
            self._append_spans()
            self._cell = []
            attributes = dict(attrs)
            self._rowspan = _positive_span(attributes.get("rowspan"))
            self._colspan = _positive_span(attributes.get("colspan"))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        self.all_text.append(data)
        if self._cell is not None:
            self._cell.append(data)

    def close(self) -> None:
        super().close()
        self._finish_cell()
        self._finish_row()

    def _finish_cell(self) -> None:
        if self._cell is None:
            return
        if self._row is None:
            self._row = []
        value = _normalize(" ".join(self._cell))
        for _ in range(self._colspan):
            self._row.append(value)
            if self._rowspan > 1:
                self._spans[self._column] = (value, self._rowspan - 1)
            self._column += 1
        self._cell = None
        self._rowspan = 1
        self._colspan = 1

    def _finish_row(self) -> None:
        self._finish_cell()
        self._append_spans()
        if self._row:
            self.rows.append(tuple(self._row))
        self._row = None

    def _append_spans(self) -> None:
        if self._row is None:
            return
        while self._column in self._spans:
            value, remaining = self._spans[self._column]
            self._row.append(value)
            if remaining == 1:
                del self._spans[self._column]
            else:
                self._spans[self._column] = (value, remaining - 1)
            self._column += 1


def _positive_span(value: str | None) -> int:
    try:
        return min(_MAX_TABLE_SPAN, max(1, int(value or 1)))
    except ValueError:
        return 1


def _parse_table_body(body: str) -> tuple[tuple[tuple[str, ...], ...], str]:
    parser = _TableHTMLParser()
    parser.feed(body)
    parser.close()
    rows = tuple(parser.rows)
    if rows:
        text = "\n".join(" | ".join(row) for row in rows)
    else:
        text = _normalize(" ".join(parser.all_text)) or _normalize(body)
    return rows, text


def _join_text(value: object) -> str:
    if isinstance(value, str):
        return _normalize(value)
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _join_text(item)))
    return ""


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _table_id(caption: str) -> str | None:
    match = _TABLE_ID_RE.search(caption)
    return f"Table {match.group(1)}" if match else None


def _validate_paper_id(paper_id: str) -> None:
    if (
        not isinstance(paper_id, str)
        or not paper_id
        or paper_id in {".", ".."}
        or "/" in paper_id
        or "\\" in paper_id
    ):
        raise ValueError("paper_id must be a single path component")
