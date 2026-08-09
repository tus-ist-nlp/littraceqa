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


@dataclass(frozen=True)
class PaperTable:
    paper_id: str
    table_id: str | None
    caption: str
    rows: tuple[tuple[str, ...], ...]
    text: str


class PaperTableSource(Protocol):
    def tables(self, paper_id: str) -> tuple[PaperTable, ...]: ...


class MinerUPaperTableSource:
    """Read table blocks from existing MinerU content lists."""

    def __init__(self, mineru_dir: str | Path, cache_size: int = 32) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.mineru_dir = Path(mineru_dir)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, tuple[PaperTable, ...]] = OrderedDict()

    def tables(self, paper_id: str) -> tuple[PaperTable, ...]:
        _validate_paper_id(paper_id)
        cached = self._cache.get(paper_id)
        if cached is not None:
            self._cache.move_to_end(paper_id)
            return cached

        tables = self._read_tables(paper_id)
        if self.cache_size:
            self._cache[paper_id] = tables
            self._cache.move_to_end(paper_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return tables

    def _read_tables(self, paper_id: str) -> tuple[PaperTable, ...]:
        path = self.mineru_dir / paper_id / "auto" / f"{paper_id}_content_list.json"
        try:
            with path.open(encoding="utf-8") as stream:
                blocks = json.load(stream)
        except (OSError, UnicodeError, ValueError):
            return ()
        if not isinstance(blocks, list):
            return ()

        tables: list[PaperTable] = []
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "table":
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
        return tuple(tables)


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self.all_text: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "tr":
            self._finish_row()
            self._row = []
        elif tag in {"td", "th"}:
            if self._row is None:
                self._row = []
            self._finish_cell()
            self._cell = []

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
        self._row.append(_normalize(" ".join(self._cell)))
        self._cell = None

    def _finish_row(self) -> None:
        self._finish_cell()
        if self._row:
            self.rows.append(tuple(self._row))
        self._row = None


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
