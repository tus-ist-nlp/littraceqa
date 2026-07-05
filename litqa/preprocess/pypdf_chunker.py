"""pypdf を使い、PDF 本文からページ単位で Chunk を抽出する Preprocessor。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

from litqa.contracts import Chunk
from litqa.registry import register

_MIN_PAGE_CHARS = 100
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _split_text(text: str, max_chars: int) -> list[str]:
    """max_chars を超えるテキストを単語境界でできるだけ壊さずに分割する。"""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


@register("preprocessor", "pypdf")
class PyPDFChunker:
    def __init__(self, pdf_dir: str, max_chars_per_chunk: int = 2000):
        self.pdf_dir = Path(pdf_dir)
        self.max_chars_per_chunk = max_chars_per_chunk

    def process(self, paper: dict) -> list[Chunk]:
        paper_id = paper["paper_id"]
        prefix = f"[{paper['venue']} {paper['year']}] {paper['title']}\n"
        metadata_base = {
            "title": paper["title"],
            "venue": paper["venue"],
            "year": paper["year"],
            "authors": paper["authors"],
        }

        chunks = [
            Chunk(
                chunk_id=f"{paper_id}#c0000",
                paper_id=paper_id,
                text=f"{prefix}{paper['abstract']}",
                chunk_type="title_abstract",
                metadata=dict(metadata_base),
            )
        ]

        pdf_path = self.pdf_dir / f"{paper_id}.pdf"
        if not pdf_path.exists():
            return chunks

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            print(f"警告: {paper_id}: PDF の読み込みに失敗しました: {exc}", file=sys.stderr)
            return chunks

        idx = 1
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                print(
                    f"警告: {paper_id}: {page_number} ページ目の抽出に失敗しました: {exc}",
                    file=sys.stderr,
                )
                continue

            page_text = _normalize_whitespace(raw_text)
            if len(page_text) < _MIN_PAGE_CHARS:
                continue

            for part in _split_text(page_text, self.max_chars_per_chunk):
                chunk_metadata = dict(metadata_base)
                chunk_metadata["page"] = page_number
                chunks.append(
                    Chunk(
                        chunk_id=f"{paper_id}#c{idx:04d}",
                        paper_id=paper_id,
                        text=f"{prefix}{part}",
                        chunk_type="text_span",
                        metadata=chunk_metadata,
                    )
                )
                idx += 1

        return chunks
