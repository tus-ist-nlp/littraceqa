"""Turn the content_list.json MinerU wrote into Chunks.

**The PDF conversion does not happen here.** MinerU's dependencies cannot coexist
with this package's, so it runs first in an isolated venv (.venv-mineru) via
scripts/run_mineru.py, and this preprocessor only reads what it produced. See the
comment at the top of requirements-mineru.txt.

    bash scripts/setup_mineru_env.sh
    .venv-mineru/bin/python scripts/run_mineru.py --paths configs/paths/default.yaml --gpus 0,1,2,3

content_list.json holds body text, headings, equations, tables and figures as
blocks, each carrying a page_idx. **That structure is what this can use, and a
plain size-based split of the page cannot:**

* A heading (text_level) becomes the section carried by the body that follows, and
  the body is grouped per section and page.
* Equations stay as LaTeX (``$$...$$``) inline, in reading order. An equation on
  its own is **not** made into a chunk — with no context it cannot serve as
  evidence. A numbered one (``\\tag{N}``) is the exception: gold evidence can point
  at it by equation_id, so it gets its own equation_algorithm chunk with the
  preceding body text attached as context.
* A table's table_body (HTML) becomes Markdown; a figure's caption becomes its text.

MinerU's page_idx is 0-indexed while the convention everywhere else here (gold's
evidence.locator.page) is 1-indexed, so it is converted with +1.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from markdownify import markdownify

from littraceqa.di_pipeline.contracts import Chunk

# Pulls the number off the front of a caption (`Figure 3`, `Table 12a`). It used to
# live with the figure_vlm preprocessor; that was deleted, so it moved to its only
# remaining caller.
_VISIBLE_ID_RE = re.compile(r"(?:Figure|Table|Fig\.?)\s*(\d+[a-zA-Z]?)", re.IGNORECASE)


def _extract_number(caption: str) -> str | None:
    """The figure/table number in a caption (e.g. "3", "12a"); None if there is none."""
    if not caption:
        return None
    match = _VISIBLE_ID_RE.search(caption)
    return match.group(1) if match else None


# MinerU writes a numbered equation as "$$ ... \tag{12} $$".
_EQUATION_TAG_RE = re.compile(r"\\tag\{(\d+[a-zA-Z]?)\}")

# Blocks that are never taken as body text: page numbers, headers and footers are
# nothing but noise to retrieval.
_SKIPPED_TYPES = frozenset({"page_number", "header", "footer", "discarded"})

_FIGURE_TYPES = frozenset({"image", "chart"})


def _extract_equation_id(text: str) -> str | None:
    match = _EQUATION_TAG_RE.search(text)
    if not match:
        return None
    return f"Equation {match.group(1)}"


def _visible_id(caption: str, prefix: str) -> str | None:
    number = _extract_number(caption)
    if not number:
        return None
    return f"{prefix} {number}"


def _join(parts: list[str]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _split_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Group paragraphs up to roughly max_chars.

    **Cutting on paragraph boundaries is what keeps a ``$$...$$`` equation whole.**
    Only when a single paragraph is longer than max_chars is it split, on a word
    boundary, and never if it is an equation.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            chunks.extend(_split_oversized(paragraph, max_chars))
            continue
        if current and current_len + 2 + len(paragraph) > max_chars:
            flush()
        # Count the separator ("\n\n") only once `current` is non-empty. Deciding
        # before flush() would charge two characters against an emptied `current`.
        separator = 2 if current else 0
        current.append(paragraph)
        current_len += len(paragraph) + separator

    flush()
    return chunks


def _split_oversized(paragraph: str, max_chars: int) -> list[str]:
    """Split one over-long paragraph on word boundaries; equations are returned whole."""
    if paragraph.lstrip().startswith("$$"):
        return [paragraph]

    parts: list[str] = []
    current = ""
    for word in paragraph.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


class MinerUChunker:
    def __init__(
        self,
        pdf_dir: str,
        mineru_dir: str | None = None,
        max_chars_per_chunk: int = 2000,
    ):
        self.pdf_dir = Path(pdf_dir)
        # Matches scripts/run_mineru.py's default output location: a sibling of
        # pdf_dir, derived rather than configured (paths are not written into the
        # configuration; configs/paths/*.yaml names pdf_dir and stops there).
        self.mineru_dir = Path(mineru_dir) if mineru_dir else self.pdf_dir.parent / "mineru"
        self.max_chars_per_chunk = max_chars_per_chunk

    def content_list_path(self, paper_id: str) -> Path:
        return self.mineru_dir / paper_id / "auto" / f"{paper_id}_content_list.json"

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

        blocks = self._load_blocks(paper_id)
        if blocks is None:
            return chunks

        chunks.extend(self._build_chunks(blocks, paper_id, prefix, metadata_base))
        return chunks

    def _load_blocks(self, paper_id: str) -> list[dict] | None:
        path = self.content_list_path(paper_id)
        if not path.exists():
            # A paper scripts/run_mineru.py has not converted yet: title/abstract only.
            return None
        try:
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"警告: {paper_id}: content_list.json の読み込みに失敗しました: {exc}", file=sys.stderr)
            return None

    def _build_chunks(
        self, blocks: list[dict], paper_id: str, prefix: str, metadata_base: dict
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        counters = {"text": 0, "table": 0, "figure": 0, "equation": 0}

        # Body text accumulates while section and page hold, then splits at max_chars.
        buffer: list[str] = []
        buffer_page: int | None = None
        section: str | None = None
        last_text: str | None = None  # context to attach to an equation chunk

        def flush() -> None:
            nonlocal buffer, buffer_page
            if not buffer or buffer_page is None:
                buffer = []
                return
            for part in _split_paragraphs(buffer, self.max_chars_per_chunk):
                counters["text"] += 1
                metadata = dict(metadata_base)
                metadata["page"] = buffer_page
                metadata["section"] = section
                chunks.append(
                    Chunk(
                        chunk_id=f"{paper_id}#c{counters['text']:04d}",
                        paper_id=paper_id,
                        text=f"{prefix}{part}",
                        chunk_type="text_span",
                        metadata=metadata,
                    )
                )
            buffer = []
            buffer_page = None

        for block in blocks:
            block_type = block.get("type")
            if block_type in _SKIPPED_TYPES:
                continue

            page = block.get("page_idx", 0) + 1  # MinerU is 0-indexed; gold is 1-indexed

            if block_type == "text":
                text = (block.get("text") or "").strip()
                if not text:
                    continue
                if block.get("text_level"):
                    # The heading is not a chunk of its own; it becomes the section
                    # of the body that follows.
                    flush()
                    section = text
                    continue
                if buffer_page is not None and page != buffer_page:
                    flush()
                buffer.append(text)
                buffer_page = page if buffer_page is None else buffer_page
                last_text = text

            elif block_type == "list":
                items = _join(block.get("list_items") or [])
                if not items:
                    continue
                if buffer_page is not None and page != buffer_page:
                    flush()
                buffer.append(items)
                buffer_page = page if buffer_page is None else buffer_page

            elif block_type == "equation":
                equation = (block.get("text") or "").strip()
                if not equation:
                    continue
                if buffer_page is not None and page != buffer_page:
                    flush()
                buffer.append(equation)
                buffer_page = page if buffer_page is None else buffer_page

                equation_id = _extract_equation_id(equation)
                if equation_id:
                    chunks.append(
                        self._equation_chunk(
                            equation, equation_id, last_text, page,
                            section, paper_id, prefix, metadata_base, counters,
                        )
                    )

            elif block_type == "table":
                chunks.append(self._table_chunk(block, page, section, paper_id, prefix, metadata_base, counters))

            elif block_type in _FIGURE_TYPES:
                figure = self._figure_chunk(block, page, section, paper_id, prefix, metadata_base, counters)
                if figure is not None:
                    chunks.append(figure)

        flush()
        return chunks

    def _equation_chunk(
        self, equation: str, equation_id: str, context: str | None, page: int,
        section: str | None, paper_id: str, prefix: str, metadata_base: dict, counters: dict,
    ) -> Chunk:
        counters["equation"] += 1
        metadata = dict(metadata_base)
        metadata["page"] = page
        metadata["section"] = section
        metadata["equation_id"] = equation_id
        # An equation alone cannot serve as evidence even when retrieved, so the
        # preceding body text comes with it as context.
        body = _join([context or "", equation])
        return Chunk(
            chunk_id=f"{paper_id}#eq{counters['equation']:04d}",
            paper_id=paper_id,
            text=f"{prefix}{body}",
            chunk_type="equation_algorithm",
            metadata=metadata,
        )

    def _table_chunk(
        self, block: dict, page: int, section: str | None,
        paper_id: str, prefix: str, metadata_base: dict, counters: dict,
    ) -> Chunk:
        counters["table"] += 1
        caption = _join(block.get("table_caption") or [])
        body = markdownify(block.get("table_body") or "").strip()
        footnote = _join(block.get("table_footnote") or [])

        metadata = dict(metadata_base)
        metadata["page"] = page
        metadata["section"] = section
        metadata["table_id"] = _visible_id(caption, "Table")
        image_path = self._image_path(paper_id, block.get("img_path"))
        if image_path:
            metadata["image_path"] = image_path

        return Chunk(
            chunk_id=f"{paper_id}#tab{counters['table']:04d}",
            paper_id=paper_id,
            text=f"{prefix}{_join([caption, body, footnote])}",
            chunk_type="table",
            metadata=metadata,
        )

    def _figure_chunk(
        self, block: dict, page: int, section: str | None,
        paper_id: str, prefix: str, metadata_base: dict, counters: dict,
    ) -> Chunk | None:
        kind = "chart" if block.get("type") == "chart" else "image"
        caption = _join(block.get(f"{kind}_caption") or [])
        footnote = _join(block.get(f"{kind}_footnote") or [])
        # For a chart, MinerU sometimes transcribes the contents as text.
        content = (block.get("content") or "").strip()
        body = _join([caption, content, footnote])
        if not body:
            # A figure with neither caption nor contents is not searchable at all
            # (decorative images and the like).
            return None

        counters["figure"] += 1
        metadata = dict(metadata_base)
        metadata["page"] = page
        metadata["section"] = section
        metadata["figure_id"] = _visible_id(caption, "Figure")
        image_path = self._image_path(paper_id, block.get("img_path"))
        if image_path:
            metadata["image_path"] = image_path

        return Chunk(
            chunk_id=f"{paper_id}#fig{counters['figure']:04d}",
            paper_id=paper_id,
            text=f"{prefix}{body}",
            chunk_type="figure",
            metadata=metadata,
        )

    def _image_path(self, paper_id: str, img_path: str | None) -> str | None:
        """Resolve a content_list.json relative path to an absolute one.

        It reaches the reading team through chunks.jsonl:
        `ChunkStore.figures()` hands back the table/figure chunks whose
        `metadata["image_path"]` still exists on disk.
        """
        if not img_path:
            return None
        return str(self.mineru_dir / paper_id / "auto" / img_path)
