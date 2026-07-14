"""Convert existing MinerU content-list artifacts into common search chunks."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from litqa.contracts import Chunk
from litqa.registry import register


_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FIGURE_ID_RE = re.compile(r"\b(?:Figure|Fig\.?)\s*([A-Z]?\d+(?:[.-]\d+)?[A-Z]?)", re.IGNORECASE)
_TABLE_ID_RE = re.compile(r"\bTable\s*([A-Z]?\d+(?:[.-]\d+)?[A-Z]?)", re.IGNORECASE)
_EQUATION_ID_RE = re.compile(r"\\tag\{([^{}]+)\}")
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_VISUAL_TYPES = {"image", "chart", "table", "equation"}


class MinerUDataError(ValueError):
    """Raised when a requested MinerU artifact is missing, unsafe, or malformed."""


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from a small HTML fragment without extra dependencies."""

    _BREAK_TAGS = {"br", "p", "tr", "li", "table", "thead", "tbody", "tfoot"}
    _CELL_TAGS = {"td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BREAK_TAGS:
            self.parts.append("\n")
        elif tag.lower() in self._CELL_TAGS:
            self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        lines = []
        for raw_line in "".join(self.parts).splitlines():
            line = _WHITESPACE_RE.sub(" ", raw_line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)


def _html_to_text(value: str) -> str:
    if not value.strip():
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return _WHITESPACE_RE.sub(" ", value).strip()
    return parser.text()


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            values.append(item.strip())
    return values


def _v2_inline_text(value: Any) -> str:
    """Flatten MinerU v2 inline content without inventing missing text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(
            text for item in value if (text := _v2_inline_text(item))
        ).strip()
    if not isinstance(value, dict):
        return ""

    content = value.get("content")
    if isinstance(content, str) and content.strip():
        text = content.strip()
        if value.get("type") == "equation_inline":
            return f"${text}$"
        return text
    for key in (
        "item_content",
        "title_content",
        "paragraph_content",
        "page_footnote_content",
        "page_footer_content",
        "page_number_content",
    ):
        if key in value:
            return _v2_inline_text(value[key])
    return ""


def _v2_text_list(value: Any) -> list[str]:
    text = _v2_inline_text(value)
    return [text] if text else []


def _v2_image_path(content: dict[str, Any]) -> str:
    image_source = content.get("image_source")
    if not isinstance(image_source, dict):
        return ""
    path = image_source.get("path")
    return path.strip() if isinstance(path, str) else ""


def _join_nonempty(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text deterministically near whitespace while keeping every character."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        newline_at = remaining.rfind("\n", 0, max_chars + 1)
        split_at = max(split_at, newline_at)
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _visible_ids(captions: list[str], kind: str) -> list[str]:
    text = " ".join(captions)
    pattern = _FIGURE_ID_RE if kind == "figure" else _TABLE_ID_RE
    label = "Figure" if kind == "figure" else "Table"
    visible_ids = []
    for match in pattern.finditer(text):
        visible_id = f"{label} {match.group(1)}"
        if visible_id not in visible_ids:
            visible_ids.append(visible_id)
    return visible_ids


def _equation_id(text: str) -> str | None:
    match = _EQUATION_ID_RE.search(text)
    if not match:
        return None
    value = match.group(1).strip()
    return f"Equation {value}" if value else None


@register("preprocessor", "mineru_v2")
@register("preprocessor", "mineru")
class MinerUChunker:
    """Read one precomputed MinerU paper and return deterministic common chunks."""

    def __init__(
        self,
        mineru_root: str,
        max_chars_per_chunk: int = 2000,
        content_version: str = "v1",
    ):
        if max_chars_per_chunk <= 0:
            raise ValueError("max_chars_per_chunk must be positive")
        if content_version not in {"v1", "v2"}:
            raise ValueError("content_version must be 'v1' or 'v2'")
        self.mineru_root = Path(mineru_root).expanduser().resolve()
        self.max_chars_per_chunk = max_chars_per_chunk
        self.content_version = content_version

    def process(self, paper: dict) -> list[Chunk]:
        paper_id = self._paper_id(paper)
        content_path = self._content_path(paper_id)
        auto_dir = content_path.parent
        blocks = self._load_content_list(content_path, self.content_version)

        metadata_base = self._metadata_base(paper)
        metadata_base["mineru_content_version"] = self.content_version
        prefix = self._prefix(paper)
        chunks = [
            Chunk(
                chunk_id=f"{paper_id}#c0000",
                paper_id=paper_id,
                text=_join_nonempty(prefix, str(paper.get("abstract") or "")),
                chunk_type="title_abstract",
                metadata=dict(metadata_base),
            )
        ]
        counters = {"text_span": 0, "figure": 0, "table": 0, "equation_algorithm": 0}

        warnings: list[dict[str, Any]] = []
        for block_index, block in enumerate(blocks):
            built = self._build_chunks(
                block=block,
                block_index=block_index,
                paper_id=paper_id,
                auto_dir=auto_dir,
                prefix=prefix,
                metadata_base=metadata_base,
                counters=counters,
            )
            chunks.extend(built)
            if not built and block["type"] in _VISUAL_TYPES:
                warnings.append(
                    {
                        "block_index": block_index,
                        "mineru_type": block["type"],
                        "reason": "no usable text or image reference",
                    }
                )
        if warnings:
            chunks[0].metadata["preprocess_status"] = "partial"
            chunks[0].metadata["mineru_warnings"] = warnings
        return chunks

    def input_paths(self, paper: dict) -> list[Path]:
        """Return the selected precomputed content list without reading it."""
        return [self._content_path(self._paper_id(paper))]

    def _paper_id(self, paper: dict) -> str:
        paper_id = str(paper.get("paper_id") or "").strip()
        if not _PAPER_ID_RE.fullmatch(paper_id) or paper_id in {".", ".."}:
            raise MinerUDataError(f"unsafe or missing paper_id: {paper_id!r}")
        return paper_id

    def _auto_dir(self, paper_id: str) -> Path:
        paper_dir = (self.mineru_root / paper_id).resolve()
        try:
            paper_dir.relative_to(self.mineru_root)
        except ValueError as exc:
            raise MinerUDataError(f"paper path escapes MinerU root: {paper_id}") from exc
        auto_dir = paper_dir / "auto"
        if not auto_dir.is_dir():
            raise MinerUDataError(f"MinerU auto directory is missing: {auto_dir}")
        return auto_dir

    def _content_path(self, paper_id: str) -> Path:
        auto_dir = self._auto_dir(paper_id)
        suffix = (
            "_content_list_v2.json"
            if self.content_version == "v2"
            else "_content_list.json"
        )
        content_path = (auto_dir / f"{paper_id}{suffix}").resolve()
        try:
            content_path.relative_to(auto_dir)
        except ValueError as exc:
            raise MinerUDataError(
                f"MinerU content list escapes the paper auto directory: {content_path}"
            ) from exc
        if not content_path.is_file():
            raise MinerUDataError(f"MinerU content list is missing: {content_path}")
        if content_path.stat().st_size == 0:
            raise MinerUDataError(f"MinerU content list is empty: {content_path}")
        return content_path

    @classmethod
    def _load_content_list(
        cls, path: Path, content_version: str
    ) -> list[dict[str, Any]]:
        if not path.is_file():
            raise MinerUDataError(f"MinerU content list is missing: {path}")
        if path.stat().st_size == 0:
            raise MinerUDataError(f"MinerU content list is empty: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MinerUDataError(f"MinerU content list is not valid JSON: {path}") from exc
        if not isinstance(payload, list):
            raise MinerUDataError(f"MinerU content list must be a JSON array: {path}")
        if not payload:
            raise MinerUDataError(f"MinerU content list has no blocks: {path}")
        if content_version == "v2":
            return cls._normalize_v2_content_list(payload, path)
        for index, block in enumerate(payload):
            if not isinstance(block, dict):
                raise MinerUDataError(f"MinerU block {index} must be a JSON object: {path}")
            if not isinstance(block.get("type"), str):
                raise MinerUDataError(f"MinerU block {index} has no valid type: {path}")
        return payload

    @classmethod
    def _normalize_v2_content_list(
        cls, pages: list[Any], path: Path
    ) -> list[dict[str, Any]]:
        """Flatten page-grouped MinerU v2 blocks into the internal v1-like shape."""
        normalized: list[dict[str, Any]] = []
        for page_index, page in enumerate(pages):
            if not isinstance(page, list):
                raise MinerUDataError(
                    f"MinerU v2 page {page_index} must be a JSON array: {path}"
                )
            for page_block_index, block in enumerate(page):
                if not isinstance(block, dict):
                    raise MinerUDataError(
                        f"MinerU v2 block {page_index}:{page_block_index} "
                        f"must be a JSON object: {path}"
                    )
                mineru_type = block.get("type")
                content = block.get("content")
                if not isinstance(mineru_type, str) or not isinstance(content, dict):
                    raise MinerUDataError(
                        f"MinerU v2 block {page_index}:{page_block_index} "
                        f"has no valid type or content: {path}"
                    )
                normalized.append(
                    cls._normalize_v2_block(
                        mineru_type=mineru_type,
                        content=content,
                        bbox=block.get("bbox"),
                        page_index=page_index,
                        page_block_index=page_block_index,
                    )
                )
        if not normalized:
            raise MinerUDataError(f"MinerU v2 content list has no blocks: {path}")
        return normalized

    @staticmethod
    def _normalize_v2_block(
        mineru_type: str,
        content: dict[str, Any],
        bbox: Any,
        page_index: int,
        page_block_index: int,
    ) -> dict[str, Any]:
        block: dict[str, Any] = {
            "type": mineru_type,
            "source_mineru_type": mineru_type,
            "page_idx": page_index,
            "mineru_page_block_index": page_block_index,
        }
        if isinstance(bbox, list):
            block["bbox"] = bbox

        if mineru_type == "title":
            block.update(
                type="text",
                text=_v2_inline_text(content.get("title_content")),
                text_level=content.get("level"),
                sub_type="title",
            )
        elif mineru_type == "paragraph":
            block.update(
                type="text",
                text=_v2_inline_text(content.get("paragraph_content")),
                sub_type="paragraph",
            )
        elif mineru_type == "page_footnote":
            block.update(
                type="page_footnote",
                text=_v2_inline_text(content.get("page_footnote_content")),
            )
        elif mineru_type == "page_footer":
            block.update(type="footer")
        elif mineru_type == "page_number":
            block.update(type="page_number")
        elif mineru_type == "list":
            items = content.get("list_items")
            list_items = []
            if isinstance(items, list):
                list_items = [
                    text
                    for item in items
                    if (text := _v2_inline_text(item))
                ]
            block.update(
                type="list",
                list_items=list_items,
                sub_type=content.get("list_type"),
            )
        elif mineru_type in {"image", "chart"}:
            caption_key = "image_caption" if mineru_type == "image" else "chart_caption"
            footnote_key = "image_footnote" if mineru_type == "image" else "chart_footnote"
            block.update(
                img_path=_v2_image_path(content),
                content=str(content.get("content") or ""),
            )
            block[caption_key] = _v2_text_list(content.get(caption_key))
            block[footnote_key] = _v2_text_list(content.get(footnote_key))
        elif mineru_type == "table":
            block.update(
                img_path=_v2_image_path(content),
                table_caption=_v2_text_list(content.get("table_caption")),
                table_footnote=_v2_text_list(content.get("table_footnote")),
                table_body=str(content.get("html") or ""),
                sub_type=content.get("table_type"),
                table_nest_level=content.get("table_nest_level"),
            )
        elif mineru_type in {"equation", "equation_interline"}:
            block.update(
                type="equation",
                text=str(content.get("math_content") or ""),
                text_format=content.get("math_type"),
                img_path=_v2_image_path(content),
            )
        return block

    @staticmethod
    def _metadata_base(paper: dict) -> dict[str, Any]:
        metadata: dict[str, Any] = {"preprocess_source": "mineru"}
        for key in ("title", "venue", "year", "authors"):
            if key in paper and paper[key] is not None:
                metadata[key] = paper[key]
        return metadata

    @staticmethod
    def _prefix(paper: dict) -> str:
        venue = str(paper.get("venue") or "").strip()
        year = str(paper.get("year") or "").strip()
        title = str(paper.get("title") or "").strip()
        context = " ".join(part for part in (venue, year) if part)
        label = f"[{context}]" if context else ""
        return " ".join(part for part in (label, title) if part)

    def _build_chunks(
        self,
        block: dict[str, Any],
        block_index: int,
        paper_id: str,
        auto_dir: Path,
        prefix: str,
        metadata_base: dict[str, Any],
        counters: dict[str, int],
    ) -> list[Chunk]:
        mineru_type = str(block["type"])
        if mineru_type in {"footer", "page_number"}:
            return []

        if mineru_type in {"text", "page_footnote"}:
            chunk_type = "text_span"
            body = str(block.get("text") or "").strip()
            captions: list[str] = []
            footnotes: list[str] = []
        elif mineru_type == "list":
            chunk_type = "text_span"
            body = "\n".join(_as_text_list(block.get("list_items")))
            captions = []
            footnotes = []
        elif mineru_type in {"image", "chart"}:
            chunk_type = "figure"
            caption_key = "image_caption" if mineru_type == "image" else "chart_caption"
            footnote_key = "image_footnote" if mineru_type == "image" else "chart_footnote"
            captions = _as_text_list(block.get(caption_key))
            footnotes = _as_text_list(block.get(footnote_key))
            body = _join_nonempty(
                str(block.get("content") or ""),
                "\n".join(captions),
                "\n".join(footnotes),
            )
        elif mineru_type == "table":
            chunk_type = "table"
            captions = _as_text_list(block.get("table_caption"))
            footnotes = _as_text_list(block.get("table_footnote"))
            body = _join_nonempty(
                "\n".join(captions),
                _html_to_text(str(block.get("table_body") or "")),
                "\n".join(footnotes),
            )
        elif mineru_type == "equation":
            chunk_type = "equation_algorithm"
            captions = []
            footnotes = []
            body = str(block.get("text") or "").strip()
        else:
            return []

        image_path = self._resolve_image_path(auto_dir, block.get("img_path"))
        if not body and image_path is None:
            return []

        metadata = dict(metadata_base)
        metadata["mineru_type"] = str(block.get("source_mineru_type") or mineru_type)
        metadata["source_block_index"] = block_index
        page_idx = block.get("page_idx")
        if isinstance(page_idx, int) and not isinstance(page_idx, bool) and page_idx >= 0:
            metadata["page"] = page_idx + 1
            metadata["mineru_page_idx"] = page_idx
        bbox = block.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            metadata["bbox"] = list(bbox)
        for key in (
            "text_level",
            "sub_type",
            "text_format",
            "table_nest_level",
            "mineru_page_block_index",
        ):
            if block.get(key) is not None:
                metadata[key] = block[key]
        if captions:
            metadata["caption"] = list(captions)
        if footnotes:
            metadata["footnote"] = list(footnotes)

        if chunk_type == "figure":
            visible_ids = _visible_ids(captions, "figure")
            if visible_ids:
                metadata["figure_id"] = visible_ids[0]
            if len(visible_ids) > 1:
                metadata["figure_ids"] = visible_ids
        elif chunk_type == "table":
            visible_ids = _visible_ids(captions, "table")
            if visible_ids:
                metadata["table_id"] = visible_ids[0]
            if len(visible_ids) > 1:
                metadata["table_ids"] = visible_ids
        elif chunk_type == "equation_algorithm":
            visible_id = _equation_id(body)
            if visible_id:
                metadata["equation_id"] = visible_id

        if image_path is not None:
            metadata["image_path"] = str(image_path)
        elif isinstance(block.get("img_path"), str) and block["img_path"].strip():
            metadata["image_reference_missing"] = True
            metadata["preprocess_status"] = "partial"

        tag = {
            "text_span": "c",
            "figure": "fig",
            "table": "tab",
            "equation_algorithm": "eq",
        }[chunk_type]
        pieces = _split_text(body, self.max_chars_per_chunk) if body else [""]
        chunks: list[Chunk] = []
        for piece_index, piece in enumerate(pieces):
            counters[chunk_type] += 1
            chunk_id = f"{paper_id}#{tag}{counters[chunk_type]:04d}"
            piece_metadata = dict(metadata)
            if piece_index > 0:
                piece_metadata.pop("image_path", None)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=_join_nonempty(prefix, piece),
                    chunk_type=chunk_type,
                    metadata=piece_metadata,
                )
            )
        return chunks

    @staticmethod
    def _resolve_image_path(auto_dir: Path, raw_path: Any) -> Path | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        candidate = (auto_dir / raw_path).resolve()
        try:
            candidate.relative_to(auto_dir.resolve())
        except ValueError as exc:
            raise MinerUDataError(
                f"image path escapes the paper auto directory: {raw_path}"
            ) from exc
        if not candidate.is_file() or candidate.stat().st_size == 0:
            return None
        return candidate
