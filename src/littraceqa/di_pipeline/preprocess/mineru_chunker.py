"""MinerU が書き出した content_list.json を読んで Chunk 化する Preprocessor。

PDF の変換自体はここでは行わない。MinerU は本体と依存が両立しないため
隔離 venv (.venv-mineru) の scripts/run_mineru.py で先に走らせておき、
本 Preprocessor はその成果物（content_list.json）を読むだけにする。
詳細は requirements-mineru.txt の先頭コメントを参照。

    bash scripts/setup_mineru_env.sh
    .venv-mineru/bin/python scripts/run_mineru.py --paths configs/paths/default.yaml --gpus 0,1,2,3

content_list.json は本文・見出し・数式・表・図をブロック単位で持ち、各ブロックに
page_idx が付く。ページ内をサイズで機械的に割るだけの単純な抽出と違い、以下ができる。

* 見出し(text_level)を節として引き継ぎ、本文を節・ページ単位でまとめる。
* 数式は LaTeX(``$$...$$``)のまま本文の読み順に埋め込む。数式だけのチャンクは
  文脈が無く根拠にならないため作らない。ただし ``\\tag{N}`` が付いた採番済みの
  数式は evidence の equation_id で参照されうるので、直前の本文を文脈として
  付けた equation_algorithm チャンクを別途立てる。
* 表は table_body(HTML)を Markdown に、図は caption を本文にする。

MinerU の page_idx は 0-indexed だが、littraceqa.di_pipeline の既存規約(gold の
evidence.locator.page)は 1-indexed なので +1 して変換する。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from markdownify import markdownify

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.preprocess.figure_vlm import _extract_number
from littraceqa.di_pipeline.registry import register

# MinerU は採番済み数式を "$$ ... \tag{12} $$" の形で出す。
_EQUATION_TAG_RE = re.compile(r"\\tag\{(\d+[a-zA-Z]?)\}")

# 本文として拾わないブロック。ページ番号・ヘッダ・フッタは検索の雑音にしかならない。
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


def _page_number(block: dict) -> int | None:
    """Convert a valid zero-based MinerU page index without guessing one."""
    page_index = block.get("page_idx")
    if isinstance(page_index, bool) or not isinstance(page_index, int):
        return None
    if page_index < 0:
        return None
    return page_index + 1


def _split_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """段落を max_chars を目安にまとめる。

    段落の境界で切るので ``$$...$$`` の数式が途中で割れない。1段落だけで
    max_chars を超える場合のみ、数式でなければ単語境界で割る。
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
        # 区切り("\n\n")の分は current が空でなくなってから数える。flush() の前に
        # 決めてしまうと、空になった current に対しても2文字を足してしまう。
        separator = 2 if current else 0
        current.append(paragraph)
        current_len += len(paragraph) + separator

    flush()
    return chunks


def _split_oversized(paragraph: str, max_chars: int) -> list[str]:
    """max_chars を超える1段落を単語境界で割る。数式は割らずにそのまま返す。"""
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


@register("preprocessor", "mineru")
class MinerUChunker:
    checkpoint_dependencies = (markdownify, _extract_number)

    def __init__(
        self,
        pdf_dir: str,
        mineru_dir: str | None = None,
        max_chars_per_chunk: int = 2000,
        strict: bool = False,
    ):
        self.pdf_dir = Path(pdf_dir)
        # scripts/run_mineru.py の既定の出力先と合わせる。pdf_dir と衝突しない
        # 兄弟ディレクトリに自動導出する(process_style yaml にはパスを書かない方針)。
        self.mineru_dir = Path(mineru_dir) if mineru_dir else self.pdf_dir.parent / "mineru"
        self.max_chars_per_chunk = max_chars_per_chunk
        self.strict = strict

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
            if self.strict:
                raise FileNotFoundError(f"MinerU content list is missing: {path}")
            return None
        try:
            with path.open(encoding="utf-8") as f:
                blocks = json.load(f)
            if not isinstance(blocks, list) or not blocks:
                raise ValueError("MinerU content list must be a non-empty JSON array")
            if not all(isinstance(block, dict) for block in blocks):
                raise ValueError("MinerU content list contains a non-object block")
            return blocks
        except Exception as exc:
            if self.strict:
                raise ValueError(f"Failed to load MinerU content list: {path}") from exc
            print(f"警告: {paper_id}: content_list.json の読み込みに失敗しました: {exc}", file=sys.stderr)
            return None

    def _build_chunks(
        self, blocks: list[dict], paper_id: str, prefix: str, metadata_base: dict
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        counters = {"text": 0, "table": 0, "figure": 0, "equation": 0}

        # 本文は「同じ節・同じページ」のあいだ溜めてから max_chars で割る。
        buffer: list[str] = []
        buffer_page: int | None = None
        section: str | None = None
        last_text: str | None = None  # 数式チャンクに付ける文脈

        def flush() -> None:
            nonlocal buffer, buffer_page
            if not buffer:
                buffer = []
                return
            for part in _split_paragraphs(buffer, self.max_chars_per_chunk):
                counters["text"] += 1
                metadata = dict(metadata_base)
                if buffer_page is not None:
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

            page = _page_number(block)

            if block_type == "text":
                text = (block.get("text") or "").strip()
                if not text:
                    continue
                if block.get("text_level"):
                    # 見出し自体はチャンクにせず、以降の本文の section として持つ。
                    flush()
                    section = text
                    continue
                if buffer and page != buffer_page:
                    flush()
                buffer.append(text)
                if len(buffer) == 1:
                    buffer_page = page
                last_text = text

            elif block_type == "list":
                items = _join(block.get("list_items") or [])
                if not items:
                    continue
                if buffer and page != buffer_page:
                    flush()
                buffer.append(items)
                if len(buffer) == 1:
                    buffer_page = page

            elif block_type == "equation":
                equation = (block.get("text") or "").strip()
                if not equation:
                    continue
                if buffer and page != buffer_page:
                    flush()
                buffer.append(equation)
                if len(buffer) == 1:
                    buffer_page = page

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
        self, equation: str, equation_id: str, context: str | None, page: int | None,
        section: str | None, paper_id: str, prefix: str, metadata_base: dict, counters: dict,
    ) -> Chunk:
        counters["equation"] += 1
        metadata = dict(metadata_base)
        if page is not None:
            metadata["page"] = page
        metadata["section"] = section
        metadata["equation_id"] = equation_id
        # 数式単独では検索でヒットしても根拠にならないので直前の本文を文脈として付ける。
        body = _join([context or "", equation])
        return Chunk(
            chunk_id=f"{paper_id}#eq{counters['equation']:04d}",
            paper_id=paper_id,
            text=f"{prefix}{body}",
            chunk_type="equation_algorithm",
            metadata=metadata,
        )

    def _table_chunk(
        self, block: dict, page: int | None, section: str | None,
        paper_id: str, prefix: str, metadata_base: dict, counters: dict,
    ) -> Chunk:
        counters["table"] += 1
        caption = _join(block.get("table_caption") or [])
        body = markdownify(block.get("table_body") or "").strip()
        footnote = _join(block.get("table_footnote") or [])

        metadata = dict(metadata_base)
        if page is not None:
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
        self, block: dict, page: int | None, section: str | None,
        paper_id: str, prefix: str, metadata_base: dict, counters: dict,
    ) -> Chunk | None:
        kind = "chart" if block.get("type") == "chart" else "image"
        caption = _join(block.get(f"{kind}_caption") or [])
        footnote = _join(block.get(f"{kind}_footnote") or [])
        # chart は MinerU が中身をテキスト化してくれることがある。
        content = (block.get("content") or "").strip()
        body = _join([caption, content, footnote])
        image_path = self._image_path(paper_id, block.get("img_path"))
        if not body and not image_path:
            return None

        counters["figure"] += 1
        metadata = dict(metadata_base)
        if page is not None:
            metadata["page"] = page
        metadata["section"] = section
        metadata["figure_id"] = _visible_id(caption, "Figure")
        if image_path:
            metadata["image_path"] = image_path
        if not body:
            # Keep image-only figures for vision retrieval without inventing text.
            metadata["image_only"] = True

        return Chunk(
            chunk_id=f"{paper_id}#fig{counters['figure']:04d}",
            paper_id=paper_id,
            text=f"{prefix}{body}",
            chunk_type="figure",
            metadata=metadata,
        )

    def _image_path(self, paper_id: str, img_path: str | None) -> str | None:
        """Resolve an existing MinerU image without inventing a locator."""
        if not img_path:
            return None
        auto_root = (self.mineru_dir / paper_id / "auto").resolve()
        path = (auto_root / img_path).resolve()
        try:
            path.relative_to(auto_root)
        except ValueError:
            return None
        return str(path) if path.is_file() else None
