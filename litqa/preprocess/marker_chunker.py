"""Marker(marker-pdf) で PDF を構造保持したまま変換し、ブロック種別ごとに Chunk 化する

Preprocessor。ページ単位の平坦テキスト抽出と違い、図・表・数式をブロック単位のまま
(キャプション統合済み)保持できるため、根拠の質(特に数式・図・表)の向上を狙う。

Marker の ChunkOutput.blocks の page は 0-indexed だが、litqa の既存規約
(gold の evidence.locator.page)は1-indexedなので +1 して変換する。
"""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path
from typing import Any, Callable

from bs4 import BeautifulSoup
from markdownify import markdownify

from litqa.contracts import Chunk
from litqa.preprocess.figure_vlm import _extract_number
from litqa.registry import register

_EQUATION_NUMBER_RE = re.compile(r"\((\d+)\)\s*$")

_FIGURE_BLOCK_TYPES = {"FigureGroup", "PictureGroup"}
_TABLE_BLOCK_TYPES = {"TableGroup"}
_EQUATION_BLOCK_TYPES = {"Equation"}
_TEXT_BLOCK_TYPES = {
    "Text",
    "SectionHeader",
    "TextInlineMath",
    "ListGroup",
    "ListItem",
    "Code",
    "Footnote",
    "ComplexRegion",
    "Caption",
}


def _extract_equation_number(text: str) -> str | None:
    match = _EQUATION_NUMBER_RE.search(text)
    if not match:
        return None
    return f"Equation {match.group(1)}"


@register("preprocessor", "marker")
class MarkerChunker:
    def __init__(
        self,
        pdf_dir: str,
        force_ocr: bool = True,
        use_llm: bool = False,
        image_dir: str | None = None,
        converter: Callable[[str], Any] | None = None,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.force_ocr = force_ocr
        self.use_llm = use_llm
        # 画像自体の保存先。指定がなければ pdf_dir と衝突しない兄弟ディレクトリに
        # 自動導出する(figure_vlm.py と同じ方針。process_style yaml には書かない)。
        self.image_dir = Path(image_dir) if image_dir else self.pdf_dir.parent / "marker_images"
        # PdfConverter は重いため、注入されなければ初回 process() 呼び出し時まで
        # 実体を構築しない(遅延ロード)。テストでは fake を注入して重い依存を避ける。
        self._converter = converter

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

        converter = self._get_converter()
        try:
            rendered = converter(str(pdf_path))
        except Exception as exc:
            print(
                f"警告: {paper_id}: Marker による変換に失敗しました: {exc}",
                file=sys.stderr,
            )
            return chunks

        counters = {"figure": 0, "table": 0, "equation": 0, "text": 0}
        for block in rendered.blocks:
            chunk = self._build_chunk(block, paper_id, prefix, metadata_base, counters)
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    def _build_chunk(
        self,
        block: Any,
        paper_id: str,
        prefix: str,
        metadata_base: dict,
        counters: dict,
    ) -> Chunk | None:
        block_type = block.block_type

        if block_type in _FIGURE_BLOCK_TYPES:
            chunk_type, tag, counter_key = "figure", "fig", "figure"
        elif block_type in _TABLE_BLOCK_TYPES:
            chunk_type, tag, counter_key = "table", "tab", "table"
        elif block_type in _EQUATION_BLOCK_TYPES:
            chunk_type, tag, counter_key = "equation_algorithm", "eq", "equation"
        elif block_type in _TEXT_BLOCK_TYPES:
            chunk_type, tag, counter_key = "text_span", "c", "text"
        else:
            return None

        if chunk_type == "table":
            text = markdownify(block.html).strip()
        else:
            text = BeautifulSoup(block.html, "html.parser").get_text(" ", strip=True)
        if not text:
            return None

        counters[counter_key] += 1
        chunk_id = f"{paper_id}#{tag}{counters[counter_key]:04d}"

        chunk_metadata = dict(metadata_base)
        chunk_metadata["page"] = block.page + 1  # Marker は0-indexed → 1-indexedへ変換

        if chunk_type == "figure":
            chunk_metadata["figure_id"] = _visible_id(text, "Figure")
        elif chunk_type == "table":
            chunk_metadata["table_id"] = _visible_id(text, "Table")
        elif chunk_type == "equation_algorithm":
            chunk_metadata["equation_id"] = _extract_equation_number(text)

        images = getattr(block, "images", None)
        if images:
            image_path = self._save_image(paper_id, chunk_id, images)
            if image_path:
                chunk_metadata["image_path"] = image_path

        return Chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=f"{prefix}{text}",
            chunk_type=chunk_type,
            metadata=chunk_metadata,
        )

    def _save_image(self, paper_id: str, chunk_id: str, images: dict) -> str | None:
        """block.images(base64 PNG想定)の最初の1枚を image_dir に保存し、パスを返す。"""
        first_image = next(iter(images.values()), None)
        if not first_image:
            return None
        try:
            image_bytes = base64.b64decode(first_image)
            path = self.image_dir / paper_id / f"{chunk_id.split('#')[-1]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_bytes)
        except Exception as exc:
            print(
                f"警告: {paper_id}: 画像 {chunk_id} の保存に失敗しました: {exc}",
                file=sys.stderr,
            )
            return None
        return str(path)

    def _get_converter(self) -> Callable[[str], Any]:
        if self._converter is None:
            self._converter = self._build_converter()
        return self._converter

    def _build_converter(self) -> Callable[[str], Any]:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        config_parser = ConfigParser(
            {
                "output_format": "chunks",
                "force_ocr": self.force_ocr,
                "use_llm": self.use_llm,
            }
        )
        return PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )


def _visible_id(text: str, prefix: str) -> str | None:
    number = _extract_number(text)
    if not number:
        return None
    return f"{prefix} {number}"
