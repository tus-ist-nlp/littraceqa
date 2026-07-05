"""Docling で PDF から図・表を抽出し、Qwen2-VL で検索用の記述文を生成する Preprocessor。

本文チャンク（pypdf 等）とは独立に、図・表の Chunk のみを返す。本文チャンクとの
結合は scripts/merge_chunks.py で行う。

抽出した図・表の画像は metadata["image_path"] にファイルパスとして保存する。
これは本 Preprocessor 自身は使わないが、画像を直接 vision encoder に通して
埋め込む indexer（例: litqa/index/siglip_image.py）が chunks.jsonl 経由で
参照するために必要。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

from litqa.contracts import Chunk
from litqa.registry import register

_VISIBLE_ID_RE = re.compile(r"(?:Figure|Table|Fig\.?)\s*(\d+[a-zA-Z]?)", re.IGNORECASE)

_VLM_PROMPT_TEMPLATE = (
    "You are helping build a search index for scientific figures.\n"
    "Describe this figure so it can be retrieved by researchers' questions.\n"
    "Include: what is being compared, axis labels, trends, key numbers, "
    "and methods mentioned.\n"
    "Caption: {caption}\n"
    "Write a concise but information-dense description (2-4 sentences)."
)


def _extract_number(caption: str) -> str | None:
    """caption から図表番号（例: "3", "12a"）を抽出する。マッチしなければ None。"""
    if not caption:
        return None
    match = _VISIBLE_ID_RE.search(caption)
    if not match:
        return None
    return match.group(1)


@register("preprocessor", "figure_vlm")
class FigureVLMChunker:
    def __init__(
        self,
        pdf_dir: str,
        vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 256,
        include_tables: bool = True,
        image_dir: str | None = None,
        docling_converter: Any | None = None,
        vlm_describer: Callable[[Any, str], str] | None = None,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.vlm_model = vlm_model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.include_tables = include_tables
        # 画像自体の保存先。指定がなければ pdf_dir と衝突しない兄弟ディレクトリに
        # 自動導出する（process_style yaml には pdf_dir/index_dir 同様パスを
        # 書かない、という CLAUDE.md の方針に合わせる）。
        self.image_dir = Path(image_dir) if image_dir else self.pdf_dir.parent / "figure_images"
        # Docling / Qwen2-VL は重いため、注入されなければ初回 process() 呼び出し時まで
        # 実体を構築しない（遅延ロード）。テストでは fake を注入して重い依存を避ける。
        self._docling_converter = docling_converter
        self._vlm_describer = vlm_describer

    def process(self, paper: dict) -> list[Chunk]:
        paper_id = paper["paper_id"]
        prefix = f"[{paper['venue']} {paper['year']}] {paper['title']}\n"
        metadata_base = {
            "title": paper["title"],
            "venue": paper["venue"],
            "year": paper["year"],
        }

        pdf_path = self.pdf_dir / f"{paper_id}.pdf"
        if not pdf_path.exists():
            return []

        converter = self._get_docling_converter()
        try:
            result = converter.convert(str(pdf_path))
            doc = result.document
        except Exception as exc:
            print(
                f"警告: {paper_id}: Docling による図表抽出に失敗しました: {exc}",
                file=sys.stderr,
            )
            return []

        chunks = self._build_chunks(doc.pictures, "figure", paper_id, prefix, metadata_base)
        if self.include_tables:
            chunks.extend(
                self._build_chunks(doc.tables, "table", paper_id, prefix, metadata_base)
            )
        return chunks

    def _build_chunks(
        self,
        items: list,
        kind: str,
        paper_id: str,
        prefix: str,
        metadata_base: dict,
    ) -> list[Chunk]:
        tag = "fig" if kind == "figure" else "tab"
        chunks: list[Chunk] = []
        idx = 0

        for item in items:
            prov = item.prov[0] if getattr(item, "prov", None) else None
            page = prov.page_no if prov is not None else None
            caption = getattr(item, "caption_text", "") or ""
            image = getattr(item, "image", None)
            pil_image = image.pil_image if image is not None else None

            if not caption and pil_image is None:
                continue

            idx += 1
            chunk_id = f"{paper_id}#{tag}{idx:04d}"
            description = self._describe(paper_id, pil_image, caption)
            image_path = self._save_image(paper_id, chunk_id, pil_image)
            number = _extract_number(caption)
            if number:
                visible_id = f"{kind.capitalize()} {number}"
            else:
                visible_id = None
            label = visible_id or f"{kind.capitalize()} {idx}"

            chunk_metadata = dict(metadata_base)
            chunk_metadata["page"] = page
            chunk_metadata[f"{kind}_id"] = visible_id
            chunk_metadata["image_path"] = image_path

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    text=f"{prefix}[{label}] {description}\nCaption: {caption}",
                    chunk_type=kind,
                    metadata=chunk_metadata,
                )
            )

        return chunks

    def _save_image(self, paper_id: str, chunk_id: str, pil_image: Any | None) -> str | None:
        """図・表の画像を PNG として image_dir に保存し、パスを返す。失敗時は None。"""
        if pil_image is None:
            return None
        path = self.image_dir / paper_id / f"{chunk_id.split('#')[-1]}.png"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(path)
        except Exception as exc:
            print(
                f"警告: {paper_id}: 画像 {chunk_id} の保存に失敗しました: {exc}",
                file=sys.stderr,
            )
            return None
        return str(path)

    def _describe(self, paper_id: str, pil_image: Any | None, caption: str) -> str:
        if pil_image is not None:
            try:
                description = self._get_vlm_describer()(pil_image, caption)
                if description and description.strip():
                    return description.strip()
            except Exception as exc:
                print(
                    f"警告: {paper_id}: VLM による記述生成に失敗しました: {exc}",
                    file=sys.stderr,
                )
        return caption

    def _get_docling_converter(self) -> Any:
        if self._docling_converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = 2.0
            pipeline_options.do_table_structure = True

            self._docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        return self._docling_converter

    def _get_vlm_describer(self) -> Callable[[Any, str], str]:
        if self._vlm_describer is None:
            self._vlm_describer = self._build_vlm_describer()
        return self._vlm_describer

    def _build_vlm_describer(self) -> Callable[[Any, str], str]:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.vlm_model, torch_dtype="auto", device_map=self.device
        )
        processor = AutoProcessor.from_pretrained(self.vlm_model)

        def describe(image: Any, caption: str) -> str:
            prompt = _VLM_PROMPT_TEMPLATE.format(caption=caption)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
            return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        return describe
