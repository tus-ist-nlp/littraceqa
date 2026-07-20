"""FigureVLMChunker の図表抽出ロジックのテスト（Docling/VLM は fake を注入）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from littraceqa.di_pipeline.preprocess.figure_vlm import FigureVLMChunker, _extract_number


class _FakeProv:
    def __init__(self, page_no):
        self.page_no = page_no


class _FakeImage:
    def __init__(self, pil_image):
        self.pil_image = pil_image


class _FakeItem:
    def __init__(self, page_no=None, caption_text="", pil_image=None):
        self.prov = [_FakeProv(page_no)] if page_no is not None else []
        self.caption_text = caption_text
        self.image = _FakeImage(pil_image) if pil_image is not None else None


class _FakeDocument:
    def __init__(self, pictures=None, tables=None):
        self.pictures = pictures or []
        self.tables = tables or []


class _FakeConvertResult:
    def __init__(self, document):
        self.document = document


class _FakeDoclingConverter:
    def __init__(self, document=None, exception=None):
        self._document = document
        self._exception = exception
        self.calls: list[str] = []

    def convert(self, pdf_path):
        self.calls.append(pdf_path)
        if self._exception is not None:
            raise self._exception
        return _FakeConvertResult(self._document)


class _RaisingVLMDescriber:
    def __call__(self, image, caption):
        raise RuntimeError("vlm unavailable")


def _fake_vlm_describer(image, caption):
    return f"description of {caption}"


def _paper(paper_id: str = "p1") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Example Paper",
        "venue": "EMNLP",
        "year": 2026,
        "authors": ["A. Author"],
        "abstract": "An abstract.",
    }


def _touch_pdf(pdf_dir, paper_id: str) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    (pdf_dir / f"{paper_id}.pdf").write_bytes(b"%PDF-1.4 fake")


def test_missing_pdf_returns_empty_list(tmp_path):
    chunker = FigureVLMChunker(
        pdf_dir=str(tmp_path), docling_converter=_FakeDoclingConverter()
    )

    chunks = chunker.process(_paper())

    assert chunks == []


def test_docling_failure_returns_empty_list(tmp_path):
    _touch_pdf(tmp_path, "p1")
    converter = _FakeDoclingConverter(exception=RuntimeError("boom"))
    chunker = FigureVLMChunker(pdf_dir=str(tmp_path), docling_converter=converter)

    chunks = chunker.process(_paper())

    assert chunks == []
    assert converter.calls


def test_figure_and_table_chunks_have_expected_shape(tmp_path):
    _touch_pdf(tmp_path, "p1")
    doc = _FakeDocument(
        pictures=[
            _FakeItem(
                page_no=3,
                caption_text="Figure 2. Accuracy comparison.",
                pil_image=object(),
            )
        ],
        tables=[
            _FakeItem(
                page_no=5,
                caption_text="Table 1. Dataset statistics.",
                pil_image=object(),
            )
        ],
    )
    converter = _FakeDoclingConverter(document=doc)
    chunker = FigureVLMChunker(
        pdf_dir=str(tmp_path),
        docling_converter=converter,
        vlm_describer=_fake_vlm_describer,
    )

    chunks = chunker.process(_paper())

    assert len(chunks) == 2
    figure_chunk, table_chunk = chunks

    assert figure_chunk.chunk_id == "p1#fig0001"
    assert figure_chunk.chunk_type == "figure"
    assert figure_chunk.paper_id == "p1"
    assert figure_chunk.metadata["figure_id"] == "Figure 2"
    assert figure_chunk.metadata["page"] == 3
    assert "[EMNLP 2026] Example Paper" in figure_chunk.text
    assert "description of Figure 2. Accuracy comparison." in figure_chunk.text
    assert "Caption: Figure 2. Accuracy comparison." in figure_chunk.text

    assert table_chunk.chunk_id == "p1#tab0001"
    assert table_chunk.chunk_type == "table"
    assert table_chunk.metadata["table_id"] == "Table 1"
    assert table_chunk.metadata["page"] == 5


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("Figure 3. Comparison of methods.", "3"),
        ("Fig. 12a shows the trend.", "12a"),
        ("Table 1: dataset stats", "1"),
        ("No number mentioned at all", None),
        ("", None),
    ],
)
def test_extract_number(caption, expected):
    assert _extract_number(caption) == expected


def test_vlm_failure_falls_back_to_caption(tmp_path):
    _touch_pdf(tmp_path, "p1")
    doc = _FakeDocument(
        pictures=[_FakeItem(page_no=1, caption_text="Figure 1. A chart.", pil_image=object())]
    )
    converter = _FakeDoclingConverter(document=doc)
    chunker = FigureVLMChunker(
        pdf_dir=str(tmp_path),
        docling_converter=converter,
        vlm_describer=_RaisingVLMDescriber(),
    )

    chunks = chunker.process(_paper())

    assert len(chunks) == 1
    assert "[Figure 1] Figure 1. A chart." in chunks[0].text


def test_include_tables_false_skips_tables(tmp_path):
    _touch_pdf(tmp_path, "p1")
    doc = _FakeDocument(
        pictures=[_FakeItem(page_no=1, caption_text="Figure 1. A chart.", pil_image=object())],
        tables=[_FakeItem(page_no=2, caption_text="Table 1. Stats.", pil_image=object())],
    )
    converter = _FakeDoclingConverter(document=doc)
    chunker = FigureVLMChunker(
        pdf_dir=str(tmp_path),
        docling_converter=converter,
        vlm_describer=_fake_vlm_describer,
        include_tables=False,
    )

    chunks = chunker.process(_paper())

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "figure"


def test_image_is_saved_and_path_recorded_in_metadata(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    _touch_pdf(tmp_path, "p1")
    real_image = Image.new("RGB", (4, 4))
    doc = _FakeDocument(
        pictures=[
            _FakeItem(page_no=1, caption_text="Figure 1. A chart.", pil_image=real_image)
        ]
    )
    converter = _FakeDoclingConverter(document=doc)
    image_dir = tmp_path / "figure_images"
    chunker = FigureVLMChunker(
        pdf_dir=str(tmp_path),
        image_dir=str(image_dir),
        docling_converter=converter,
        vlm_describer=_fake_vlm_describer,
    )

    chunks = chunker.process(_paper())

    assert len(chunks) == 1
    image_path = chunks[0].metadata["image_path"]
    assert image_path is not None
    assert Path(image_path).exists()
    assert Path(image_path).is_relative_to(image_dir)


def test_image_dir_defaults_to_sibling_of_pdf_dir(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    chunker = FigureVLMChunker(pdf_dir=str(pdf_dir), docling_converter=_FakeDoclingConverter())

    assert chunker.image_dir == tmp_path / "figure_images"


def test_image_save_failure_falls_back_to_none(tmp_path):
    _touch_pdf(tmp_path, "p1")
    doc = _FakeDocument(
        pictures=[_FakeItem(page_no=1, caption_text="Figure 1. A chart.", pil_image=object())]
    )
    converter = _FakeDoclingConverter(document=doc)
    chunker = FigureVLMChunker(
        pdf_dir=str(tmp_path),
        docling_converter=converter,
        vlm_describer=_fake_vlm_describer,
    )

    chunks = chunker.process(_paper())

    assert chunks[0].metadata["image_path"] is None


def test_item_without_caption_or_image_is_skipped(tmp_path):
    _touch_pdf(tmp_path, "p1")
    doc = _FakeDocument(pictures=[_FakeItem(page_no=1, caption_text="", pil_image=None)])
    converter = _FakeDoclingConverter(document=doc)
    chunker = FigureVLMChunker(
        pdf_dir=str(tmp_path),
        docling_converter=converter,
        vlm_describer=_fake_vlm_describer,
    )

    chunks = chunker.process(_paper())

    assert chunks == []
