"""MarkerChunker のブロック種別マッピング・ページ番号変換のテスト(PdfConverter は fake を注入)。"""

from __future__ import annotations

from littraceqa.di_pipeline.preprocess.marker_chunker import MarkerChunker


class _FakeBlock:
    def __init__(self, block_type, html, page, images=None):
        self.block_type = block_type
        self.html = html
        self.page = page
        self.images = images


class _FakeRendered:
    def __init__(self, blocks):
        self.blocks = blocks


class _FakeConverter:
    def __init__(self, rendered=None, exception=None):
        self._rendered = rendered
        self._exception = exception
        self.calls: list[str] = []

    def __call__(self, pdf_path):
        self.calls.append(pdf_path)
        if self._exception is not None:
            raise self._exception
        return self._rendered


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


def test_missing_pdf_returns_only_title_abstract_chunk(tmp_path):
    chunker = MarkerChunker(pdf_dir=str(tmp_path), converter=_FakeConverter())

    chunks = chunker.process(_paper())

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "p1#c0000"
    assert chunks[0].chunk_type == "title_abstract"


def test_converter_failure_returns_only_title_abstract_chunk(tmp_path):
    _touch_pdf(tmp_path, "p1")
    converter = _FakeConverter(exception=RuntimeError("boom"))
    chunker = MarkerChunker(pdf_dir=str(tmp_path), converter=converter)

    chunks = chunker.process(_paper())

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "title_abstract"
    assert converter.calls


def test_title_abstract_chunk_is_always_first(tmp_path):
    _touch_pdf(tmp_path, "p1")
    rendered = _FakeRendered([_FakeBlock("Text", "<p>Some body text.</p>", page=0)])
    chunker = MarkerChunker(pdf_dir=str(tmp_path), converter=_FakeConverter(rendered=rendered))

    chunks = chunker.process(_paper())

    assert chunks[0].chunk_id == "p1#c0000"
    assert chunks[0].chunk_type == "title_abstract"


def test_text_block_maps_to_text_span_and_page_is_converted_to_1_indexed(tmp_path):
    _touch_pdf(tmp_path, "p1")
    rendered = _FakeRendered([_FakeBlock("Text", "<p>Some body text.</p>", page=0)])
    chunker = MarkerChunker(pdf_dir=str(tmp_path), converter=_FakeConverter(rendered=rendered))

    chunks = chunker.process(_paper())

    text_chunk = chunks[1]
    assert text_chunk.chunk_id == "p1#c0001"
    assert text_chunk.chunk_type == "text_span"
    assert text_chunk.metadata["page"] == 1
    assert "Some body text." in text_chunk.text
    assert "[EMNLP 2026] Example Paper" in text_chunk.text


def test_equation_block_extracts_trailing_number(tmp_path):
    _touch_pdf(tmp_path, "p1")
    rendered = _FakeRendered(
        [_FakeBlock("Equation", "<p>x = y + z (6)</p>", page=6)]
    )
    chunker = MarkerChunker(pdf_dir=str(tmp_path), converter=_FakeConverter(rendered=rendered))

    chunks = chunker.process(_paper())

    eq_chunk = chunks[1]
    assert eq_chunk.chunk_id == "p1#eq0001"
    assert eq_chunk.chunk_type == "equation_algorithm"
    assert eq_chunk.metadata["page"] == 7
    assert eq_chunk.metadata["equation_id"] == "Equation 6"


def test_table_group_block_maps_to_table_and_uses_markdownify(tmp_path):
    _touch_pdf(tmp_path, "p1")
    html = "<p>Table 3. Dataset statistics.</p><table><tr><td>a</td><td>b</td></tr></table>"
    rendered = _FakeRendered([_FakeBlock("TableGroup", html, page=4)])
    chunker = MarkerChunker(pdf_dir=str(tmp_path), converter=_FakeConverter(rendered=rendered))

    chunks = chunker.process(_paper())

    table_chunk = chunks[1]
    assert table_chunk.chunk_id == "p1#tab0001"
    assert table_chunk.chunk_type == "table"
    assert table_chunk.metadata["table_id"] == "Table 3"
    assert table_chunk.metadata["page"] == 5
    assert "| a | b |" in table_chunk.text


def test_figure_group_block_extracts_figure_id_and_saves_image(tmp_path):
    _touch_pdf(tmp_path, "p1")
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
        b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01"
        b"\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    import base64

    b64_png = base64.b64encode(png_1x1).decode()
    html = "<p>Figure 2. Accuracy comparison.</p>"
    rendered = _FakeRendered(
        [_FakeBlock("FigureGroup", html, page=2, images={"img1": b64_png})]
    )
    image_dir = tmp_path / "marker_images"
    chunker = MarkerChunker(
        pdf_dir=str(tmp_path),
        image_dir=str(image_dir),
        converter=_FakeConverter(rendered=rendered),
    )

    chunks = chunker.process(_paper())

    figure_chunk = chunks[1]
    assert figure_chunk.chunk_id == "p1#fig0001"
    assert figure_chunk.chunk_type == "figure"
    assert figure_chunk.metadata["figure_id"] == "Figure 2"
    assert figure_chunk.metadata["page"] == 3
    image_path = figure_chunk.metadata["image_path"]
    assert image_path is not None
    from pathlib import Path

    assert Path(image_path).exists()
    assert Path(image_path).is_relative_to(image_dir)


def test_noise_block_types_are_skipped(tmp_path):
    _touch_pdf(tmp_path, "p1")
    rendered = _FakeRendered(
        [
            _FakeBlock("PageHeader", "<p>Running head</p>", page=0),
            _FakeBlock("PageFooter", "<p>1</p>", page=0),
            _FakeBlock("Text", "<p>Kept text.</p>", page=0),
        ]
    )
    chunker = MarkerChunker(pdf_dir=str(tmp_path), converter=_FakeConverter(rendered=rendered))

    chunks = chunker.process(_paper())

    assert len(chunks) == 2
    assert chunks[1].chunk_type == "text_span"
    assert "Kept text." in chunks[1].text
