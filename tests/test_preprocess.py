"""MinerUChunker: block type mapping, page number conversion, and the split rules."""

from __future__ import annotations

import json

from littraceqa.search.preprocess import MinerUChunker


def _paper(paper_id: str = "p1") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Example Paper",
        "venue": "EMNLP",
        "year": 2026,
        "authors": ["A. Author"],
        "abstract": "An abstract.",
    }


def _write_content_list(mineru_dir, paper_id: str, blocks: list[dict]) -> None:
    path = mineru_dir / paper_id / "auto" / f"{paper_id}_content_list.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blocks), encoding="utf-8")


def _chunker(tmp_path, **kwargs) -> MinerUChunker:
    return MinerUChunker(pdf_dir=str(tmp_path / "pdfs"), mineru_dir=str(tmp_path / "mineru"), **kwargs)


def _by_type(chunks, chunk_type):
    return [c for c in chunks if c.chunk_type == chunk_type]


def test_missing_content_list_returns_only_title_abstract_chunk(tmp_path):
    chunker = _chunker(tmp_path)

    chunks = chunker.process(_paper())

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "p1#c0000"
    assert chunks[0].chunk_type == "title_abstract"


def test_broken_content_list_returns_only_title_abstract_chunk(tmp_path):
    path = tmp_path / "mineru" / "p1" / "auto" / "p1_content_list.json"
    path.parent.mkdir(parents=True)
    path.write_text("{ not json", encoding="utf-8")

    chunks = _chunker(tmp_path).process(_paper())

    assert [c.chunk_type for c in chunks] == ["title_abstract"]


def test_mineru_dir_defaults_to_sibling_of_pdf_dir(tmp_path):
    chunker = MinerUChunker(pdf_dir=str(tmp_path / "corpus" / "pdfs"))

    assert chunker.mineru_dir == tmp_path / "corpus" / "mineru"


def test_page_idx_is_converted_to_1_indexed(tmp_path):
    _write_content_list(tmp_path / "mineru", "p1", [{"type": "text", "text": "Body.", "page_idx": 0}])

    chunks = _chunker(tmp_path).process(_paper())

    assert _by_type(chunks, "text_span")[0].metadata["page"] == 1


def test_heading_becomes_section_and_is_not_its_own_chunk(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": "Body.", "page_idx": 0},
        ],
    )

    spans = _by_type(_chunker(tmp_path).process(_paper()), "text_span")

    assert len(spans) == 1
    assert spans[0].metadata["section"] == "1 Introduction"
    assert spans[0].text.endswith("Body.")


def test_page_number_and_footer_blocks_are_skipped(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {"type": "page_number", "text": "3", "page_idx": 0},
            {"type": "footer", "text": "Proceedings", "page_idx": 0},
            {"type": "text", "text": "Body.", "page_idx": 0},
        ],
    )

    spans = _by_type(_chunker(tmp_path).process(_paper()), "text_span")

    assert len(spans) == 1
    assert "Proceedings" not in spans[0].text


def test_text_on_different_pages_is_not_merged_into_one_chunk(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {"type": "text", "text": "First page.", "page_idx": 0},
            {"type": "text", "text": "Second page.", "page_idx": 1},
        ],
    )

    spans = _by_type(_chunker(tmp_path).process(_paper()), "text_span")

    assert [c.metadata["page"] for c in spans] == [1, 2]


def test_tagged_equation_gets_its_own_chunk_with_preceding_text_as_context(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {"type": "text", "text": "We define the score.", "page_idx": 2},
            {"type": "equation", "text": "$$\nx = y\\tag{12}\n$$", "page_idx": 2},
        ],
    )

    chunks = _chunker(tmp_path).process(_paper())
    equations = _by_type(chunks, "equation_algorithm")

    assert len(equations) == 1
    assert equations[0].chunk_id == "p1#eq0001"
    assert equations[0].metadata["equation_id"] == "Equation 12"
    assert equations[0].metadata["page"] == 3
    # Not the equation alone: the preceding body text comes with it as context.
    assert "We define the score." in equations[0].text
    assert "x = y" in equations[0].text


def test_equation_is_also_inlined_into_the_surrounding_text_span(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {"type": "text", "text": "We define the score.", "page_idx": 0},
            {"type": "equation", "text": "$$\nx = y\\tag{1}\n$$", "page_idx": 0},
            {"type": "text", "text": "Therefore it holds.", "page_idx": 0},
        ],
    )

    spans = _by_type(_chunker(tmp_path).process(_paper()), "text_span")

    assert len(spans) == 1
    assert "x = y" in spans[0].text
    assert "Therefore it holds." in spans[0].text


def test_untagged_equation_does_not_create_an_equation_chunk(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {"type": "text", "text": "Context.", "page_idx": 0},
            {"type": "equation", "text": "$$\nx = y\n$$", "page_idx": 0},
        ],
    )

    chunks = _chunker(tmp_path).process(_paper())

    assert _by_type(chunks, "equation_algorithm") == []
    assert "x = y" in _by_type(chunks, "text_span")[0].text


def test_table_body_html_becomes_markdown_and_table_id_comes_from_caption(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {
                "type": "table",
                "table_caption": ["Table 4: Results."],
                "table_body": "<table><tr><td>a</td><td>b</td></tr></table>",
                "table_footnote": ["Bold is best."],
                "img_path": "images/t.jpg",
                "page_idx": 1,
            }
        ],
    )

    tables = _by_type(_chunker(tmp_path).process(_paper()), "table")

    assert len(tables) == 1
    assert tables[0].chunk_id == "p1#tab0001"
    assert tables[0].metadata["table_id"] == "Table 4"
    assert tables[0].metadata["page"] == 2
    assert tables[0].metadata["image_path"] == str(tmp_path / "mineru" / "p1" / "auto" / "images" / "t.jpg")
    assert "Table 4: Results." in tables[0].text
    assert "Bold is best." in tables[0].text
    assert "<table>" not in tables[0].text


def test_image_becomes_figure_chunk_with_figure_id_and_image_path(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {
                "type": "image",
                "image_caption": ["Figure 2: An illustration."],
                "image_footnote": [],
                "img_path": "images/f.jpg",
                "page_idx": 0,
            }
        ],
    )

    figures = _by_type(_chunker(tmp_path).process(_paper()), "figure")

    assert len(figures) == 1
    assert figures[0].chunk_id == "p1#fig0001"
    assert figures[0].metadata["figure_id"] == "Figure 2"
    assert figures[0].metadata["image_path"] == str(tmp_path / "mineru" / "p1" / "auto" / "images" / "f.jpg")


def test_chart_content_is_used_as_figure_text(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [{"type": "chart", "chart_caption": [], "content": "acc rises with size", "page_idx": 0}],
    )

    figures = _by_type(_chunker(tmp_path).process(_paper()), "figure")

    assert len(figures) == 1
    assert "acc rises with size" in figures[0].text


def test_figure_without_caption_or_content_is_dropped(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [{"type": "chart", "chart_caption": [], "content": "", "img_path": "images/x.jpg", "page_idx": 0}],
    )

    assert _by_type(_chunker(tmp_path).process(_paper()), "figure") == []


def test_list_items_are_joined_into_the_text_buffer(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [{"type": "list", "sub_type": "ref_text", "list_items": ["Ref one.", "Ref two."], "page_idx": 0}],
    )

    spans = _by_type(_chunker(tmp_path).process(_paper()), "text_span")

    assert len(spans) == 1
    assert "Ref one." in spans[0].text and "Ref two." in spans[0].text


def test_long_text_is_split_at_max_chars(tmp_path):
    paragraph = " ".join(["word"] * 100)  # 499 characters
    _write_content_list(
        tmp_path / "mineru", "p1",
        [{"type": "text", "text": paragraph, "page_idx": 0} for _ in range(4)],
    )

    spans = _by_type(_chunker(tmp_path, max_chars_per_chunk=1000).process(_paper()), "text_span")

    assert len(spans) == 2
    prefix_len = len("[EMNLP 2026] Example Paper\n")
    assert all(len(c.text) - prefix_len <= 1000 for c in spans)


def test_oversized_equation_paragraph_is_not_split_mid_equation(tmp_path):
    equation = "$$\n" + " + ".join(["x"] * 400) + "\\tag{1}\n$$"
    _write_content_list(tmp_path / "mineru", "p1", [{"type": "equation", "text": equation, "page_idx": 0}])

    spans = _by_type(_chunker(tmp_path, max_chars_per_chunk=100).process(_paper()), "text_span")

    assert len(spans) == 1
    assert spans[0].text.rstrip().endswith("$$")


def test_chunk_ids_are_unique_and_numbered_per_type(tmp_path):
    _write_content_list(
        tmp_path / "mineru", "p1",
        [
            {"type": "text", "text": "One.", "page_idx": 0},
            {"type": "text", "text": "Two.", "page_idx": 1},
            {"type": "image", "image_caption": ["Figure 1: A."], "page_idx": 0},
            {"type": "image", "image_caption": ["Figure 2: B."], "page_idx": 1},
        ],
    )

    chunks = _chunker(tmp_path).process(_paper())
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids))
    assert ids[0] == "p1#c0000"
    assert [c.chunk_id for c in _by_type(chunks, "text_span")] == ["p1#c0001", "p1#c0002"]
    assert [c.chunk_id for c in _by_type(chunks, "figure")] == ["p1#fig0001", "p1#fig0002"]
