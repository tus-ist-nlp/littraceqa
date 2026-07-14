"""Tests for converting existing MinerU content-list artifacts into chunks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litqa.preprocess.mineru_chunker import (
    MinerUChunker,
    MinerUDataError,
    _visible_ids,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "mineru"


def _paper(paper_id: str = "fixture_paper") -> dict:
    return {
        "paper_id": paper_id,
        "title": "Fixture Paper",
        "abstract": "Fixture abstract.",
        "venue": "TEST",
        "year": 2026,
        "authors": ["Ada Example"],
    }


@pytest.mark.parametrize(
    ("content_version", "filename"),
    [
        ("v1", "fixture_paper_content_list.json"),
        ("v2", "fixture_paper_content_list_v2.json"),
    ],
)
def test_input_paths_selects_one_versioned_content_list(
    content_version: str, filename: str
):
    chunker = MinerUChunker(str(FIXTURE_ROOT), content_version=content_version)

    paths = chunker.input_paths(_paper())

    assert paths == [(FIXTURE_ROOT / "fixture_paper" / "auto" / filename).resolve()]


def test_input_paths_rejects_unsafe_paper_id():
    chunker = MinerUChunker(str(FIXTURE_ROOT))

    with pytest.raises(MinerUDataError, match="unsafe or missing paper_id"):
        chunker.input_paths(_paper("../escape"))


def test_converts_modalities_and_provenance_deterministically():
    chunker = MinerUChunker(str(FIXTURE_ROOT))

    first = chunker.process(_paper())
    second = chunker.process(_paper())

    assert [chunk.to_dict() for chunk in first] == [chunk.to_dict() for chunk in second]
    assert [chunk.chunk_type for chunk in first] == [
        "title_abstract",
        "text_span",
        "figure",
        "table",
        "equation_algorithm",
    ]
    assert all(chunk.paper_id == "fixture_paper" for chunk in first)

    text, figure, table, equation = first[1:]
    assert text.metadata["page"] == 1
    assert text.metadata["bbox"] == [10, 20, 300, 80]
    assert figure.metadata["page"] == 2
    assert figure.metadata["figure_id"] == "Figure 2"
    assert figure.metadata["caption"] == ["Figure 2: A small synthetic figure."]
    assert Path(figure.metadata["image_path"]).is_file()
    assert table.metadata["table_id"] == "Table 3"
    assert "Method" in table.text and "1.0" in table.text
    assert "image_path" not in table.metadata
    assert equation.metadata["page"] == 3
    assert equation.metadata["equation_id"] == "Equation 4"
    assert "image_path" not in equation.metadata


def test_extracts_all_visible_ids_from_a_combined_visual_block():
    assert _visible_ids(
        ["Table 3: First results.", "Table 4: Cross-domain results."], "table"
    ) == ["Table 3", "Table 4"]
    assert _visible_ids(
        ["Figures 1 and 2 are shown together.", "Figure 2: Detail."], "figure"
    ) == ["Figure 2"]


def test_converts_v2_page_structure_to_the_same_common_modalities():
    chunks = MinerUChunker(str(FIXTURE_ROOT), content_version="v2").process(_paper())

    assert [chunk.chunk_type for chunk in chunks] == [
        "title_abstract",
        "text_span",
        "figure",
        "table",
        "equation_algorithm",
    ]
    text, figure, table, equation = chunks[1:]
    assert text.metadata["page"] == 1
    assert text.metadata["mineru_type"] == "paragraph"
    assert figure.metadata["page"] == 2
    assert figure.metadata["figure_id"] == "Figure 2"
    assert table.metadata["table_id"] == "Table 3"
    assert table.metadata["table_nest_level"] == 1
    assert equation.metadata["page"] == 3
    assert equation.metadata["mineru_type"] == "equation_interline"
    assert equation.metadata["equation_id"] == "Equation 4"
    assert all(chunk.metadata["mineru_content_version"] == "v2" for chunk in chunks)


def test_rejects_malformed_v2_page_structure(tmp_path: Path):
    paper_id = "bad_v2"
    auto_dir = tmp_path / paper_id / "auto"
    auto_dir.mkdir(parents=True)
    (auto_dir / f"{paper_id}_content_list_v2.json").write_text(
        json.dumps([{"type": "paragraph", "content": {}}]), encoding="utf-8"
    )

    with pytest.raises(MinerUDataError, match="v2 page 0"):
        MinerUChunker(str(tmp_path), content_version="v2").process(_paper(paper_id))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("", "empty"),
        ("not json", "not valid JSON"),
        (json.dumps({"type": "text"}), "JSON array"),
        (json.dumps([]), "no blocks"),
        (json.dumps(["bad block"]), "JSON object"),
    ],
)
def test_rejects_invalid_content_lists(tmp_path: Path, payload: str, message: str):
    paper_id = "bad_paper"
    auto_dir = tmp_path / paper_id / "auto"
    auto_dir.mkdir(parents=True)
    (auto_dir / f"{paper_id}_content_list.json").write_text(payload, encoding="utf-8")

    with pytest.raises(MinerUDataError, match=message):
        MinerUChunker(str(tmp_path)).process(_paper(paper_id))


def test_rejects_missing_artifact(tmp_path: Path):
    (tmp_path / "missing_paper" / "auto").mkdir(parents=True)
    with pytest.raises(MinerUDataError, match="content list is missing"):
        MinerUChunker(str(tmp_path)).process(_paper("missing_paper"))


@pytest.mark.parametrize("paper_id", ["../escape", "/absolute", "..", ""])
def test_rejects_unsafe_paper_ids(paper_id: str):
    with pytest.raises(MinerUDataError, match="unsafe or missing paper_id"):
        MinerUChunker(str(FIXTURE_ROOT)).process(_paper(paper_id))


def test_rejects_image_path_traversal(tmp_path: Path):
    paper_id = "unsafe_image"
    auto_dir = tmp_path / paper_id / "auto"
    auto_dir.mkdir(parents=True)
    payload = [
        {
            "type": "image",
            "img_path": "../../outside.jpg",
            "image_caption": ["Figure 1: Unsafe path."],
            "page_idx": 0,
            "bbox": [0, 0, 1, 1],
        }
    ]
    (auto_dir / f"{paper_id}_content_list.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(MinerUDataError, match="image path escapes"):
        MinerUChunker(str(tmp_path)).process(_paper(paper_id))


def test_keeps_image_only_figure_without_inventing_caption_or_visible_id(tmp_path: Path):
    paper_id = "image_only"
    auto_dir = tmp_path / paper_id / "auto"
    image_dir = auto_dir / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "figure.jpg").write_bytes(b"image")
    payload = [
        {
            "type": "image",
            "img_path": "images/figure.jpg",
            "image_caption": [],
            "image_footnote": [],
            "page_idx": 0,
            "bbox": [0, 0, 1, 1],
        }
    ]
    (auto_dir / f"{paper_id}_content_list.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    chunks = MinerUChunker(str(tmp_path)).process(_paper(paper_id))

    assert [chunk.chunk_type for chunk in chunks] == ["title_abstract", "figure"]
    assert "image_path" in chunks[1].metadata
    assert "caption" not in chunks[1].metadata
    assert "figure_id" not in chunks[1].metadata


def test_marks_empty_visual_blocks_as_partial(tmp_path: Path):
    paper_id = "partial_paper"
    auto_dir = tmp_path / paper_id / "auto"
    auto_dir.mkdir(parents=True)
    payload = [
        {"type": "text", "text": "usable", "page_idx": 0, "bbox": [0, 0, 1, 1]},
        {
            "type": "table",
            "img_path": "",
            "table_caption": [],
            "table_body": "",
            "page_idx": 1,
            "bbox": [0, 0, 1, 1],
        },
    ]
    (auto_dir / f"{paper_id}_content_list.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    chunks = MinerUChunker(str(tmp_path)).process(_paper(paper_id))

    assert chunks[0].metadata["preprocess_status"] == "partial"
    assert chunks[0].metadata["mineru_warnings"] == [
        {
            "block_index": 1,
            "mineru_type": "table",
            "reason": "no usable text or image reference",
        }
    ]


def test_marks_retained_caption_with_missing_image_as_partial(tmp_path: Path):
    paper_id = "missing_visual_image"
    auto_dir = tmp_path / paper_id / "auto"
    auto_dir.mkdir(parents=True)
    payload = [
        {
            "type": "image",
            "img_path": "images/missing.jpg",
            "image_caption": ["Figure 1: Retained caption."],
            "page_idx": 0,
            "bbox": [0, 0, 1, 1],
        }
    ]
    (auto_dir / f"{paper_id}_content_list.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    chunks = MinerUChunker(str(tmp_path)).process(_paper(paper_id))

    assert chunks[1].chunk_type == "figure"
    assert chunks[1].metadata["image_reference_missing"] is True
    assert chunks[1].metadata["preprocess_status"] == "partial"
