from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.retrieve.paper_tables import (
    MinerUPaperEvidenceSource,
    MinerUPaperTableSource,
)


def _content_list_path(tmp_path, paper_id: str):
    return tmp_path / paper_id / "auto" / f"{paper_id}_content_list.json"


def _write_content_list(tmp_path, paper_id: str, content: object) -> None:
    path = _content_list_path(tmp_path, paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


def test_old_table_source_name_remains_compatible():
    assert MinerUPaperTableSource is MinerUPaperEvidenceSource


def test_reads_caption_visible_id_rows_and_searchable_text(tmp_path):
    _write_content_list(
        tmp_path,
        "paper-1",
        [
            {"type": "text", "text": "Ignored."},
            {
                "type": "table",
                "table_caption": ["Table 5: Results on six datasets."],
                "table_body": (
                    "<table><tr><th>Dataset</th><th>Score</th></tr>"
                    "<tr><td>SUN RGB-D</td><td><b>64.2</b></td></tr></table>"
                ),
                "table_footnote": ["Bold is best."],
            },
        ],
    )

    tables = MinerUPaperTableSource(tmp_path).tables("paper-1")

    assert len(tables) == 1
    table = tables[0]
    assert table.paper_id == "paper-1"
    assert table.table_id == "Table 5"
    assert table.caption == "Table 5: Results on six datasets."
    assert table.rows == (("Dataset", "Score"), ("SUN RGB-D", "64.2"))
    assert table.text == (
        "Table 5: Results on six datasets.\n"
        "Dataset | Score\nSUN RGB-D | 64.2\nBold is best."
    )


def test_reads_title_and_text_blocks_without_loading_images(tmp_path):
    _write_content_list(
        tmp_path,
        "paper-1",
        [
            {"type": "text", "text_level": 1, "text": "  A Paper Title  "},
            {"type": "image", "img_path": "images/does-not-exist.jpg"},
            {"type": "text", "text": "First body paragraph."},
            {"type": "text", "text": ["Second", "body paragraph."]},
        ],
    )

    document = MinerUPaperTableSource(tmp_path).document("paper-1")

    assert document.paper_id == "paper-1"
    assert document.title == "A Paper Title"
    assert document.text_blocks == (
        "A Paper Title",
        "First body paragraph.",
        "Second\nbody paragraph.",
    )
    assert document.tables == ()


def test_reads_reference_list_items_separately_from_body_text(tmp_path):
    _write_content_list(
        tmp_path,
        "paper-1",
        [
            {"type": "list", "sub_type": "text", "list_items": ["Not a ref"]},
            {
                "type": "list",
                "sub_type": "ref_text",
                "list_items": [
                    "[1] A. Author. First paper.",
                    "[2] B. Author. Second paper.",
                ],
            },
        ],
    )

    document = MinerUPaperTableSource(tmp_path).document("paper-1")

    assert document.reference_entries == (
        "[1] A. Author. First paper.",
        "[2] B. Author. Second paper.",
    )
    assert document.text_blocks == ()


def test_tables_and_document_share_the_same_cached_content_list(tmp_path):
    _write_content_list(
        tmp_path,
        "paper-1",
        [
            {"type": "text", "text_level": 1, "text": "Original title"},
            {
                "type": "table",
                "table_caption": ["Table 1: Original results."],
                "table_body": "<table><tr><td>Original row</td></tr></table>",
            },
        ],
    )
    source = MinerUPaperTableSource(tmp_path)

    assert source.tables("paper-1")[0].table_id == "Table 1"
    _write_content_list(
        tmp_path,
        "paper-1",
        [{"type": "text", "text_level": 1, "text": "Changed title"}],
    )

    document = source.document("paper-1")
    assert document.title == "Original title"
    assert document.text_blocks == ("Original title",)
    assert document.tables[0].caption == "Table 1: Original results."


def test_expands_rowspan_and_colspan_into_logical_rows(tmp_path):
    _write_content_list(
        tmp_path,
        "paper-1",
        [
            {
                "type": "table",
                "table_caption": ["Table 1: Results."],
                "table_body": (
                    '<table><tr><td rowspan="2">Adversarial</td>'
                    '<td colspan="2">LLaVA-1.5</td></tr>'
                    "<tr><td>Accuracy</td><td>F1</td></tr>"
                    "<tr><td>VTI</td><td>82.5</td><td>82.1</td></tr></table>"
                ),
            }
        ],
    )

    rows = MinerUPaperTableSource(tmp_path).tables("paper-1")[0].rows

    assert rows == (
        ("Adversarial", "LLaVA-1.5", "LLaVA-1.5"),
        ("Adversarial", "Accuracy", "F1"),
        ("VTI", "82.5", "82.1"),
    )


def test_clamps_untrusted_table_spans(tmp_path):
    _write_content_list(
        tmp_path,
        "paper-1",
        [
            {
                "type": "table",
                "table_body": (
                    '<table><tr><td colspan="100000000">value</td></tr></table>'
                ),
            }
        ],
    )

    row = MinerUPaperTableSource(tmp_path).tables("paper-1")[0].rows[0]

    assert len(row) == 128
    assert set(row) == {"value"}


@pytest.mark.parametrize("case", ["missing", "broken", "object"])
def test_invalid_content_lists_return_no_tables(tmp_path, case):
    path = _content_list_path(tmp_path, "paper-1")
    if case == "broken":
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
    elif case == "object":
        _write_content_list(tmp_path, "paper-1", {"type": "table"})

    source = MinerUPaperTableSource(tmp_path)
    assert source.tables("paper-1") == ()
    document = source.document("paper-1")
    assert document.paper_id == "paper-1"
    assert document.title == ""
    assert document.text_blocks == ()
    assert document.tables == ()
    assert document.reference_entries == ()


@pytest.mark.parametrize("paper_id", ["", ".", "..", "../paper", "dir/paper", r"dir\paper"])
def test_rejects_paper_ids_that_are_not_one_path_component(tmp_path, paper_id):
    with pytest.raises(ValueError, match="single path component"):
        MinerUPaperTableSource(tmp_path).tables(paper_id)
