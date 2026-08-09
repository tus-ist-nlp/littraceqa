from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.retrieve.paper_tables import MinerUPaperTableSource


def _content_list_path(tmp_path, paper_id: str):
    return tmp_path / paper_id / "auto" / f"{paper_id}_content_list.json"


def _write_content_list(tmp_path, paper_id: str, content: object) -> None:
    path = _content_list_path(tmp_path, paper_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(content), encoding="utf-8")


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


@pytest.mark.parametrize("case", ["missing", "broken", "object"])
def test_invalid_content_lists_return_no_tables(tmp_path, case):
    path = _content_list_path(tmp_path, "paper-1")
    if case == "broken":
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
    elif case == "object":
        _write_content_list(tmp_path, "paper-1", {"type": "table"})

    assert MinerUPaperTableSource(tmp_path).tables("paper-1") == ()


@pytest.mark.parametrize("paper_id", ["", ".", "..", "../paper", "dir/paper", r"dir\paper"])
def test_rejects_paper_ids_that_are_not_one_path_component(tmp_path, paper_id):
    with pytest.raises(ValueError, match="single path component"):
        MinerUPaperTableSource(tmp_path).tables(paper_id)
