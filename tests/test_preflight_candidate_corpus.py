from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

from littraceqa.candidate_handoff import CandidateHandoff, CandidatePaper
from littraceqa.chunk_store import ChunkStore
from littraceqa.corpus_preflight import requires_visual_image
from littraceqa.di_pipeline.contracts import Query

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_candidate_corpus.py"
SPEC = importlib.util.spec_from_file_location("preflight_candidate_corpus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _image_root(tmp_path: Path) -> Path:
    return tmp_path / "trusted-images"


def _image_path(tmp_path: Path, paper_id: str, filename: str) -> Path:
    return _image_root(tmp_path) / paper_id / "auto" / "images" / filename


def _image_store(chunks: Path, tmp_path: Path) -> ChunkStore:
    return ChunkStore(chunks, image_root=_image_root(tmp_path))


def test_preflight_reports_real_locator_and_image_coverage(tmp_path):
    image = _image_path(tmp_path, "p1", "figure.jpg")
    image.parent.mkdir(parents=True)
    image.write_bytes(VALID_PNG)
    chunks = tmp_path / "chunks.jsonl"
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#1",
            "chunk_type": "figure",
            "text": "Figure",
            "metadata": {
                "page": 2,
                "figure_id": "Figure 1",
                "image_path": str(image),
            },
        }
    ]
    chunks.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    handoff = CandidateHandoff(
        Query("q1", "According to Figure 1, what is shown?", ["freeform"], None),
        (CandidatePaper("p1", 1),),
    )

    report, errors = MODULE.inspect_corpus(
        [handoff], _image_store(chunks, tmp_path), image_workers=4
    )
    serial_report, serial_errors = MODULE.inspect_corpus(
        [handoff], _image_store(chunks, tmp_path), image_workers=1
    )

    assert errors == []
    assert (report, errors) == (serial_report, serial_errors)
    assert report["chunk_types"] == {"figure": 1}
    assert report["image_paths"]["declared"] == 1
    assert report["image_paths"]["existing"] == 1
    assert report["image_paths"]["content_sha256"]


def test_preflight_fails_for_missing_candidate_paper(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "other",
                "chunk_id": "other#1",
                "chunk_type": "text_span",
                "text": "text",
                "metadata": {"page": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query("q1", "question", ["freeform"], None),
        (CandidatePaper("missing", 1),),
    )

    report, errors = MODULE.inspect_corpus([handoff], ChunkStore(chunks))

    assert errors
    assert report["missing_candidate_papers"] == ["missing"]


def test_preflight_checks_full_corpus_against_canonical_metadata(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#1",
                "chunk_type": "text_span",
                "text": "text",
                "metadata": {"page": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query("q1", "question", ["freeform"], None),
        (CandidatePaper("p1", 1),),
    )

    report, errors = MODULE.inspect_corpus(
        [handoff], ChunkStore(chunks), {"p1", "p2"}
    )

    assert errors
    assert report["canonical_papers_missing_from_corpus"] == ["p2"]


def test_preflight_rejects_corrupt_image_for_figure_query(tmp_path):
    image = _image_path(tmp_path, "p1", "figure.jpg")
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-an-image")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#figure",
                "chunk_type": "figure",
                "text": "Figure 1",
                "metadata": {
                    "page": 1,
                    "figure_id": "Figure 1",
                    "image_path": str(image),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query("q1", "What does Figure 1 show?", ["freeform"], None),
        (CandidatePaper("p1", 1),),
    )

    report, errors = MODULE.inspect_corpus(
        [handoff], _image_store(chunks, tmp_path), {"p1"}
    )

    assert errors
    assert report["image_paths"]["unreadable"] == 1
    assert report["queries_without_figure_images"] == ["q1"]

    # The per-query gate can be downgraded explicitly, but a corpus whose entire
    # declared image set is unreadable remains a non-overridable root/data error.
    allowed_report, allowed_errors = MODULE.inspect_corpus(
        [handoff],
        _image_store(chunks, tmp_path),
        {"p1"},
        allow_missing_figure_images=True,
    )
    assert len(allowed_errors) == 1
    assert "all declared table/figure images are unavailable" in allowed_errors[0]
    assert not any(
        "explicit visual-reading queries" in error for error in allowed_errors
    )
    assert any(
        "1 declared image files are unreadable or corrupt" in warning
        for warning in allowed_report["warnings"]
    )
    assert allowed_report["allow_missing_figure_images"] is True


def test_preflight_can_explicitly_warn_for_missing_figure_images(tmp_path):
    missing_image = _image_path(tmp_path, "p1", "missing-figure.png")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#figure",
                "chunk_type": "figure",
                "text": "Figure 1 caption and extracted text remain readable.",
                "metadata": {
                    "page": 1,
                    "figure_id": "Figure 1",
                    "image_path": str(missing_image),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query("q1", "What does Figure 1 show?", ["freeform"], None),
        (CandidatePaper("p1", 1),),
    )
    store = _image_store(chunks, tmp_path)

    strict_report, strict_errors = MODULE.inspect_corpus(
        [handoff], store, {"p1"}
    )
    allowed_report, allowed_errors = MODULE.inspect_corpus(
        [handoff],
        store,
        {"p1"},
        allow_missing_figure_images=True,
    )

    assert strict_report["allow_missing_figure_images"] is False
    assert any("explicit visual-reading queries" in error for error in strict_errors)
    assert any(
        "all declared table/figure images are unavailable" in error
        for error in strict_errors
    )
    # The per-query escape hatch cannot hide a globally wrong image root.
    assert len(allowed_errors) == 1
    assert "all declared table/figure images are unavailable" in allowed_errors[0]
    assert allowed_report["queries_without_figure_images"] == ["q1"]
    assert any(
        "allowed by --allow-missing-required-visual-images" in warning
        for warning in allowed_report["warnings"]
    )


def test_preflight_table_answer_type_does_not_require_table_source(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#text",
                "chunk_type": "text_span",
                "text": "The paper reports Alpha=1 and Beta=2.",
                "metadata": {"page": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query(
            "q1",
            "Compile the reported Alpha and Beta values.",
            ["table"],
            [
                {"name": "name", "type": "string", "is_row_key": True},
                {"name": "value", "type": "number", "is_row_key": False},
            ],
        ),
        (CandidatePaper("p1", 1),),
    )

    report, errors = MODULE.inspect_corpus([handoff], ChunkStore(chunks), {"p1"})

    assert errors == []
    assert report["missing_source_hints"] == []
    assert report["visual_image_required_queries"] == []


def test_preflight_source_word_hints_are_warning_only(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#text",
                "chunk_type": "text_span",
                "text": "The extracted prose contains the requested values.",
                "metadata": {"page": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query(
            "q1",
            "Report the values discussed around Table 1 and Equation 2.",
            ["freeform"],
            None,
        ),
        (CandidatePaper("p1", 1),),
    )

    report, errors = MODULE.inspect_corpus([handoff], ChunkStore(chunks), {"p1"})

    assert errors == []
    assert report["missing_source_hints"] == [
        {"query_id": "q1", "source_type": "equation_algorithm"},
        {"query_id": "q1", "source_type": "table"},
    ]
    assert any("diagnostic only" in warning for warning in report["warnings"])
    assert report["queries_without_required_visual_images"] == []


def test_missing_image_root_is_fatal_even_for_nonvisual_query_and_override(
    tmp_path,
):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#table",
                "chunk_type": "table",
                "text": "Table 1 text is still extractable.",
                "metadata": {
                    "page": 1,
                    "table_id": "Table 1",
                    "image_path": str(tmp_path / "wrong-root" / "table.png"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query("q1", "What value does the paper report?", ["freeform"], None),
        (CandidatePaper("p1", 1),),
    )

    report, errors = MODULE.inspect_corpus(
        [handoff],
        ChunkStore(chunks),
        {"p1"},
        allow_missing_figure_images=True,
    )

    assert report["visual_image_required_queries"] == []
    assert len(errors) == 1
    assert "image paths are unsafe" in errors[0]
    assert report["image_paths"]["unsafe"] == 1
    assert "image_root is required" in report["image_paths"]["unsafe_examples"][0][
        "reason"
    ]


def test_preflight_fails_for_image_path_outside_configured_root(tmp_path):
    image_root = tmp_path / "trusted-images"
    image_root.mkdir()
    external = tmp_path / "external.png"
    external.write_bytes(VALID_PNG)
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "chunk_id": "p1#figure",
                "chunk_type": "figure",
                "text": "Figure 1",
                "metadata": {
                    "page": 1,
                    "figure_id": "Figure 1",
                    # No p1/auto/images tail: this must never be retained or sent.
                    "image_path": str(external),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = CandidateHandoff(
        Query("q1", "What value is reported?", ["freeform"], None),
        (CandidatePaper("p1", 1),),
    )

    store = ChunkStore(chunks, image_root=image_root)
    loaded = store.load_paper("p1")[0]
    report, errors = MODULE.inspect_corpus([handoff], store, {"p1"})

    assert loaded["metadata"]["image_path"] == ""
    assert report["image_paths"]["unsafe"] == 1
    assert report["image_paths"]["unsafe_examples"][0]["path"] == str(external)
    assert any("image paths are unsafe" in error for error in errors)


def test_isolated_corrupt_image_is_warning_for_nonvisual_query(tmp_path):
    corrupt = _image_path(tmp_path, "p1", "corrupt.png")
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not-an-image")
    readable = _image_path(tmp_path, "p2", "readable.png")
    readable.parent.mkdir(parents=True)
    readable.write_bytes(VALID_PNG)
    chunks = tmp_path / "chunks.jsonl"
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#table",
            "chunk_type": "table",
            "text": "Table 1",
            "metadata": {
                "page": 1,
                "table_id": "Table 1",
                "image_path": str(corrupt),
            },
        },
        {
            "paper_id": "p2",
            "chunk_id": "p2#table",
            "chunk_type": "table",
            "text": "Table 2",
            "metadata": {
                "page": 2,
                "table_id": "Table 2",
                "image_path": str(readable),
            },
        },
    ]
    chunks.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    handoff = CandidateHandoff(
        Query("q1", "What values do the papers report?", ["freeform"], None),
        (CandidatePaper("p1", 1), CandidatePaper("p2", 2)),
    )

    report, errors = MODULE.inspect_corpus(
        [handoff], _image_store(chunks, tmp_path), {"p1", "p2"}
    )

    assert errors == []
    assert report["image_paths"]["unreadable"] == 1
    assert any("unreadable or corrupt" in warning for warning in report["warnings"])


def test_isolated_missing_visual_image_can_be_explicitly_allowed(tmp_path):
    readable_table = _image_path(tmp_path, "p2", "table.png")
    readable_table.parent.mkdir(parents=True)
    readable_table.write_bytes(VALID_PNG)
    chunks = tmp_path / "chunks.jsonl"
    records = [
        {
            "paper_id": "p1",
            "chunk_id": "p1#figure",
            "chunk_type": "figure",
            "text": "Figure 4 caption",
            "metadata": {
                "page": 2,
                "figure_id": "Figure 4",
                "image_path": str(_image_path(tmp_path, "p1", "missing-figure.png")),
            },
        },
        {
            "paper_id": "p2",
            "chunk_id": "p2#table",
            "chunk_type": "table",
            "text": "Table 1",
            "metadata": {
                "page": 3,
                "table_id": "Table 1",
                "image_path": str(readable_table),
            },
        },
    ]
    chunks.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    handoff = CandidateHandoff(
        Query("q1", "How many panels are shown in Figure 4?", ["freeform"], None),
        (CandidatePaper("p1", 1), CandidatePaper("p2", 2)),
    )

    strict_report, strict_errors = MODULE.inspect_corpus(
        [handoff], _image_store(chunks, tmp_path), {"p1", "p2"}
    )
    allowed_report, allowed_errors = MODULE.inspect_corpus(
        [handoff],
        _image_store(chunks, tmp_path),
        {"p1", "p2"},
        allow_missing_figure_images=True,
    )

    assert strict_report["image_paths"]["existing"] == 1
    assert strict_errors == [
        "1 explicit visual-reading queries have no readable candidate figure/chart image"
    ]
    assert allowed_errors == []
    assert allowed_report["queries_without_required_visual_images"] == ["q1"]


def test_visual_image_requirement_is_conservative():
    required = [
        "According to Figure 4(a), how many curves are shown?",
        "What does the chart show?",
        "How many panels are visible?",
        "What value is visible in the image?",
        "What is the plotted ratio at the lowest difficulty?",
        (
            "Which NAACL 2025 papers explicitly mention or reference MCTS "
            "(Monte Carlo Tree Search) in their primary method/framework figure?"
        ),
    ]
    not_required = [
        "What speedup is reported for 2K image generation?",
        "Compile the reported values as a table answer.",
        "Which image dataset is used for training?",
        "Across these graph-focused works, how many categories are reported?",
        "What value does the paper report?",
        "Which method improves performance in figure generation?",
    ]

    assert all(requires_visual_image(question) for question in required)
    assert not any(requires_visual_image(question) for question in not_required)
