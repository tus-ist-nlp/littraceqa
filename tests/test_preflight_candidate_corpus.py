from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from littraceqa.candidate_handoff import CandidateHandoff, CandidatePaper
from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import Query

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight_candidate_corpus.py"
SPEC = importlib.util.spec_from_file_location("preflight_candidate_corpus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_reports_real_locator_and_image_coverage(tmp_path):
    image = tmp_path / "figure.jpg"
    image.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
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

    report, errors = MODULE.inspect_corpus([handoff], ChunkStore(chunks))

    assert errors == []
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
    image = tmp_path / "figure.jpg"
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

    report, errors = MODULE.inspect_corpus([handoff], ChunkStore(chunks), {"p1"})

    assert errors
    assert report["image_paths"]["unreadable"] == 1
    assert report["queries_without_figure_images"] == ["q1"]

    # The fallback is intentionally limited to missing images. A declared but
    # corrupt image remains a fatal, independently reported preflight error.
    allowed_report, allowed_errors = MODULE.inspect_corpus(
        [handoff],
        ChunkStore(chunks),
        {"p1"},
        allow_missing_figure_images=True,
    )
    assert allowed_errors == ["1 declared image files are unreadable or corrupt"]
    assert allowed_report["allow_missing_figure_images"] is True


def test_preflight_can_explicitly_warn_for_missing_figure_images(tmp_path):
    missing_image = tmp_path / "missing-figure.png"
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
    store = ChunkStore(chunks)

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
    assert strict_errors == [
        "1 figure queries have no readable candidate image"
    ]
    assert allowed_errors == []
    assert allowed_report["queries_without_figure_images"] == ["q1"]
    assert any(
        "allowed by --allow-missing-figure-images" in warning
        for warning in allowed_report["warnings"]
    )
