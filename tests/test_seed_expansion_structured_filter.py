"""Venue + modality filtering for enumeration questions."""

from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.retrieve.seed_expansion.structured_filter import (
    StructuredFilterSearch,
    detect_constraint,
)

_CVPR_QUESTION = (
    "Which CVPR 2025 papers cite UniAD and use it as a baseline in their "
    "main comparison table?"
)


def _metadata(tmp_path, records):
    path = tmp_path / "papers.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


class _FakeChunkIndex:
    """Chunk-level index: it searches but exposes no paper document lookup."""

    def __init__(self, hits: list[RetrievalResult]) -> None:
        self.hits = hits
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.queries.append((query, top_k))
        return self.hits[:top_k]


class _FakePaperIndexWithDocuments(_FakeChunkIndex):
    def get_document(self, paper_id: str) -> Chunk | None:
        return None


def _hit(paper_id: str, chunk_type: str = "table") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{paper_id}#c0",
        paper_id=paper_id,
        score=1.0,
        text=f"UniAD row in {paper_id}",
        chunk_type=chunk_type,
        metadata={},
        source="bm25s",
    )


def test_venue_year_and_modality_are_all_detected() -> None:
    constraint = detect_constraint(_CVPR_QUESTION)

    assert constraint is not None
    assert (constraint.venue, constraint.year, constraint.chunk_type) == (
        "CVPR",
        2025,
        "table",
    )


def test_framework_figure_questions_select_the_figure_modality() -> None:
    constraint = detect_constraint(
        "Which NAACL 2025 papers reference MCTS in their primary method figure?"
    )

    assert constraint is not None
    assert constraint.chunk_type == "figure"
    assert constraint.venue == "NAACL"


@pytest.mark.parametrize(
    "question",
    [
        "Which CVPR papers use UniAD in their comparison table?",  # no year
        "Which CVPR 2025 papers cite UniAD?",  # no modality
        "Which 2025 papers show MCTS in a figure?",  # no venue
        "How many subfigures are in Figure 4 of DynaPipe?",  # no venue or year
    ],
)
def test_incomplete_constraints_do_not_trigger_the_lane(question: str) -> None:
    assert detect_constraint(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "For the two ICCV 2025 papers, compare the optimization iterations "
        "Stable Score Distillation reports in their framework figure.",
        "In the two ACL 2025 in-context learning papers that formalize their "
        "setups with recursive generative equations, what symbol is used?",
        "Comparing the two ICCV 2025 papers, one whose framework figure aligns "
        "sample-wise saliencies, which reports the higher score?",
    ],
)
def test_comparisons_over_named_papers_do_not_trigger_the_lane(
    question: str,
) -> None:
    """A question that already fixes its papers keeps its own ranking.

    These state a venue, a year and a modality, so the constraint alone accepts
    them. Promoting a venue-wide shortlist over them dropped papers the
    question names from 3rd to 21st and from 2nd to 22nd on the test set.
    """

    assert detect_constraint(question) is None


def test_enumeration_still_triggers_with_the_same_constraint() -> None:
    assert detect_constraint(
        "Which ICCV 2025 papers show a framework figure of their pipeline?"
    ) is not None


def test_only_matching_venue_and_modality_survive(tmp_path) -> None:
    path = _metadata(
        tmp_path,
        [
            {"paper_id": "cvpr_a", "title": "A", "venue": "CVPR", "year": 2025},
            {"paper_id": "cvpr_b", "title": "B", "venue": "CVPR", "year": 2025},
            {"paper_id": "iccv_c", "title": "C", "venue": "ICCV", "year": 2025},
            {"paper_id": "cvpr_old", "title": "D", "venue": "CVPR", "year": 2024},
        ],
    )
    index = _FakeChunkIndex(
        [
            _hit("iccv_c"),  # wrong venue
            _hit("cvpr_old"),  # wrong year
            _hit("cvpr_a", chunk_type="text_span"),  # wrong modality
            _hit("cvpr_a"),
            _hit("cvpr_b"),
        ]
    )
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    results = stage.candidates(
        _CVPR_QUESTION, [index], exclude_paper_ids=set()
    )

    assert [result.paper_id for result in results] == ["cvpr_a", "cvpr_b"]


def test_results_record_the_constraint_they_matched(tmp_path) -> None:
    path = _metadata(
        tmp_path,
        [{"paper_id": "cvpr_a", "title": "A", "venue": "CVPR", "year": 2025}],
    )
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    result = stage.candidates(
        _CVPR_QUESTION,
        [_FakeChunkIndex([_hit("cvpr_a")])],
        exclude_paper_ids=set(),
    )[0]

    assert result.source == "structured_filter"
    assert result.metadata["structured_filter_venue"] == "CVPR"
    assert result.metadata["structured_filter_year"] == 2025
    assert result.metadata["structured_filter_chunk_type"] == "table"


def test_named_entities_replace_the_sentence_as_the_search_term(tmp_path) -> None:
    """The venue is already enforced, so it must not dilute the term search."""

    path = _metadata(
        tmp_path,
        [{"paper_id": "cvpr_a", "title": "A", "venue": "CVPR", "year": 2025}],
    )
    index = _FakeChunkIndex([_hit("cvpr_a")])
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    stage.candidates(_CVPR_QUESTION, [index], exclude_paper_ids=set())

    query, _ = index.queries[0]
    assert "UniAD" in query
    assert "CVPR" not in query
    assert "papers" not in query


def test_one_paper_appears_once_even_with_several_matching_chunks(tmp_path) -> None:
    path = _metadata(
        tmp_path,
        [{"paper_id": "cvpr_a", "title": "A", "venue": "CVPR", "year": 2025}],
    )
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    results = stage.candidates(
        _CVPR_QUESTION,
        [_FakeChunkIndex([_hit("cvpr_a"), _hit("cvpr_a")])],
        exclude_paper_ids=set(),
    )

    assert len(results) == 1


def test_result_count_is_bounded(tmp_path) -> None:
    path = _metadata(
        tmp_path,
        [
            {"paper_id": f"cvpr_{i}", "title": "A", "venue": "CVPR", "year": 2025}
            for i in range(6)
        ],
    )
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=3,
        search_depth=100,
        seed_text_chars=64,
    )

    results = stage.candidates(
        _CVPR_QUESTION,
        [_FakeChunkIndex([_hit(f"cvpr_{i}") for i in range(6)])],
        exclude_paper_ids=set(),
    )

    assert len(results) == 3


def test_disabled_lane_returns_nothing(tmp_path) -> None:
    stage = StructuredFilterSearch(
        enabled=False,
        metadata_path=str(tmp_path / "missing.jsonl"),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    assert stage.candidates(_CVPR_QUESTION, [], exclude_paper_ids=set()) == []


def test_unreadable_metadata_degrades_quietly(tmp_path) -> None:
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(tmp_path / "missing.jsonl"),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    assert stage.candidates(
        _CVPR_QUESTION,
        [_FakeChunkIndex([_hit("cvpr_a")])],
        exclude_paper_ids=set(),
    ) == []


def test_a_failing_index_does_not_propagate(tmp_path) -> None:
    class _FailingIndex:
        def search(self, query: str, top_k: int):
            raise RuntimeError("index unavailable")

    path = _metadata(
        tmp_path,
        [{"paper_id": "cvpr_a", "title": "A", "venue": "CVPR", "year": 2025}],
    )
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    assert stage.candidates(
        _CVPR_QUESTION, [_FailingIndex()], exclude_paper_ids=set()
    ) == []


def test_the_paper_level_index_is_not_used_for_chunk_search(tmp_path) -> None:
    """Chunk types only exist below the paper level, so the search must be there."""

    path = _metadata(
        tmp_path,
        [{"paper_id": "cvpr_a", "title": "A", "venue": "CVPR", "year": 2025}],
    )
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    assert stage.candidates(
        _CVPR_QUESTION,
        [_FakePaperIndexWithDocuments([_hit("cvpr_a")])],
        exclude_paper_ids=set(),
    ) == []


def test_questions_without_a_constraint_never_search(tmp_path) -> None:
    path = _metadata(
        tmp_path,
        [{"paper_id": "cvpr_a", "title": "A", "venue": "CVPR", "year": 2025}],
    )
    index = _FakeChunkIndex([_hit("cvpr_a")])
    stage = StructuredFilterSearch(
        enabled=True,
        metadata_path=str(path),
        max_papers=10,
        search_depth=100,
        seed_text_chars=64,
    )

    assert stage.candidates(
        "How many subfigures are in Figure 4 of the DynaPipe paper?",
        [index],
        exclude_paper_ids=set(),
    ) == []
    assert index.queries == []
