"""Seed query preparation: hint extraction, seed text and expanded queries."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import SearchHints
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.query import (
    is_open_set_enumeration,
)
from seed_expansion_doubles import (
    _FakeReranker,
    _FakeRetriever,
    _result,
)


@pytest.mark.parametrize(
    "query",
    [
        "Which NAACL 2025 papers explicitly mention MCTS?",
        (
            "Among ICML 2025 papers, what is each proposed method "
            "and what objective does each optimize?"
        ),
        (
            "Across all venues, among 2025 scaling methods, "
            "what base model does each method build on?"
        ),
    ],
)
def test_detects_open_set_enumeration_queries(query):
    assert is_open_set_enumeration(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "Which paper achieves the highest driving score?",
        "What scores do TCM, sCT, ECM-XL, and IMM report?",
        (
            "What are the scores for TCM, ECM-XL, iCT-deep, and SiD "
            "as reported in their respective papers?"
        ),
        (
            "What are the accuracies for MoST, PMA, PointLoRA, and "
            "RISurConv as reported in their respective papers?"
        ),
        "What values do sCM and IMM use, and do they match?",
        "How many papers cite the method?",
        "",
    ],
)
def test_open_set_enumeration_gate_is_conservative(query):
    assert is_open_set_enumeration(query) is False


def test_expands_from_top_result_and_forwards_hints():
    representative_text = f"{'x' * 300}\n\n{'y' * 300}"
    inner = _FakeRetriever(
        [
            [_result("p1", title="A paper", text=representative_text)],
            [_result("p2")],
        ]
    )
    retriever = SeedExpansionRetriever(inner)
    hints = SearchHints(venues=("ACL",), years=(2024,))

    retriever.retrieve("original question", 10, hints=hints)

    assert len(inner.calls) == 2
    assert inner.calls[0] == ("original question", 50, hints)
    assert inner.calls[1] == (
        f"original question A paper {'x' * 300} {'y' * 211}",
        50,
        hints,
    )


def test_prefers_paper_context_and_does_not_duplicate_leading_title():
    seed = _result("p1", title="A paper", text="Query-matched table")
    seed.metadata["paper_rank_expansion_text"] = (
        "A paper\nUseful abstract and introduction"
    )
    seed.metadata["paper_rank_expansion_source"] = "paper_bm25"
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(inner)

    results = retriever.retrieve("original question", 10)

    assert len(inner.calls) == 2
    assert inner.calls[1][0] == (
        "original question A paper Useful abstract and introduction"
    )
    assert "Query-matched table" not in inner.calls[1][0]
    assert all(
        "seed_expansion_local_rank" not in result.metadata
        for result in results
    )


def test_opt_in_local_expansion_uses_legacy_seed_query_and_forwards_hints():
    seed = _result(
        "p1",
        title="A paper",
        text="Query-matched table details",
    )
    seed.metadata["paper_rank_expansion_text"] = "A paper\nUseful abstract"
    inner = _FakeRetriever(
        [
            [seed],
            [_result("p2")],
            [_result("p3")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        seed_text_chars=19,
        local_expansion_weight=0.5,
    )
    hints = SearchHints(venues=("ACL",), years=(2024,))

    retriever.retrieve("original question", 10, hints=hints)

    assert len(inner.calls) == 3
    assert inner.calls[1] == (
        "original question A paper Useful abst",
        50,
        hints,
    )
    assert inner.calls[2] == (
        "original question A paper Query-matched table",
        50,
        hints,
    )


def test_opt_in_literal_attribute_hints_are_forwarded_to_every_lane():
    seed = _result("p1", title="A paper", text="Local evidence")
    seed.metadata["paper_rank_expansion_text"] = "Paper overview"
    inner = _FakeRetriever(
        [
            [seed],
            [_result("p2")],
            [_result("p3")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        local_expansion_weight=0.5,
        literal_attribute_hints=True,
    )

    retriever.retrieve(
        "Which CVPR 2025 papers cite UniAD "
        "(Planning-oriented Autonomous Driving, CVPR2023)?",
        10,
    )

    expected = SearchHints(venues=("CVPR",), years=(2025,))
    assert len(inner.calls) == 3
    assert all(hints == expected for _, _, hints in inner.calls)


def test_opt_in_target_method_hints_are_deferred_until_final_ranking():
    seed = _result("p1", title="A paper", text="Local evidence")
    seed.metadata["paper_rank_expansion_text"] = "Paper overview"
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(
        inner,
        literal_attribute_hints=True,
        literal_method_hints=True,
    )

    retriever.retrieve("In the TCM paper, what is the batch size?", 10)

    expected = SearchHints()
    assert len(inner.calls) == 2
    assert all(hints == expected for _, _, hints in inner.calls)


@pytest.mark.parametrize(
    "provided",
    [
        SearchHints(),
        SearchHints(venues=("ACL",), years=(2024,)),
        SearchHints(
            venues=("ACL",),
            years=(2024,),
            methods=("TCM",),
        ),
    ],
)
def test_caller_hints_take_precedence_over_literal_extraction(provided):
    inner = _FakeRetriever(
        [
            [_result("p1")],
            [_result("p2")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        literal_attribute_hints=True,
    )

    retriever.retrieve("Which CVPR 2025 papers are relevant?", 10, hints=provided)

    assert len(inner.calls) == 2
    expected = SearchHints(
        venues=provided.venues,
        years=provided.years,
    )
    assert all(hints == expected for _, _, hints in inner.calls)


def test_default_does_not_extract_literal_attribute_hints():
    inner = _FakeRetriever(
        [
            [_result("p1")],
            [_result("p2")],
        ]
    )
    retriever = SeedExpansionRetriever(inner)

    retriever.retrieve("Which CVPR 2025 papers are relevant?", 10)

    assert len(inner.calls) == 2
    assert all(hints is None for _, _, hints in inner.calls)


def test_local_expansion_still_runs_when_paper_context_search_is_empty():
    seed = _result("p1", title="A paper", text="Local evidence")
    seed.metadata["paper_rank_expansion_text"] = "Paper overview"
    inner = _FakeRetriever(
        [
            [seed],
            [],
            [_result("local")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        local_expansion_weight=0.5,
    )

    results = retriever.retrieve("question", 10)

    assert len(inner.calls) == 3
    assert [result.paper_id for result in results] == ["p1", "local"]
    assert results[0].metadata["seed_expansion_expanded_rank"] is None
    assert results[1].metadata["seed_expansion_local_rank"] == 1


def test_local_expansion_is_not_duplicated_without_paper_context():
    initial = [_result("p1", title="A paper", text="Local evidence")]
    inner = _FakeRetriever(
        [
            initial,
            [_result("p2")],
            [_result("unexpected")],
        ]
    )
    retriever = SeedExpansionRetriever(
        inner,
        local_expansion_weight=0.5,
    )

    results = retriever.retrieve("question", 10)

    assert len(inner.calls) == 2
    assert inner.calls[1][0] == "question A paper Local evidence"
    assert all(
        "seed_expansion_local_rank" not in result.metadata
        for result in results
    )


def test_paper_context_keeps_title_when_context_does_not_begin_with_it():
    seed = _result("p1", title="A paper", text="Query-matched table")
    seed.metadata["paper_rank_expansion_text"] = "Useful abstract"
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(inner)

    retriever.retrieve("original question", 10)

    assert inner.calls[1][0] == "original question A paper Useful abstract"


def test_empty_paper_context_falls_back_to_legacy_seed_text():
    seed = _result("p1", title="A paper", text="Query-matched table")
    seed.metadata["paper_rank_expansion_text"] = "  "
    inner = _FakeRetriever([[seed], [_result("p2")]])
    retriever = SeedExpansionRetriever(inner)

    retriever.retrieve("original question", 10)

    assert inner.calls[1][0] == "original question A paper Query-matched table"


@pytest.mark.parametrize(
    ("title", "text"),
    [
        ("", "Representative text"),
        ("Seed title", ""),
    ],
)
def test_missing_seed_content_falls_back_without_second_retrieval(title, text):
    initial = [
        _result("p1", title=title, text=text, chunk_id="p1#c0000"),
        _result("p1", title="Duplicate", chunk_id="p1#c0001"),
        _result("p2"),
    ]
    inner = _FakeRetriever([initial])
    retriever = SeedExpansionRetriever(inner)

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["p1", "p2"]
    assert len(inner.calls) == 1
    assert results[0] is initial[0]


def test_final_reranker_also_applies_when_seed_content_is_missing():
    initial = [
        _result("p1", title="", text="Representative text"),
        _result("p2"),
        _result("p3"),
    ]
    inner = _FakeRetriever([initial])
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        inner,
        reranker=reranker,
        rerank_pool_k=2,
    )

    results = retriever.retrieve("question", 2)

    assert len(inner.calls) == 1
    assert len(reranker.calls) == 1
    assert [candidate.paper_id for candidate in reranker.calls[0][1]] == [
        "p1",
        "p2",
    ]
    assert [result.paper_id for result in results] == ["p2", "p1"]
