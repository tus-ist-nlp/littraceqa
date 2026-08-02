"""Explicit title guard: restoring papers the ranking stages dropped."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from seed_expansion_doubles import (
    _FakeReranker,
    _FakeRetriever,
    _result,
)


@pytest.mark.parametrize(
    ("alias", "question"),
    [
        ("EasySpec", "Which reference is cited in the EasySpec paper?"),
        ("SecEmb", "What is reported in paper ＳｅｃＥｍｂ?"),
        ("PMA", "What accuracy does PMA achieve?"),
        ("D-FINE", "Which setting is used by D-FINE?"),
        ("Model2", "How does Model2 perform?"),
    ],
)
def test_explicit_title_guard_restores_identifier_like_aliases(alias, question):
    candidates = [
        _result("p1", title="Unrelated paper"),
        _result("p2", title="Another unrelated paper"),
        _result("named", title=f"{alias}: A descriptive subtitle"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve(question, 2)

    assert [result.paper_id for result in results] == ["p1", "named"]
    assert results[-1].metadata["explicit_title_guard_alias"] == alias
    assert results[-1].metadata["pre_title_guard_rank"] == 3


def test_explicit_title_guard_keeps_selected_ranking_unchanged():
    candidates = [
        _result("p1", title="Unrelated paper"),
        _result("named", title="SecEmb: A descriptive subtitle"),
        _result("p3", title="Another unrelated paper"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in the SecEmb paper?", 2)

    assert [result.paper_id for result in results] == ["p1", "named"]
    assert "explicit_title_guard_alias" not in results[1].metadata
    assert "pre_title_guard_rank" not in results[1].metadata


def test_explicit_title_guard_restores_candidate_dropped_by_reranker():
    candidates = [
        _result("named", title="SecEmb: A descriptive subtitle"),
        _result("p2", title="Second paper"),
        _result("p3", title="Third paper"),
        _result("p4", title="Fourth paper"),
    ]
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        reranker=reranker,
        rerank_pool_k=4,
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in paper SecEmb?", 2)

    assert [result.paper_id for result in results] == ["p4", "named"]
    assert results[-1].metadata["explicit_title_guard_alias"] == "SecEmb"
    assert results[-1].metadata["pre_title_guard_rank"] == 1
    assert results[0].metadata["pre_rerank_candidate_papers"] == [
        "named",
        "p2",
        "p3",
        "p4",
    ]


def test_explicit_title_guard_ignores_generic_alias():
    candidates = [
        _result("p1", title="First paper"),
        _result("p2", title="Second paper"),
        _result("generic", title="RAG: A descriptive subtitle"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=2,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in the RAG paper?", 2)

    assert [result.paper_id for result in results] == ["p1", "p2"]


def test_explicit_title_guard_ignores_ambiguous_alias():
    candidates = [
        _result("p1", title="First paper"),
        _result("ambiguous-1", title="SecEmb: First interpretation"),
        _result("ambiguous-2", title="SecEmb: Second interpretation"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=1,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("What is reported in paper SecEmb?", 1)

    assert [result.paper_id for result in results] == ["p1"]


def test_explicit_title_guard_requires_standalone_alias_occurrence():
    candidates = [
        _result("p1", title="First paper"),
        _result("named", title="SecEmb: A descriptive subtitle"),
    ]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=1,
        protect_explicit_title_matches=True,
    )

    results = retriever.retrieve("How does SecEmbedding perform?", 1)

    assert [result.paper_id for result in results] == ["p1"]
