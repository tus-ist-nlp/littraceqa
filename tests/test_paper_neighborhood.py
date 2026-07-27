"""Tests for strict, gold-free paper-neighborhood reranking."""

from __future__ import annotations

import math

import pytest

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.retrieve.paper_neighborhood import (
    PaperNeighborhoodReranker,
    _conservative_title_alias,
    _title_pattern,
)


def _chunk(
    paper_id: str,
    *,
    title: str | None,
    text: str,
) -> Chunk:
    metadata = {} if title is None else {"title": title}
    return Chunk(
        chunk_id=f"{paper_id}#paper",
        paper_id=paper_id,
        text=text,
        chunk_type="paper",
        metadata=metadata,
    )


def _result(
    paper_id: str,
    *,
    title: str | None,
    score: float = 10.0,
    source: str = "baseline",
) -> RetrievalResult:
    metadata = {"kept": paper_id}
    if title is not None:
        metadata["title"] = title
    return RetrievalResult(
        chunk_id=f"{paper_id}#c0000",
        paper_id=paper_id,
        score=score,
        text=f"representative text for {paper_id}",
        chunk_type="text_span",
        metadata=metadata,
        source=source,
    )


def test_bidirectional_title_mentions_add_a_relation_lane():
    seed_title = "SeedNet: A Structured Method for Scientific Retrieval"
    related_title = "LinkFormer: Cross-Paper Evidence Discovery"
    candidates = [
        _result("seed", title=seed_title),
        _result("distractor", title="Unrelated Baseline for Document Search"),
        _result("related", title=related_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=(
                f"{seed_title}\nWe compare against "
                "LINKFORMER -- cross paper evidence discovery."
            ),
        ),
        "distractor": _chunk(
            "distractor",
            title="Unrelated Baseline for Document Search",
            text="This paper discusses an independent baseline.",
        ),
        "related": _chunk(
            "related",
            title=related_title,
            text=(
                f"{related_title}\nOur analysis extends "
                "seednet / a structured method for scientific retrieval."
            ),
        ),
    }
    reranker = PaperNeighborhoodReranker(documents.get)

    results = reranker.rerank("unused question", candidates, top_k=3)

    assert [result.paper_id for result in results] == [
        "related",
        "seed",
        "distractor",
    ]
    related = results[0]
    assert related.metadata["paper_neighborhood_baseline_rank"] == 3
    assert related.metadata["paper_neighborhood_relation_rank"] == 1
    assert related.metadata["paper_neighborhood_relation_strength"] == 6
    assert related.metadata["kept"] == "related"
    assert related.chunk_id == "related#c0000"
    assert related.text == "representative text for related"
    assert related.chunk_type == "text_span"
    assert related.source == "paper_neighborhood_rrf"
    assert related.score == pytest.approx(1 / 63 + 0.2 / 61)

    seed = results[1]
    assert seed.metadata["paper_neighborhood_baseline_rank"] == 1
    assert seed.metadata["paper_neighborhood_relation_rank"] is None
    assert seed.metadata["paper_neighborhood_relation_strength"] == 0


def test_relation_rank_uses_strength_then_baseline_order():
    seed_title = "SeedModel: Reliable Evaluation for Long Documents"
    weak_title = "WeakLink: Candidate Retrieval with Sparse Features"
    strong_title = "StrongLink: Candidate Retrieval with Explicit Relations"
    candidates = [
        _result("seed", title=seed_title),
        _result("weak", title=weak_title),
        _result("strong", title=strong_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=f"{seed_title}\n{weak_title}\n{strong_title}",
        ),
        "weak": _chunk(
            "weak",
            title=weak_title,
            text="No reverse reference appears here.",
        ),
        "strong": _chunk(
            "strong",
            title=strong_title,
            text=f"We extend {seed_title}.",
        ),
    }

    results = PaperNeighborhoodReranker(documents.get).rerank(
        "query",
        candidates,
        top_k=3,
    )
    by_id = {result.paper_id: result for result in results}

    assert by_id["weak"].metadata["paper_neighborhood_relation_strength"] == 3
    assert by_id["weak"].metadata["paper_neighborhood_relation_rank"] == 2
    assert by_id["strong"].metadata["paper_neighborhood_relation_strength"] == 6
    assert by_id["strong"].metadata["paper_neighborhood_relation_rank"] == 1


def test_two_hop_lane_promotes_a_related_non_direct_candidate():
    seed_title = "SeedNet: Reliable Retrieval for Scientific Documents"
    hub_title = "HubNet: Explicit Links Between Related Scientific Papers"
    target_title = "TargetNet: Recovering Related Work Through Citations"
    candidates = [
        _result("seed", title=seed_title),
        _result(
            "distractor",
            title="DistractorNet: Independent Scientific Document Ranking",
        ),
        _result("target", title=target_title),
        _result("hub", title=hub_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=f"We compare our approach with {hub_title}.",
        ),
        "distractor": _chunk(
            "distractor",
            title="DistractorNet: Independent Scientific Document Ranking",
            text="This document contains no related paper titles.",
        ),
        "target": _chunk(
            "target",
            title=target_title,
            text="This document does not name the seed.",
        ),
        "hub": _chunk(
            "hub",
            title=hub_title,
            text=f"The comparison also includes {target_title}.",
        ),
    }
    reranker = PaperNeighborhoodReranker(
        documents.get,
        two_hop_weight=0.05,
    )

    results = reranker.rerank("query", candidates, top_k=4)
    by_id = {result.paper_id: result for result in results}

    assert [result.paper_id for result in results].index("target") < (
        [result.paper_id for result in results].index("distractor")
    )
    assert by_id["target"].metadata["paper_neighborhood_two_hop_rank"] == 1
    assert (
        by_id["target"].metadata["paper_neighborhood_two_hop_strength"] == 3
    )
    assert (
        by_id["target"].metadata["paper_neighborhood_two_hop_path_count"] == 1
    )
    assert by_id["target"].score == pytest.approx(1 / 63 + 0.05 / 61)


def test_direct_candidate_is_not_boosted_again_by_two_hop_lane():
    seed_title = "SeedNet: Reliable Retrieval for Scientific Documents"
    hub_title = "HubNet: Explicit Links Between Related Scientific Papers"
    direct_title = "DirectNet: A Directly Related Scientific Retrieval Method"
    candidates = [
        _result("seed", title=seed_title),
        _result("direct", title=direct_title),
        _result("hub", title=hub_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=f"We compare {direct_title} and {hub_title}.",
        ),
        "direct": _chunk(
            "direct",
            title=direct_title,
            text="No additional titles appear here.",
        ),
        "hub": _chunk(
            "hub",
            title=hub_title,
            text=f"Our closest comparison is {direct_title}.",
        ),
    }

    results = PaperNeighborhoodReranker(
        documents.get,
        two_hop_weight=0.05,
    ).rerank("query", candidates, top_k=3)
    direct = next(result for result in results if result.paper_id == "direct")

    assert direct.metadata["paper_neighborhood_relation_rank"] == 1
    assert direct.metadata["paper_neighborhood_two_hop_rank"] is None
    assert direct.metadata["paper_neighborhood_two_hop_strength"] == 0
    assert direct.metadata["paper_neighborhood_two_hop_path_count"] == 0
    assert direct.score == pytest.approx(1 / 62 + 0.2 / 61)


def test_two_hop_rejects_a_weak_second_edge():
    seed_title = "SeedNet: Reliable Retrieval for Scientific Documents"
    hub_title = "HubNet: Explicit Links Between Related Scientific Papers"
    target_title = "TargetNet: Recovering Related Work Through Citations"
    candidates = [
        _result("seed", title=seed_title),
        _result("target", title=target_title),
        _result("hub", title=hub_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=f"We compare our approach with {hub_title}.",
        ),
        "target": _chunk(
            "target",
            title=target_title,
            text="No reverse paper-title mention appears here.",
        ),
        "hub": _chunk(
            "hub",
            title=hub_title,
            text="TargetNet is included in one experiment.",
        ),
    }

    results = PaperNeighborhoodReranker(
        documents.get,
        two_hop_weight=0.05,
    ).rerank("query", candidates, top_k=3)
    target = next(result for result in results if result.paper_id == "target")

    assert target.metadata["paper_neighborhood_two_hop_rank"] is None
    assert target.metadata["paper_neighborhood_two_hop_strength"] == 0


def test_two_hop_rejects_a_weak_seed_to_hub_edge():
    seed_title = "SeedNet: Reliable Retrieval for Scientific Documents"
    hub_title = "HubNet: Explicit Links Between Related Scientific Papers"
    target_title = "TargetNet: Recovering Related Work Through Citations"
    candidates = [
        _result("seed", title=seed_title),
        _result("target", title=target_title),
        _result("hub", title=hub_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text="HubNet is included in one experiment.",
        ),
        "target": _chunk(
            "target",
            title=target_title,
            text="No reverse paper-title mention appears here.",
        ),
        "hub": _chunk(
            "hub",
            title=hub_title,
            text=f"The comparison also includes {target_title}.",
        ),
    }

    results = PaperNeighborhoodReranker(
        documents.get,
        two_hop_weight=0.05,
    ).rerank("query", candidates, top_k=3)
    target = next(result for result in results if result.paper_id == "target")

    assert target.metadata["paper_neighborhood_two_hop_rank"] is None
    assert target.metadata["paper_neighborhood_two_hop_strength"] == 0


def test_two_hop_rejects_a_hub_above_the_degree_cap():
    seed_title = "SeedNet: Reliable Retrieval for Scientific Documents"
    hub_title = "HubNet: Explicit Links Between Related Scientific Papers"
    target_title = "TargetNet: Recovering Related Work Through Citations"
    extra_title = "ExtraNet: An Additional Related Scientific Baseline"
    candidates = [
        _result("seed", title=seed_title),
        _result("target", title=target_title),
        _result("extra", title=extra_title),
        _result("hub", title=hub_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=f"We compare our approach with {hub_title}.",
        ),
        "target": _chunk(
            "target",
            title=target_title,
            text="No reverse paper-title mention appears here.",
        ),
        "extra": _chunk(
            "extra",
            title=extra_title,
            text="No reverse paper-title mention appears here.",
        ),
        "hub": _chunk(
            "hub",
            title=hub_title,
            text=f"We compare {target_title} and {extra_title}.",
        ),
    }

    results = PaperNeighborhoodReranker(
        documents.get,
        two_hop_weight=0.05,
        max_hub_degree=2,
    ).rerank("query", candidates, top_k=4)
    by_id = {result.paper_id: result for result in results}

    assert by_id["target"].metadata["paper_neighborhood_two_hop_rank"] is None
    assert by_id["extra"].metadata["paper_neighborhood_two_hop_rank"] is None


def test_two_hop_rank_uses_path_count_after_bottleneck_strength():
    seed_title = "SeedNet: Reliable Retrieval for Scientific Documents"
    one_path_title = "Related Work Reached Through One Scientific Paper"
    two_path_title = "Related Work Reached Through Two Scientific Papers"
    strong_path_title = (
        "StrongPathNet: Related Work With a Strong Explicit Link"
    )
    hub_one_title = "HubOneNet: First Explicit Scientific Relation Hub"
    hub_two_title = "HubTwoNet: Second Explicit Scientific Relation Hub"
    candidates = [
        _result("seed", title=seed_title),
        _result("one_path", title=one_path_title),
        _result("two_path", title=two_path_title),
        _result("strong_path", title=strong_path_title),
        _result("hub_one", title=hub_one_title),
        _result("hub_two", title=hub_two_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=f"We compare {hub_one_title} and {hub_two_title}.",
        ),
        "one_path": _chunk(
            "one_path",
            title=one_path_title,
            text="No reverse paper-title mention appears here.",
        ),
        "two_path": _chunk(
            "two_path",
            title=two_path_title,
            text="No reverse paper-title mention appears here.",
        ),
        "strong_path": _chunk(
            "strong_path",
            title=strong_path_title,
            text="No reverse paper-title mention appears here.",
        ),
        "hub_one": _chunk(
            "hub_one",
            title=hub_one_title,
            text=(
                f"We compare {one_path_title}, {two_path_title}, and "
                f"{strong_path_title}."
            ),
        ),
        "hub_two": _chunk(
            "hub_two",
            title=hub_two_title,
            text=f"We compare only {two_path_title}.",
        ),
    }
    reranker = PaperNeighborhoodReranker(
        documents.get,
        two_hop_weight=0.05,
    )

    first = reranker.rerank("query", candidates, top_k=6)
    second = reranker.rerank("different query", candidates, top_k=6)
    by_id = {result.paper_id: result for result in first}

    assert by_id["strong_path"].metadata[
        "paper_neighborhood_two_hop_rank"
    ] == 1
    assert by_id["strong_path"].metadata[
        "paper_neighborhood_two_hop_strength"
    ] == 3
    assert by_id["two_path"].metadata["paper_neighborhood_two_hop_rank"] == 2
    assert by_id["two_path"].metadata[
        "paper_neighborhood_two_hop_strength"
    ] == 2
    assert by_id["two_path"].metadata[
        "paper_neighborhood_two_hop_path_count"
    ] == 2
    assert by_id["one_path"].metadata["paper_neighborhood_two_hop_rank"] == 3
    assert by_id["one_path"].metadata[
        "paper_neighborhood_two_hop_path_count"
    ] == 1
    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]


def test_zero_two_hop_weight_preserves_direct_results_exactly():
    seed_title = "SeedNet: Reliable Retrieval for Scientific Documents"
    hub_title = "HubNet: Explicit Links Between Related Scientific Papers"
    target_title = "TargetNet: Recovering Related Work Through Citations"
    candidates = [
        _result("seed", title=seed_title),
        _result("target", title=target_title),
        _result("hub", title=hub_title),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title=seed_title,
            text=f"We compare our approach with {hub_title}.",
        ),
        "target": _chunk(
            "target",
            title=target_title,
            text="No reverse paper-title mention appears here.",
        ),
        "hub": _chunk(
            "hub",
            title=hub_title,
            text=f"The comparison also includes {target_title}.",
        ),
    }

    default_results = PaperNeighborhoodReranker(documents.get).rerank(
        "query",
        candidates,
        top_k=3,
    )
    explicit_zero_results = PaperNeighborhoodReranker(
        documents.get,
        two_hop_weight=0.0,
    ).rerank("query", candidates, top_k=3)

    assert [result.to_dict() for result in explicit_zero_results] == [
        result.to_dict() for result in default_results
    ]
    assert all(
        "paper_neighborhood_two_hop_rank" not in result.metadata
        for result in explicit_zero_results
    )


def test_full_title_pattern_normalizes_width_case_and_punctuation():
    pattern = _title_pattern(
        "ＡＢＣNet: Cross-Modal Retrieval",
        min_alnum_chars=20,
    )

    assert pattern is not None
    assert pattern.search("We use abcnet / CROSS modal retrieval.") is not None
    assert pattern.search("xabcnet / cross modal retrieval") is None
    assert pattern.search("abcnet / cross modal retrieval2") is None


def test_short_full_title_does_not_create_a_pattern():
    assert (
        _title_pattern("Short title", min_alnum_chars=20)
        is None
    )


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("EasySpec: Finding Specification Bugs", "EasySpec"),
        ("D-FINE: Redefine Regression", "D-FINE"),
        ("YOLOv12: Attention-Centric Detection", "YOLOv12"),
        ("ACL: Annual Meeting", None),
        ("ordinary words: A Paper", None),
        ("AB: Too Short", None),
        ("No Colon Here", None),
    ],
)
def test_aliases_are_conservative(title, expected):
    assert _conservative_title_alias(title) == expected


@pytest.mark.parametrize("missing_paper_id", ["seed", "candidate"])
def test_missing_document_falls_back_to_original_results(missing_paper_id):
    candidates = [
        _result(
            "seed",
            title="SeedNet: Reliable Retrieval for Scientific Documents",
        ),
        _result(
            "candidate",
            title="LinkNet: Related Evidence Across Scientific Papers",
        ),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title="SeedNet: Reliable Retrieval for Scientific Documents",
            text="LinkNet: Related Evidence Across Scientific Papers",
        ),
        "candidate": _chunk(
            "candidate",
            title="LinkNet: Related Evidence Across Scientific Papers",
            text="SeedNet: Reliable Retrieval for Scientific Documents",
        ),
    }
    documents[missing_paper_id] = None

    results = PaperNeighborhoodReranker(documents.get).rerank(
        "query",
        candidates,
        top_k=2,
    )

    assert results == candidates
    assert all(result.source == "baseline" for result in results)
    assert all(
        "paper_neighborhood_baseline_rank" not in result.metadata
        for result in results
    )


def test_provider_exception_falls_back_to_original_results():
    candidates = [
        _result(
            "seed",
            title="SeedNet: Reliable Retrieval for Scientific Documents",
        )
    ]

    def fail(_paper_id: str) -> Chunk | None:
        raise OSError("unavailable")

    results = PaperNeighborhoodReranker(fail).rerank(
        "query",
        candidates,
        top_k=1,
    )

    assert results == candidates


def test_missing_title_falls_back_without_guessing_from_text():
    candidates = [_result("seed", title=None)]
    documents = {
        "seed": _chunk(
            "seed",
            title=None,
            text="A title-looking first line must not be inferred.",
        )
    }

    results = PaperNeighborhoodReranker(documents.get).rerank(
        "query",
        candidates,
        top_k=1,
    )

    assert results == candidates


def test_provider_is_called_once_per_distinct_paper():
    candidates = [
        _result(
            "seed",
            title="SeedNet: Reliable Retrieval for Scientific Documents",
        ),
        _result(
            "seed",
            title="SeedNet: Reliable Retrieval for Scientific Documents",
        ),
    ]
    document = _chunk(
        "seed",
        title="SeedNet: Reliable Retrieval for Scientific Documents",
        text="A complete paper without a related title.",
    )
    calls: list[str] = []

    def get_document(paper_id: str) -> Chunk:
        calls.append(paper_id)
        return document

    results = PaperNeighborhoodReranker(get_document).rerank(
        "query",
        candidates,
        top_k=2,
    )

    assert calls == ["seed"]
    assert [result.paper_id for result in results] == ["seed"]


def test_results_are_deterministic_and_top_k_is_respected():
    candidates = [
        _result(
            "seed",
            title="SeedNet: Reliable Retrieval for Scientific Documents",
        ),
        _result(
            "other",
            title="OtherNet: Independent Retrieval for Scientific Documents",
        ),
    ]
    documents = {
        "seed": _chunk(
            "seed",
            title="SeedNet: Reliable Retrieval for Scientific Documents",
            text="No explicit related title.",
        ),
        "other": _chunk(
            "other",
            title="OtherNet: Independent Retrieval for Scientific Documents",
            text="No explicit seed title.",
        ),
    }
    reranker = PaperNeighborhoodReranker(documents.get)

    first = reranker.rerank("first query", candidates, top_k=1)
    second = reranker.rerank("different query", candidates, top_k=1)

    assert [result.to_dict() for result in first] == [
        result.to_dict() for result in second
    ]
    assert [result.paper_id for result in first] == ["seed"]


@pytest.mark.parametrize("value", [None, 1, "provider"])
def test_rejects_non_callable_document_provider(value):
    with pytest.raises(TypeError, match="get_document"):
        PaperNeighborhoodReranker(value)


@pytest.mark.parametrize("value", [True, "60", None])
def test_rejects_non_numeric_rrf_k(value):
    with pytest.raises(TypeError, match="rrf_k"):
        PaperNeighborhoodReranker(lambda _paper_id: None, rrf_k=value)


@pytest.mark.parametrize("value", [-1, math.inf, math.nan])
def test_rejects_invalid_rrf_k(value):
    with pytest.raises(ValueError, match="rrf_k"):
        PaperNeighborhoodReranker(lambda _paper_id: None, rrf_k=value)


@pytest.mark.parametrize("value", [True, "0.2", None])
def test_rejects_non_numeric_relation_weight(value):
    with pytest.raises(TypeError, match="relation_weight"):
        PaperNeighborhoodReranker(
            lambda _paper_id: None,
            relation_weight=value,
        )


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_rejects_invalid_relation_weight(value):
    with pytest.raises(ValueError, match="relation_weight"):
        PaperNeighborhoodReranker(
            lambda _paper_id: None,
            relation_weight=value,
        )


@pytest.mark.parametrize("value", [True, "0.05", None])
def test_rejects_non_numeric_two_hop_weight(value):
    with pytest.raises(TypeError, match="two_hop_weight"):
        PaperNeighborhoodReranker(
            lambda _paper_id: None,
            two_hop_weight=value,
        )


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_rejects_invalid_two_hop_weight(value):
    with pytest.raises(ValueError, match="two_hop_weight"):
        PaperNeighborhoodReranker(
            lambda _paper_id: None,
            two_hop_weight=value,
        )


@pytest.mark.parametrize("value", [True, 4.0, "4", None])
def test_rejects_non_integer_max_hub_degree(value):
    with pytest.raises(TypeError, match="max_hub_degree"):
        PaperNeighborhoodReranker(
            lambda _paper_id: None,
            max_hub_degree=value,
        )


@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_max_hub_degree(value):
    with pytest.raises(ValueError, match="max_hub_degree"):
        PaperNeighborhoodReranker(
            lambda _paper_id: None,
            max_hub_degree=value,
        )


def test_non_positive_top_k_returns_no_results_without_loading_documents():
    calls: list[str] = []
    reranker = PaperNeighborhoodReranker(
        lambda paper_id: calls.append(paper_id)
    )

    assert reranker.rerank("query", [_result("p", title="A title")], 0) == []
    assert calls == []
