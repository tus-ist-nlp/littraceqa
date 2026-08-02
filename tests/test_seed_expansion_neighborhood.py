"""Paper-neighbourhood reranking and its position in the stage order."""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import (
    Chunk,
    SearchHints,
)
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from seed_expansion_doubles import (
    _FakePaperIndex,
    _FakeReranker,
    _FakeRetriever,
    _result,
)


def test_opt_in_paper_neighborhood_recovers_a_related_candidate():
    seed = _result(
        "seed",
        title="AnchorNet: A Reliable Anchor Method",
    )
    related = _result(
        "related",
        title="Related Method for Scientific Document Retrieval",
    )
    distractors = [
        _result(f"distractor-{index}", title=f"Distractor paper number {index}")
        for index in range(1, 12)
    ]
    candidates = [seed, *distractors, related]
    paper_index = _FakePaperIndex(
        {
            candidate.paper_id: Chunk(
                chunk_id=f"{candidate.paper_id}#paper",
                paper_id=candidate.paper_id,
                text=(
                    "This paper discusses Related Method for Scientific "
                    "Document Retrieval."
                    if candidate.paper_id == "seed"
                    else "Independent full-paper text."
                ),
                chunk_type="paper",
                metadata={"title": candidate.metadata["title"]},
            )
            for candidate in candidates
        }
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        paper_neighborhood_weight=0.2,
    )

    results = retriever.retrieve("What does AnchorNet report?", 10)

    assert "related" in [result.paper_id for result in results]
    assert "related" not in [result.paper_id for result in candidates[:10]]
    assert paper_index.calls


def test_paper_neighborhood_is_disabled_by_default():
    candidates = [_result("seed"), _result("other")]
    paper_index = _FakePaperIndex({})
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        )
    )

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["seed", "other"]
    assert paper_index.calls == []


def test_enabled_paper_neighborhood_without_provider_keeps_ranking():
    candidates = [_result("seed"), _result("other")]
    retriever = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        paper_neighborhood_weight=0.2,
    )

    results = retriever.retrieve("question", 10)

    assert [result.paper_id for result in results] == ["seed", "other"]


def test_method_relations_run_after_paper_neighborhood():
    owner_result = _result(
        "owner",
        title="TCM: Truncated Consistency Models",
    )
    baseline = [owner_result, _result("other", title="Other paper")]
    related = Chunk(
        chunk_id="related#paper",
        paper_id="related",
        text="Related full paper",
        chunk_type="paper",
        metadata={"title": "A Related Consistency Method"},
    )
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Independent owner paper.",
                chunk_type="paper",
                metadata=owner_result.metadata,
            ),
            "other": Chunk(
                chunk_id="other#paper",
                paper_id="other",
                text="Independent other paper.",
                chunk_type="paper",
                metadata=baseline[1].metadata,
            ),
            "related": related,
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["TCM"],
                "strength": 2,
            },
        ),
        neighbors={
            "owner": (
                {
                    "paper_id": "related",
                    "aliases": ["TCM"],
                    "strength": 1,
                },
            )
        },
    )
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [baseline, baseline],
            indexers=[paper_index],
        ),
        paper_neighborhood_weight=0.2,
        method_owner_weight=1.0,
        method_relation_weight=0.5,
    )

    results = retriever.retrieve(
        "question",
        10,
        hints=SearchHints(methods=("TCM",)),
    )

    by_id = {result.paper_id: result for result in results}
    assert "paper_neighborhood_baseline_rank" in by_id["owner"].metadata
    assert by_id["owner"].metadata["method_owner_aliases"] == ["TCM"]
    assert (
        "paper_neighborhood_baseline_rank"
        not in by_id["related"].metadata
    )


def test_neighborhood_runs_before_final_reranker_and_title_guard():
    candidates = [
        _result("named", title="SecEmb: A descriptive subtitle"),
        _result("p2", title="Second paper"),
        _result("p3", title="Third paper"),
        _result("p4", title="Fourth paper"),
    ]
    paper_index = _FakePaperIndex(
        {
            candidate.paper_id: Chunk(
                chunk_id=f"{candidate.paper_id}#paper",
                paper_id=candidate.paper_id,
                text="Independent full-paper text.",
                chunk_type="paper",
                metadata={"title": candidate.metadata["title"]},
            )
            for candidate in candidates
        }
    )
    reranker = _FakeReranker()
    retriever = SeedExpansionRetriever(
        _FakeRetriever(
            [candidates, candidates],
            indexers=[paper_index],
        ),
        reranker=reranker,
        rerank_pool_k=4,
        max_results=2,
        protect_explicit_title_matches=True,
        paper_neighborhood_weight=0.2,
    )

    results = retriever.retrieve("What is reported in paper SecEmb?", 2)

    assert all(
        candidate.source == "paper_neighborhood_rrf"
        for candidate in reranker.calls[0][1]
    )
    assert [result.paper_id for result in results] == ["p4", "named"]
    assert results[-1].metadata["explicit_title_guard_alias"] == "SecEmb"
