"""Open-set enumeration lane: the single guarded exploration slot."""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from littraceqa.di_pipeline.retrieve.seed_expansion.candidates import (
    OpenSetExploration,
)
from seed_expansion_doubles import (
    _FakePaperIndex,
    _FakeRetriever,
    _PushPaperLastReranker,
    _result,
)


def test_open_set_exploration_preserves_top_nineteen_and_fills_slot_twenty():
    baseline = [
        _result(f"p{rank:02d}", score=float(30 - rank))
        for rank in range(1, 26)
    ]
    selected = _result("new-paper", score=99.0)
    stage = OpenSetExploration(
        min_support=2,
        max_seed_rank=2,
        slot_k=20,
    )

    results = stage.insert(
        baseline,
        [
            ("seed-2", [selected]),
            ("seed-3", [_result("other"), selected]),
            ("seed-4", [_result("unrelated")]),
        ],
    )

    assert [result.paper_id for result in results[:19]] == [
        result.paper_id for result in baseline[:19]
    ]
    assert results[19].paper_id == "new-paper"
    assert results[20].paper_id == "p20"
    assert len(results) == len(baseline)
    assert results[19].metadata["open_set_expansion_support"] == 2
    assert results[19].metadata["open_set_expansion_best_rank"] == 1
    assert results[19].metadata["open_set_expansion_original_rank"] is None
    assert results[19].metadata["open_set_expansion_via_papers"] == [
        "seed-2",
        "seed-3",
    ]


def test_open_set_exploration_does_not_select_weak_consensus():
    baseline = [_result(f"p{rank:02d}") for rank in range(1, 21)]
    stage = OpenSetExploration(
        min_support=2,
        max_seed_rank=2,
        slot_k=20,
    )

    results = stage.insert(
        baseline,
        [
            ("seed-2", [_result("one-run-only")]),
            (
                "seed-3",
                [_result("r1"), _result("r2"), _result("too-low")],
            ),
            (
                "seed-4",
                [_result("r3"), _result("r4"), _result("too-low")],
            ),
        ],
    )

    assert [result.paper_id for result in results] == [
        result.paper_id for result in baseline
    ]
    assert results[0].metadata["open_set_expansion_attempted"] is True
    assert results[0].metadata["open_set_expansion_selected_paper_id"] is None


def test_open_set_exploration_uses_the_best_ranked_representative():
    baseline = [_result(f"p{rank:02d}") for rank in range(1, 21)]
    stage = OpenSetExploration(
        min_support=2,
        max_seed_rank=2,
        slot_k=20,
    )

    results = stage.insert(
        baseline,
        [
            (
                "seed-2",
                [
                    _result("other"),
                    _result("candidate", title="Lower-ranked copy"),
                ],
            ),
            ("seed-3", [_result("candidate", title="Best-ranked copy")]),
        ],
    )

    assert results[19].paper_id == "candidate"
    assert results[19].metadata["title"] == "Best-ranked copy"


def test_open_set_expansion_is_inert_for_normal_queries_when_enabled():
    baseline = [_result(f"p{rank:02d}") for rank in range(1, 21)]
    inner = _FakeRetriever([baseline, baseline])
    retriever = SeedExpansionRetriever(
        inner,
        candidate_k=20,
        max_results=20,
        open_set_seed_k=5,
        open_set_min_support=2,
        open_set_max_seed_rank=2,
        open_set_slot_k=20,
    )

    results = retriever.retrieve(
        "What scores do TCM, sCT, ECM-XL, and IMM report?",
        20,
    )

    assert len(inner.calls) == 2
    assert [result.paper_id for result in results] == [
        result.paper_id for result in baseline
    ]


def test_open_set_seed_searches_are_bounded_and_fail_soft():
    initial = [
        _result(
            f"p{rank:02d}",
            title=f"Seed {rank}",
            text=f"Seed text {rank}",
        )
        for rank in range(1, 21)
    ]
    consensus = _result("consensus")
    inner = _FakeRetriever(
        [
            initial,
            initial,
            [],
            [consensus],
            [consensus],
            [],
        ],
        fail_on_calls={3},
    )
    retriever = SeedExpansionRetriever(
        inner,
        candidate_k=20,
        max_results=20,
        open_set_seed_k=5,
        open_set_min_support=2,
        open_set_max_seed_rank=2,
        open_set_slot_k=20,
    )

    results = retriever.retrieve(
        "Which conference papers propose a scaling method?",
        20,
    )

    assert len(inner.calls) == 6
    assert results[19].paper_id == "consensus"
    assert results[19].metadata["open_set_expansion_run_count"] == 3
    assert results[19].metadata["open_set_expansion_support"] == 2


def test_open_set_candidate_is_inserted_after_final_qwen_reranking():
    baseline = [
        _result(
            f"p{rank:02d}",
            title=f"Seed {rank}",
            text=f"Seed text {rank}",
        )
        for rank in range(1, 26)
    ]
    consensus = _result("consensus")
    documents = {
        candidate.paper_id: Chunk(
            chunk_id=f"{candidate.paper_id}#paper",
            paper_id=candidate.paper_id,
            text=f"Paper document for {candidate.paper_id}",
            chunk_type="paper",
            metadata={},
        )
        for candidate in [*baseline, consensus]
    }
    final_reranker = _PushPaperLastReranker("consensus")
    inner = _FakeRetriever(
        [
            baseline,
            baseline,
            [consensus],
            [consensus],
            [],
            [],
        ],
        indexers=[_FakePaperIndex(documents)],
    )
    retriever = SeedExpansionRetriever(
        inner,
        candidate_k=25,
        max_results=25,
        reranker=final_reranker,
        rerank_pool_k=25,
        rerank_final_candidates=True,
        final_rerank_protected_top_k=20,
        open_set_seed_k=5,
        open_set_min_support=2,
        open_set_max_seed_rank=2,
        open_set_slot_k=20,
    )

    results = retriever.retrieve(
        "Which conference papers propose a scaling method?",
        25,
    )

    assert all(
        candidate.paper_id != "consensus"
        for candidate in final_reranker.calls[0][1]
    )
    assert [result.paper_id for result in results[:19]] == [
        result.paper_id for result in baseline[:19]
    ]
    assert results[19].paper_id == "consensus"
    assert results[19].metadata["open_set_expansion_selected"] is True
