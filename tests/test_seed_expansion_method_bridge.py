"""Method bridge: the one final slot spent on a method-connected paper."""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.retrieve.seed_expansion import (
    SeedExpansionRetriever,
)
from seed_expansion_doubles import (
    _FakePaperIndex,
    _FakeRetriever,
    _result,
)


def test_method_bridge_replaces_only_final_slot_with_dual_support():
    candidates = [
        _result("owner"),
        *[_result(f"p{index}") for index in range(2, 21)],
    ]
    linked_document = Chunk(
        chunk_id="linked#paper",
        paper_id="linked",
        text="Linked paper text",
        chunk_type="paper",
        metadata={"title": "Linked paper"},
    )
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner topic document",
                chunk_type="paper",
                metadata={"title": "Owner paper"},
            ),
            "linked": linked_document,
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["OWNER"],
                "strength": 2,
            },
        ),
        neighbors={
            "p11": (
                {
                    "paper_id": "linked",
                    "aliases": ["LINK"],
                    "strength": 1,
                },
            ),
        },
    )
    inner = _FakeRetriever(
        [
            candidates,
            candidates,
            [_result("p2"), _result("linked")],
        ],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        max_results=20,
        stable_prefix_k=10,
        literal_attribute_hints=True,
        literal_method_hints=True,
        method_topic_seed_chars=100,
        method_topic_max_results=10,
        method_bridge_topic_max_rank=2,
    )

    results = retriever.retrieve(
        "In the OWNER paper, what is reported?",
        20,
    )

    assert [result.paper_id for result in results[:-1]] == [
        result.paper_id for result in candidates[:-1]
    ]
    assert results[-1].paper_id == "linked"
    assert results[-1].source == "method_bridge_exploration"
    assert results[-1].metadata == {
        "title": "Linked paper",
        "method_bridge_topic_rank": 2,
        "method_bridge_owner_papers": ["owner"],
        "method_bridge_via_papers": ["p11"],
        "method_bridge_aliases": ["LINK"],
        "method_bridge_strength": 1,
        "method_bridge_replaced_paper_id": "p20",
        "method_bridge_is_new": True,
    }


def test_method_bridge_requires_owner_topic_support():
    candidates = [
        _result("owner"),
        *[_result(f"p{index}") for index in range(2, 21)],
    ]
    paper_index = _FakePaperIndex(
        {
            "owner": Chunk(
                chunk_id="owner#paper",
                paper_id="owner",
                text="Owner topic document",
                chunk_type="paper",
                metadata={},
            ),
        },
        owners=(
            {
                "paper_id": "owner",
                "aliases": ["OWNER"],
                "strength": 2,
            },
        ),
        neighbors={
            "p11": (
                {
                    "paper_id": "unsupported",
                    "aliases": ["METHOD"],
                    "strength": 1,
                },
            ),
        },
    )
    inner = _FakeRetriever(
        [
            candidates,
            candidates,
            [_result("different")],
        ],
        indexers=[paper_index],
    )
    retriever = SeedExpansionRetriever(
        inner,
        max_results=20,
        stable_prefix_k=10,
        literal_attribute_hints=True,
        literal_method_hints=True,
        method_topic_max_results=10,
        method_bridge_topic_max_rank=5,
    )
    baseline = SeedExpansionRetriever(
        _FakeRetriever([candidates, candidates]),
        max_results=20,
        stable_prefix_k=10,
    ).retrieve("In the OWNER paper, what is reported?", 20)

    results = retriever.retrieve(
        "In the OWNER paper, what is reported?",
        20,
    )

    assert [result.to_dict() for result in results] == [
        result.to_dict() for result in baseline
    ]
