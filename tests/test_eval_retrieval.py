"""Pure aggregation tests for the retriever-only evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from littraceqa.di_pipeline.evaluation.checkpoint import (
    build_checkpoint,
    load_resume_state,
)
from littraceqa.di_pipeline.evaluation.diagnostics import (
    paper_ranking_details,
    pre_rerank_papers,
    query_diagnostic,
)
from littraceqa.di_pipeline.evaluation.gold import parse_ks, select_records
from littraceqa.di_pipeline.evaluation.metrics import aggregate_rankings
from littraceqa.di_pipeline.evaluation.output import (
    build_output_payload,
    validate_output_path,
    write_output_atomic,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_retrieval import (  # noqa: E402  (CLI-only guards stay in the script)
    validate_retrieval_cutoffs,
    validate_shared_index_load,
)


def test_aggregate_rankings_splits_by_actual_gold_count() -> None:
    rankings = [
        ({"p1"}, ["p1", "n1", "n2"]),
        ({"p9"}, ["n1", "n2", "n3"]),
        ({"p1", "p2"}, ["p1", "n1", "p2"]),
    ]

    metrics = aggregate_rankings(rankings, ks=[1, 3])

    assert metrics["single"][1] == {
        "query_count": 2,
        "recall": 0.5,
        "precision": 0.5,
        "hit_rate": 0.5,
        "all_gold": 0.5,
    }
    assert metrics["multi"][1] == {
        "query_count": 1,
        "recall": 0.5,
        "precision": 1.0,
        "hit_rate": 1.0,
        "all_gold": 0.0,
    }
    assert metrics["total"][1] == {
        "query_count": 3,
        "recall": 0.5,
        "precision": pytest.approx(2 / 3),
        "hit_rate": pytest.approx(2 / 3),
        "all_gold": pytest.approx(1 / 3),
    }
    assert metrics["multi"][3]["recall"] == 1.0
    assert metrics["multi"][3]["precision"] == pytest.approx(2 / 3)
    assert metrics["multi"][3]["all_gold"] == 1.0


def test_aggregate_rankings_reports_empty_group_as_none() -> None:
    metrics = aggregate_rankings([({"p1"}, ["p1"])], ks=[5])

    assert metrics["multi"][5] == {
        "query_count": 0,
        "recall": None,
        "precision": None,
        "hit_rate": None,
        "all_gold": None,
    }


def test_query_without_gold_is_total_only() -> None:
    metrics = aggregate_rankings([(set(), ["p1"])], ks=[1])

    assert metrics["total"][1] == {
        "query_count": 1,
        "recall": 1.0,
        "precision": 0.0,
        "hit_rate": 0.0,
        "all_gold": 1.0,
    }
    assert metrics["single"][1]["query_count"] == 0
    assert metrics["multi"][1]["query_count"] == 0


@pytest.mark.parametrize("ks", [[], [0], [-1, 5]])
def test_aggregate_rankings_rejects_invalid_cutoffs(ks: list[int]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        aggregate_rankings([], ks)


def test_parse_ks_sorts_and_deduplicates() -> None:
    assert parse_ks("20,5,20,1") == (1, 5, 20)


@pytest.mark.parametrize("value", ["", "0", "1,-2", "five"])
def test_parse_ks_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_ks(value)


def test_validate_retrieval_cutoffs_rejects_wrapper_truncation() -> None:
    retriever = {
        "retriever_wrapper": {
            "name": "seed_expansion",
            "params": {"max_results": 10},
        }
    }

    validate_retrieval_cutoffs(retriever, [5, 10])
    with pytest.raises(ValueError, match=r"20.*max_results \(10\)"):
        validate_retrieval_cutoffs(retriever, [5, 20])


def test_validate_retrieval_cutoffs_allows_unbounded_retriever() -> None:
    validate_retrieval_cutoffs({}, [5, 50])


def test_query_diagnostic_reports_each_gold_rank_without_task_family() -> None:
    record = {
        "query_id": "q1",
        "question": "Compare the methods.",
        "gold_papers": [{"paper_id": "p1"}, {"paper_id": "p3"}],
    }

    diagnostic = query_diagnostic(record, ["p1", "p2", "p4"], [1, 3])

    assert diagnostic["scenario"] == "multi"
    assert diagnostic["gold_ranks"] == {"p1": 1, "p3": None}
    assert diagnostic["all_gold_at_k"] == {"1": False, "3": False}


def test_rerank_diagnostics_preserve_scores_and_original_paper_order() -> None:
    results = [
        SimpleNamespace(
            paper_id="p2",
            score=0.9,
            source="qwen3",
            chunk_id="p2#paper",
            chunk_type="paper",
            metadata={"pre_rerank_rank": 2, "pre_rerank_score": 0.4},
        ),
        SimpleNamespace(
            paper_id="p1",
            score=0.8,
            source="qwen3",
            chunk_id="p1#c1",
            chunk_type="text",
            metadata={"pre_rerank_rank": 1, "pre_rerank_score": 0.7},
        ),
    ]

    details = paper_ranking_details(results, max_papers=2)
    original = pre_rerank_papers(results)
    diagnostic = query_diagnostic(
        {
            "query_id": "q1",
            "question": "question",
            "gold_papers": [{"paper_id": "p1"}],
        },
        ["p2", "p1"],
        [1, 2],
        pre_rerank_ranked_papers=original,
        ranking_details=details,
        elapsed_seconds=1.25,
    )

    assert original == ["p1", "p2"]
    assert details == [
        {
            "paper_id": "p2",
            "score": 0.9,
            "source": "qwen3",
            "representative_chunk_id": "p2#paper",
            "chunk_type": "paper",
            "pre_rerank_rank": 2,
            "pre_rerank_score": 0.4,
        },
        {
            "paper_id": "p1",
            "score": 0.8,
            "source": "qwen3",
            "representative_chunk_id": "p1#c1",
            "chunk_type": "text",
            "pre_rerank_rank": 1,
            "pre_rerank_score": 0.7,
        },
    ]
    assert diagnostic["pre_rerank_gold_ranks"] == {"p1": 1}
    assert diagnostic["pre_rerank_all_gold_at_k"] == {"1": True, "2": True}
    assert diagnostic["ranking_details"] == details
    assert diagnostic["elapsed_seconds"] == 1.25


def test_ranking_details_include_typed_final_rerank_provenance() -> None:
    result = SimpleNamespace(
        paper_id="reranked",
        score=0.75,
        source="qwen3",
        chunk_id="reranked#paper",
        chunk_type="paper",
        metadata={
            "qwen3_score": 4.25,
            "qwen3_rank": 2,
            "rank_fusion_base_weight": 0.75,
            "rank_fusion_k": 60.0,
            "final_rerank_status": "applied",
            "final_rerank_candidate_set_preserved": True,
            "final_rerank_error_type": None,
            "attribute_matches": ["venue"],
            "final_rerank_pre_protection_rank": 4,
            "final_rerank_pre_protection_score": 0.625,
            "final_rerank_protected_top_k": 20,
            "final_rerank_prefix_protected": True,
            "open_set_expansion_attempted": True,
            "open_set_expansion_best_rank": 2,
            "open_set_expansion_original_rank": None,
            "open_set_expansion_run_count": 4,
            "open_set_expansion_selected": True,
            "open_set_expansion_selected_paper_id": "reranked",
            "open_set_expansion_slot_k": 20,
            "open_set_expansion_support": 2,
            "open_set_expansion_via_papers": ["seed-2", "seed-4"],
        },
    )

    detail = paper_ranking_details([result])[0]

    assert detail["qwen3_score"] == 4.25
    assert detail["qwen3_rank"] == 2
    assert detail["rank_fusion_base_weight"] == 0.75
    assert detail["rank_fusion_k"] == 60.0
    assert detail["final_rerank_status"] == "applied"
    assert detail["final_rerank_candidate_set_preserved"] is True
    assert detail["final_rerank_error_type"] is None
    assert detail["attribute_matches"] == ["venue"]
    assert detail["final_rerank_pre_protection_rank"] == 4
    assert detail["final_rerank_pre_protection_score"] == 0.625
    assert detail["final_rerank_protected_top_k"] == 20
    assert detail["final_rerank_prefix_protected"] is True
    assert detail["open_set_expansion_attempted"] is True
    assert detail["open_set_expansion_best_rank"] == 2
    assert detail["open_set_expansion_original_rank"] is None
    assert detail["open_set_expansion_run_count"] == 4
    assert detail["open_set_expansion_selected"] is True
    assert detail["open_set_expansion_selected_paper_id"] == "reranked"
    assert detail["open_set_expansion_slot_k"] == 20
    assert detail["open_set_expansion_support"] == 2
    assert detail["open_set_expansion_via_papers"] == ["seed-2", "seed-4"]
    json.dumps(detail)


def test_ranking_details_include_typed_method_provenance_without_text() -> None:
    result = SimpleNamespace(
        paper_id="related",
        score=0.75,
        source="method_relation_rrf",
        chunk_id="related#paper",
        chunk_type="paper",
        text="SECRET DOCUMENT TEXT",
        metadata={
            "method_relation_baseline_rank": 7,
            "method_owner_rank": None,
            "method_relation_rank": 2,
            "method_owner_aliases": ["TCM"],
            "method_relation_aliases": ["TCM", "sCT"],
            "method_relation_via_papers": ["owner"],
            "method_relation_strength": 3,
            "method_topic_rank": 1,
            "method_topic_search_rank": 4,
            "method_topic_via_papers": ["owner"],
            "output_order_rank": 2,
            "pre_output_order_score": 0.8,
        },
    )

    details = paper_ranking_details([result])

    assert details == [
        {
            "paper_id": "related",
            "score": 0.75,
            "source": "method_relation_rrf",
            "representative_chunk_id": "related#paper",
            "chunk_type": "paper",
            "pre_rerank_rank": None,
            "pre_rerank_score": None,
            "method_relation_baseline_rank": 7,
            "method_owner_rank": None,
            "method_relation_rank": 2,
            "method_owner_aliases": ["TCM"],
            "method_relation_aliases": ["TCM", "sCT"],
            "method_relation_via_papers": ["owner"],
            "method_relation_strength": 3,
            "method_topic_rank": 1,
            "method_topic_search_rank": 4,
            "method_topic_via_papers": ["owner"],
            "output_order_rank": 2,
            "pre_output_order_score": 0.8,
        }
    ]
    encoded = json.dumps(details)
    assert "SECRET DOCUMENT TEXT" not in encoded


def test_ranking_details_include_typed_dense_tail_provenance() -> None:
    result = SimpleNamespace(
        paper_id="neighbor",
        score=0.5,
        source="method_dense_tail_rrf",
        chunk_id="neighbor#paper",
        chunk_type="title_abstract",
        metadata={
            "method_dense_tail_baseline_rank": None,
            "method_dense_tail_rank": 2,
            "method_dense_tail_best_neighbor_rank": 3,
            "method_dense_tail_best_similarity": 0.91,
            "method_dense_tail_via_papers": ["owner"],
            "method_dense_tail_rrf_score": 0.04,
            "method_dense_tail_is_new": True,
            "paper_dense_tail_baseline_rank": 7,
            "paper_dense_tail_rank": 4,
            "paper_dense_tail_best_neighbor_rank": 5,
            "paper_dense_tail_best_similarity": 0.89,
            "paper_dense_tail_via_papers": ["rank-one"],
            "paper_dense_tail_rrf_score": 0.03,
            "paper_dense_tail_is_new": False,
            "paper_dense_consensus_support": 2,
            "paper_dense_consensus_best_neighbor_rank": 1,
            "paper_dense_consensus_best_similarity": 0.95,
            "paper_dense_consensus_via_papers": ["seed-two", "seed-three"],
            "paper_dense_consensus_rrf_score": 0.032,
            "paper_dense_consensus_replaced_paper_id": "old-tail",
            "paper_dense_consensus_is_new": True,
        },
    )

    detail = paper_ranking_details([result])[0]

    assert detail["method_dense_tail_baseline_rank"] is None
    assert detail["method_dense_tail_rank"] == 2
    assert detail["method_dense_tail_best_neighbor_rank"] == 3
    assert detail["method_dense_tail_best_similarity"] == 0.91
    assert detail["method_dense_tail_via_papers"] == ["owner"]
    assert detail["method_dense_tail_rrf_score"] == 0.04
    assert detail["method_dense_tail_is_new"] is True
    assert detail["paper_dense_tail_baseline_rank"] == 7
    assert detail["paper_dense_tail_rank"] == 4
    assert detail["paper_dense_tail_best_neighbor_rank"] == 5
    assert detail["paper_dense_tail_best_similarity"] == 0.89
    assert detail["paper_dense_tail_via_papers"] == ["rank-one"]
    assert detail["paper_dense_tail_rrf_score"] == 0.03
    assert detail["paper_dense_tail_is_new"] is False
    assert detail["paper_dense_consensus_support"] == 2
    assert detail["paper_dense_consensus_best_neighbor_rank"] == 1
    assert detail["paper_dense_consensus_best_similarity"] == 0.95
    assert detail["paper_dense_consensus_via_papers"] == [
        "seed-two",
        "seed-three",
    ]
    assert detail["paper_dense_consensus_rrf_score"] == 0.032
    assert detail["paper_dense_consensus_replaced_paper_id"] == "old-tail"
    assert detail["paper_dense_consensus_is_new"] is True
    json.dumps(detail)


def test_ranking_details_include_method_bridge_provenance() -> None:
    result = SimpleNamespace(
        paper_id="linked",
        score=0.25,
        source="method_bridge_exploration",
        chunk_id="linked#paper",
        chunk_type="paper",
        metadata={
            "method_bridge_topic_rank": 4,
            "method_bridge_strength": 1,
            "method_bridge_owner_papers": ["owner"],
            "method_bridge_via_papers": ["bridge"],
            "method_bridge_aliases": ["PAI"],
            "method_bridge_replaced_paper_id": "old-tail",
            "method_bridge_is_new": True,
        },
    )

    detail = paper_ranking_details([result])[0]

    assert detail["method_bridge_topic_rank"] == 4
    assert detail["method_bridge_strength"] == 1
    assert detail["method_bridge_owner_papers"] == ["owner"]
    assert detail["method_bridge_via_papers"] == ["bridge"]
    assert detail["method_bridge_aliases"] == ["PAI"]
    assert detail["method_bridge_replaced_paper_id"] == "old-tail"
    assert detail["method_bridge_is_new"] is True


def test_ranking_details_include_dense_reciprocal_provenance() -> None:
    result = SimpleNamespace(
        paper_id="reciprocal",
        score=0.2,
        source="paper_dense_reciprocal_exploration",
        chunk_id="reciprocal#paper",
        chunk_type="paper",
        metadata={
            "paper_dense_reciprocal_seed_count": 8,
            "paper_dense_reciprocal_discovered_candidates": 28,
            "paper_dense_reciprocal_examined_candidates": 28,
            "paper_dense_reciprocal_support": 6,
            "paper_dense_reciprocal_forward_support": 1,
            "paper_dense_reciprocal_best_forward_rank": 16,
            "paper_dense_reciprocal_best_reverse_rank": 1,
            "paper_dense_reciprocal_best_similarity": 0.96,
            "paper_dense_reciprocal_forward_rrf_score": 0.013,
            "paper_dense_reciprocal_reverse_rrf_score": 0.091,
            "paper_dense_reciprocal_forward_via_papers": ["alpha"],
            "paper_dense_reciprocal_reverse_via_papers": [
                "alpha",
                "owner",
            ],
            "paper_dense_reciprocal_replaced_paper_id": "old-tail",
            "paper_dense_reciprocal_is_new": True,
        },
    )

    detail = paper_ranking_details([result])[0]

    assert detail["paper_dense_reciprocal_seed_count"] == 8
    assert detail["paper_dense_reciprocal_discovered_candidates"] == 28
    assert detail["paper_dense_reciprocal_examined_candidates"] == 28
    assert detail["paper_dense_reciprocal_support"] == 6
    assert detail["paper_dense_reciprocal_forward_support"] == 1
    assert detail["paper_dense_reciprocal_best_forward_rank"] == 16
    assert detail["paper_dense_reciprocal_best_reverse_rank"] == 1
    assert detail["paper_dense_reciprocal_best_similarity"] == 0.96
    assert detail["paper_dense_reciprocal_forward_rrf_score"] == 0.013
    assert detail["paper_dense_reciprocal_reverse_rrf_score"] == 0.091
    assert detail["paper_dense_reciprocal_forward_via_papers"] == ["alpha"]
    assert detail["paper_dense_reciprocal_reverse_via_papers"] == [
        "alpha",
        "owner",
    ]
    assert detail["paper_dense_reciprocal_replaced_paper_id"] == "old-tail"
    assert detail["paper_dense_reciprocal_is_new"] is True
    json.dumps(detail)


def test_ranking_details_fail_closed_for_invalid_method_provenance() -> None:
    result = SimpleNamespace(
        paper_id="invalid",
        score=0.5,
        source="method_relation_rrf",
        chunk_id="invalid#paper",
        chunk_type="paper",
        metadata={
            "method_relation_baseline_rank": True,
            "method_owner_rank": 1.5,
            "method_relation_rank": "2",
            "method_owner_aliases": ("TCM",),
            "method_relation_aliases": ["TCM", 3],
            "method_relation_via_papers": {"owner"},
            "method_relation_strength": False,
            "method_topic_rank": 1.5,
            "method_topic_search_rank": "4",
            "method_topic_via_papers": ("owner",),
            "output_order_rank": 1.5,
            "pre_output_order_score": float("inf"),
            "method_dense_tail_is_new": 1,
            "paper_dense_tail_is_new": "false",
            "paper_dense_consensus_support": True,
            "paper_dense_consensus_via_papers": ("seed-one",),
            "paper_dense_consensus_best_similarity": float("nan"),
            "paper_dense_consensus_is_new": 1,
            "paper_dense_consensus_replaced_paper_id": "",
            "qwen3_score": float("inf"),
            "qwen3_rank": False,
            "rank_fusion_base_weight": "0.75",
            "rank_fusion_k": float("nan"),
            "final_rerank_status": "",
            "final_rerank_candidate_set_preserved": 1,
            "final_rerank_error_type": [],
        },
    )

    detail = paper_ranking_details([result])[0]

    assert detail["method_relation_baseline_rank"] is None
    assert detail["method_owner_rank"] is None
    assert detail["method_relation_rank"] is None
    assert detail["method_owner_aliases"] == []
    assert detail["method_relation_aliases"] == []
    assert detail["method_relation_via_papers"] == []
    assert detail["method_relation_strength"] is None
    assert detail["method_topic_rank"] is None
    assert detail["method_topic_search_rank"] is None
    assert detail["method_topic_via_papers"] == []
    assert detail["output_order_rank"] is None
    assert detail["pre_output_order_score"] is None
    assert detail["method_dense_tail_is_new"] is None
    assert detail["paper_dense_tail_is_new"] is None
    assert detail["paper_dense_consensus_support"] is None
    assert detail["paper_dense_consensus_via_papers"] == []
    assert detail["paper_dense_consensus_best_similarity"] is None
    assert detail["paper_dense_consensus_is_new"] is None
    assert detail["paper_dense_consensus_replaced_paper_id"] is None
    assert detail["qwen3_score"] is None
    assert detail["qwen3_rank"] is None
    assert detail["rank_fusion_base_weight"] is None
    assert detail["rank_fusion_k"] is None
    assert detail["final_rerank_status"] is None
    assert detail["final_rerank_candidate_set_preserved"] is None
    assert detail["final_rerank_error_type"] is None
    json.dumps(detail)


def test_pre_rerank_papers_requires_complete_valid_provenance() -> None:
    missing_rank = SimpleNamespace(paper_id="p1", metadata={})
    invalid_rank = SimpleNamespace(
        paper_id="p1", metadata={"pre_rerank_rank": True}
    )

    assert pre_rerank_papers([]) is None
    assert pre_rerank_papers([missing_rank]) is None
    assert pre_rerank_papers([invalid_rank]) is None


def test_pre_rerank_papers_prefers_the_recorded_full_candidate_pool() -> None:
    results = [
        SimpleNamespace(
            paper_id="p2",
            metadata={
                "pre_rerank_rank": 2,
                "pre_rerank_candidate_papers": ["p1", "p2", "p3"],
            },
        ),
        SimpleNamespace(
            paper_id="p1",
            metadata={"pre_rerank_rank": 1},
        ),
    ]

    assert pre_rerank_papers(results) == ["p1", "p2", "p3"]


def test_select_records_bounds_evaluation_in_explicit_request_order() -> None:
    records = [
        {"query_id": "q_001", "question": "single"},
        {"query_id": "q_037", "question": "multi"},
        {"query_id": "q_038", "question": "multi"},
    ]

    selected = select_records(records, ["q_037", "q_001", "q_037"])

    assert [record["query_id"] for record in selected] == ["q_037", "q_001"]
    assert select_records(records, []) == records


def test_select_records_rejects_unknown_query_id() -> None:
    with pytest.raises(ValueError, match="q_missing"):
        select_records([{"query_id": "q_001"}], ["q_missing"])


@pytest.mark.parametrize(
    "records, message",
    [
        ([{"query_id": ""}], "empty query_id"),
        ([{"query_id": "q1"}, {"query_id": "q1"}], "duplicate query_id"),
    ],
)
def test_select_records_rejects_invalid_query_ids(
    records: list[dict], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_records(records, [])


def _write_queries(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_checkpoint_signature_covers_query_content_and_requested_order(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "queries.jsonl"
    records = [
        {"query_id": "q1", "question": "first"},
        {"query_id": "q2", "question": "second"},
    ]
    _write_queries(query_path, records)

    first = build_checkpoint({"retriever": {"name": "test"}}, [5, 20], query_path, records)
    identical = build_checkpoint(
        {"retriever": {"name": "test"}}, [5, 20], query_path, records
    )
    reversed_order = build_checkpoint(
        {"retriever": {"name": "test"}},
        [5, 20],
        query_path,
        list(reversed(records)),
    )
    _write_queries(
        query_path,
        [
            {"query_id": "q1", "question": "changed"},
            {"query_id": "q2", "question": "second"},
        ],
    )
    changed_content = build_checkpoint(
        {"retriever": {"name": "test"}}, [5, 20], query_path, records
    )

    assert first["run_signature"] == identical["run_signature"]
    assert first["run_signature"] != reversed_order["run_signature"]
    assert first["run_signature"] != changed_content["run_signature"]
    assert first["run_spec"]["runtime"]["python_version"]
    assert "transformers" in first["run_spec"]["runtime"]["packages"]


def test_checkpoint_signature_changes_when_index_state_changes(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    records = [{"query_id": "q1", "question": "first"}]
    _write_queries(query_path, records)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index_file = index_dir / "chunks.jsonl"
    index_file.write_text("{}\n", encoding="utf-8")
    cfg = {
        "retriever": {
            "indexers": [
                {"name": "bm25s", "params": {"index_dir": str(index_dir)}}
            ]
        }
    }

    before = build_checkpoint(cfg, [1], query_path, records)
    index_file.write_text('{"changed": true}\n', encoding="utf-8")
    after = build_checkpoint(cfg, [1], query_path, records)

    assert before["run_signature"] != after["run_signature"]


def test_output_payload_recomputes_metrics_and_tracks_pending_queries() -> None:
    records = [
        {"query_id": "q1"},
        {"query_id": "q2"},
        {"query_id": "q3"},
    ]
    diagnostics = {
        "q2": {
            "query_id": "q2",
            "gold_papers": ["p2"],
            "ranked_papers": ["p2", "n1"],
        }
    }
    failures = {
        "q3": {
            "query_id": "q3",
            "error_type": "RuntimeError",
            "error": "failed",
            "attempts": 1,
        }
    }

    payload = build_output_payload(
        records,
        diagnostics,
        failures,
        {"run_signature": "test"},
        [1, 2],
    )

    assert [item["query_id"] for item in payload["queries"]] == ["q2"]
    assert [item["query_id"] for item in payload["failures"]] == ["q3"]
    assert payload["metrics"]["total"][1]["recall"] == 1.0
    assert payload["summary"] == {
        "requested_query_count": 3,
        "successful_query_count": 1,
        "failed_query_count": 1,
        "pending_query_count": 1,
        "completed": False,
        "metrics_include_successful_queries_only": True,
    }

    pending_without_failures = build_output_payload(
        records,
        diagnostics,
        {},
        {"run_signature": "test"},
        [1],
    )
    assert pending_without_failures["summary"]["pending_query_count"] == 2
    assert pending_without_failures["summary"]["completed"] is False
    assert (
        pending_without_failures["summary"]["metrics_include_successful_queries_only"]
        is True
    )


def test_output_payload_counts_one_final_rerank_status_per_query() -> None:
    records = [{"query_id": "q1"}, {"query_id": "q2"}, {"query_id": "q3"}]
    diagnostics = {
        "q1": {
            "query_id": "q1",
            "gold_papers": ["p1"],
            "ranked_papers": ["p1"],
            "ranking_details": [
                {"paper_id": "p1", "final_rerank_status": "applied"},
                {"paper_id": "p2", "final_rerank_status": "applied"},
            ],
        },
        "q2": {
            "query_id": "q2",
            "gold_papers": ["p2"],
            "ranked_papers": ["p2"],
            "ranking_details": [
                {"paper_id": "p2", "final_rerank_status": "fallback"}
            ],
        },
        "q3": {
            "query_id": "q3",
            "gold_papers": ["p3"],
            "ranked_papers": ["p3"],
            "ranking_details": [{"paper_id": "p3"}],
        },
    }

    payload = build_output_payload(
        records,
        diagnostics,
        {},
        {"run_signature": "test"},
        [1],
    )

    assert payload["summary"]["final_rerank_status_counts"] == {
        "applied": 1,
        "fallback": 1,
    }


def test_resume_restores_successes_and_retryable_failures(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    records = [
        {"query_id": "q1", "question": "first"},
        {"query_id": "q2", "question": "second"},
    ]
    _write_queries(query_path, records)
    checkpoint = build_checkpoint({"retriever": {}}, [1], query_path, records)
    diagnostics = {
        "q1": {
            "query_id": "q1",
            "gold_papers": ["p1"],
            "ranked_papers": ["p1"],
        }
    }
    failures = {
        "q2": {
            "query_id": "q2",
            "error_type": "RuntimeError",
            "error": "temporary failure",
            "attempts": 2,
        }
    }
    output = tmp_path / "nested" / "results.json"
    write_output_atomic(
        output,
        build_output_payload(records, diagnostics, failures, checkpoint, [1]),
    )

    loaded_diagnostics, loaded_failures = load_resume_state(output, checkpoint)

    assert loaded_diagnostics == diagnostics
    assert loaded_failures == failures
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_resume_rejects_incompatible_or_corrupted_checkpoint(tmp_path: Path) -> None:
    query_path = tmp_path / "queries.jsonl"
    records = [{"query_id": "q1", "question": "first"}]
    _write_queries(query_path, records)
    checkpoint = build_checkpoint({"retriever": {}}, [1], query_path, records)
    output = tmp_path / "results.json"

    write_output_atomic(
        output,
        {
            "_checkpoint": {**checkpoint, "run_signature": "different"},
            "queries": [],
            "failures": [],
        },
    )
    with pytest.raises(ValueError, match="different inputs"):
        load_resume_state(output, checkpoint)

    write_output_atomic(
        output,
        {
            "_checkpoint": checkpoint,
            "queries": [],
            "failures": [
                {"query_id": "q1", "attempts": 1},
                {"query_id": "q1", "attempts": 2},
            ],
        },
    )
    with pytest.raises(ValueError, match="duplicate failed"):
        load_resume_state(output, checkpoint)

    output.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_resume_state(output, checkpoint)

    write_output_atomic(
        output,
        {
            "_checkpoint": checkpoint,
            "queries": [],
            "failures": [{"query_id": "q1", "attempts": []}],
        },
    )
    with pytest.raises(ValueError, match="invalid attempts"):
        load_resume_state(output, checkpoint)


def test_evaluation_output_rejects_shared_input(tmp_path) -> None:
    shared = tmp_path / "shared"

    assert validate_output_path(tmp_path / "results.json", shared) == (
        tmp_path / "results.json"
    ).resolve()
    with pytest.raises(ValueError, match="refusing to write"):
        validate_output_path(shared / "metrics.json", shared)


def test_shared_index_load_requires_explicit_opt_in(tmp_path) -> None:
    shared = tmp_path / "shared"
    local = tmp_path / "local" / "index"

    validate_shared_index_load([local], shared, allow=False)
    validate_shared_index_load([shared / "index"], shared, allow=True)
    with pytest.raises(ValueError, match="--allow-shared-index-load"):
        validate_shared_index_load([shared / "index"], shared, allow=False)
