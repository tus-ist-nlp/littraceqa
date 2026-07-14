"""Tests for the gold-isolated ranking evaluation command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import evaluate_rankings
from evaluate_retrieval_rankings import (
    build_evaluation_manifest,
    build_comparison_summary,
    build_question_comparison,
    discover_ranking_files,
    write_evaluation_outputs,
)


def _evaluation(ranked_ids: list[str]) -> dict:
    gold = [
        {
            "query_id": "q1",
            "gold_papers": [{"paper_id": "p1"}],
        }
    ]
    rankings = [
        {
            "query_id": "q1",
            "papers": [{"paper_id": paper_id} for paper_id in ranked_ids],
        }
    ]
    return evaluate_rankings(gold, rankings)


def test_comparison_reports_paired_deltas_and_question_details():
    evaluations = {
        "baseline": _evaluation(["other", "p1"]),
        "better": _evaluation(["p1", "other"]),
    }

    comparison = build_comparison_summary(evaluations, baseline="baseline")
    questions = build_question_comparison(evaluations)

    assert comparison["primary_metric"] == "paper_f1_at_5_macro"
    assert comparison["methods"]["better"]["delta_from_baseline"]["mrr"] == 0.5
    assert comparison["methods"]["better"]["paired_query_f1_at_5"] == {
        "improved": 0,
        "worsened": 0,
        "unchanged": 1,
    }
    assert comparison["methods"]["better"][
        "paired_query_f1_at_5_query_ids"
    ] == {"improved": [], "worsened": [], "unchanged": ["q1"]}
    assert "paper_precision_at_20_macro" in comparison["methods"]["better"][
        "delta_from_baseline"
    ]
    assert questions[0]["gold_paper_ids"] == ["p1"]
    assert questions[0]["methods"]["better"]["ranked_paper_ids"][0] == "p1"
    assert questions[0]["methods"]["better"]["ranking_missing"] is False


def test_comparison_lists_improved_and_worsened_query_ids():
    gold = [
        {"query_id": "improved", "gold_papers": [{"paper_id": "p1"}]},
        {"query_id": "worsened", "gold_papers": [{"paper_id": "p2"}]},
    ]
    baseline = evaluate_rankings(
        gold,
        [
            {"query_id": "improved", "papers": []},
            {"query_id": "worsened", "papers": [{"paper_id": "p2"}]},
        ],
    )
    candidate = evaluate_rankings(
        gold,
        [
            {"query_id": "improved", "papers": [{"paper_id": "p1"}]},
            {"query_id": "worsened", "papers": []},
        ],
    )

    method = build_comparison_summary(
        {"baseline": baseline, "candidate": candidate}, baseline="baseline"
    )["methods"]["candidate"]

    assert method["paired_query_f1_at_5"] == {
        "improved": 1,
        "worsened": 1,
        "unchanged": 0,
    }
    assert method["paired_query_f1_at_5_query_ids"] == {
        "improved": ["improved"],
        "worsened": ["worsened"],
        "unchanged": [],
    }


def test_comparison_treats_missing_ranking_records_as_empty_and_sorts_ids():
    gold = [
        {"query_id": "z-gained", "gold_papers": [{"paper_id": "pz"}]},
        {"query_id": "a-gained", "gold_papers": [{"paper_id": "pa"}]},
        {"query_id": "m-lost", "gold_papers": [{"paper_id": "pm"}]},
    ]
    baseline = evaluate_rankings(
        gold,
        [{"query_id": "m-lost", "papers": [{"paper_id": "pm"}]}],
    )
    candidate = evaluate_rankings(
        gold,
        [
            {"query_id": "z-gained", "papers": [{"paper_id": "pz"}]},
            {"query_id": "a-gained", "papers": [{"paper_id": "pa"}]},
        ],
    )

    method = build_comparison_summary(
        {"candidate": candidate, "baseline": baseline}, baseline="baseline"
    )["methods"]["candidate"]

    assert method["paired_query_f1_at_5_query_ids"] == {
        "improved": ["a-gained", "z-gained"],
        "worsened": ["m-lost"],
        "unchanged": [],
    }
    assert method["paired_all_gold_recovery"]["at_5"]["overall"][
        "query_ids"
    ] == {
        "gained": ["a-gained", "z-gained"],
        "lost": ["m-lost"],
        "retained": [],
        "neither": [],
    }
    assert next(
        query for query in candidate["queries"] if query["query_id"] == "m-lost"
    )["ranking_missing"] is True


def test_comparison_deduplicates_gold_and_ranked_papers_at_fixed_cutoffs():
    gold = [
        {
            "query_id": "duplicate-papers",
            "gold_papers": [{"paper_id": "p1"}, {"paper_id": "p1"}],
        }
    ]
    baseline = evaluate_rankings(
        gold,
        [
            {
                "query_id": "duplicate-papers",
                "papers": [
                    {"paper_id": "other"},
                    {"paper_id": "other"},
                    {"paper_id": "p1"},
                    {"paper_id": "p1"},
                ],
            }
        ],
    )
    candidate = evaluate_rankings(
        gold,
        [{"query_id": "duplicate-papers", "papers": [{"paper_id": "p1"}]}],
    )

    comparison = build_comparison_summary(
        {"baseline": baseline, "candidate": candidate}, baseline="baseline"
    )
    baseline_query = baseline["queries"][0]
    rank_diagnostic = comparison["methods"]["candidate"][
        "capped_relevant_paper_rank_at_20"
    ]

    assert baseline_query["gold_paper_count"] == 1
    assert baseline_query["ranked_paper_ids"] == ["other", "p1"]
    assert baseline_query["metrics"]["paper_precision_at_20"] == 1 / 20
    assert baseline_query["metrics"]["all_gold_at_20"] is True
    assert baseline_query["metrics"]["reciprocal_rank"] == 0.5
    assert rank_diagnostic["overall"]["relevant_paper_count"] == 1
    assert rank_diagnostic["overall"]["baseline_mean_rank"] == 2.0
    assert rank_diagnostic["overall"]["method_mean_rank"] == 1.0
    assert rank_diagnostic["overall"]["mean_rank_change"] == 1.0


def test_comparison_excludes_zero_gold_queries_from_metrics_and_pairs():
    gold = [
        {"query_id": "no-gold", "gold_papers": []},
        {"query_id": "scored", "gold_papers": [{"paper_id": "p1"}]},
    ]
    baseline = evaluate_rankings(gold, [])
    candidate = evaluate_rankings(
        gold,
        [
            {"query_id": "no-gold", "papers": [{"paper_id": "unjudged"}]},
            {"query_id": "scored", "papers": [{"paper_id": "p1"}]},
        ],
    )

    method = build_comparison_summary(
        {"baseline": baseline, "candidate": candidate}, baseline="baseline"
    )["methods"]["candidate"]

    assert baseline["metrics"]["count"] == 1
    assert candidate["metrics"]["count"] == 1
    assert method["delta_from_baseline"]["paper_recall_at_5_macro"] == 1.0
    assert method["paired_query_recall"]["at_5"]["overall"]["counts"] == {
        "improved": 1,
        "worsened": 0,
        "unchanged": 0,
    }
    assert method["paired_query_recall"]["at_5"]["overall"]["query_ids"] == {
        "improved": ["scored"],
        "worsened": [],
        "unchanged": [],
    }
    assert method["capped_relevant_paper_rank_at_20"]["overall"][
        "relevant_paper_count"
    ] == 1


def test_comparison_rejects_unpaired_evaluation_query_sets():
    baseline = _evaluation(["p1"])
    candidate = _evaluation(["p1"])
    candidate["queries"] = []

    with pytest.raises(ValueError, match=r"missing query IDs: \['q1'\]"):
        build_comparison_summary(
            {"baseline": baseline, "candidate": candidate}, baseline="baseline"
        )


def test_comparison_rejects_different_gold_papers_for_the_same_query():
    baseline = _evaluation(["p1"])
    candidate = _evaluation(["p1"])
    candidate["queries"][0]["gold_paper_ids"] = ["different-paper"]

    with pytest.raises(ValueError, match="uses different gold papers"):
        build_comparison_summary(
            {"baseline": baseline, "candidate": candidate}, baseline="baseline"
        )


def test_comparison_reports_reranking_diagnostics_by_gold_count():
    gold = [
        {"query_id": "single-up", "gold_papers": [{"paper_id": "p1"}]},
        {"query_id": "single-down", "gold_papers": [{"paper_id": "p4"}]},
        {
            "query_id": "multi-recovered",
            "gold_papers": [{"paper_id": "p2"}, {"paper_id": "p3"}],
        },
    ]
    baseline = evaluate_rankings(
        gold,
        [
            {
                "query_id": "single-up",
                "papers": [{"paper_id": "x1"}, {"paper_id": "x2"}, {"paper_id": "p1"}],
            },
            {"query_id": "single-down", "papers": [{"paper_id": "p4"}]},
            {
                "query_id": "multi-recovered",
                "papers": [{"paper_id": "p2"}, {"paper_id": "x3"}],
            },
        ],
    )
    candidate = evaluate_rankings(
        gold,
        [
            {"query_id": "single-up", "papers": [{"paper_id": "p1"}]},
            {"query_id": "single-down", "papers": []},
            {
                "query_id": "multi-recovered",
                "papers": [
                    {"paper_id": "x4"},
                    {"paper_id": "p2"},
                    {"paper_id": "x5"},
                    {"paper_id": "p3"},
                ],
            },
        ],
    )

    method = build_comparison_summary(
        {"baseline": baseline, "candidate": candidate}, baseline="baseline"
    )["methods"]["candidate"]

    recall_at_5 = method["paired_query_recall"]["at_5"]
    assert recall_at_5["overall"]["counts"] == {
        "improved": 1,
        "worsened": 1,
        "unchanged": 1,
    }
    assert recall_at_5["multiple_gold_papers"]["query_ids"]["improved"] == [
        "multi-recovered"
    ]
    assert recall_at_5["single_gold_paper"]["query_ids"]["worsened"] == [
        "single-down"
    ]

    all_gold_at_5 = method["paired_all_gold_recovery"]["at_5"]
    assert all_gold_at_5["overall"]["counts"] == {
        "gained": 1,
        "lost": 1,
        "retained": 1,
        "neither": 0,
    }
    assert all_gold_at_5["multiple_gold_papers"]["query_ids"]["gained"] == [
        "multi-recovered"
    ]

    ranks = method["capped_relevant_paper_rank_at_20"]
    assert ranks["missing_rank"] == 21
    assert ranks["overall"]["relevant_paper_count"] == 4
    assert ranks["overall"]["baseline_mean_rank"] == pytest.approx(6.5)
    assert ranks["overall"]["method_mean_rank"] == pytest.approx(7.0)
    assert ranks["overall"]["mean_rank_change"] == pytest.approx(-0.5)
    assert ranks["overall"]["paper_outcome_counts"] == {
        "improved": 2,
        "worsened": 2,
        "unchanged": 0,
    }
    assert ranks["multiple_gold_papers"]["query_outcome_counts"] == {
        "improved": 1,
        "worsened": 0,
        "unchanged": 0,
    }


def test_discovery_and_output_writes_are_deterministic_and_non_overwriting(tmp_path):
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    (rankings_dir / "zeta.jsonl").write_text(
        '{"query_id":"q1","papers":[]}\n', encoding="utf-8"
    )
    (rankings_dir / "alpha.jsonl").write_text(
        '{"query_id":"q1","papers":[]}\n', encoding="utf-8"
    )
    (rankings_dir / "notes.txt").write_text("ignored\n", encoding="utf-8")

    discovered = discover_ranking_files(rankings_dir)
    assert list(discovered) == ["alpha", "zeta"]

    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        '{"query_id":"q1","gold_papers":[{"paper_id":"p1"}]}\n',
        encoding="utf-8",
    )
    manifest = build_evaluation_manifest(
        gold_path,
        discovered,
        baseline="baseline",
        gold_query_count=1,
    )
    assert manifest["gold_used_only_by_evaluation"] is True
    assert manifest["gold"]["query_count"] == 1
    assert set(manifest["rankings"]) == {"alpha", "zeta"}
    assert manifest["protocol"]["rank_diagnostic_cutoff"] == 20
    assert manifest["protocol"]["rank_diagnostic_missing_rank"] == 21

    evaluations = {"baseline": _evaluation(["p1"])}
    comparison = build_comparison_summary(evaluations, baseline="baseline")
    questions = build_question_comparison(evaluations)
    output_dir = tmp_path / "evaluation"
    write_evaluation_outputs(
        output_dir,
        evaluations,
        comparison,
        questions,
        manifest=manifest,
    )

    saved_summary = json.loads(
        (output_dir / "comparison_summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["baseline"] == "baseline"
    assert (output_dir / "methods" / "baseline.json").is_file()
    assert (output_dir / "evaluation_manifest.json").is_file()
    assert len(
        (output_dir / "question_comparison.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_evaluation_outputs(output_dir, evaluations, comparison, questions)
