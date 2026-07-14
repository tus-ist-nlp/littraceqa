"""Tests for set-based and ordered paper retrieval metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import math

import pytest

from evaluate import evaluate, evaluate_rankings, ranked_query_metrics


def test_single_and_multiple_groups_ignore_development_labels():
    gold = [
        {
            "query_id": "single",
            "task_family": "multi_paper",
            "gold_papers": [{"paper_id": "p1"}],
            "answer_types": [],
        },
        {
            "query_id": "multiple",
            "task_family": "hidden_source_single_paper",
            "gold_papers": [{"paper_id": "p2"}, {"paper_id": "p3"}],
            "answer_types": [],
        },
    ]
    predictions = [
        {
            "query_id": "single",
            "gold_papers": [{"paper_id": "p1"}, {"paper_id": "extra"}],
        },
        {
            "query_id": "multiple",
            "gold_papers": [{"paper_id": "p2"}],
        },
    ]

    groups = evaluate(gold, predictions)["paper_metrics_by_gold_count"]

    single = groups["single_gold_paper"]
    assert single["count"] == 1
    assert single["paper_precision_macro"] == 0.5
    assert single["paper_recall_macro"] == 1.0
    assert single["all_gold_count"] == 1

    multiple = groups["multiple_gold_papers"]
    assert multiple["count"] == 1
    assert multiple["paper_precision_macro"] == 1.0
    assert multiple["paper_recall_macro"] == 0.5
    assert multiple["all_gold_count"] == 0


def test_ranked_query_metrics_use_fixed_cutoffs_and_binary_ndcg():
    metrics = ranked_query_metrics(
        {"p1", "p3"},
        ["p1", "irrelevant", "p3", "p1"],
    )

    assert metrics["paper_precision_at_5"] == 2 / 5
    assert metrics["paper_recall_at_5"] == 1.0
    assert metrics["paper_f1_at_5"] == pytest.approx(4 / 7)
    assert metrics["relevant_paper_count_at_5"] == 2
    assert metrics["all_gold_at_5"] is True
    assert metrics["ranked_paper_count"] == 3
    assert metrics["reciprocal_rank"] == 1.0
    assert metrics["paper_precision_at_10"] == 2 / 10
    assert metrics["paper_recall_at_10"] == 1.0
    assert metrics["paper_f1_at_10"] == pytest.approx(1 / 3)
    assert metrics["paper_precision_at_20"] == 2 / 20
    assert metrics["paper_recall_at_20"] == 1.0
    assert metrics["paper_f1_at_20"] == pytest.approx(2 / 11)

    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert metrics["ndcg_at_10"] == pytest.approx(dcg / idcg)


def test_evaluate_rankings_preserves_order_and_groups_by_gold_count():
    gold = [
        {
            "query_id": "single",
            "task_family": "multi_paper",
            "gold_papers": [{"paper_id": "p1"}],
        },
        {
            "query_id": "multiple",
            "task_family": "single_paper",
            "gold_papers": [{"paper_id": "p2"}, {"paper_id": "p3"}],
        },
        {
            "query_id": "missing",
            "gold_papers": [{"paper_id": "p4"}],
        },
    ]
    rankings = [
        {
            "query_id": "single",
            "papers": [
                {"paper_id": "other"},
                {"paper_id": "p1"},
                {"paper_id": "p1"},
            ],
        },
        {
            "query_id": "multiple",
            "papers": [
                {"paper_id": "p2"},
                *[{"paper_id": f"other-{index}"} for index in range(4)],
                {"paper_id": "p3"},
            ],
        },
        {"query_id": "extra", "papers": [{"paper_id": "p9"}]},
    ]

    result = evaluate_rankings(gold, rankings)

    assert result["details"]["missing_rankings"] == ["missing"]
    assert result["details"]["extra_rankings"] == ["extra"]
    assert result["metrics"]["count"] == 3
    assert result["metrics"]["ranked_paper_count_min"] == 0
    assert result["metrics"]["ranked_paper_count_max"] == 6

    single = result["paper_metrics_by_gold_count"]["single_gold_paper"]
    multiple = result["paper_metrics_by_gold_count"]["multiple_gold_papers"]
    assert single["count"] == 2
    assert multiple["count"] == 1
    assert multiple["paper_recall_at_5_macro"] == 0.5
    assert multiple["paper_recall_at_10_macro"] == 1.0
    assert multiple["all_gold_at_5_count"] == 0
    assert multiple["all_gold_at_10_count"] == 1

    single_query = result["queries"][0]
    assert single_query["query_id"] == "single"
    assert single_query["ranked_paper_ids"] == ["other", "p1"]
    assert single_query["metrics"]["reciprocal_rank"] == 0.5
    missing_query = next(
        query for query in result["queries"] if query["query_id"] == "missing"
    )
    assert missing_query["ranking_missing"] is True
    assert missing_query["ranked_paper_ids"] == []
    assert missing_query["metrics"]["paper_recall_at_20"] == 0.0


def test_evaluate_rankings_rejects_duplicate_query_ids():
    gold = [
        {"query_id": "q1", "gold_papers": [{"paper_id": "p1"}]},
        {"query_id": "q1", "gold_papers": [{"paper_id": "p2"}]},
    ]

    with pytest.raises(ValueError, match="duplicate query_id"):
        evaluate_rankings(gold, [])


def test_evaluate_rankings_counts_a_missing_ranking_as_an_empty_ranking():
    gold = [{"query_id": "q1", "gold_papers": [{"paper_id": "p1"}]}]

    result = evaluate_rankings(gold, [])

    assert result["metrics"]["paper_precision_at_5_macro"] == 0.0
    assert result["metrics"]["paper_recall_at_20_macro"] == 0.0
    assert result["metrics"]["mrr"] == 0.0
    assert result["metrics"]["ndcg_at_10_macro"] == 0.0
    assert result["details"]["missing_ranking_count"] == 1
    assert result["queries"][0]["ranking_missing"] is True


def test_evaluate_rankings_audits_but_excludes_queries_without_gold_papers():
    gold = [
        {"query_id": "no-gold", "gold_papers": []},
        {"query_id": "scored", "gold_papers": [{"paper_id": "p1"}]},
    ]
    rankings = [
        {"query_id": "no-gold", "papers": [{"paper_id": "unjudged"}]},
        {"query_id": "scored", "papers": [{"paper_id": "p1"}]},
    ]

    result = evaluate_rankings(gold, rankings)

    assert result["metrics"]["count"] == 1
    assert result["metrics"]["paper_recall_at_5_macro"] == 1.0
    assert result["metrics"]["all_gold_at_5_rate"] == 1.0
    assert result["details"]["total"] == 2
    assert result["details"]["paper_metric_query_count"] == 1
    assert result["details"]["no_gold_paper_query_count"] == 1
    no_gold = next(
        query for query in result["queries"] if query["query_id"] == "no-gold"
    )
    assert no_gold["gold_count_group"] == "no_gold_paper"
    assert no_gold["metrics"]["paper_recall_at_5"] == 0.0
