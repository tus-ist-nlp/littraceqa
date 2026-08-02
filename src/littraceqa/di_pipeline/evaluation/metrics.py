"""Aggregate per-query rankings into paper-level retrieval metrics.

Queries are split by how many gold papers they have, because a single-paper
question and a four-paper enumeration fail in very different ways.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


SCENARIOS = ("total", "single", "multi")
METRIC_NAMES = ("recall", "precision", "hit_rate", "all_gold")
Ranking = tuple[set[str], list[str]]
MetricValues = dict[str, float | int | None]
RetrievalMetrics = dict[str, dict[int, MetricValues]]


def _scenario_for_gold(gold: set[str]) -> str | None:
    if len(gold) == 1:
        return "single"
    if len(gold) > 1:
        return "multi"
    return None


def aggregate_rankings(
    rankings: Iterable[Ranking], ks: Sequence[int]
) -> RetrievalMetrics:
    """Aggregate paper-ranking metrics for total, single, and multi groups.

    Precision uses ``k`` as its denominator, matching conventional
    ``precision@k`` and the previous version of this script. ``hit_rate`` is the
    fraction of queries with at least one gold paper in the top k, while
    ``all_gold`` is the fraction whose complete gold set is in the top k.

    Queries without gold papers remain in ``total`` but are not classified as
    ``single`` or ``multi``. Empty groups report ``None`` instead of a misleading
    zero score.
    """
    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("ks must contain at least one positive integer")

    counts = {scenario: 0 for scenario in SCENARIOS}
    sums = {
        scenario: {
            k: {metric: 0.0 for metric in METRIC_NAMES}
            for k in normalized_ks
        }
        for scenario in SCENARIOS
    }

    for gold, ranked_papers in rankings:
        scenarios = ["total"]
        specific_scenario = _scenario_for_gold(gold)
        if specific_scenario is not None:
            scenarios.append(specific_scenario)
        for scenario in scenarios:
            counts[scenario] += 1

        for k in normalized_ks:
            top_k = set(ranked_papers[:k])
            hit_count = len(gold & top_k)
            values = {
                "recall": hit_count / len(gold) if gold else 1.0,
                "precision": hit_count / k,
                "hit_rate": float(hit_count > 0),
                "all_gold": float(gold.issubset(top_k)),
            }
            for scenario in scenarios:
                for metric, value in values.items():
                    sums[scenario][k][metric] += value

    metrics: RetrievalMetrics = {}
    for scenario in SCENARIOS:
        query_count = counts[scenario]
        metrics[scenario] = {}
        for k in normalized_ks:
            metrics[scenario][k] = {"query_count": query_count}
            for metric in METRIC_NAMES:
                metrics[scenario][k][metric] = (
                    sums[scenario][k][metric] / query_count
                    if query_count
                    else None
                )
    return metrics
