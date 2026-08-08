"""Score a submitted paper set the way the official evaluation does.

``scripts/evaluate.py`` compares the submitted ``gold_papers`` set against the
official one per question and macro-averages precision, recall and F1.  Rank is
never consulted, so a high Recall@20 says nothing about the submitted score:
handing in twenty papers for a one-paper question scores F1 0.095.

This module reproduces exactly that arithmetic on an in-memory selection so a
selector can be tuned without running the agent or writing a predictions file.
The ``prf`` edge cases mirror ``scripts/evaluate.py`` deliberately; changing one
without the other would make local tuning disagree with the official score.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperSelectionMetrics:
    """Macro-averaged precision, recall and F1 over a set of questions."""

    paper_precision_macro: float
    paper_recall_macro: float
    paper_f1_macro: float
    query_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "paper_precision_macro": self.paper_precision_macro,
            "paper_recall_macro": self.paper_recall_macro,
            "paper_f1_macro": self.paper_f1_macro,
            "query_count": self.query_count,
        }


def prf(gold: set[str], predicted: set[str]) -> tuple[float, float, float]:
    """Precision, recall and F1 for one question.

    Mirrors ``scripts/evaluate.py``: an empty prediction scores zero against a
    non-empty gold set rather than raising, so a selector that abstains is
    penalised the same way the official script penalises it.
    """

    if not gold and not predicted:
        return (1.0, 1.0, 1.0)
    if not predicted:
        return (0.0, 0.0, 0.0) if gold else (0.0, 1.0, 0.0)

    correct = len(gold & predicted)
    precision = correct / len(predicted)
    recall = correct / len(gold) if gold else 1.0
    if not precision + recall:
        return (precision, recall, 0.0)
    return (precision, recall, 2 * precision * recall / (precision + recall))


def score_selection(
    gold_by_query: Mapping[str, set[str]],
    selected_by_query: Mapping[str, Iterable[str]],
) -> PaperSelectionMetrics:
    """Macro-average ``prf`` over every question that has a gold set.

    A question missing from ``selected_by_query`` counts as an empty
    submission, matching what the official script does with a missing record.
    """

    if not gold_by_query:
        raise ValueError("gold_by_query must not be empty")

    totals = [0.0, 0.0, 0.0]
    for query_id, gold in gold_by_query.items():
        predicted = set(selected_by_query.get(query_id) or ())
        for index, value in enumerate(prf(gold, predicted)):
            totals[index] += value

    count = len(gold_by_query)
    return PaperSelectionMetrics(
        paper_precision_macro=totals[0] / count,
        paper_recall_macro=totals[1] / count,
        paper_f1_macro=totals[2] / count,
        query_count=count,
    )


def load_gold_paper_sets(path: Path | str) -> dict[str, set[str]]:
    """Read ``query_id -> gold paper ids`` from an official gold JSONL file."""

    gold: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query_id = str(record["query_id"])
            papers = record.get("gold_papers") or []
            gold[query_id] = {
                str(item["paper_id"]) if isinstance(item, dict) else str(item)
                for item in papers
            }
    if not gold:
        raise ValueError(f"{path} contains no queries")
    return gold
