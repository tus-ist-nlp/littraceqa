"""Choose the paper set to submit from a relevance-ordered candidate list.

The official score compares the submitted set with the gold set per question
and macro-averages F1, so the size of the submission decides most of the score.
On the 55 validation questions, submitting the top 20 papers scores F1 0.169
and submitting the top 1 scores 0.620, from the same ranking.

The selector therefore answers "how many", not "which": it takes the count the
question states and cuts the ranking there. Reordering is left to retrieval and
to the reading agent, which are the stages that can actually read the papers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from littraceqa.di_pipeline.select.cardinality import (
    MAX_EXPECTED_PAPERS,
    expected_paper_count,
    is_open_ended_enumeration,
)


@dataclass(frozen=True)
class PaperSelection:
    """The submitted paper ids plus why that many were submitted."""

    paper_ids: tuple[str, ...]
    expected_count: int
    reason: str


class CardinalityPaperSelector:
    """Cut a ranked candidate list at the count the question states.

    ``default_count`` applies when the wording states nothing, which is the
    common case. One is the right default: on validation the top-ranked paper
    is correct for 92.7% of questions, so a larger default trades that
    precision away on every unmarked question.

    ``open_set_count`` applies to "which papers ..." questions, whose answer
    set has no stated size. It is separate because those questions genuinely
    admit many papers, but it is deliberately conservative: only three
    validation questions exercise it, which is far too few to tune on.
    """

    def __init__(
        self,
        default_count: int = 1,
        open_set_count: int = 1,
        max_papers: int = MAX_EXPECTED_PAPERS,
    ) -> None:
        for name, value in (
            ("default_count", default_count),
            ("open_set_count", open_set_count),
            ("max_papers", max_papers),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if default_count > max_papers:
            raise ValueError("default_count must not exceed max_papers")
        if open_set_count > max_papers:
            raise ValueError("open_set_count must not exceed max_papers")

        self.default_count = default_count
        self.open_set_count = open_set_count
        self.max_papers = max_papers

    def expected_count(self, question: object) -> tuple[int, str]:
        """Return how many papers to submit and which rule decided it."""

        stated = expected_paper_count(question, default=0)
        if stated >= 2:
            return min(stated, self.max_papers), "stated_in_question"
        if is_open_ended_enumeration(question):
            return self.open_set_count, "open_set_enumeration"
        return self.default_count, "default"

    def select(
        self,
        question: object,
        candidates: Sequence[str] | Iterable[str],
    ) -> PaperSelection:
        """Take the leading papers of ``candidates`` up to the expected count."""

        ordered: list[str] = []
        seen: set[str] = set()
        for paper_id in candidates:
            if not isinstance(paper_id, str) or not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            ordered.append(paper_id)

        count, reason = self.expected_count(question)
        return PaperSelection(
            paper_ids=tuple(ordered[:count]),
            expected_count=count,
            reason=reason,
        )
