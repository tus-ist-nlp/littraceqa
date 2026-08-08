"""Choose the paper set to submit from a relevance-ordered candidate list.

The official score compares the submitted set with the gold set per question
and macro-averages F1, so the size of the submission decides most of the score.
On the 55 validation questions, submitting the top 20 papers scores F1 0.169
and submitting the top 1 scores 0.620, from the same ranking.

The selector therefore answers "how many", not "which": it takes the count the
question states and cuts the ranking there. Reordering is left to retrieval and
to the reading agent, which are the stages that can actually read the papers.

Its parameters are deliberately not fitted on validation. The multi-paper gold
sets there are cluster annotations -- twelve questions share one four-paper set,
and a question naming only TCM still has the other three papers as gold -- so a
threshold tuned to reproduce them would reward submitting papers the question
never refers to. The shipped configurations instead bracket the
precision/recall trade-off so the held-out evaluator can decide between them.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
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
    dropped_without_evidence: tuple[str, ...] = ()


class CardinalityPaperSelector:
    """Cut a ranked candidate list at the count the question states.

    ``default_count`` applies when the wording states nothing, which is the
    common case. One is the safest default: on validation the top-ranked paper
    is correct for 92.7% of questions, so a larger default trades that
    precision away on every unmarked question.

    ``open_set_count`` applies to "which papers ..." questions, whose answer
    set has no stated size.

    ``stated_count_margin`` widens a count the question does state. It exists
    for the recall-oriented configuration; at zero the stated count is taken
    literally.

    ``require_evidence`` drops candidates the reading agent could not find
    supporting evidence for. It only takes effect when ``select`` is given an
    ``evidence_paper_ids`` argument, so a retrieval-only caller is unaffected.
    At least one paper is always submitted: an empty submission scores zero,
    which is never better than submitting the top-ranked candidate.
    """

    def __init__(
        self,
        default_count: int = 1,
        open_set_count: int = 1,
        stated_count_margin: int = 0,
        max_papers: int = MAX_EXPECTED_PAPERS,
        require_evidence: bool = False,
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
        if isinstance(stated_count_margin, bool) or not isinstance(
            stated_count_margin, int
        ):
            raise TypeError("stated_count_margin must be an integer")
        if stated_count_margin < 0:
            raise ValueError("stated_count_margin must be non-negative")
        if not isinstance(require_evidence, bool):
            raise TypeError("require_evidence must be a boolean")
        if default_count > max_papers:
            raise ValueError("default_count must not exceed max_papers")
        if open_set_count > max_papers:
            raise ValueError("open_set_count must not exceed max_papers")

        self.default_count = default_count
        self.open_set_count = open_set_count
        self.stated_count_margin = stated_count_margin
        self.max_papers = max_papers
        self.require_evidence = require_evidence

    def expected_count(self, question: object) -> tuple[int, str]:
        """Return how many papers to submit and which rule decided it."""

        stated = expected_paper_count(question, default=0)
        if stated >= 2:
            count = min(stated + self.stated_count_margin, self.max_papers)
            return count, "stated_in_question"
        if is_open_ended_enumeration(question):
            return self.open_set_count, "open_set_enumeration"
        return self.default_count, "default"

    def select(
        self,
        question: object,
        candidates: Sequence[str] | Iterable[str],
        evidence_paper_ids: Collection[str] | None = None,
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
        dropped: tuple[str, ...] = ()
        if self.require_evidence and evidence_paper_ids is not None:
            supported = [
                paper_id for paper_id in ordered if paper_id in evidence_paper_ids
            ]
            # Falling back to the raw ranking beats submitting nothing, which
            # scores zero on both precision and recall.
            if supported:
                dropped = tuple(
                    paper_id
                    for paper_id in ordered[:count]
                    if paper_id not in evidence_paper_ids
                )
                ordered = supported
                reason = f"{reason}+evidence"

        return PaperSelection(
            paper_ids=tuple(ordered[:count]),
            expected_count=count,
            reason=reason,
            dropped_without_evidence=dropped,
        )
