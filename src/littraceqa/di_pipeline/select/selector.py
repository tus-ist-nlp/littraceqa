"""Choose a small submitted paper set from ranked retrieval candidates."""

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


def ordered_paper_ids(candidates: Iterable[str]) -> list[str]:
    """Keep valid paper ids in first-seen order."""

    ordered: list[str] = []
    seen: set[str] = set()
    for paper_id in candidates:
        if not isinstance(paper_id, str) or not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        ordered.append(paper_id)
    return ordered


def select_papers(
    candidates: Iterable[str],
    *,
    count: int,
    reason: str,
    require_evidence: bool = False,
    evidence_paper_ids: Collection[str] | None = None,
) -> PaperSelection:
    """Apply optional evidence filtering and cut an ordered candidate list."""

    ordered = ordered_paper_ids(candidates)
    dropped: tuple[str, ...] = ()
    if require_evidence and evidence_paper_ids is not None:
        supported = [
            paper_id for paper_id in ordered if paper_id in evidence_paper_ids
        ]
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


class CardinalityPaperSelector:
    """Cut a ranked candidate list at the count the question states.

    ``default_count`` applies when wording states nothing, ``open_set_count``
    handles open-ended enumerations, and ``stated_count_margin`` widens an
    explicit count. ``require_evidence`` is inactive when no evidence set is
    supplied.
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
        count, reason = self.expected_count(question)
        return select_papers(
            candidates,
            count=count,
            reason=reason,
            require_evidence=self.require_evidence,
            evidence_paper_ids=evidence_paper_ids,
        )
