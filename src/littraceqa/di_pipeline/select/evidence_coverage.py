"""Run conservative evidence-based paper-selection refinements in order."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.retrieve.paper_tables import (
    PaperDocumentSource,
    PaperEvidenceSource,
    PaperTableSource,
)
from littraceqa.di_pipeline.select.citation_table_coverage import (
    CitationTableOpenSetRefiner,
)
from littraceqa.di_pipeline.select.multi_paper_coverage import (
    MultiPaperCoverageRefiner,
)
from littraceqa.di_pipeline.select.selector import (
    PaperSelection,
    ordered_paper_ids,
)
from littraceqa.di_pipeline.select.table_rules import (
    ExplicitTableAnchorRefiner,
    SingleTableCoverageRefiner,
)


class PaperSelectionRefiner(Protocol):
    """A conservative post-ranking paper-selection rule."""

    def refine(
        self,
        query: Query,
        candidates: Sequence[str] | Iterable[str],
        selection: PaperSelection,
    ) -> PaperSelection: ...


class EvidenceCoverageRefiner:
    """Return the first high-confidence refinement that changes a selection."""

    def __init__(
        self,
        table_source: PaperTableSource,
        candidate_limit: int = 20,
        *,
        evidence_source: PaperDocumentSource | None = None,
        paper_metadata: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.explicit_table = ExplicitTableAnchorRefiner(
            table_source,
            candidate_limit=candidate_limit,
        )
        self.single_table = SingleTableCoverageRefiner(
            table_source,
            candidate_limit=candidate_limit,
        )
        self.multi_paper = (
            MultiPaperCoverageRefiner(
                evidence_source,
                candidate_limit=candidate_limit,
            )
            if evidence_source is not None
            else None
        )
        self.citation_table = (
            CitationTableOpenSetRefiner(
                evidence_source,
                paper_metadata,
                candidate_limit=candidate_limit,
            )
            if evidence_source is not None and paper_metadata is not None
            else None
        )
        refiners: list[PaperSelectionRefiner] = [
            self.explicit_table,
            self.single_table,
        ]
        if self.multi_paper is not None:
            refiners.append(self.multi_paper)
        if self.citation_table is not None:
            refiners.append(self.citation_table)
        self._refiners = tuple(refiners)

    @classmethod
    def from_evidence_source(
        cls,
        source: PaperEvidenceSource,
        candidate_limit: int = 20,
        *,
        paper_metadata: Mapping[str, Mapping[str, object]] | None = None,
    ) -> EvidenceCoverageRefiner:
        """Use one source for table, body-text, and citation checks."""

        return cls(
            source,
            candidate_limit,
            evidence_source=source,
            paper_metadata=paper_metadata,
        )

    def refine(
        self,
        query: Query,
        candidates: Sequence[str] | Iterable[str],
        selection: PaperSelection,
    ) -> PaperSelection:
        ranked = ordered_paper_ids(candidates)
        for refiner in self._refiners:
            refined = refiner.refine(query, ranked, selection)
            if refined != selection:
                return refined
        return selection


__all__ = ["EvidenceCoverageRefiner", "PaperSelectionRefiner"]
