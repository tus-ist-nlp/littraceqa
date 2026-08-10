"""Prepare optional MinerU evidence refinement for evaluation commands."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from littraceqa.di_pipeline.contracts import Query
from littraceqa.di_pipeline.evaluation.selection_input import load_paper_metadata
from littraceqa.di_pipeline.retrieve.paper_tables import MinerUPaperEvidenceSource
from littraceqa.di_pipeline.select.citation_table_coverage import (
    citation_table_candidate_ids,
)
from littraceqa.di_pipeline.select.evidence_coverage import EvidenceCoverageRefiner


class MissingPaperMetadataError(ValueError):
    """Raised when an evidence condition needs unavailable paper metadata."""


@dataclass(frozen=True)
class EvidenceCoverageSetup:
    refiner: EvidenceCoverageRefiner
    paper_metadata: dict[str, dict]


def prepare_evidence_coverage(
    mineru_dir: Path,
    paper_metadata_path: Path,
    queries: Mapping[str, Query],
    rankings: Mapping[str, Sequence[str]],
    *,
    abstract_chars: int = 0,
) -> EvidenceCoverageSetup:
    """Build a refiner and load metadata only for eligible top candidates."""

    source = MinerUPaperEvidenceSource(mineru_dir)
    wanted_ids = citation_table_candidate_ids(queries, rankings)
    metadata = load_paper_metadata(
        paper_metadata_path,
        wanted_ids,
        abstract_chars=abstract_chars,
    )
    missing_ids = wanted_ids - metadata.keys()
    if missing_ids:
        raise MissingPaperMetadataError(
            f"paper metadata is missing {next(iter(sorted(missing_ids)))}"
        )
    return EvidenceCoverageSetup(
        refiner=EvidenceCoverageRefiner.from_evidence_source(
            source,
            paper_metadata=metadata,
        ),
        paper_metadata=metadata,
    )


__all__ = [
    "EvidenceCoverageSetup",
    "MissingPaperMetadataError",
    "prepare_evidence_coverage",
]
