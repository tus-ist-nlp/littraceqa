"""Compatibility imports for table-based evidence refinement."""

from littraceqa.di_pipeline.select.evidence_coverage import EvidenceCoverageRefiner
from littraceqa.di_pipeline.select.table_rules import (
    ExplicitTableAnchorRefiner,
    SingleTableCoverageRefiner,
)

__all__ = [
    "EvidenceCoverageRefiner",
    "ExplicitTableAnchorRefiner",
    "SingleTableCoverageRefiner",
]
