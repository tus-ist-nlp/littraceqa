"""Choosing the paper set to submit, separately from ranking it."""

from littraceqa.di_pipeline.select.selector import (
    CardinalityPaperSelector,
    PaperSelection,
)

__all__ = ["CardinalityPaperSelector", "PaperSelection"]
