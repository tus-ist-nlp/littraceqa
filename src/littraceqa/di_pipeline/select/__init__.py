"""Choosing the paper set to submit, separately from ranking it."""

from typing import Any

from littraceqa.di_pipeline.select.selector import (
    CardinalityPaperSelector,
    PaperSelection,
)

_SELECTORS = {"cardinality": CardinalityPaperSelector}


def build_paper_selector(spec: dict[str, Any] | None):
    """Build the selector a composed ``select_style`` block names.

    ``None`` means no select_style was chosen, and the agent keeps its own
    ``paper_cutoff`` behaviour.
    """

    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise TypeError("paper_selector must be a mapping or None")
    name = spec.get("name")
    if name not in _SELECTORS:
        raise ValueError(
            f"unknown paper selector: {name!r} "
            f"(expected one of {sorted(_SELECTORS)})"
        )
    return _SELECTORS[name](**dict(spec.get("params") or {}))


__all__ = ["CardinalityPaperSelector", "PaperSelection", "build_paper_selector"]
