"""Choosing the paper set to submit, separately from ranking it."""

from typing import Any

from littraceqa.di_pipeline.select.owner_aware import OwnerAwarePaperSelector
from littraceqa.di_pipeline.select.selector import (
    CardinalityPaperSelector,
    PaperSelection,
)

_SELECTORS = {
    "cardinality": CardinalityPaperSelector,
    "owner_aware": OwnerAwarePaperSelector,
}


def build_paper_selector(
    spec: dict[str, Any] | None,
    *,
    method_owner_index_path: str | None = None,
):
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
    params = dict(spec.get("params") or {})
    if name == "owner_aware" and method_owner_index_path is not None:
        params["method_owner_index_path"] = method_owner_index_path
    return _SELECTORS[name](**params)


__all__ = [
    "CardinalityPaperSelector",
    "OwnerAwarePaperSelector",
    "PaperSelection",
    "build_paper_selector",
]
