"""Look up which papers own the method names carried by the search hints.

Shared by the method relation, method bridge and dense tail lanes: each of them
starts from the papers that introduce a hinted method.
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import RetrievalResult

METHOD_PROVIDER_METHODS = (
    "get_document",
    "get_method_neighbors",
    "find_method_owners",
)


def find_method_owner_records(
    provider,
    methods: tuple[str, ...],
    candidates: list[RetrievalResult],
    *,
    limit: int,
) -> tuple[dict, ...]:
    """Merge a prebuilt owner index with bounded live extraction."""

    try:
        indexed_records = tuple(provider.find_method_owners(methods, limit=limit))
    except Exception:
        indexed_records = ()

    live_records: tuple = ()
    live_finder = getattr(provider, "find_method_owners_in_papers", None)
    if not indexed_records and callable(live_finder):
        try:
            live_records = tuple(
                live_finder(
                    methods,
                    (candidate.paper_id for candidate in candidates[:10]),
                    limit=limit,
                )
            )
        except Exception:
            live_records = ()

    merged: dict[str, dict] = {}
    for record in (*indexed_records, *live_records):
        if not isinstance(record, dict):
            continue
        paper_id = record.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            continue
        state = merged.setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "aliases": set(),
                "strength": 0,
            },
        )
        aliases = record.get("aliases")
        if isinstance(aliases, (list, tuple)):
            state["aliases"].update(
                alias for alias in aliases if isinstance(alias, str)
            )
        strength = record.get("strength")
        if isinstance(strength, int) and not isinstance(strength, bool):
            state["strength"] = max(state["strength"], max(strength, 0))

    records = [
        {
            "paper_id": state["paper_id"],
            "aliases": sorted(state["aliases"]),
            "strength": state["strength"],
        }
        for state in merged.values()
    ]
    records.sort(key=lambda record: (-record["strength"], record["paper_id"]))
    return tuple(records[:limit])
