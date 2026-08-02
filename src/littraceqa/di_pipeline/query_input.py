"""Load evaluation queries using only the fields production input carries.

``task_family`` and ``primary_evidence_type`` exist in the validation file but
not in the real input, so they are never read here.
"""

from __future__ import annotations

import json
from pathlib import Path

from littraceqa.di_pipeline.contracts import Query


# Production input is guaranteed to contain only these four fields.
# It does not include multiple-choice options, so they are excluded here.
_PRODUCTION_FIELDS = ("query_id", "question", "answer_types", "table_schema")


def load_mc_options(path: Path) -> dict[str, dict]:
    """Load multiple-choice options by query ID without reading gold labels.

    Production input does not include options. Joining them therefore creates
    an oracle setting that measures performance when choices are provided, not
    production performance. This does not expose the correct choice, but it can
    materially affect scores: 41 of 55 validation queries are multiple choice,
    and 21 of those do not provide a free-form answer.
    """
    options_map: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            mc = (record.get("answer") or {}).get("multiple_choice") or {}
            options = mc.get("options") if isinstance(mc, dict) else None
            if options:
                options_map[record["query_id"]] = options
    return options_map


def load_queries(
    path: Path, production_input: bool = True, options_path: Path | None = None
) -> list[Query]:
    """Load queries using production fields by default.

    Validation-only labels are discarded unless an explicit oracle run passes
    ``production_input=False``. Multiple-choice options are also oracle-only
    because the production input does not provide them.
    """
    options_map = (
        load_mc_options(options_path)
        if options_path is not None and not production_input
        else {}
    )
    queries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if production_input:
                record = {k: v for k, v in record.items() if k in _PRODUCTION_FIELDS}
            if not record.get("options") and record["query_id"] in options_map:
                record["options"] = options_map[record["query_id"]]
            queries.append(Query.from_dict(record))
    return queries
