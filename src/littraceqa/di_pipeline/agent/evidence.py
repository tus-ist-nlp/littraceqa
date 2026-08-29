"""Turn a RetrievalResult into the Evidence record that gets submitted.

A MinerU chunk already carries page / table_id / figure_id / section / equation_id
in its metadata, and the chunk_type vocabulary (table / figure / text_span /
equation_algorithm / citation_context) matches the values observed in
Evidence.source_type exactly. **So once it is settled which chunk is the evidence,
building the Evidence is mechanical** — nothing has to be inferred.

Scoring keys on the 4-tuple (paper_id, source_type, page, object_id), where
object_id is table_id for a table and figure_id for a figure and nothing else
(`coarse_evidence_key` in scripts/evaluate.py). The finer gold fields — row,
column, sentence_start — can be left empty and evidence F1 still scores.
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import Evidence, EvidenceLocator, RetrievalResult


def evidence_from_result(result: RetrievalResult) -> Evidence:
    """One retrieved chunk becomes one submitted Evidence."""
    metadata = result.metadata or {}
    locator = EvidenceLocator(
        page=metadata.get("page"),
        table_id=metadata.get("table_id"),
        figure_id=metadata.get("figure_id"),
        section=metadata.get("section"),
        equation_id=metadata.get("equation_id"),
    )
    return Evidence(
        paper_id=result.paper_id,
        source_type=result.chunk_type,
        locator=locator,
        evidence_text_or_value=result.text,
    )
