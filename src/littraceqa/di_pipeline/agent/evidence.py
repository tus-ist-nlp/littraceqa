"""RetrievalResult から提出用の Evidence を組み立てるヘルパ。

MinerU のチャンクは metadata に page / table_id / figure_id / section / equation_id を
持っており、chunk_type の語彙（table / figure / text_span / equation_algorithm /
citation_context）は Evidence.source_type の観測値とそのまま一致する。
つまり「どのチャンクが根拠か」さえ決まれば Evidence は機械的に組める。

scripts/evaluate.py の採点キーは (paper_id, source_type, page, object_id) の4つ組で、
table なら table_id、figure なら figure_id しか見ない（coarse_evidence_key）。
gold にある row / column / sentence_start などは埋めなくても evidence F1 は取れる。
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import Evidence, EvidenceLocator, RetrievalResult


def evidence_from_result(result: RetrievalResult) -> Evidence:
    """検索でヒットしたチャンク1件を、提出用の Evidence 1件に変換する。"""
    metadata = result.metadata or {}
    locator = EvidenceLocator(
        page=metadata.get("page"),
        table_id=metadata.get("table_id"),
        row=metadata.get("row"),
        column=metadata.get("column"),
        figure_id=metadata.get("figure_id"),
        region=metadata.get("region"),
        section=metadata.get("section"),
        paragraph_id=metadata.get("paragraph_id"),
        sentence_start=metadata.get("sentence_start"),
        sentence_end=metadata.get("sentence_end"),
        equation_id=metadata.get("equation_id"),
        citation_id=metadata.get("citation_id"),
        cited_paper=metadata.get("cited_paper"),
    )
    return Evidence(
        paper_id=result.paper_id,
        source_type=result.chunk_type,
        locator=locator,
        evidence_text_or_value=result.text,
    )
