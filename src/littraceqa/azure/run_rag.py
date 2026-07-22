#!/usr/bin/env python3
"""Run LitTraceQA RAG over Azure AI Search and Azure OpenAI."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from .azure_config import (
    OpenAISettings,
    SearchSettings,
    build_openai_client,
    build_search_client,
    load_environment,
)
from ..common import (
    DEFAULT_CHUNKS_DIR,
    DEFAULT_METADATA,
    ROOT,
    Record,
    SEMANTIC_CONFIGURATION_NAME,
    clean_text,
    compact_text,
    parse_locator_json,
    read_jsonl,
    retry_chat_completion,
    try_parse_json_object,
    write_json,
    write_jsonl,
)
from ..fix_chunk_locators import normalize_object_id


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Index fields fetched for every chunk hit (everything the evidence/locator
# and context builders need).
SEARCH_SELECT = [
    "id",
    "paper_id",
    "chunk_id",
    "title",
    "authors",
    "venue",
    "year",
    "track",
    "source_type",
    "section",
    "page_numbers",
    "source_url",
    "pdf_url",
    "anthology_id",
    "locator_json",
    "content",
]

# Primary evidence labels that are noisy in the dataset; see
# evidence_from_citations for the both-labels emission strategy.
RELABEL_PRIMARY_TYPES = {"citation_context", "equation_algorithm"}
SINGLE_PAPER_FAMILY = "hidden_source_single_paper"

# Submitted evidence caps: extra items dilute evaluator precision, so 4 slots
# normally and 2 for the blind retrieval-order fallback.
EVIDENCE_CAP = 4
FALLBACK_EVIDENCE_CAP = 2
# evidence_text_or_value cap; the evaluator keys on locators, not text.
EVIDENCE_SNIPPET_CHARS = 300

# Reciprocal-rank-fusion damping (the k in 1/(k+rank)); 60 is the standard
# value from the original RRF paper.
RRF_RANK_OFFSET = 60

# Default context budget in characters; also the base of the P4 prepend cap.
DEFAULT_CONTEXT_CHARS = 18000
# A truncated tail block shorter than this is a header plus noise: drop it.
CONTEXT_MIN_TAIL_CHARS = 500

# NOTE: DEFAULT_CHUNKS_DIR (imported above) points at the local chunk JSONL
# files - the source of truth for which evidence keys (paper_id, source_type,
# page, table/figure id) are actually achievable.

_CHUNK_CACHE: dict[str, Optional[list[Record]]] = {}
_CHUNK_CACHE_LOCK = threading.Lock()


SYSTEM_PROMPT = """You are solving LitTraceQA, a scientific-paper grounded QA task.
Use only the retrieved context. Copy exact values verbatim from the source; never paraphrase numbers.
Return JSON only. Do not include markdown."""


# ---------------------------------------------------------------------------
# Retrieval primitives (embedding, hybrid/keyword search, paper ranking)
# ---------------------------------------------------------------------------


def embedding_kwargs(settings: OpenAISettings, text: str) -> Record:
    """Kwargs for a single-text embeddings.create call."""
    kwargs: Record = {
        "model": settings.embedding_deployment,
        "input": [text],
    }
    if settings.request_embedding_dimensions:
        kwargs["dimensions"] = settings.embedding_dimensions
    return kwargs


def embed_query(client: Any, settings: OpenAISettings, query: str) -> list[float]:
    """Embed one query string, validating the returned dimensionality."""
    response = client.embeddings.create(**embedding_kwargs(settings, query))
    vector = list(response.data[0].embedding)
    if len(vector) != settings.embedding_dimensions:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"expected {settings.embedding_dimensions}, got {len(vector)}"
        )
    return vector


def parse_search_fields(raw: Optional[str]) -> Optional[list[str]]:
    """Split the --search-fields CSV into field names; None keeps all fields."""
    if not raw:
        return None
    fields = [field.strip() for field in raw.split(",") if field.strip()]
    return fields or None


def _escape_odata(value: str) -> str:
    """Escape a string for use inside an OData filter literal."""
    return str(value).replace("'", "''")


def build_filter_expression(venue: Optional[str], year: Optional[int]) -> Optional[str]:
    """OData filter for validated venue/year restrictions; None when neither."""
    parts: list[str] = []
    if venue:
        parts.append(f"venue eq '{_escape_odata(venue)}'")
    if year is not None:
        parts.append(f"year eq {year}")
    return " and ".join(parts) if parts else None


def _materialize_hits(results: Any) -> list[Record]:
    """Turn SDK result iterables into plain dicts with a float search_score."""
    records: list[Record] = []
    for result in results:
        item = dict(result)
        item["search_score"] = float(item.pop("@search.score", 0.0) or 0.0)
        records.append(item)
    return records


def search_chunks(
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    *,
    query: str,
    top_chunks: int,
    vector_k: int,
    query_type: str = "hybrid",
    filter_expr: Optional[str] = None,
    search_fields: Optional[list[str]] = None,
) -> list[Record]:
    from azure.search.documents.models import VectorizedQuery

    query_vector = embed_query(openai_client, openai_settings, query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=vector_k,
        fields="content_vector",
    )
    kwargs: Record = {
        "search_text": query,
        "vector_queries": [vector_query],
        "top": top_chunks,
        "select": SEARCH_SELECT,
    }
    if search_fields:
        # Restricts only the keyword (BM25) leg; the vector leg always runs
        # against content_vector.
        kwargs["search_fields"] = search_fields
    if filter_expr:
        kwargs["filter"] = filter_expr
    if query_type == "semantic":
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = SEMANTIC_CONFIGURATION_NAME
    return _materialize_hits(search_client.search(**kwargs))


def keyword_search(
    search_client: Any,
    query: str,
    *,
    fields: list[str],
    top: int,
    filter_expr: Optional[str] = None,
    select: Optional[list[str]] = None,
) -> list[Record]:
    """Keyword-only (BM25) search; no embedding call."""
    kwargs: Record = {
        "search_text": query,
        "search_fields": fields,
        "top": top,
        "select": select or SEARCH_SELECT,
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
    return _materialize_hits(search_client.search(**kwargs))


def _rrf_accumulate(by_paper: dict[str, Record], hits: list[Record]) -> None:
    """Fold one ranked hit list into a paper-level RRF aggregate in place."""
    for rank, hit in enumerate(hits, start=1):
        paper_id = str(hit.get("paper_id") or "")
        if not paper_id:
            continue
        paper = by_paper.setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "title": hit.get("title") or "",
                "venue": hit.get("venue") or "",
                "year": hit.get("year"),
                "score": 0.0,
                "best_rank": rank,
            },
        )
        paper["score"] = float(paper["score"]) + 1.0 / (RRF_RANK_OFFSET + rank)
        paper["best_rank"] = min(int(paper["best_rank"]), rank)


def _rrf_ranked(by_paper: dict[str, Record]) -> list[Record]:
    """Order an RRF aggregate by (score desc, best rank asc)."""
    return sorted(
        by_paper.values(),
        key=lambda item: (-float(item["score"]), int(item["best_rank"])),
    )


def rank_papers(results: list[Record], top_papers: int) -> list[Record]:
    """Aggregate chunk hits to papers with summed reciprocal-rank fusion."""
    by_paper: dict[str, Record] = {}
    _rrf_accumulate(by_paper, results)
    return _rrf_ranked(by_paper)[:top_papers]


def _chunk_key(hit: Record) -> str:
    """Stable chunk identity for deduping hits across searches."""
    return str(hit.get("id") or f"{hit.get('paper_id')}::{hit.get('chunk_id')}")


def merge_hit_lists(primary: list[Record], secondary: list[Record]) -> list[Record]:
    """Concatenate hit lists, deduping by chunk identity (primary wins)."""
    merged: list[Record] = []
    seen: set[str] = set()
    for hit in [*primary, *secondary]:
        key = _chunk_key(hit)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


# ---------------------------------------------------------------------------
# Locators and evaluator evidence keys
# ---------------------------------------------------------------------------


def parse_locator(result: Record) -> Record:
    """Chunk locator dict, falling back to page_numbers/section fields."""
    locator = parse_locator_json(result.get("locator_json"))
    pages = result.get("page_numbers") or []
    if pages and "page" not in locator:
        locator["page"] = pages[0]
    if result.get("section") and "section" not in locator:
        locator["section"] = result["section"]
    return locator


def evaluator_evidence_key(
    paper_id: str,
    source_type: str,
    locator: Record,
) -> tuple[str, str, str, str]:
    """Coarse key used by scripts/evaluate.py; duplicates collapse to one score."""
    page = str(locator.get("page", "")).strip()
    object_id = ""
    if source_type == "table":
        object_id = normalize_object_id(locator.get("table_id"), "table")
    elif source_type == "figure":
        object_id = normalize_object_id(locator.get("figure_id"), "figure")
    return (paper_id.strip(), source_type.strip(), page, object_id)


def _page_missing(page: Any) -> bool:
    """True when a locator has no usable page (evaluate.py drops such items)."""
    return page is None or str(page).strip() == ""


def _evidence_item(
    paper_id: str,
    source_type: str,
    page: Any,
    chunk_locator: Record,
    text: str,
) -> Record:
    """One submission evidence item; carries table_id/figure_id when present."""
    locator: Record = {"page": page}
    if source_type == "table":
        object_id = str(chunk_locator.get("table_id") or "")
        if object_id:
            locator["table_id"] = object_id
    elif source_type == "figure":
        object_id = str(chunk_locator.get("figure_id") or "")
        if object_id:
            locator["figure_id"] = object_id
    return {
        "evidence_id": "ev_000",  # renumbered by merge_evidence_items
        "paper_id": paper_id,
        "source_type": source_type,
        "evidence_text_or_value": text,
        "locator": locator,
    }


# ---------------------------------------------------------------------------
# Evidence building (retrieval-order fallback and LLM citation mapping)
# ---------------------------------------------------------------------------


def evidence_from_results(results: list[Record], selected_paper_ids: set[str], limit: int) -> list[Record]:
    """Blind fallback: first deduped, page-bearing hits of the selected papers."""
    evidence: list[Record] = []
    seen: set[tuple[str, str, str, str]] = set()
    candidates = [
        result
        for result in results
        if result.get("paper_id") in selected_paper_ids
        and result.get("source_type") != "metadata"
    ]
    if not candidates:
        candidates = [result for result in results if result.get("paper_id") in selected_paper_ids]

    for result in candidates:
        paper_id = str(result.get("paper_id") or "")
        source_type = str(result.get("source_type") or "text_span")
        if source_type == "metadata":
            source_type = "text_span"
        locator = parse_locator(result)
        if _page_missing(locator.get("page")):
            # evaluate.py silently drops page-less items; never waste a slot.
            continue
        key = evaluator_evidence_key(paper_id, source_type, locator)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "evidence_id": f"ev_{len(evidence) + 1:03d}",
                "paper_id": paper_id,
                "source_type": source_type,
                "evidence_text_or_value": compact_text(
                    result.get("content"), max_chars=EVIDENCE_SNIPPET_CHARS
                ),
                "locator": locator,
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


def evidence_from_citations(
    llm_evidence: Any,
    context_map: dict[str, Record],
    sample: Record,
    selected_paper_ids: set[str],
    *,
    cap: int = EVIDENCE_CAP,
) -> list[Record]:
    """Map LLM-cited [C#] context blocks back to chunk locators."""
    if not isinstance(llm_evidence, list):
        return []
    primary_type = str(sample.get("primary_evidence_type") or "")
    evidence: list[Record] = []
    seen: set[tuple[str, str, str, str]] = set()
    # For citation_context/equation_algorithm questions the gold locus is
    # almost always a single equation/ablation location: additional citations
    # must share the FIRST citation's paper and page (off-page results-table
    # citations only dilute precision).
    first_locus: Optional[tuple[str, str]] = None
    for item in llm_evidence:
        if not isinstance(item, dict):
            continue
        raw_context_id = str(item.get("context_id") or item.get("id") or "")
        # Tolerate "[C3]", " c3 ", etc. -> "C3".
        context_id = re.sub(r"[\[\]\s]+", "", raw_context_id).upper()
        chunk = context_map.get(context_id)
        if chunk is None:
            continue
        paper_id = str(chunk.get("paper_id") or "")
        if paper_id not in selected_paper_ids:
            continue
        source_type = str(chunk.get("source_type") or "text_span")
        if source_type == "metadata":
            source_type = "text_span"
        chunk_locator = parse_locator(chunk)
        if _page_missing(chunk_locator.get("page")):
            continue
        if primary_type in RELABEL_PRIMARY_TYPES and source_type == "text_span":
            # Primary labels are noisy (~3/7 citation-labeled questions actually
            # have text_span gold): emit BOTH the relabeled item (first) and the
            # original text_span item instead of gambling on one label.
            emit_types = [primary_type, "text_span"]
        else:
            emit_types = [source_type]
        quote = item.get("quote")
        if isinstance(quote, str) and quote.strip():
            text = compact_text(quote, max_chars=EVIDENCE_SNIPPET_CHARS)
            matched_text = quote
        else:
            text = compact_text(chunk.get("content"), max_chars=EVIDENCE_SNIPPET_CHARS)
            matched_text = ""
        # P2: a chunk spanning multiple pages fans out to ONE extra page item
        # (gold pages often land on the chunk's later page); the extra page is
        # emitted for the primary emit type only. Never empty here: the page
        # check above guarantees fanout_pages finds a primary page.
        pages = fanout_pages(chunk, matched_text)
        if primary_type in RELABEL_PRIMARY_TYPES:
            locus = (paper_id, str(pages[0]))
            if first_locus is None:
                first_locus = locus
            elif locus != first_locus:
                continue
        emit_pairs: list[tuple[str, Any]] = [(etype, pages[0]) for etype in emit_types]
        if len(pages) > 1:
            emit_pairs.append((emit_types[0], pages[1]))
        for emit_type, page_value in emit_pairs:
            candidate = _evidence_item(paper_id, emit_type, page_value, chunk_locator, text)
            key = evaluator_evidence_key(paper_id, emit_type, candidate["locator"])
            if key in seen:
                continue
            seen.add(key)
            candidate["evidence_id"] = f"ev_{len(evidence) + 1:03d}"
            evidence.append(candidate)
            if len(evidence) >= cap:
                break
        if len(evidence) >= cap:
            break
    return evidence


# ---------------------------------------------------------------------------
# P1/P2: local chunk-file re-anchoring, page fan-out and evidence merge
# (chunk-loading helpers here are shared with the P5 lookup below)
# ---------------------------------------------------------------------------


def load_paper_chunks(paper_id: str, chunks_dir: Optional[Path] = None) -> list[Record]:
    """Load a paper's local chunk JSONL, cached; missing files return []."""
    directory = Path(chunks_dir) if chunks_dir else DEFAULT_CHUNKS_DIR
    path = directory / f"{paper_id}.jsonl"
    cache_key = str(path)
    with _CHUNK_CACHE_LOCK:
        cached = _CHUNK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        records = read_jsonl(path) if path.exists() else []
    except Exception:  # noqa: BLE001 - a corrupt chunk file must not break a run.
        records = []
    with _CHUNK_CACHE_LOCK:
        _CHUNK_CACHE[cache_key] = records
    return records


def _norm_match_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _strip_thousands(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)


def fanout_pages(chunk: Record, matched_text: str = "") -> list[Any]:
    """Primary locator page plus at most ONE extra page for multi-page chunks.

    When the matched text's position within the chunk content maps to a later
    page (uniform-length heuristic), that page is preferred as the extra one.
    """
    locator = parse_locator(chunk)
    primary = locator.get("page")
    pages = [p for p in (chunk.get("page_numbers") or []) if p is not None]
    if primary is None:
        primary = pages[0] if pages else None
    if _page_missing(primary):
        return []
    result: list[Any] = [primary]
    extra_candidates = [p for p in pages if str(p) != str(primary)]
    if not extra_candidates:
        return result
    extra = extra_candidates[0]
    if matched_text:
        content_norm = _norm_match_text(chunk.get("content"))
        needle = _norm_match_text(matched_text)
        index = content_norm.find(needle) if needle else -1
        if index >= 0 and content_norm and pages:
            position_page = pages[min(len(pages) - 1, index * len(pages) // len(content_norm))]
            if str(position_page) != str(primary):
                extra = position_page
            else:
                # The quote localizes to the primary page itself: emitting the
                # later page too would just be a precision-diluting twin. The
                # unconditional extra page is kept only when there is no quote
                # to localize.
                return result
    result.append(extra)
    return result


_TABLE_REF_RE = re.compile(r"\btable\s*(\d+[a-z]?)\b", re.IGNORECASE)
_FIGURE_REF_RE = re.compile(r"\bfig(?:ure|\.)?\s*(\d+[a-z]?)\b", re.IGNORECASE)


def question_object_refs(question: str) -> dict[str, set[str]]:
    """Normalized ids of tables/figures the question names ('Table 4', 'Fig. 2')."""
    return {
        "table": {normalize_object_id(m, "table") for m in _TABLE_REF_RE.findall(question)},
        "figure": {normalize_object_id(m, "figure") for m in _FIGURE_REF_RE.findall(question)},
    }


def answer_match_values(answer: Record, options: Optional[dict[str, str]]) -> list[str]:
    """Final answer strings worth locating in chunk content (freeform + MC option text)."""
    values: list[str] = []
    freeform = answer.get("freeform") if isinstance(answer.get("freeform"), dict) else {}
    text = str((freeform or {}).get("text") or "").strip()
    if text:
        values.append(text)
    multiple_choice = (
        answer.get("multiple_choice")
        if isinstance(answer.get("multiple_choice"), dict)
        else {}
    )
    letter = str((multiple_choice or {}).get("gold") or "").strip().upper()
    if letter and options:
        by_key = {str(key).strip().upper(): str(value) for key, value in options.items()}
        option_text = str(by_key.get(letter) or "").strip()
        if option_text:
            values.append(option_text)
    filtered: list[str] = []
    seen_norms: set[str] = set()
    for value in values:
        norm = _norm_match_text(value)
        if not norm or norm in seen_norms:
            continue
        # Too-short values match everywhere; require 2+ chars for numerics,
        # 3+ for words.
        if re.fullmatch(r"[\d.,%±\s+-]+", norm):
            if len(norm) < 2:
                continue
        elif len(norm) < 3:
            continue
        seen_norms.add(norm)
        filtered.append(value)
    return filtered


def _numeric_table_cell_values(answer: Record) -> list[str]:
    """Numeric-looking string cells of answer.table.rows ('37.55', '96.23').

    Row-key/name cells are excluded on purpose: method names occur in every
    table of a paper and would defeat value-based disambiguation.
    """
    table = answer.get("table") if isinstance(answer.get("table"), dict) else {}
    rows = (table or {}).get("rows")
    values: list[str] = []
    seen: set[str] = set()
    if not isinstance(rows, list):
        return values
    for row in rows:
        if not isinstance(row, dict):
            continue
        for cell in row.values():
            if not isinstance(cell, str):
                continue
            norm = _norm_match_text(cell)
            if not norm or norm in seen:
                continue
            if re.fullmatch(r"[\d.,%±\s+-]+", norm) and len(norm) >= 2:
                seen.add(norm)
                values.append(cell)
    return values


def _content_contains_value(content_norm: str, content_norm_nc: str, value: str) -> bool:
    needle = _norm_match_text(value)
    if not needle:
        return False
    for variant in {needle, _strip_thousands(needle)}:
        if re.fullmatch(r"[\d.,%±\s+-]+", variant):
            # Numeric: require non-digit boundaries so 0.09 does not hit 10.09.
            pattern = re.compile(r"(?<![\d.])" + re.escape(variant) + r"(?![\d.])")
            if pattern.search(content_norm) or pattern.search(content_norm_nc):
                return True
        elif variant in content_norm or variant in content_norm_nc:
            return True
    return False


def build_reanchor_candidates(
    sample: Record,
    submitted_paper_ids: list[str],
    answer: Record,
    options: Optional[dict[str, str]],
    llm_evidence: list[Record],
    chunks_dir: Optional[Path] = None,
) -> list[Record]:
    """P1: evidence items re-anchored onto local table/figure chunks.

    Priority order: (i) table/figure chunks the question names, (ii) chunks
    whose content contains the final answer value (preferred source_type
    first), (iii) for table/figure primary types with no such match, the
    table/figure chunk sharing a page with an LLM-cited text_span.
    """
    question = str(sample.get("question") or "")
    primary = str(sample.get("primary_evidence_type") or "")
    answer_types = set(sample.get("answer_types") or [])
    refs = question_object_refs(question)
    table_ish = (
        primary == "table"
        or bool(refs["table"])
        or "table" in answer_types
        or bool(re.search(r"\btable\b", question, re.IGNORECASE))
    )
    figure_ish = primary == "figure" or bool(refs["figure"])
    values = answer_match_values(answer, options)

    named: list[tuple[Record, str]] = []
    value_hits: list[tuple[Record, str]] = []
    tf_by_paper: dict[str, list[Record]] = {}
    for paper_id in submitted_paper_ids:
        chunks = load_paper_chunks(paper_id, chunks_dir)
        tf_chunks = [c for c in chunks if str(c.get("source_type") or "") in ("table", "figure")]
        tf_by_paper[paper_id] = tf_chunks
        # (i) question-named tables/figures.
        for chunk in tf_chunks:
            chunk_locator = parse_locator(chunk)
            source_type = str(chunk.get("source_type") or "")
            if source_type == "table" and refs["table"]:
                if normalize_object_id(chunk_locator.get("table_id"), "table") in refs["table"]:
                    named.append((chunk, ""))
            elif source_type == "figure" and refs["figure"]:
                if normalize_object_id(chunk_locator.get("figure_id"), "figure") in refs["figure"]:
                    named.append((chunk, ""))
        # (ii) answer-value containment (only for table/figure-ish questions).
        if values and (table_ish or figure_ish):
            if table_ish and not figure_ish:
                type_order = ["table", "figure"]
            elif figure_ish and not table_ish:
                type_order = ["figure", "table"]
            else:
                type_order = ["figure", "table"] if primary == "figure" else ["table", "figure"]
            for source_type in type_order:
                for chunk in tf_chunks:
                    if str(chunk.get("source_type") or "") != source_type:
                        continue
                    content_norm = _norm_match_text(chunk.get("content"))
                    content_norm_nc = _strip_thousands(content_norm)
                    for value in values:
                        if _content_contains_value(content_norm, content_norm_nc, value):
                            value_hits.append((chunk, value))
                            break

    candidates: list[tuple[Record, str]] = named + value_hits
    # (iii) same-page table/figure next to an LLM-cited text_span.
    if not candidates and primary in ("table", "figure"):
        cited_pages: set[tuple[str, str]] = set()
        for item in llm_evidence:
            if str(item.get("source_type") or "") != "text_span":
                continue
            locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
            page = (locator or {}).get("page")
            if _page_missing(page):
                continue
            cited_pages.add((str(item.get("paper_id") or ""), str(page)))
        same_page: list[tuple[Record, str]] = []
        for paper_id in submitted_paper_ids:
            for chunk in tf_by_paper.get(paper_id, []):
                if str(chunk.get("source_type") or "") != primary:
                    continue
                chunk_pages = {str(p) for p in (chunk.get("page_numbers") or []) if p is not None}
                chunk_locator = parse_locator(chunk)
                if chunk_locator.get("page") is not None:
                    chunk_pages.add(str(chunk_locator.get("page")))
                if any((paper_id, page) in cited_pages for page in chunk_pages):
                    same_page.append((chunk, ""))
        # Multiple table/figure chunks share the cited page (Tables 1+2 both
        # on p7): prefer the chunk(s) actually containing a predicted table
        # cell value; fall back to all same-page chunks only when no chunk
        # contains any value.
        cell_values = _numeric_table_cell_values(answer)
        if cell_values and len(same_page) > 1:
            value_filtered: list[tuple[Record, str]] = []
            for chunk, matched in same_page:
                content_norm = _norm_match_text(chunk.get("content"))
                content_norm_nc = _strip_thousands(content_norm)
                if any(
                    _content_contains_value(content_norm, content_norm_nc, value)
                    for value in cell_values
                ):
                    value_filtered.append((chunk, matched))
            if value_filtered:
                same_page = value_filtered
        # Multi-paper questions: one same-page chunk per paper, so one page's
        # tables cannot fill every priority evidence slot.
        if str(sample.get("task_family") or "") == "multi_paper":
            per_paper_seen: set[str] = set()
            capped_same_page: list[tuple[Record, str]] = []
            for chunk, matched in same_page:
                paper_id = str(chunk.get("paper_id") or "")
                if paper_id in per_paper_seen:
                    continue
                per_paper_seen.add(paper_id)
                capped_same_page.append((chunk, matched))
            same_page = capped_same_page
        candidates.extend(same_page)

    items: list[Record] = []
    for chunk, matched in candidates:
        source_type = str(chunk.get("source_type") or "")
        chunk_locator = parse_locator(chunk)
        pages = fanout_pages(chunk, matched)
        text = compact_text(chunk.get("content"), max_chars=EVIDENCE_SNIPPET_CHARS)
        for page in pages:
            items.append(
                _evidence_item(str(chunk.get("paper_id") or ""), source_type, page, chunk_locator, text)
            )
    return items


def merge_evidence_items(
    priority: list[Record],
    llm_items: list[Record],
    cap: int = EVIDENCE_CAP,
    primary_type: str = "",
) -> list[Record]:
    """P1 merge: re-anchor/lookup items first, then LLM citations.

    Dedupes by the official coarse key and caps at ``cap``; when LLM
    citations exist, one slot is always reserved so they are never all
    dropped. For figure-primary questions where a figure item was re-anchored,
    at most ONE text_span LLM citation is kept (fan-out twins of the same
    quote on adjacent pages only dilute precision next to the gold figure).
    """
    if primary_type == "figure" and any(
        str(item.get("source_type") or "") == "figure" for item in priority
    ):
        kept: list[Record] = []
        text_spans = 0
        for item in llm_items:
            if str(item.get("source_type") or "") == "text_span":
                text_spans += 1
                if text_spans > 1:
                    continue
            kept.append(item)
        llm_items = kept
    merged: list[Record] = []
    seen: set[tuple[str, str, str, str]] = set()

    def try_add(item: Record) -> None:
        locator = item.get("locator") if isinstance(item.get("locator"), dict) else {}
        if _page_missing((locator or {}).get("page")):
            return
        key = evaluator_evidence_key(
            str(item.get("paper_id") or ""),
            str(item.get("source_type") or ""),
            locator or {},
        )
        if key in seen:
            return
        seen.add(key)
        merged.append(dict(item))

    reserve = 1 if llm_items else 0
    for item in priority:
        if len(merged) >= cap - reserve:
            break
        try_add(item)
    for item in llm_items:
        if len(merged) >= cap:
            break
        try_add(item)
    for item in priority:  # backfill any slot the LLM items left unused
        if len(merged) >= cap:
            break
        try_add(item)
    for index, item in enumerate(merged, start=1):
        item["evidence_id"] = f"ev_{index:03d}"
    return merged


# ---------------------------------------------------------------------------
# P5: reference/equation local lookup
# ---------------------------------------------------------------------------

# A single bibliography entry fits well inside 600 chars; the cap stops a
# missing next-entry marker from swallowing the rest of the chunk.
REFERENCE_ENTRY_MAX_CHARS = 600
# Injected excerpt budgets: the whole matching References region for counting
# questions, a tight window for single reference/equation lookups.
REFERENCE_COUNT_SNIPPET_CHARS = 4500
LOOKUP_SNIPPET_CHARS = 1500

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

_MATH_CHARS = "=≈≤≥±∑∏∫∇∂√·×→↦"


def _merge_overlapping_texts(texts: list[str], min_overlap: int = 40) -> str:
    """Concatenate chunk texts, dropping the duplicated suffix/prefix overlap
    DI chunking leaves between consecutive chunks (a reference entry repeated
    at a chunk boundary would otherwise be counted twice)."""
    merged = ""
    for text in texts:
        text = str(text or "")
        if not merged:
            merged = text
            continue
        overlap = 0
        for size in range(min(len(merged), len(text)), min_overlap - 1, -1):
            if merged.endswith(text[:size]):
                overlap = size
                break
        merged = merged + ("" if overlap else "\n") + text[overlap:]
    return merged


def detect_lookup_request(question: str) -> Optional[Record]:
    """Detect 'Nth/last reference', 'Equation (N)' and 'how many references
    ... <Surname>' question patterns."""
    q = str(question or "").lower()
    if re.search(r"\bhow many references\b", q) and re.search(
        r"\b(?:include|includes|list|lists|cite|cites|mention|mentions|by|with)\b", q
    ):
        # 'How many references ... include <Surname> as an author?' -> count
        # over the local References region; retrieval routinely surfaces only
        # part of a split reference list, which misleads the count.
        surname = re.search(
            r"\b(?:include|includes|list|lists|cite|cites|mention|mentions|by|with)\s+"
            r"([A-Z][a-z][A-Za-z'’-]+)",
            str(question or ""),
        )
        if surname:
            return {"kind": "reference_count", "surname": surname.group(1)}
    if re.search(r"\b(?:last|final)\s+(?:cited\s+|listed\s+)?reference\b", q):
        return {"kind": "reference", "last": True, "number": None}
    match = re.search(r"\b(\d+)\s*(?:st|nd|rd|th)\s+(?:cited\s+|listed\s+)?reference\b", q)
    if match:
        return {"kind": "reference", "last": False, "number": int(match.group(1))}
    match = re.search(
        r"\b(" + "|".join(_ORDINAL_WORDS) + r")\s+(?:cited\s+|listed\s+)?reference\b", q
    )
    if match:
        return {"kind": "reference", "last": False, "number": _ORDINAL_WORDS[match.group(1)]}
    match = re.search(r"\breference\s*(?:number\s*)?\[?(\d+)\]?", q)
    if match:
        return {"kind": "reference", "last": False, "number": int(match.group(1))}
    match = re.search(r"\b(?:equation|eq)\.?\s*\(?(\d+)\)?", q)
    if match:
        return {"kind": "equation", "number": int(match.group(1))}
    return None


_REFERENCE_ENTRY_RE = re.compile(r"\[\d{1,3}\]\s+[A-Z]")


def _reference_region_chunks(chunks: list[Record]) -> list[Record]:
    """Chunks of the References region.

    Section-labeled chunks win; only when no chunk is labeled do we fall back
    to entry-shaped content ('[N] Capitalized...'), so body-text chunks full
    of citation markers never hijack the lookup.
    """
    labeled: list[Record] = []
    entry_shaped: list[Record] = []
    for chunk in chunks:
        if str(chunk.get("source_type") or "") in ("table", "figure", "metadata"):
            continue
        section = str(chunk.get("section") or "").lower()
        content = str(chunk.get("content") or "")
        if "reference" in section:
            labeled.append(chunk)
        elif len(_REFERENCE_ENTRY_RE.findall(content)) >= 3:
            entry_shaped.append(chunk)
    return labeled or entry_shaped


def _extract_reference_entry(content: str, number: int) -> tuple[str, int]:
    """Return (entry text, start offset) of '[N] ...'; ('', -1) when absent."""
    matches = list(re.finditer(r"\[(\d{1,3})\]", content))
    for index, match in enumerate(matches):
        if int(match.group(1)) != number:
            continue
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )
        end = min(end, match.start() + REFERENCE_ENTRY_MAX_CHARS)
        entry = content[match.start() : end].strip()
        if entry:
            return entry, match.start()
    return "", -1


def find_reference_entry(
    chunks: list[Record], number: Optional[int], last: bool
) -> Optional[tuple[Record, str]]:
    region = _reference_region_chunks(chunks)
    if not region:
        return None
    if last:
        best = -1
        for chunk in region:
            for match in re.finditer(r"\[(\d{1,3})\]", str(chunk.get("content") or "")):
                best = max(best, int(match.group(1)))
        if best < 0:
            return None
        number = best
    if number is None:
        return None
    # The same entry can straddle two chunks (tail of one, head of the next);
    # prefer the chunk where the entry starts earliest, whose locator page is
    # the page the entry actually sits on.
    best: Optional[tuple[int, Record, str]] = None
    for chunk in region:
        entry, offset = _extract_reference_entry(str(chunk.get("content") or ""), number)
        if entry and (best is None or offset < best[0]):
            best = (offset, chunk, entry)
    if best is None:
        return None
    return best[1], best[2]


def find_equation_chunk(chunks: list[Record], number: int) -> Optional[tuple[Record, str]]:
    """Find the chunk defining equation (N): a line ENDING in '(N)' that looks
    like math (contains '=' or math symbols), not a bracketed citation."""
    pattern = re.compile(rf"\(\s*{number}\s*\)$")
    fallback: Optional[tuple[Record, str]] = None
    for chunk in chunks:
        if str(chunk.get("source_type") or "") in ("table", "figure", "metadata"):
            continue
        content = str(chunk.get("content") or "")
        lines = content.splitlines()
        for index, line in enumerate(lines):
            stripped = line.rstrip()
            if not stripped or not pattern.search(stripped):
                continue
            window = "\n".join(lines[max(0, index - 2) : index + 1]).strip()
            if "=" in stripped or any(ch in stripped for ch in _MATH_CHARS):
                return chunk, window
            if fallback is None:
                fallback = (chunk, window)
    return fallback


def perform_local_lookup(
    sample: Record, paper_id: str, chunks_dir: Optional[Path] = None
) -> Optional[Record]:
    """Run the P5 lookup for one paper; never raises."""
    try:
        request = detect_lookup_request(str(sample.get("question") or ""))
        if not request or not paper_id:
            return None
        chunks = load_paper_chunks(paper_id, chunks_dir)
        if not chunks:
            return None
        if request["kind"] == "reference_count":
            # Inject EVERY References-region chunk containing the queried
            # surname: DI chunking splits reference lists mid-region, so a
            # single retrieved chunk shows a misleadingly partial list.
            surname = str(request.get("surname") or "")
            if not surname:
                return None
            surname_re = re.compile(
                r"(?<![A-Za-z])" + re.escape(surname) + r"(?![a-z])"
            )
            matching = [
                chunk
                for chunk in _reference_chunks_extended(chunks)
                if surname_re.search(str(chunk.get("content") or ""))
            ]
            if not matching:
                return None
            chunk = matching[0]
            locator = parse_locator(chunk)
            page = locator.get("page")
            if _page_missing(page):
                return None
            # Consecutive DI chunks overlap: merge before showing, otherwise
            # a boundary-straddling entry is shown (and counted) twice.
            snippet = compact_text(
                _merge_overlapping_texts(
                    [str(item.get("content") or "") for item in matching[:3]]
                ),
                max_chars=REFERENCE_COUNT_SNIPPET_CHARS,
            )
            header = (
                f"[LOOKUP] paper_id={paper_id} source_type=citation_context page={page} "
                f"(locally verified References-region excerpt: ALL reference entries "
                f"mentioning '{surname}' are shown below; count them here; "
                "authoritative for this question)"
            )
            return {
                "paper_id": paper_id,
                "evidence_type": "citation_context",
                "snippet": snippet,
                "chunk": chunk,
                # snippet is already compact_text output, safe to embed as-is.
                "context_block": header + "\n" + snippet,
            }
        if request["kind"] == "reference":
            found = find_reference_entry(
                chunks, request.get("number"), bool(request.get("last"))
            )
            evidence_type = "citation_context"
        else:
            found = find_equation_chunk(chunks, int(request["number"]))
            evidence_type = "equation_algorithm"
        if not found:
            return None
        chunk, snippet = found
        locator = parse_locator(chunk)
        page = locator.get("page")
        if _page_missing(page):
            return None
        header = (
            f"[LOOKUP] paper_id={paper_id} source_type={evidence_type} page={page} "
            "(locally verified excerpt from this paper; authoritative for this question)"
        )
        return {
            "paper_id": paper_id,
            "evidence_type": evidence_type,
            "snippet": snippet,
            "chunk": chunk,
            "context_block": header + "\n" + compact_text(snippet, max_chars=LOOKUP_SNIPPET_CHARS),
        }
    except Exception:  # noqa: BLE001 - lookup must never break the run.
        return None


def lookup_evidence_items(lookup: Record) -> list[Record]:
    """Evidence items for a P5 lookup result, with P2 page fan-out."""
    chunk = lookup.get("chunk") or {}
    chunk_locator = parse_locator(chunk)
    # The snippet head is enough to localize the excerpt within the chunk.
    pages = fanout_pages(chunk, str(lookup.get("snippet") or "")[:120])
    text = compact_text(lookup.get("snippet"), max_chars=EVIDENCE_SNIPPET_CHARS)
    return [
        _evidence_item(
            str(lookup.get("paper_id") or ""),
            str(lookup.get("evidence_type") or ""),
            page,
            chunk_locator,
            text,
        )
        for page in pages
    ]


# ---------------------------------------------------------------------------
# P4: entity title/abstract search and scoped boost
# ---------------------------------------------------------------------------

# Generic ML/CS acronyms and dataset names that must never be treated as a
# paper-identifying entity (checked uppercased).
ENTITY_STOPWORDS = {
    "AI", "API", "AUC", "ACC", "ADAM", "BERT", "BLEU", "CNN", "COCO", "CPU",
    "CIFAR", "CV", "DNN", "DPO", "ELBO", "EM", "FID", "FLOP", "FLOPS", "GAN",
    "GPT", "GPU", "GPUS", "GRU", "HTML", "ID", "III", "IOU", "JSON", "KL",
    "LLM", "LLMS", "LORA", "LSTM", "MAE", "MAP", "MC", "MCQ", "ML", "MLP",
    "MLLM", "MNIST", "MSE", "NAS", "NLP", "NN", "OCR", "ODE", "OOD", "PDE",
    "PDF", "PSNR", "QA", "RAG", "RAM", "RELU", "RL", "RLHF", "RMSE", "RNN",
    "ROUGE", "SDE", "SGD", "SOTA", "SSIM", "SVM", "TPU", "URL", "VAE", "VIT",
    "VLM", "VLMS", "VQA", "XML",
    # Venues (never paper entities).
    "ACL", "AAAI", "CVPR", "ECCV", "EMNLP", "ICCV", "ICLR", "ICML", "IJCAI",
    "NAACL", "NEURIPS", "NIPS",
    # Common datasets/benchmarks with camel-case spellings.
    "IMAGENET", "WIKITEXT", "TRIVIAQA", "NATURALQ", "TRUTHFULQA", "WEBQA",
    "COMMONSENSEQA",
    # Benchmark/backbone names: they resolve to whichever unrelated paper
    # happens to title-match ('LLaVA' -> LLaVA-MoD) and burn boost slots.
    "POPE", "LLAVA", "QWEN", "GEMMA", "LLAMA", "VICUNA",
}

# BM25 acceptance floors, calibrated on the live index (see Stage A2 probes):
# exact named-entity title hits score ~8-10, full paper titles 18-63, generic
# words ~3.5; entity+question content hits on the right paper score 33-44.
TITLE_TOKEN_SCORE_FLOOR = 4.0
TITLE_STRONG_SCORE_FLOOR = 15.0
ABSTRACT_TOKEN_SCORE_FLOOR = 8.0
CONTENT_CONTEXT_SCORE_FLOOR = 25.0

# At most this many entity candidates are extracted per question, and at most
# this many resolved papers get a scoped per-paper search.
ENTITY_CANDIDATE_CAP = 6
ENTITY_PAPER_CAP = 4
# Scoped chunks fetched per boosted paper (before the 3+-papers fair share).
ENTITY_SCOPED_CHUNKS = 5

_QUOTED_ENTITY_RE = re.compile(r'["“]([^"”“]{2,90})["”]')
_HYPHEN_ENTITY_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")
_WORD_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]{1,29}\b")


def _looks_like_paper_title(text: str) -> bool:
    """Long, mostly-capitalized strings (MC options that are paper titles)."""
    text = str(text or "").strip()
    if len(text) < 40:
        return False
    words = re.findall(r"[A-Za-z][\w-]*", text)
    if len(words) < 4:
        return False
    capitalized = sum(1 for word in words if word[0].isupper())
    return capitalized / len(words) >= 0.45


def extract_entity_candidates(
    question: str,
    options: Optional[dict[str, str]] = None,
    cap: int = ENTITY_CANDIDATE_CAP,
) -> list[Record]:
    """P4: candidate proper nouns worth a title search.

    Priority: quoted strings, MC option texts that look like paper titles,
    CamelCase tokens (SecEmb, sCM, SimLingo), ALLCAPS tokens (IMM, ORION),
    hyphenated method names (D-FINE). Generic ML acronyms are filtered.
    """
    # NFKC folds superscripts and friends so 'D²PO' tokenizes as 'D2PO' and
    # resolves to its paper instead of silently producing no candidate.
    question = unicodedata.normalize("NFKC", str(question or ""))
    ordered: list[Record] = []
    seen: set[str] = set()

    def add(text: str, is_title: bool) -> None:
        text = re.sub(r"\s+", " ", str(text or "")).strip().strip(".,;:")
        if not text or text.upper() in ENTITY_STOPWORDS:
            return
        # Hyphenated dataset/backbone variants ('CIFAR-10', 'LLaVA-1.5'):
        # test the head token against the stopword list too.
        if "-" in text and text.split("-", 1)[0].upper() in ENTITY_STOPWORDS:
            return
        # Backbone-model names ('Gemma2-9B-Instruct', 'Llama-3-8B-Chat') are
        # never paper-identifying entities: a parameter-size segment or an
        # -Instruct/-Chat suffix marks them.
        if not is_title and re.search(
            r"-\d+(?:\.\d+)?[BbMm](?![A-Za-z0-9])|-(?:Instruct|Chat)(?![A-Za-z0-9])", text
        ):
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append({"text": text, "is_title": is_title})

    for match in _QUOTED_ENTITY_RE.findall(question):
        add(match, _looks_like_paper_title(match))
    for option_text in (options or {}).values():
        if _looks_like_paper_title(option_text):
            add(option_text, True)
    camel: list[str] = []
    allcaps: list[str] = []
    for token in _WORD_TOKEN_RE.findall(question):
        if re.fullmatch(r"[A-Z]{3,}[0-9]{0,2}", token):
            allcaps.append(token)
        elif re.search(r"[a-z0-9][A-Z]", token):
            camel.append(token)
    for token in camel:
        add(token, False)
    for token in allcaps:
        add(token, False)
    for token in _HYPHEN_ENTITY_RE.findall(question):
        if any(ch.isupper() for ch in token) and len(token) >= 4:
            add(token, False)
    return ordered[:cap]


def _entity_token_re(name: str) -> "re.Pattern[str]":
    """Case-SENSITIVE standalone-token match ('sCM' must not hit 'SCM')."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])")


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _titles_equivalent(query_title: str, hit_title: str) -> bool:
    """Exact normalized match, or one title is a prefix of the other (subtitle
    added/dropped). Blocks superstring collisions like 'Denoising Diffusion
    Probabilistic Models' -> 'Fingerprinting Denoising Diffusion ...'."""
    a = _norm_title(query_title)
    b = _norm_title(hit_title)
    if not a or not b or min(len(a), len(b)) < 15:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def _paper_metadata_token_match(paper_id: str, token_re: "re.Pattern[str]") -> bool:
    """True when the entity token appears in the paper's OWN title/abstract
    (local metadata chunk). A method's own paper names its acronym there
    ('Mixture of Decoding (MoD)'); a merely-citing hub never does."""
    for chunk in load_paper_chunks(paper_id):
        if str(chunk.get("source_type") or "") == "metadata":
            return bool(token_re.search(str(chunk.get("content") or "")))
    return False


def _paper_info(hit: Record) -> Record:
    return {
        "paper_id": str(hit.get("paper_id") or ""),
        "title": hit.get("title") or "",
        "venue": hit.get("venue") or "",
        "year": hit.get("year"),
        "score": float(hit.get("search_score") or 0.0),
        "best_rank": 1,
    }


def resolve_entity_papers(
    search_client: Any,
    candidate: Record,
    question: str,
    *,
    max_papers: int = 2,
) -> list[Record]:
    """Resolve one entity candidate to corpus papers via a search ladder.

    Rung 1: title BM25 (token-verified >= TITLE_TOKEN_SCORE_FLOOR, or a
    strong un-verified score for long title strings). Short acronyms
    (<= 4 chars: EDM, DDPM, IMM ...) collide across domains, so a title match
    must be CONFIRMED by the same paper appearing token-verified in a content
    search of '<name> <question>' — the question terms disambiguate. Rung 2:
    best token-verified hit of that contextual content search. Rung 3
    (5+ char names only): abstract BM25, token-verified. Returns [] when
    nothing credible; never raises.
    """
    try:
        name = str(candidate.get("text") or "")
        if not name:
            return []
        token_re = _entity_token_re(name)
        select = SEARCH_SELECT + ["abstract"]
        short_acronym = len(name) <= 4 and not candidate.get("is_title")

        title_accepts: list[Record] = []
        seen: set[str] = set()
        for hit in keyword_search(search_client, name, fields=["title"], top=3, select=select):
            score = float(hit.get("search_score") or 0.0)
            title = str(hit.get("title") or "")
            paper_id = str(hit.get("paper_id") or "")
            if not paper_id or paper_id in seen:
                continue
            if (token_re.search(title) and score >= TITLE_TOKEN_SCORE_FLOOR) or (
                candidate.get("is_title")
                and score >= TITLE_STRONG_SCORE_FLOOR
                and _titles_equivalent(name, title)
            ):
                seen.add(paper_id)
                info = _paper_info(hit)
                info["resolved_via"] = "title"
                title_accepts.append(info)
        if title_accepts and not short_acronym:
            return title_accepts[:max_papers]
        if candidate.get("is_title"):
            # Long title strings that miss on the title field are not worth
            # noisy abstract/content fallbacks.
            return title_accepts[:max_papers]

        contextual = f"{name} {question}".strip()
        content_first_hit: dict[str, Record] = {}  # best-ranked token-verified hit
        content_counts: dict[str, int] = {}  # token-verified hits per paper (top-20)
        for hit in keyword_search(search_client, contextual, fields=["content"], top=20, select=select):
            if not token_re.search(str(hit.get("content") or "")):
                continue
            paper_id = str(hit.get("paper_id") or "")
            if not paper_id:
                continue
            content_counts[paper_id] = content_counts.get(paper_id, 0) + 1
            content_first_hit.setdefault(paper_id, hit)
        content_token_papers = list(content_first_hit)  # rank order preserved
        content_accepts: list[Record] = []  # token-verified AND above floor
        for paper_id in content_token_papers:
            hit = content_first_hit[paper_id]
            if float(hit.get("search_score") or 0.0) < CONTENT_CONTEXT_SCORE_FLOOR:
                continue
            # A method's OWN paper names its acronym in its title/abstract
            # (strong signal, checked against the local metadata chunk) and
            # token-matches the contextual search across many chunks; a
            # merely-citing paper does neither, matching only in 1-2
            # appendix/comparison-table chunks, and must not be accepted on a
            # single such hit.
            own_paper = _paper_metadata_token_match(paper_id, token_re)
            if not own_paper and content_counts.get(paper_id, 0) < 2:
                continue
            info = _paper_info(hit)
            info["resolved_via"] = "abstract" if own_paper else "content"
            content_accepts.append(info)
        if title_accepts:
            confirmed = [
                paper for paper in title_accepts if paper["paper_id"] in content_token_papers
            ]
            if confirmed:
                if (
                    short_acronym
                    and content_accepts
                    and content_accepts[0]["paper_id"]
                    not in {paper["paper_id"] for paper in confirmed}
                ):
                    # Short acronym whose STRONGEST content evidence points at
                    # a different paper than the merely-confirmed title match
                    # (hyphen compounds like 'OW-VAP' token-verify 'VAP'):
                    # trust the content evidence.
                    return content_accepts[:1]
                return confirmed[:max_papers]
        if content_accepts:
            return content_accepts[:1]

        if not short_acronym:
            for hit in keyword_search(search_client, name, fields=["abstract"], top=5, select=select):
                score = float(hit.get("search_score") or 0.0)
                if score < ABSTRACT_TOKEN_SCORE_FLOOR:
                    break
                if token_re.search(str(hit.get("abstract") or "")):
                    info = _paper_info(hit)
                    info["resolved_via"] = "abstract"
                    return [info]
        return []
    except Exception:  # noqa: BLE001 - entity resolution is best effort.
        return []


def entity_boost_hits(
    sample: Record,
    options: Optional[dict[str, str]],
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    args: argparse.Namespace,
) -> tuple[list[Record], list[Record]]:
    """P4: chunks of question-named papers, fetched by scoped hybrid search.

    Entities are resolved to papers via title/abstract search; for each
    resolved paper the question is re-searched WITHIN that paper. Returns
    (prepend_hits, append_hits): only title/abstract-verified resolutions get
    the strong prepend (rank boost into rank_papers and build_context);
    content-rung resolutions are appended after the organic results so a
    citing 'hub' paper can never displace organically-retrieved evidence.
    Never raises.
    """
    try:
        question = str(sample.get("question") or "")
        candidates = extract_entity_candidates(question, options)
        if not candidates:
            return [], []
        resolved: dict[str, Record] = {}
        for candidate in candidates:
            for paper in resolve_entity_papers(search_client, candidate, question):
                resolved.setdefault(paper["paper_id"], paper)
        if not resolved:
            return [], []
        strong_ids = [
            paper_id
            for paper_id, paper in resolved.items()
            if str(paper.get("resolved_via") or "title") != "content"
        ]
        content_ids = [
            paper_id
            for paper_id, paper in resolved.items()
            if str(paper.get("resolved_via") or "") == "content"
        ]
        # With >=3 confidently (title/abstract) resolved papers, generic
        # contextual-content resolutions are junk-prone slot fillers ('IPC'
        # -> an unrelated dataset-distillation paper): skip them entirely.
        if len(strong_ids) >= 3:
            content_ids = []
        boosted_ids = (strong_ids + content_ids)[:ENTITY_PAPER_CAP]
        # Paper-fair share: with 3+ resolved papers, shrink the per-paper
        # scoped fetch so boosted papers cannot monopolize the context.
        per_paper = (
            ENTITY_SCOPED_CHUNKS
            if len(boosted_ids) < 3
            else max(2, 10 // len(boosted_ids))
        )
        scoped_by_paper: dict[str, list[Record]] = {}
        for paper_id in boosted_ids:
            try:
                scoped_by_paper[paper_id] = search_chunks(
                    search_client,
                    openai_client,
                    openai_settings,
                    query=question,
                    top_chunks=per_paper,
                    vector_k=20,
                    query_type="hybrid",
                    filter_expr=f"paper_id eq '{_escape_odata(paper_id)}'",
                )
            except Exception:  # noqa: BLE001 - one bad scope must not kill the boost.
                scoped_by_paper[paper_id] = []

        def interleave(paper_ids: list[str]) -> list[Record]:
            # Round-robin: every resolved paper lands its top scoped chunk in
            # context before any paper gets its 2nd..Nth chunk.
            out: list[Record] = []
            for index in range(per_paper):
                for paper_id in paper_ids:
                    scoped = scoped_by_paper.get(paper_id) or []
                    if index < len(scoped):
                        out.append(scoped[index])
            return out

        strong_set = set(strong_ids)
        prepend_hits = interleave([pid for pid in boosted_ids if pid in strong_set])
        append_hits = interleave([pid for pid in boosted_ids if pid not in strong_set])
        if len(boosted_ids) < 3:
            # 1-2 verified papers cannot flood the pool the way 4 could, and
            # their DEEP scoped chunks (an appendix table holding the asked
            # value is often scoped rank 4-5) are exactly what the boost is
            # for: keep the full scoped set prepended.
            return prepend_hits, append_hits
        # 3+ boosted papers: cap the prepended share of the context (~1/3 of
        # the char budget) so mid-rank organic evidence can never be wholly
        # evicted; overflow chunks stay available at the tail instead of
        # being dropped.
        budget = (
            max(0, int(getattr(args, "context_chars", DEFAULT_CONTEXT_CHARS) or DEFAULT_CONTEXT_CHARS))
            // 3
        )
        capped: list[Record] = []
        used = 0
        for hit in prepend_hits:
            size = len(str(hit.get("content") or ""))
            if capped and used + size > budget:
                append_hits.append(hit)
                continue
            capped.append(hit)
            used += size
        return capped, append_hits
    except Exception:  # noqa: BLE001 - the boost is best effort.
        return [], []


# ---------------------------------------------------------------------------
# P3: comparison-group expansion (forward extraction + citing-hub mediation)
# ---------------------------------------------------------------------------

# Multi-paper gold sets almost always hold the 4 co-compared methods; expand
# short submissions toward this size, with a cap on hub excerpts per pass.
COMPARISON_TARGET_PAPERS = 4
COMPARISON_EXCERPT_CAP = 8

# Both P3 extraction prompts say "at most 12 entries"; the resolver slices
# to the same bound.
METHOD_ENTRY_CAP = 12

# Reverse-citation hub mining: quoted-title BM25 search depth, and how many
# hubs are excerpted per expansion pass.
HUB_TITLE_SEARCH_TOP = 50
HUB_CAP = 5

# max_tokens for the two P3 JSON extraction calls (12 short entries fit).
EXPAND_MAX_TOKENS = 500
HUB_EXTRACT_MAX_TOKENS = 600

_REF_DENSITY_RE = re.compile(r",\s*20\d\d[a-z]?\.|arXiv|\bIn Proceedings\b|\bIn International\b")


def _reference_chunks_extended(chunks: list[Record]) -> list[Record]:
    """The labeled References region plus following reference-dense chunks.

    Two-column reference lists often continue into chunks whose section label
    is a stale running header, so the labeled region alone misses entries.
    """
    region = _reference_region_chunks(chunks)
    if not region:
        return region
    region_ids = {id(chunk) for chunk in region}
    last_index = max(
        (index for index, chunk in enumerate(chunks) if id(chunk) in region_ids),
        default=-1,
    )
    extended = list(region)
    for chunk in chunks[last_index + 1 : last_index + 7]:
        if str(chunk.get("source_type") or "") in ("table", "figure", "metadata"):
            continue
        content = str(chunk.get("content") or "")
        if len(_REF_DENSITY_RE.findall(content)) >= 8:
            extended.append(chunk)
        else:
            break
    return extended


def _selected_references_text(
    paper_ids: list[str],
    chunks_dir: Optional[Path],
    max_chars: int = 16000,
) -> str:
    """References-region text of the selected papers (local chunk files);
    lets the P3 expansion LLM map method abbreviations to full cited titles."""
    parts: list[str] = []
    used = 0
    for paper_id in paper_ids:
        chunks = load_paper_chunks(paper_id, chunks_dir)
        for chunk in _reference_chunks_extended(chunks):
            text = compact_text(chunk.get("content"), max_chars=5000)
            if not text:
                continue
            block = f"[references of {paper_id}]\n{text}"
            if used + len(block) > max_chars:
                return "\n\n".join(parts)
            parts.append(block)
            used += len(block)
    return "\n\n".join(parts)


def _title_phrase_re(title: str) -> Optional["re.Pattern[str]"]:
    """Whitespace/punctuation-flexible regex for an exact paper title phrase.

    None for short/one-word titles ('Attention'), which would match far too
    much text to be treated as a citation of that specific paper.
    """
    tokens = _norm_title(title).split()
    if len(tokens) < 2 or len(" ".join(tokens)) < 15:
        return None
    return re.compile(r"[^a-z0-9]+".join(tokens))


def _reverse_citing_hubs(
    search_client: Any,
    titles: list[str],
    exclude_ids: set[str],
) -> list[str]:
    """Corpus papers whose text quotes a group paper's exact title, i.e.
    papers citing/comparing the group (the forward direction cannot see
    concurrent work: TCM never mentions sCT/IMM, but both quote ECM's title).

    These 'hubs' typically benchmark the group members side by side, so their
    comparison tables reveal the remaining group members. Ranked by how many
    group titles they quote, then by BM25 score. Phrase-verified (quoted BM25
    alone still returns near-misses).
    """
    quoted: dict[str, set[str]] = {}
    best_score: dict[str, float] = {}
    for title in titles:
        phrase_re = _title_phrase_re(title)
        if phrase_re is None:
            continue
        query = '"' + title.replace('"', " ").strip() + '"'
        for hit in keyword_search(
            search_client, query, fields=["content"], top=HUB_TITLE_SEARCH_TOP, select=SEARCH_SELECT
        ):
            content = str(hit.get("content") or "")
            if not phrase_re.search(content.lower()):
                continue
            paper_id = str(hit.get("paper_id") or "")
            if not paper_id or paper_id in exclude_ids:
                continue
            quoted.setdefault(paper_id, set()).add(title)
            score = float(hit.get("search_score") or 0.0)
            best_score[paper_id] = max(best_score.get(paper_id, 0.0), score)
    return sorted(
        quoted,
        key=lambda paper_id: (-len(quoted[paper_id]), -best_score[paper_id]),
    )


def _group_alias_tokens(
    group: list[Record], chunks_dir: Optional[Path]
) -> list[str]:
    """Distinctive short method tokens of the group papers ('TCM', 'sCT',
    'ECT' ...), mined from each paper's own non-references text: short tokens
    that are not a simply-capitalized word (so 'The'/'Figure' drop out, 'TCM'
    and 'sCT' stay), are not generic ML acronyms, and repeat like a method
    name does. Used only to RANK hub chunks (a results-table chunk containing
    a group alias is likely the comparison table naming the whole group).
    Tokens repeating across SEVERAL other group papers (NFE, EDM - field-wide
    jargon, not method names) are dropped; a real method alias like 'TCM'
    repeats in its own paper and at most in a group member citing it."""
    per_paper: list[dict[str, int]] = []
    for paper in group:
        counts: dict[str, int] = {}
        chunks = load_paper_chunks(str(paper.get("paper_id") or ""), chunks_dir)
        if chunks:
            reference_ids = {id(chunk) for chunk in _reference_chunks_extended(chunks)}
            for chunk in chunks:
                if id(chunk) in reference_ids:
                    continue
                content = str(chunk.get("content") or "")
                for token in re.findall(
                    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+-]{1,7}(?![A-Za-z0-9])", content
                ):
                    if not re.search(r"[A-Z]", token):
                        continue
                    if re.fullmatch(r"[A-Z][a-z0-9+-]*", token):
                        continue
                    if token.upper() in ENTITY_STOPWORDS:
                        continue
                    counts[token] = counts.get(token, 0) + 1
        per_paper.append(counts)
    aliases: list[str] = []
    for index, counts in enumerate(per_paper):
        candidates = [
            token
            for token, count in counts.items()
            if count >= 8
            and sum(
                1
                for position, other in enumerate(per_paper)
                if position != index and other.get(token, 0) >= 3
            )
            < 2
        ]
        for token in sorted(candidates, key=lambda token: -counts[token])[:5]:
            if token not in aliases:
                aliases.append(token)
    return aliases


def _first_author_surname(paper_id: str, chunks_dir: Optional[Path]) -> str:
    """First author's surname from the paper's local metadata chunk."""
    for chunk in load_paper_chunks(paper_id, chunks_dir):
        if str(chunk.get("source_type") or "") != "metadata":
            continue
        match = re.search(
            r"Authors?:\s*([^,\n]+)", str(chunk.get("content") or "")
        )
        if not match:
            return ""
        tokens = match.group(1).strip().split()
        return tokens[-1] if tokens else ""
    return ""


def _hub_comparison_excerpts(
    group: list[Record],
    hub_ids: list[str],
    chunks_dir: Optional[Path],
    cap: int = COMPARISON_EXCERPT_CAP,
) -> list[tuple[str, str]]:
    """(hub_id, excerpt) comparison-table/prose pairs from citing 'hub' papers.

    A hub whose results table lists a group member as a baseline row usually
    lists the OTHER group members right next to it ('TCM (Lee et al., 2025)' /
    'sCT (Lu & Song, 2025)' / 'IMM [52]' ...), which is exactly the evidence
    the extraction call needs to see who is compared with whom. Chunks are
    scored by (a) group method alias tokens present ('sCT', 'TCM' - the
    strongest sign of a comparison table row naming a group member; numeric
    citations like 'IMM [52]' carry no surname), (b) '<first-author surname>
    ... <year>' citation matches, (c) looking like a results table. The hub's
    references region is skipped because bibliography adjacency is
    alphabetical noise, not co-comparison.
    """
    surname_res: list["re.Pattern[str]"] = []
    for paper in group:
        surname = _first_author_surname(str(paper.get("paper_id") or ""), chunks_dir)
        if len(surname) >= 2:
            surname_res.append(
                re.compile(
                    r"(?<![A-Za-z])" + re.escape(surname) + r"(?![a-z])[\s\S]{0,40}?20\d\d"
                )
            )
    alias_res = [
        _entity_token_re(alias) for alias in _group_alias_tokens(group, chunks_dir)
    ]
    excerpts: list[tuple[str, str]] = []
    for hub_id in hub_ids:
        chunks = load_paper_chunks(hub_id, chunks_dir)
        if not chunks:
            continue
        reference_ids = {id(chunk) for chunk in _reference_chunks_extended(chunks)}
        # (score, chunk order, chunk, offset of the first alias/surname match)
        scored: list[tuple[int, int, Record, int]] = []
        for order, chunk in enumerate(chunks):
            if id(chunk) in reference_ids:
                continue
            if str(chunk.get("source_type") or "") in ("metadata", "figure"):
                continue
            content = str(chunk.get("content") or "")
            matches = [
                match
                for pattern in alias_res + surname_res
                for match in [pattern.search(content)]
                if match
            ]
            alias_hits = sum(1 for pattern in alias_res if pattern.search(content))
            surname_hits = len(matches) - alias_hits
            if not matches:
                continue
            is_table = (
                str(chunk.get("source_type") or "") == "table" or "Table" in content
            )
            score = 10 * alias_hits + 10 * surname_hits + (5 if is_table else 0)
            first = min(match.start() for match in matches)
            scored.append((score, order, chunk, first))
        if not scored:
            # Fallback: results-table chunks with a 'Method' header column
            # (vs. ablation/config tables) still show co-compared methods.
            for order, chunk in enumerate(chunks):
                if str(chunk.get("source_type") or "") != "table":
                    continue
                content = str(chunk.get("content") or "")
                cells = len(re.findall(r"\d+\.\d", content))
                if re.search(r"(?i)\bmethods?\b", content[:200]) and cells >= 3:
                    scored.append((cells, order, chunk, 0))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for _, _, chunk, first in scored[:2]:
            content = str(chunk.get("content") or "")
            if first > 400:
                # Long tables: keep the header row but window on the match,
                # otherwise the group member's row falls beyond the cut.
                window = content[:200] + " ... " + content[first - 300 : first + 700]
            else:
                window = content
            excerpts.append((hub_id, compact_text(window, max_chars=1200)))
            if len(excerpts) >= cap:
                return excerpts
    return excerpts


def _numbered_reference_entries(
    hub_id: str,
    numbers: list[int],
    chunks_dir: Optional[Path],
) -> list[str]:
    """Bibliography entries '[N] ...' of a hub for the given citation numbers.

    Hubs with numeric citations name baselines as 'IMM [52]'; the referenced
    entry (with the full paper title) is what makes that name resolvable, and
    a whole-bibliography dump often gets truncated before entry [52].
    """
    if not numbers:
        return []
    text = "\n".join(
        str(chunk.get("content") or "")
        for chunk in _reference_chunks_extended(load_paper_chunks(hub_id, chunks_dir))
    )
    if not text:
        return []
    entries: list[str] = []
    bibliography_like = re.compile(r",\s*20\d\d|arXiv|preprint|Proceedings|Conference|Journal")
    for number in sorted(set(numbers))[:20]:
        # '[N]' also occurs as an in-text citation marker (table rows); only
        # a slice that looks like a bibliography entry qualifies.
        for match in re.finditer(rf"\[{number}\]", text):
            following = re.search(r"\[\d+\]", text[match.end() :])
            end = match.end() + (following.start() if following else 300)
            entry = text[match.start() : end]
            if bibliography_like.search(entry):
                entries.append(compact_text(entry, max_chars=300))
                break
    return entries


def _resolve_method_entries(
    search_client: Any,
    methods: list[Any],
    question: str,
) -> list[tuple[int, int, Record]]:
    """Resolve LLM-listed {name,title} method entries against the corpus.

    Returns (preference, order, paper) tuples: bibliography-title resolutions
    (high precision) are preferred over bare method names (acronym
    collisions). Methods whose papers are not in the corpus (e.g. pre-2025
    baselines like iCT/CTM) simply fail to resolve and drop out.
    """
    resolutions: list[tuple[int, int, Record]] = []
    for order, method in enumerate(methods[:METHOD_ENTRY_CAP]):
        if not isinstance(method, dict):
            continue
        name = str(method.get("name") or "").strip()
        title = str(method.get("title") or "").strip()
        if title.lower() in ("none", "null"):
            title = ""
        hits: list[Record] = []
        via_title = False
        if title:
            hits = resolve_entity_papers(
                search_client, {"text": title, "is_title": True}, question, max_papers=1
            )
            via_title = bool(hits)
        if not hits and name and name.upper() not in ENTITY_STOPWORDS:
            hits = resolve_entity_papers(
                search_client, {"text": name, "is_title": False}, question, max_papers=1
            )
        if hits:
            resolutions.append((0 if via_title else 1, order, hits[0]))
    return resolutions


def _extract_hub_cocompared_methods(
    question: str,
    group: list[Record],
    excerpts: list[tuple[str, str]],
    hub_ids: list[str],
    openai_client: Any,
    openai_settings: OpenAISettings,
    chunks_dir: Optional[Path],
) -> list[Any]:
    """One JSON call: method names co-compared with the group members inside
    the hub excerpts (baseline rows of the same results table / explicit
    comparisons), each mapped to its cited full title via the hubs'
    bibliographies (author-year or '[N]' citation numbers)."""
    group_lines = "\n".join(
        f"- {paper.get('title') or paper.get('paper_id')}" for paper in group
    )
    # Per-hub bibliography budget (one long bibliography must not starve the
    # others) plus the exact '[N]' entries cited inside each hub's excerpts.
    reference_parts: list[str] = []
    for hub_id in hub_ids:
        hub_text = " ".join(text for owner, text in excerpts if owner == hub_id)
        numbers = [int(number) for number in re.findall(r"\[(\d{1,3})\]", hub_text)]
        for entry in _numbered_reference_entries(hub_id, numbers, chunks_dir):
            reference_parts.append(f"[bibliography entry of {hub_id}] {entry}")
        block = _selected_references_text([hub_id], chunks_dir, max_chars=5200)
        if block:
            reference_parts.append(block)
    references = "\n\n".join(reference_parts)
    prompt_parts = [
        "You are analyzing excerpts from scientific papers that benchmark "
        "methods against each other. The following papers form a comparison "
        "group of directly competing methods:",
        group_lines,
        "From the excerpts below, list the OTHER method names that are "
        "compared side by side WITH the group members' methods: baseline rows "
        "of the same results table (same category/section of the table as the "
        "group member's row) or explicit head-to-head comparisons. Be "
        "exhaustive: include EVERY such co-listed method, even one appearing "
        "in a single excerpt. Exclude the excerpt paper's own method (rows "
        "marked 'Ours'), methods from unrelated table categories, datasets, "
        "metrics, and generic terms. Order entries by how directly and "
        "frequently they are compared with the group members.",
        "For each method, find its cited paper in the bibliography text: "
        "match the author names/year attached to the method (e.g. 'sCT (Lu & "
        "Song, 2025)' -> the Lu & Song 2025 entry) or the citation number "
        "(e.g. 'IMM [52]' -> entry [52] of that paper's bibliography) and "
        "copy that entry's paper title EXACTLY. Use null when the title is "
        "not visible.",
        'Return JSON: {"methods": [{"name": "<short method name>", '
        '"title": "<full paper title or null>"}]} with at most 12 entries.',
        f"Question being answered: {question}",
        "Excerpts:\n"
        + "\n".join(f"[from {owner}] {text}" for owner, text in excerpts),
    ]
    if references:
        prompt_parts.append(f"Bibliography text of the excerpt papers:\n{references}")
    kwargs: Record = {
        "model": openai_settings.chat_deployment,
        "messages": [{"role": "user", "content": "\n\n".join(prompt_parts)}],
        "temperature": 0,
        "max_tokens": HUB_EXTRACT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    response = retry_chat_completion(openai_client, kwargs)
    payload, ok = try_parse_json_object(response.choices[0].message.content or "")
    if not ok:
        return []
    methods = payload.get("methods")
    return methods if isinstance(methods, list) else []


def expand_comparison_group(
    sample: Record,
    papers: list[Record],
    context: str,
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    chunks_dir: Optional[Path],
) -> list[Record]:
    """P3: fill a short multi_paper submission up to 4 with co-compared methods.

    27/29 multi_paper gold sets contain exactly the 4 papers whose methods are
    compared together. Stage 1 (forward): a small JSON call lists the
    co-compared method names visible in the context (tables/baselines); each
    name is resolved against the corpus with the P4 ladder. Stage 2 (when
    slots remain): forward extraction cannot see CONCURRENT group members the
    selected papers never mention (TCM never names sCT/IMM), so 'hub' papers
    that quote a group member's exact title are located, their comparison
    tables/discussions are excerpted, and the method names co-listed with the
    group members there are extracted and resolved with the same P4 ladder
    (out-of-corpus baselines drop out at resolution). Existing papers are
    never removed and junk below the search-score floors is never appended.
    Never raises.
    """
    try:
        needed = COMPARISON_TARGET_PAPERS - len(papers)
        if needed <= 0:
            return []
        question = str(sample.get("question") or "")
        existing_ids = {str(paper.get("paper_id") or "") for paper in papers}
        references = _selected_references_text(sorted(existing_ids), chunks_dir)
        prompt_parts = [
            "You are analyzing retrieved context from scientific papers.",
            "List the method/model/system names that are compared AGAINST EACH OTHER "
            "in the context below (baseline rows in results tables, explicit comparison "
            "discussions) or explicitly named in the question. Exclude datasets, metrics, "
            "and generic terms. For each method, look for its cited paper in the "
            "bibliography text: match the author names/year attached to the method in "
            "the context (e.g. 'ECM (Geng et al., 2024)' -> the Geng et al., 2024 "
            "entry) and copy that entry's paper title EXACTLY. Use null when the title "
            "is not visible.",
            'Return JSON: {"methods": [{"name": "<short method name>", '
            '"title": "<full paper title or null>"}]} with at most 12 entries.',
            f"Question: {question}",
            f"Context:\n{compact_text(context, max_chars=10000)}",
        ]
        if references:
            prompt_parts.append(f"Bibliography text:\n{references}")
        kwargs: Record = {
            "model": openai_settings.chat_deployment,
            "messages": [{"role": "user", "content": "\n\n".join(prompt_parts)}],
            "temperature": 0,
            "max_tokens": EXPAND_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        response = retry_chat_completion(openai_client, kwargs)
        payload, ok = try_parse_json_object(response.choices[0].message.content or "")
        if not ok:
            return []
        methods = payload.get("methods")
        if not isinstance(methods, list):
            return []
        # Resolve every listed method first, then fill slots preferring
        # bibliography-title resolutions (high precision) over bare method
        # names (acronym collisions).
        added: list[Record] = []
        for _, _, paper in sorted(_resolve_method_entries(search_client, methods, question)):
            if len(added) >= needed:
                break
            paper_id = str(paper.get("paper_id") or "")
            if paper_id and paper_id not in existing_ids:
                existing_ids.add(paper_id)
                added.append(paper)

        # Stage 2: hub-mediated co-comparison. At most 2 picks per invocation
        # on purpose: papers accepted here contribute their own reverse
        # citations to the caller's next expansion pass (IMM only becomes
        # reachable through papers citing ECM, i.e. after ECM joins).
        if len(added) < needed:
            group = papers + added
            titles = [str(paper.get("title") or "") for paper in group]
            hub_ids = _reverse_citing_hubs(
                search_client, [title for title in titles if title], existing_ids
            )[:HUB_CAP]
            excerpts = _hub_comparison_excerpts(group, hub_ids, chunks_dir)
            if excerpts:
                hub_methods = _extract_hub_cocompared_methods(
                    question,
                    group,
                    excerpts,
                    hub_ids,
                    openai_client,
                    openai_settings,
                    chunks_dir,
                )
                slots = min(needed, len(added) + 2)
                for pref, _, paper in sorted(
                    _resolve_method_entries(search_client, hub_methods, question)
                ):
                    if len(added) >= slots:
                        break
                    if pref != 0:
                        # Hub tables list many out-of-corpus baselines (PD,
                        # TRACT, iCT ...) whose bare acronyms collide with
                        # unrelated corpus papers; only bibliography-title
                        # verified resolutions are trusted here.
                        continue
                    paper_id = str(paper.get("paper_id") or "")
                    if paper_id and paper_id not in existing_ids:
                        existing_ids.add(paper_id)
                        added.append(paper)
        return added
    except Exception:  # noqa: BLE001 - expansion is best effort.
        return []


# ---------------------------------------------------------------------------
# P6: bounded venue/corpus enumeration
# ---------------------------------------------------------------------------

# Candidate pool fed to the batched verification call, the submission cap,
# and the minimum qualifying count for a verdict confident enough to
# suppress P3 expansion.
ENUM_CANDIDATE_CAP = 20
ENUM_SUBMIT_CAP = 8
ENUM_MIN_QUALIFYING = 2

# Wide-search sizes: enumeration recall needs a much deeper pool than normal
# retrieval; figure-caption searches stay narrower.
ENUM_WIDE_TOP = 60
ENUM_FIGURE_TOP = 30

# max_tokens for the batched verification call (a short paper_ids array).
ENUM_VERIFY_MAX_TOKENS = 400

# Plural 'papers' is required on purpose: singular phrasings like 'Which
# paper reports the highest BLEU?' are selection questions, not enumeration,
# and must never let the enumeration verdict replace the normal paper set.
# Genuinely plural-but-oddly-phrased enumerations are still caught by the
# decomposition LLM's enumeration flag, which is OR'ed with this fallback.
_ENUM_QUESTION_RES = [
    re.compile(r"\bwhich\s+(?:\w+[\s,-]+){0,3}papers\b", re.IGNORECASE),
    re.compile(r"\ball\s+(?:the\s+)?papers\s+(?:that|which|with)\b", re.IGNORECASE),
    re.compile(r"\b(?:list|find|identify|enumerate)\s+(?:all\s+)?(?:the\s+)?papers\b", re.IGNORECASE),
]


def question_is_enumeration(question: str) -> bool:
    """P6 regex fallback for 'Which <VENUE> papers ...' / 'all papers that ...'.

    Matches only plural 'papers'; see the comment on _ENUM_QUESTION_RES.
    """
    question = str(question or "")
    return any(pattern.search(question) for pattern in _ENUM_QUESTION_RES)


def _local_paper_abstract(paper_id: str, chunks_dir: Optional[Path], max_chars: int = 700) -> str:
    """Title/authors/abstract text from the paper's local metadata chunk."""
    for chunk in load_paper_chunks(paper_id, chunks_dir):
        if str(chunk.get("source_type") or "") == "metadata":
            return compact_text(chunk.get("content"), max_chars=max_chars)
    for chunk in load_paper_chunks(paper_id, chunks_dir):
        if "abstract" in str(chunk.get("section") or "").lower():
            return compact_text(chunk.get("content"), max_chars=max_chars)
    return ""


def run_corpus_enumeration(
    sample: Record,
    decomposition: Record,
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    args: argparse.Namespace,
) -> Optional[tuple[list[Record], list[Record]]]:
    """P6: bounded venue/corpus enumeration.

    Re-runs the decomposition subqueries wide (top=60) with the venue/year
    filter, plus explicit figure-caption searches (several enumeration
    questions hinge on method/framework figures), aggregates up to 20
    paper-level candidates, then ONE batched LLM verification call decides
    which candidates actually qualify. Returns (qualifying paper records,
    boost chunk hits) or None; never raises.
    """
    try:
        question = str(sample.get("question") or "")
        queries: list[str] = [question]
        for subquery in decomposition.get("subqueries") or []:
            text = str(subquery or "").strip()
            if text and text not in queries:
                queries.append(text)
            if len(queries) >= 4:
                break
        filter_expr = build_filter_expression(
            decomposition.get("venue"), decomposition.get("year")
        )
        figure_filter = "source_type eq 'figure'"
        if filter_expr:
            figure_filter = f"{filter_expr} and {figure_filter}"

        hit_lists: list[list[Record]] = []
        for search_query in queries:
            hit_lists.append(
                search_chunks(
                    search_client,
                    openai_client,
                    openai_settings,
                    query=search_query,
                    top_chunks=ENUM_WIDE_TOP,
                    vector_k=ENUM_WIDE_TOP,
                    query_type=args.query_type,
                    filter_expr=filter_expr,
                    search_fields=parse_search_fields(args.search_fields),
                )
            )
        for search_query in queries[:2]:
            hit_lists.append(
                search_chunks(
                    search_client,
                    openai_client,
                    openai_settings,
                    query=search_query,
                    top_chunks=ENUM_FIGURE_TOP,
                    vector_k=ENUM_FIGURE_TOP,
                    query_type="hybrid",
                    filter_expr=figure_filter,
                )
            )

        # Paper-level RRF aggregation across all lists; remember each paper's
        # first two hits (encounter order) as its representative snippets.
        agg: dict[str, Record] = {}
        best_hits: dict[str, list[Record]] = {}
        for hits in hit_lists:
            _rrf_accumulate(agg, hits)
            for hit in hits:
                paper_id = str(hit.get("paper_id") or "")
                if not paper_id:
                    continue
                kept = best_hits.setdefault(paper_id, [])
                if len(kept) < 2:
                    kept.append(hit)
        for paper_id, entry in agg.items():
            entry["best_hits"] = best_hits.get(paper_id, [])
        candidates = _rrf_ranked(agg)[:ENUM_CANDIDATE_CAP]
        if not candidates:
            return None

        chunks_dir: Optional[Path] = getattr(args, "chunks_dir", None)
        blocks: list[str] = []
        for candidate in candidates:
            paper_id = str(candidate["paper_id"])
            abstract = _local_paper_abstract(paper_id, chunks_dir)
            snippet = ""
            if candidate["best_hits"]:
                snippet = compact_text(candidate["best_hits"][0].get("content"), max_chars=400)
            blocks.append(
                f"paper_id: {paper_id}\n"
                f"title: {candidate.get('title')}\n"
                f"venue: {candidate.get('venue')} year: {candidate.get('year')}\n"
                f"abstract: {abstract}\n"
                f"best matching snippet: {snippet}"
            )
        prompt = (
            "A question asks to identify ALL papers in a corpus matching a condition. "
            "For each candidate below, decide from its title/abstract/snippet whether "
            "the PAPER ITSELF satisfies every condition in the question (not merely "
            "mentions the topic). Be strict.\n"
            'Return JSON: {"paper_ids": [qualifying paper_id strings]}.\n\n'
            f"Question: {question}\n\n"
            "Candidates:\n\n" + "\n\n".join(blocks)
        )
        kwargs: Record = {
            "model": openai_settings.chat_deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": ENUM_VERIFY_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        response = retry_chat_completion(openai_client, kwargs)
        payload, ok = try_parse_json_object(response.choices[0].message.content or "")
        if not ok:
            return None
        by_id = {str(candidate["paper_id"]): candidate for candidate in candidates}
        qualifying: list[Record] = []
        raw_ids = payload.get("paper_ids")
        if isinstance(raw_ids, list):
            for raw_id in raw_ids:
                candidate = by_id.get(str(raw_id or "").strip())
                if candidate is not None and candidate not in qualifying:
                    qualifying.append(candidate)
                if len(qualifying) >= ENUM_SUBMIT_CAP:
                    break
        boost_hits: list[Record] = []
        for candidate in qualifying:
            # Metadata chunks carry no citable evidence (title/abstract are
            # already in the verification prompt); boosting them evicts the
            # figure-mention/body spans the answer LLM actually needs.
            boost_hits.extend(
                hit
                for hit in candidate["best_hits"]
                if str(hit.get("source_type") or "") != "metadata"
            )
        papers = [
            {
                "paper_id": candidate["paper_id"],
                "title": candidate.get("title") or "",
                "venue": candidate.get("venue") or "",
                "year": candidate.get("year"),
                "score": float(candidate.get("score") or 0.0),
                "best_rank": int(candidate.get("best_rank") or 1),
            }
            for candidate in qualifying
        ]
        return papers, boost_hits
    except Exception:  # noqa: BLE001 - enumeration is best effort.
        return None


# ---------------------------------------------------------------------------
# Context rendering and answer prompts
# ---------------------------------------------------------------------------


def build_context(
    results: list[Record],
    selected_paper_ids: set[str],
    context_chars: int,
) -> tuple[str, dict[str, Record]]:
    """Render [C#]-labeled context blocks and return the id -> chunk map."""
    parts: list[str] = []
    context_map: dict[str, Record] = {}
    used_chars = 0
    label_index = 0
    for result in results:
        if result.get("paper_id") not in selected_paper_ids:
            continue
        content = clean_text(result.get("content") or "")
        if not content:
            continue
        locator = parse_locator(result)
        label_index += 1
        label = f"C{label_index}"
        header_fields = [
            f"[{label}]",
            f"paper_id={result.get('paper_id')}",
            f"source_type={result.get('source_type')}",
            f"page={locator.get('page')}",
        ]
        if locator.get("table_id"):
            header_fields.append(f"table_id={locator.get('table_id')}")
        if locator.get("figure_id"):
            header_fields.append(f"figure_id={locator.get('figure_id')}")
        header_fields.append(f"venue={result.get('venue')}")
        header_fields.append(f"year={result.get('year')}")
        header_fields.append(f"title={result.get('title')}")
        header = " ".join(header_fields)
        block = f"{header}\n{content}"
        if used_chars + len(block) > context_chars:
            remaining = context_chars - used_chars
            if remaining < CONTEXT_MIN_TAIL_CHARS:
                label_index -= 1
                break
            block = block[:remaining].rstrip()
        parts.append(block)
        context_map[label] = result
        used_chars += len(block)
        if used_chars >= context_chars:
            break
    return "\n\n".join(parts), context_map


# figure_answer keeps its own byte-identical COPY of this rule (it does not
# import this constant) so both answer paths stay extractive. The exact bytes
# are part of the tuned prompt: any edit here must be mirrored in
# figure_answer.FREEFORM_RULE or the two prompts will drift.
FREEFORM_RULE = (
    '- For "freeform": output ONLY the shortest verbatim value/name/phrase exactly as printed in the source. '
    "Preserve the printed formatting exactly (answer 14.70, not 14.7). "
    "Always emit the value as a quoted JSON string, never a bare JSON number, so trailing zeros survive. "
    "No sentence, no explanation, no trailing period, no units unless printed as part of the value. "
    'Examples: an F1 score question -> "14.70"; an author-name question -> "Jane Smith"; '
    'a hardware question -> "a single NVIDIA A100 GPU".'
)


def answer_schema_instruction(answer_types: list[str]) -> str:
    """JSON shape line of the answer prompt, matching the answer types."""
    pieces = [
        '"paper_ids": array of paper_id strings for ONLY the papers that answer the question',
        '"evidence": array of {"context_id": "C3", "quote": "short verbatim supporting snippet"}',
    ]
    if "freeform" in answer_types:
        pieces.append(
            '"freeform": JSON string containing the shortest verbatim value/name/phrase '
            "from the source (always a quoted string, never a bare number)"
        )
    if "multiple_choice" in answer_types:
        pieces.append('"multiple_choice": one option letter such as "B"')
    if "table" in answer_types:
        pieces.append('"table": {"schema": [...], "rows": [objects keyed by the required column names]}')
    return "{ " + ", ".join(pieces) + " }"


def render_options(options: dict[str, str]) -> str:
    """Multiple-choice options block of the answer prompt."""
    lines = [f"{key}: {text}" for key, text in options.items()]
    return "Options:\n" + "\n".join(lines)


def render_table_schema(table_schema: list[Record]) -> str:
    """Required-columns block of the answer prompt for table questions."""
    lines: list[str] = []
    for column in table_schema:
        if not isinstance(column, dict):
            continue
        name = column.get("name")
        if not name:
            continue
        column_type = column.get("type") or "string"
        row_key = ", row key" if column.get("is_row_key") else ""
        lines.append(f'- "{name}" (type: {column_type}{row_key})')
    return "Required table columns:\n" + "\n".join(lines)


def answer_rules(sample: Record, options: Optional[dict[str, str]]) -> str:
    """Rules block of the answer prompt, tuned per answer type/task family."""
    answer_types = sample.get("answer_types") or []
    task_family = str(sample.get("task_family") or "")
    rules: list[str] = []
    if task_family == SINGLE_PAPER_FAMILY:
        rules.append(
            "- Exactly ONE paper contains the answer. In \"paper_ids\" return exactly one paper_id."
        )
        evidence_count = "1-2"
    else:
        rules.append(
            "- Return in \"paper_ids\" only the paper_ids of papers that actually answer or match the question (typically up to 4)."
        )
        evidence_count = "up to 4"
    rules.append(
        f"- In \"evidence\" cite ONLY the {evidence_count} context blocks whose content directly supports your answer, "
        "using their [C#] ids, each with a short verbatim quote."
    )
    if "freeform" in answer_types:
        rules.append(FREEFORM_RULE)
    if "multiple_choice" in answer_types:
        if options:
            rules.append(
                '- For "multiple_choice": answer with exactly one of the letter keys listed under Options. '
                "Output the letter only."
            )
        else:
            rules.append('- For "multiple_choice": answer with a single letter A-D.')
    if "table" in answer_types:
        rules.append(
            '- For "table": emit answer.table.rows as JSON objects keyed EXACTLY by the required column names above. '
            "Copy each cell value exactly as printed in the paper, as a quoted JSON string "
            '(e.g. "14.70"), never a bare number. Also fill answer.table.schema.'
        )
        rules.append(
            "- Row-key cells: when the question itself names the methods/entities, key each row by the "
            "QUESTION's spelling of that name and emit exactly ONE row per method named in the question; "
            "use the paper's printed spelling only when the question's spelling appears nowhere in the "
            "sources. Never key a row by a descriptive phrase from a sentence "
            '(not "inference scaling" or "our search framework" - use the method\'s proper name, e.g. "SANA-1.5"). '
            'Drop trailing parenthetical qualifiers (write "ECM-XL", not "ECM-XL (100k iterations)") '
            "UNLESS the question itself spells the name with that qualifier."
        )
        rules.append(
            "- Include only rows for the methods the question asks about; never substitute or add other "
            "methods (e.g. baseline rows quoted from another paper's comparison table). If a "
            "question-named method's value is not visible in the context, still emit its row with the "
            "best supported value from that method's OWN paper, or an empty string if none is visible - "
            "do not replace it with a different method."
        )
        rules.append(
            "- If the question asks about multiple conditions/settings/datasets, output a separate row "
            "for EVERY requested condition, not just one."
        )
        rules.append(
            '- Write plus-minus values with no spaces around the sign: "44.5±0.6", never "44.5 ± 0.6".'
        )
    return "\n".join(rules)


def user_prompt(sample: Record, context: str, options: Optional[dict[str, str]]) -> str:
    """Assemble the full user message for the answering call."""
    answer_types = sample.get("answer_types") or []
    sections = [
        f"Question:\n{sample.get('question')}",
        (
            f"Task family: {sample.get('task_family')}\n"
            f"Primary evidence type: {sample.get('primary_evidence_type')}\n"
            f"Answer types: {answer_types}"
        ),
    ]
    if options and "multiple_choice" in answer_types:
        sections.append(render_options(options))
    table_schema = sample.get("table_schema")
    if "table" in answer_types and isinstance(table_schema, list) and table_schema:
        sections.append(render_table_schema(table_schema))
    sections.append(
        "Return a JSON object shaped like:\n" + answer_schema_instruction(answer_types)
    )
    sections.append("Rules:\n" + answer_rules(sample, options))
    sections.append(f"Retrieved context:\n{context}")
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Chat plumbing (JSON-mode calls with length and parse retries)
# ---------------------------------------------------------------------------


def call_chat_model(
    client: Any,
    settings: OpenAISettings,
    messages: list[Record],
    *,
    max_tokens: int,
) -> tuple[str, str]:
    """One JSON-mode chat call; returns (content, finish_reason)."""
    kwargs: Record = {
        "model": settings.chat_deployment,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = retry_chat_completion(client, kwargs)
    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = str(getattr(choice, "finish_reason", "") or "")
    return content, finish_reason


def call_chat_with_length_retry(
    client: Any,
    settings: OpenAISettings,
    messages: list[Record],
    *,
    max_tokens: int,
) -> str:
    """Retry once with a doubled token budget when the answer was truncated."""
    content, finish_reason = call_chat_model(client, settings, messages, max_tokens=max_tokens)
    if finish_reason == "length":
        content, _ = call_chat_model(client, settings, messages, max_tokens=max_tokens * 2)
    return content


def chat_json(
    client: Any,
    settings: OpenAISettings,
    sample: Record,
    context: str,
    options: Optional[dict[str, str]],
    *,
    max_tokens: int,
    failure_writer: Callable[[Record], None],
) -> Record:
    """Answering call returning a parsed JSON payload ({} on double failure).

    Unparseable responses are logged via ``failure_writer`` and retried once
    with an explicit corrective message.
    """
    messages: list[Record] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(sample, context, options)},
    ]
    content = call_chat_with_length_retry(client, settings, messages, max_tokens=max_tokens)
    payload, _ = try_parse_json_object(content)
    if payload:
        return payload
    failure_writer({"query_id": sample.get("query_id"), "raw_content": content})
    retry_messages = messages + [
        {"role": "assistant", "content": content or "{}"},
        {
            "role": "user",
            "content": (
                "Your previous response was not a valid JSON object. "
                "Return ONLY a valid JSON object matching the requested shape. "
                "No markdown, no commentary."
            ),
        },
    ]
    content = call_chat_with_length_retry(client, settings, retry_messages, max_tokens=max_tokens)
    payload, _ = try_parse_json_object(content)
    if not payload:
        failure_writer({"query_id": sample.get("query_id"), "raw_content": content})
    return payload


# ---------------------------------------------------------------------------
# Query decomposition and multi-query retrieval
# ---------------------------------------------------------------------------

# max_tokens for the decomposition pre-pass (4 short subqueries + flags).
DECOMPOSE_MAX_TOKENS = 300


def decompose_query(
    client: Any,
    settings: OpenAISettings,
    sample: Record,
    venues: set[str],
) -> Optional[Record]:
    """Pre-pass extracting subqueries and validated venue/year filters."""
    try:
        venue_list = ", ".join(sorted(venues)) if venues else "(none known)"
        prompt = (
            "Plan a literature search for the question below. Return a JSON object:\n"
            '{"subqueries": [up to 4 short, focused search queries], '
            '"venue": "<venue name or null>", "year": <publication year integer or null>, '
            '"enumeration": <true or false>}\n'
            "Set venue/year ONLY if the question explicitly restricts to that venue or year.\n"
            'Set enumeration=true ONLY if the question asks to identify/enumerate the SET of '
            'papers in a venue or corpus matching a condition (e.g. "Which <venue> papers ...", '
            '"all papers that ..."), not when it asks about specific named papers.\n'
            f"Known venue values: {venue_list}\n\n"
            f"Question: {sample.get('question')}"
        )
        kwargs: Record = {
            "model": settings.chat_deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": DECOMPOSE_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        response = retry_chat_completion(client, kwargs)
        payload, ok = try_parse_json_object(response.choices[0].message.content or "")
        if not ok:
            return None
        subqueries: list[str] = []
        raw_subqueries = payload.get("subqueries")
        if isinstance(raw_subqueries, list):
            for item in raw_subqueries:
                text = str(item or "").strip()
                if text and text not in subqueries:
                    subqueries.append(text)
                if len(subqueries) >= 4:
                    break
        canonical_venues = {venue.lower(): venue for venue in venues}
        venue = canonical_venues.get(str(payload.get("venue") or "").strip().lower())
        year: Optional[int] = None
        raw_year = payload.get("year")
        if isinstance(raw_year, int) or (isinstance(raw_year, str) and raw_year.strip().isdigit()):
            candidate = int(raw_year)
            if 1900 <= candidate <= 2100:
                year = candidate
        return {
            "subqueries": subqueries,
            "venue": venue,
            "year": year,
            "enumeration": bool(payload.get("enumeration")),
        }
    except Exception:  # noqa: BLE001 - decomposition must never break the run.
        return None


def search_with_decomposition(
    sample: Record,
    query: str,
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    venues: set[str],
    args: argparse.Namespace,
    decomposition: Optional[Record] = None,
) -> Optional[list[Record]]:
    """Multi-query retrieval; returns None so the caller falls back on failure."""
    try:
        if decomposition is None:
            decomposition = decompose_query(openai_client, openai_settings, sample, venues)
        if not decomposition:
            return None
        filter_expr = build_filter_expression(
            decomposition.get("venue"), decomposition.get("year")
        )
        queries: list[str] = [query]
        for subquery in decomposition.get("subqueries") or []:
            if subquery not in queries:
                queries.append(subquery)
        if len(queries) == 1 and not filter_expr:
            return None
        merged: dict[str, tuple[int, Record]] = {}
        for search_query in queries:
            hits = search_chunks(
                search_client,
                openai_client,
                openai_settings,
                query=search_query,
                top_chunks=args.top_chunks,
                vector_k=args.vector_k,
                query_type=args.query_type,
                filter_expr=filter_expr,
                search_fields=parse_search_fields(args.search_fields),
            )
            for position, hit in enumerate(hits, start=1):
                chunk_key = _chunk_key(hit)
                existing = merged.get(chunk_key)
                if existing is None or position < existing[0]:
                    merged[chunk_key] = (position, hit)
        if not merged:
            return None
        ordered = sorted(
            merged.values(),
            key=lambda pair: (pair[0], -float(pair[1].get("search_score") or 0.0)),
        )
        return [hit for _, hit in ordered[: args.top_chunks]]
    except Exception:  # noqa: BLE001 - fall back to the single-query path.
        return None


# ---------------------------------------------------------------------------
# Answer normalization and submission building
# ---------------------------------------------------------------------------


def paper_submission(papers: list[Record]) -> list[Record]:
    """Project ranked paper records onto the submission's gold_papers shape."""
    return [
        {
            "paper_id": paper["paper_id"],
            "title": paper.get("title") or "",
            "venue": paper.get("venue") or "",
            "year": paper.get("year"),
        }
        for paper in papers
    ]


def _normalize_option_text(value: Any) -> str:
    text = str(value or "").strip().strip("\"'")
    return re.sub(r"\s+", " ", text).lower()


def normalize_choice(
    value: Any,
    options: Optional[dict[str, str]] = None,
    fallback_texts: tuple[Any, ...] = (),
) -> str:
    """Return a letter that is one of the question's option keys, else ''.

    When the actual option keys are unknown (no options), any single A-Z
    letter from the model passes through untouched (e.g. "E" is kept).
    """
    keys = (
        [str(key).strip().upper() for key in options if str(key).strip()]
        if options
        else None
    )

    def letter_ok(letter: str) -> bool:
        if keys is not None:
            return letter in keys
        return bool(re.fullmatch(r"[A-Z]", letter))

    text = str(value or "").strip()
    if text:
        upper = text.upper()
        if letter_ok(upper):
            return upper
        match = re.fullmatch(r"\(?([A-Za-z])[\).:]?", text)
        if match and letter_ok(match.group(1).upper()):
            return match.group(1).upper()
        match = re.match(r"^\(?([A-Za-z])\)?\s*[:.\-]\s+", text)
        if match and letter_ok(match.group(1).upper()):
            return match.group(1).upper()
    if not options:
        return ""
    normalized_options = {
        str(key).strip().upper(): _normalize_option_text(option_text)
        for key, option_text in options.items()
    }
    candidates = [text, *[str(item or "") for item in fallback_texts]]
    for candidate in candidates:
        normalized = _normalize_option_text(candidate)
        if not normalized:
            continue
        for key, option_text in normalized_options.items():
            if option_text and option_text == normalized:
                return key
    for candidate in candidates:
        normalized = _normalize_option_text(candidate)
        if len(normalized) < 3:
            continue
        matches = [
            key
            for key, option_text in normalized_options.items()
            if option_text and (option_text in normalized or normalized in option_text)
        ]
        if len(matches) == 1:
            return matches[0]
    return ""


def normalize_paper_ids(value: Any) -> list[str]:
    """Coerce the model's paper_ids field (string or list) to a string list."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


# Opening quote -> required closing quote; only matched pairs are stripped.
QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`"}


def strip_matched_quotes(text: str) -> str:
    """Drop surrounding matched quote pairs the model sometimes emits.

    Safe for scoring: evaluate.py's normalize_text strips the same quote
    characters anyway; doing it here just keeps the artifacts clean.
    """
    while len(text) >= 2 and QUOTE_PAIRS.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    return text


def freeform_text(llm_payload: Record) -> str:
    """Free-form answer text from the payload, tolerating shape drift."""
    raw = llm_payload.get("freeform")
    if raw is None or raw == "":
        raw = llm_payload.get("answer")
    if isinstance(raw, dict):
        raw = raw.get("text")
    if isinstance(raw, bool):
        # bool before int/float: bool is an int subclass, and 0/False must not
        # collapse to "" via clean_text's str(value or "").
        return str(raw).lower()
    if isinstance(raw, (int, float)):
        return repr(raw)
    return compact_text(raw)


def extract_choice_value(llm_payload: Record) -> Any:
    """Pull the raw multiple-choice value, tolerating dict payloads.

    The model sometimes wraps the letter, e.g. {"multiple_choice": {"gold": "C"}}
    or {"answer": "C"}; a valid letter must never be lost to shape mismatch.
    """
    for key in ("multiple_choice", "choice", "gold"):
        value = llm_payload.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, dict):
            for inner_key in ("gold", "answer", "predicted_answer_id", "choice", "letter"):
                inner = value.get(inner_key)
                if inner is not None and inner != "":
                    return inner
            continue
        return value
    return ""


def _normalize_plus_minus(text: str) -> str:
    """P7: gold strings print '±' with no surrounding spaces ("44.5±0.6")."""
    return re.sub(r"\s*±\s*", "±", text)


def _normalize_table_cell(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_plus_minus(value).strip()
    return value


def build_answer(
    sample: Record,
    llm_payload: Record,
    options: Optional[dict[str, str]] = None,
) -> Record:
    """Normalize the LLM payload into the submission's answer object."""
    answer_types = set(sample.get("answer_types") or [])
    answer: Record = {}
    text = _normalize_plus_minus(strip_matched_quotes(freeform_text(llm_payload)))
    if "freeform" in answer_types:
        answer["freeform"] = {"text": text}
    if "multiple_choice" in answer_types:
        answer["multiple_choice"] = {
            "gold": normalize_choice(
                extract_choice_value(llm_payload),
                options,
                fallback_texts=(text,),
            )
        }
    if "table" in answer_types:
        table = llm_payload.get("table") if isinstance(llm_payload.get("table"), dict) else {}
        schema = table.get("schema") if isinstance(table.get("schema"), list) else []
        if not schema and isinstance(sample.get("table_schema"), list):
            schema = sample["table_schema"]
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        answer["table"] = {
            "schema": schema,
            "rows": [
                {
                    str(key).strip(): _normalize_table_cell(value)
                    for key, value in row.items()
                }
                for row in rows
                if isinstance(row, dict)
            ],
        }
    return answer


def final_paper_count(sample: Record, args: argparse.Namespace) -> int:
    """Fallback submission size when the LLM returns no usable paper subset."""
    if str(sample.get("task_family") or "") == SINGLE_PAPER_FAMILY:
        return max(1, args.papers_single)
    return max(1, args.papers_multi)


# ---------------------------------------------------------------------------
# Per-question orchestration
# ---------------------------------------------------------------------------


def run_one(
    sample: Record,
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    args: argparse.Namespace,
    options: Optional[dict[str, str]],
    venues: set[str],
    failure_writer: Callable[[Record], None],
) -> Record:
    """Produce one prediction row: retrieve, boost, answer, build evidence.

    Pass order (see the inline P# comments): retrieval (optionally decomposed)
    -> P4 entity boost -> P6 enumeration -> candidate ranking/context -> P5
    lookup -> answer LLM -> paper-set assembly (P6 union, P3 expansion, P5
    retry) -> evidence assembly (P1/P2 re-anchoring, lookup items, citations).
    """
    query = str(sample.get("question") or "")
    task_family = str(sample.get("task_family") or "")
    decompose_enabled = args.decompose == "on" or (
        args.decompose == "auto" and task_family == "multi_paper"
    )
    results: Optional[list[Record]] = None
    decomposition: Optional[Record] = None
    if decompose_enabled and not args.retrieval_only:
        decomposition = decompose_query(openai_client, openai_settings, sample, venues)
        results = search_with_decomposition(
            sample,
            query,
            search_client=search_client,
            openai_client=openai_client,
            openai_settings=openai_settings,
            venues=venues,
            args=args,
            decomposition=decomposition,
        )
    if results is None:
        results = search_chunks(
            search_client,
            openai_client,
            openai_settings,
            query=query,
            top_chunks=args.top_chunks,
            vector_k=args.vector_k,
            query_type=args.query_type,
            search_fields=parse_search_fields(args.search_fields),
        )

    # EXPERIMENTAL --rewrite-prompt-file (prompt iteration; see query_lab):
    # one extra AOAI call rewrites the question into additional search
    # queries whose hits are APPENDED after the organic results via
    # merge_hit_lists (concatenation dedupe, primary wins) before
    # rank_papers. This is deliberately precision-conservative and is NOT
    # the rank-interleaved merge used by search_with_decomposition or by
    # query_lab.merge_query_hits: a paper surfaced only by a rewritten
    # query enters RRF at a list position after the ~top_chunks organic
    # hits, so query_lab recall deltas OVERSTATE the end-to-end gain of
    # this flag. Always re-score with scripts/evaluate.py + compare_runs
    # (RUNBOOK, "Query-rewrite experimentation"). Best effort: any failure
    # leaves the results untouched.
    if getattr(args, "rewrite_prompt_file", None):
        # Lazy import: query_lab imports run_rag at module level, so the
        # reverse import must happen at call time to avoid a cycle.
        from . import query_lab

        try:
            rewritten, _raw, _ok = query_lab.rewrite_queries(
                openai_client,
                openai_settings,
                query_lab.load_prompt_template(args.rewrite_prompt_file),
                sample,
            )
            extra_hits: list[Record] = []
            for extra_query in rewritten[1:]:  # query 0 is the original question
                extra_hits = merge_hit_lists(
                    extra_hits,
                    search_chunks(
                        search_client,
                        openai_client,
                        openai_settings,
                        query=extra_query,
                        top_chunks=args.top_chunks,
                        vector_k=args.vector_k,
                        query_type=args.query_type,
                        search_fields=parse_search_fields(args.search_fields),
                    ),
                )
            if extra_hits:
                results = merge_hit_lists(results, extra_hits)
        except Exception as exc:  # noqa: BLE001 - the rewrite must never break a run.
            print(
                f"WARN {sample.get('query_id')}: query rewrite failed ({exc})",
                file=sys.stderr,
            )

    # P4: papers named in the question (or in title-like MC options) are
    # force-retrieved and their chunks prepended (strong rank boost).
    if getattr(args, "entity_search", True):
        entity_prepend, entity_append = entity_boost_hits(
            sample,
            options,
            search_client=search_client,
            openai_client=openai_client,
            openai_settings=openai_settings,
            args=args,
        )
        if entity_prepend:
            results = merge_hit_lists(entity_prepend, results)
        if entity_append:
            # Content-rung resolutions: available to rank_papers/build_context
            # but never ahead of the organic results.
            results = merge_hit_lists(results, entity_append)

    # P6: bounded corpus enumeration (wide filtered searches + one batched
    # LLM verification call). The qualifying papers' best chunks are boosted
    # into the context; the final paper set is adjusted after answering.
    enum_papers: Optional[list[Record]] = None
    if (
        getattr(args, "enumeration", True)
        and not args.retrieval_only
        and decomposition is not None
        and (bool(decomposition.get("enumeration")) or question_is_enumeration(query))
    ):
        enum_outcome = run_corpus_enumeration(
            sample,
            decomposition,
            search_client=search_client,
            openai_client=openai_client,
            openai_settings=openai_settings,
            args=args,
        )
        if enum_outcome is not None:
            enum_papers, enum_hits = enum_outcome
            if enum_hits:
                results = merge_hit_lists(enum_hits, results)

    candidates = rank_papers(results, args.top_papers)
    candidate_ids = {paper["paper_id"] for paper in candidates}
    context, context_map = build_context(results, candidate_ids, args.context_chars)
    final_k = final_paper_count(sample, args)

    # P5: 'Nth/last reference' / 'Equation (N)' local lookup on the top
    # retrieval candidate; the excerpt is injected as an extra context block.
    # If the LLM then selects a DIFFERENT paper, the lookup is redone on that
    # paper and the answer re-asked (see the P5 retry below).
    chunks_dir: Optional[Path] = getattr(args, "chunks_dir", None)
    lookup: Optional[Record] = None
    lookup_base_context = context
    if candidates and not args.retrieval_only:
        lookup = perform_local_lookup(sample, str(candidates[0]["paper_id"]), chunks_dir)
        if lookup:
            context = context + "\n\n" + str(lookup["context_block"])

    llm_payload: Record = {}
    if not args.retrieval_only:
        llm_payload = chat_json(
            openai_client,
            openai_settings,
            sample,
            context,
            options,
            max_tokens=args.max_answer_tokens,
            failure_writer=failure_writer,
        )

    llm_paper_ids = normalize_paper_ids(llm_payload.get("paper_ids"))
    paper_lookup = {paper["paper_id"]: paper for paper in candidates}
    selected: list[Record] = []
    seen_ids: set[str] = set()
    for paper_id in llm_paper_ids:
        if paper_id in paper_lookup and paper_id not in seen_ids:
            selected.append(paper_lookup[paper_id])
            seen_ids.add(paper_id)
    if selected:
        if task_family == SINGLE_PAPER_FAMILY:
            papers = selected[:1]
        else:
            # Multi-paper: submit the LLM's validated subset AS-IS (already
            # bounded by the candidate pool); --papers-multi applies only when
            # the LLM returns nothing usable. Handles the 9-gold-paper outlier.
            papers = selected
    else:
        papers = candidates[:final_k]

    # P6: the enumeration verdict is a UNION with the normal paper set, never
    # a replacement: the answer-grounded LLM picks are kept first (the
    # verifier judges from title/abstract/snippet and cannot check
    # figure-level conditions), then the qualifying enum papers are appended
    # up to ENUM_SUBMIT_CAP.
    enum_applied = False
    if enum_papers is not None:
        existing_ids = {str(paper.get("paper_id") or "") for paper in papers}
        for paper in enum_papers:
            paper_id = str(paper.get("paper_id") or "")
            if paper_id and paper_id not in existing_ids:
                papers.append(paper)
                existing_ids.add(paper_id)
        papers = papers[:ENUM_SUBMIT_CAP]
        if len(enum_papers) >= ENUM_MIN_QUALIFYING:
            # A confident enum verdict still suppresses P3 expansion below.
            enum_applied = True

    # P3: multi_paper gold sets are almost always the 4 co-compared methods;
    # top up a short submission with corpus-resolved comparison-group papers
    # (never removes existing papers, skips when nothing credible is found).
    if (
        getattr(args, "compare_expand", True)
        and not args.retrieval_only
        and not enum_applied
        and task_family == "multi_paper"
    ):
        # Up to three passes: papers added in one pass contribute their
        # bibliographies, reverse citations and neighbors to the next (e.g.
        # paper A's references name the full title of co-compared paper B;
        # only papers citing B reveal group member C). Stage 2 of the
        # expansion accepts at most 2 papers per pass on purpose.
        for _ in range(3):
            if len(papers) >= COMPARISON_TARGET_PAPERS:
                break
            expanded = expand_comparison_group(
                sample,
                papers,
                context,
                search_client=search_client,
                openai_client=openai_client,
                openai_settings=openai_settings,
                chunks_dir=chunks_dir,
            )
            if not expanded:
                break
            papers = papers + expanded

    # P5 retry: the pre-answer lookup ran on the top RETRIEVAL candidate,
    # which is not always the paper the LLM actually selects (e.g. the top
    # candidate has unnumbered ACL-style references, so the '[N]' lookup found
    # nothing while the selected paper would have matched). Redo the lookup on
    # the selected papers and, when it yields a verified excerpt for a paper
    # the first pass did not have one for, re-ask the LLM with that excerpt.
    # The paper set stays fixed; only the answer/evidence are refreshed.
    if (
        not args.retrieval_only
        and papers
        and detect_lookup_request(query) is not None
        and (
            lookup is None
            or str(lookup.get("paper_id") or "") != str(papers[0]["paper_id"])
        )
    ):
        retry_lookup: Optional[Record] = None
        for paper in papers[:3]:
            retry_lookup = perform_local_lookup(sample, str(paper["paper_id"]), chunks_dir)
            if retry_lookup:
                break
        if retry_lookup:
            lookup = retry_lookup
            context = lookup_base_context + "\n\n" + str(retry_lookup["context_block"])
            try:
                retry_payload = chat_json(
                    openai_client,
                    openai_settings,
                    sample,
                    context,
                    options,
                    max_tokens=args.max_answer_tokens,
                    failure_writer=failure_writer,
                )
            except Exception:  # noqa: BLE001 - keep the first-pass answer.
                retry_payload = {}
            if retry_payload:
                llm_payload = retry_payload
    selected_paper_ids = {paper["paper_id"] for paper in papers}

    answer = build_answer(sample, llm_payload, options)

    if args.retrieval_only:
        evidence = evidence_from_results(
            results, selected_paper_ids, limit=args.evidence_limit
        )
    else:
        llm_evidence = evidence_from_citations(
            llm_payload.get("evidence"),
            context_map,
            sample,
            selected_paper_ids,
            cap=EVIDENCE_CAP,
        )
        # P1/P5: re-anchor evidence onto local table/figure chunks and the
        # lookup chunk; re-anchored items go first (they carry the
        # table_id/figure_id keys), LLM citations are never all dropped.
        priority: list[Record] = []
        if lookup and str(lookup.get("paper_id") or "") in selected_paper_ids:
            try:
                lookup_items = lookup_evidence_items(lookup)
                priority.extend(lookup_items)
                if lookup_items:
                    # The lookup item is locally VERIFIED: same-paper LLM
                    # citations pointing at other pages are companion noise
                    # that only dilutes precision next to it.
                    lookup_paper = str(lookup.get("paper_id") or "")
                    lookup_pages = {
                        str((item.get("locator") or {}).get("page"))
                        for item in lookup_items
                    }
                    llm_evidence = [
                        item
                        for item in llm_evidence
                        if str(item.get("paper_id") or "") != lookup_paper
                        or str((item.get("locator") or {}).get("page")) in lookup_pages
                    ]
            except Exception:  # noqa: BLE001 - re-anchoring is best effort.
                pass
        try:
            priority.extend(
                build_reanchor_candidates(
                    sample,
                    [str(paper["paper_id"]) for paper in papers],
                    answer,
                    options,
                    llm_evidence,
                    chunks_dir=chunks_dir,
                )
            )
        except Exception:  # noqa: BLE001 - re-anchoring is best effort.
            pass
        evidence = merge_evidence_items(
            priority,
            llm_evidence,
            cap=EVIDENCE_CAP,
            primary_type=str(sample.get("primary_evidence_type") or ""),
        )
        if not evidence:
            evidence = evidence_from_results(
                results,
                selected_paper_ids,
                limit=min(args.evidence_limit, FALLBACK_EVIDENCE_CAP),
            )

    return {
        "query_id": sample.get("query_id"),
        "gold_papers": paper_submission(papers),
        "evidence": evidence,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# Input/output helpers (options join, resume, provenance)
# ---------------------------------------------------------------------------


def extract_options(record: Record) -> Optional[dict[str, str]]:
    """Accept a top-level "options" object or the gold answer.multiple_choice.options shape."""
    options = record.get("options")
    if not isinstance(options, dict) or not options:
        answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
        multiple_choice = (
            answer.get("multiple_choice")
            if isinstance(answer.get("multiple_choice"), dict)
            else {}
        )
        options = multiple_choice.get("options")
    if isinstance(options, dict) and options:
        return {str(key): str(value) for key, value in options.items()}
    return None


def load_options_map(path: Optional[Path]) -> dict[str, Record]:
    """Read ONLY query_id -> options (plus question text for a safety
    cross-check); never gold answers, papers, or evidence."""
    if path is None or not path.exists():
        return {}
    options_map: dict[str, Record] = {}
    for record in read_jsonl(path):
        query_id = str(record.get("query_id") or "")
        if not query_id:
            continue
        options = extract_options(record)
        if options:
            options_map[query_id] = {
                "options": options,
                "question": str(record.get("question") or ""),
            }
    return options_map


def options_for_sample(sample: Record, options_map: dict[str, Record]) -> Optional[dict[str, str]]:
    """Options carried in the input record win; joined options must match the question."""
    inline = extract_options(sample)
    if inline:
        return inline
    entry = options_map.get(str(sample.get("query_id") or ""))
    if not entry:
        return None
    entry_question = _normalize_option_text(entry.get("question"))
    sample_question = _normalize_option_text(sample.get("question"))
    if entry_question and sample_question and entry_question != sample_question:
        print(
            f"WARN {sample.get('query_id')}: --options-file question text does not match "
            "the input question; dropping the joined options",
            file=sys.stderr,
        )
        return None
    return entry.get("options")


def default_validation_options(input_path: Path) -> Optional[Path]:
    """Default --options-file: the validation gold rows, ONLY for the default
    validation input.

    Any other input (e.g. the hidden test set) must opt in explicitly,
    otherwise a query_id collision would silently attach the WRONG options.
    figure_answer duplicates this guard inline; changes here must be mirrored
    there.
    """
    default_options = ROOT / "data" / "validation.jsonl"
    default_input = ROOT / "data" / "validation_inputs.jsonl"
    if default_options.exists() and input_path.resolve() == default_input.resolve():
        return default_options
    return None


def load_venues(path: Path) -> set[str]:
    """Known venue strings from the metadata file (decomposition validation)."""
    try:
        return {
            str(record.get("venue")).strip()
            for record in read_jsonl(path)
            if record.get("venue")
        }
    except Exception:  # noqa: BLE001 - venue validation is optional.
        return set()


def git_sha() -> Optional[str]:
    """Current commit hash for run provenance; None outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort.
        pass
    return None


def write_run_meta(
    args: argparse.Namespace,
    search_settings: SearchSettings,
    openai_settings: OpenAISettings,
    search_client: Any,
) -> None:
    """Write <output>.meta.json capturing args, code and index provenance."""
    document_count: Optional[int] = None
    try:
        document_count = int(search_client.get_document_count())
    except Exception:  # noqa: BLE001 - provenance is best effort.
        pass
    meta = {
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "git_sha": git_sha(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "chat_deployment": openai_settings.chat_deployment,
        "embedding_deployment": openai_settings.embedding_deployment,
        "index_name": search_settings.index_name,
        "index_document_count": document_count,
    }
    meta_path = args.output.with_name(args.output.name + ".meta.json")
    write_json(meta_path, meta)


def answer_is_empty(answer: Any) -> bool:
    """True when no answer field carries usable content."""
    if not isinstance(answer, dict) or not answer:
        return True
    freeform = answer.get("freeform")
    if isinstance(freeform, dict) and str(freeform.get("text") or "").strip():
        return False
    multiple_choice = answer.get("multiple_choice")
    if isinstance(multiple_choice, dict) and str(multiple_choice.get("gold") or "").strip():
        return False
    table = answer.get("table")
    if isinstance(table, dict) and table.get("rows"):
        return False
    return True


def row_needs_retry(record: Record) -> bool:
    """Exception-path placeholder rows: empty answer AND empty papers list."""
    return answer_is_empty(record.get("answer")) and not record.get("gold_papers")


def load_resume_rows(path: Path) -> list[Record]:
    """Load prior output rows worth keeping on --resume.

    Drops corrupt lines (e.g. a trailing partial write from an interrupted
    run) and retryable placeholder rows so those query_ids run again.
    Later duplicates of a query_id win.
    """
    if not path.exists():
        return []
    kept: dict[str, Record] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            query_id = str(record.get("query_id") or "")
            if not query_id or row_needs_retry(record):
                continue
            kept[query_id] = record
    return list(kept.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LitTraceQA predictions with Azure Search + AOAI RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "validation_inputs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "submission.jsonl")
    parser.add_argument(
        "--options-file",
        type=Path,
        default=None,
        help=(
            "JSONL joining multiple-choice options by query_id. Accepts full "
            "gold rows (answer.multiple_choice.options) or bare sidecar records "
            "{'query_id': ..., 'options': {...}}, matching validate_submission. "
            "Default: data/validation.jsonl, applied ONLY when --input is the "
            "default data/validation_inputs.jsonl; any other input (e.g. the "
            "hidden test set) never gets validation options unless this flag is "
            "passed explicitly. Only options are read, never answers."
        ),
    )
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--query-id",
        action="append",
        default=None,
        dest="query_ids",
        help="Run only these query_ids (repeatable).",
    )
    parser.add_argument("--top-chunks", type=int, default=24)
    parser.add_argument("--vector-k", type=int, default=50)
    parser.add_argument("--top-papers", type=int, default=8, help="Retrieval candidate paper pool size.")
    parser.add_argument("--papers-single", type=int, default=1, help="Submitted papers for hidden_source_single_paper.")
    parser.add_argument("--papers-multi", type=int, default=4, help="Submitted papers for multi_paper.")
    parser.add_argument("--evidence-limit", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    parser.add_argument("--max-answer-tokens", type=int, default=1200)
    parser.add_argument(
        "--decompose",
        choices=["auto", "off", "on"],
        default="auto",
        help="Query decomposition pre-pass (auto = multi_paper questions only).",
    )
    parser.add_argument(
        "--query-type",
        choices=["hybrid", "semantic"],
        default="hybrid",
        help="semantic requires the index to define the littraceqa-semantic configuration.",
    )
    parser.add_argument(
        "--search-fields",
        default=None,
        help=(
            "Comma-separated index fields for the keyword (BM25) leg of the "
            "hybrid search, e.g. 'content,section'. Restricting BM25 to "
            "content,section counters the title/abstract text duplicated into "
            "every chunk of a paper. Default: all searchable fields (previous "
            "behavior). The vector leg always uses content_vector."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help=(
            "Local chunk JSONL directory (one {paper_id}.jsonl per paper) used "
            "for evidence re-anchoring and reference/equation lookup. Missing "
            "files are tolerated (those steps just no-op)."
        ),
    )
    parser.add_argument(
        "--rewrite-prompt-file",
        type=Path,
        default=None,
        help=(
            "EXPERIMENTAL, for query-rewrite prompt iteration with "
            "littraceqa.azure.query_lab: rewrite each question into extra "
            "search queries via one AOAI call using this prompt template "
            "('#' lines stripped, {question} placeholder required) and merge "
            "those queries' hits into the candidate results before paper "
            "ranking. Default: unset (behavior identical to before)."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Skip query_ids already in the output file and append.")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--entity-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "P4: resolve proper nouns in the question (and title-like MC "
            "options) via title/abstract search and boost those papers into "
            "the candidate pool. Disable with --no-entity-search."
        ),
    )
    parser.add_argument(
        "--compare-expand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "P3: top up multi_paper submissions below 4 papers with the "
            "co-compared methods visible in the context, resolved against "
            "the corpus. Disable with --no-compare-expand."
        ),
    )
    parser.add_argument(
        "--enumeration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "P6: for venue/corpus enumeration questions, run wide filtered "
            "searches plus one batched LLM verification call to pick the "
            "qualifying papers. Disable with --no-enumeration."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run predictions over the input questions with a thread-pool of workers."""
    args = build_parser().parse_args(argv)
    load_environment(args.env_file)
    search_settings = SearchSettings.from_env()
    openai_settings = OpenAISettings.from_env()
    search_client = build_search_client(search_settings)
    # max_retries=6: the SDK retries 429s honoring Retry-After, which matters
    # when --workers > 1 hammer the same deployment.
    openai_client = build_openai_client(openai_settings, max_retries=6)

    if args.options_file is None:
        args.options_file = default_validation_options(args.input)
    options_map = load_options_map(args.options_file)

    venues: set[str] = set()
    if args.decompose != "off":
        venues = load_venues(args.metadata_file)

    samples = read_jsonl(args.input)
    if args.query_ids:
        wanted = {str(query_id) for query_id in args.query_ids}
        samples = [sample for sample in samples if str(sample.get("query_id")) in wanted]
    if args.limit:
        samples = samples[: args.limit]

    if options_map:
        sample_ids = {str(sample.get("query_id") or "") for sample in samples}
        if sample_ids and not sample_ids & set(options_map):
            print(
                f"WARN: no query_id overlap between --options-file ({args.options_file}) "
                "and --input; multiple-choice prompts will run without joined options",
                file=sys.stderr,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    if args.resume:
        good_rows = load_resume_rows(args.output)
        # Atomic rewrite (temp + os.replace) drops retryable placeholder rows
        # and any trailing corrupt partial line before appending fresh rows.
        write_jsonl(args.output, good_rows)
        done_ids = {str(row.get("query_id")) for row in good_rows}
        pending = [sample for sample in samples if str(sample.get("query_id")) not in done_ids]
        skipped = len(samples) - len(pending)
    else:
        pending = samples
    mode = "a" if args.resume else "w"

    raw_failures_path = args.output.with_name(args.output.stem + "_raw_failures.jsonl")
    if not args.resume:
        # Fresh run: truncate the sidecar so stale failures do not accumulate.
        raw_failures_path.unlink(missing_ok=True)
    raw_failures_lock = threading.Lock()

    def failure_writer(record: Record) -> None:
        with raw_failures_lock:
            raw_failures_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_failures_path.open("a", encoding="utf-8") as failure_handle:
                failure_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def process_sample(sample: Record) -> tuple[Record, bool]:
        query_id = sample.get("query_id")
        options = options_for_sample(sample, options_map)
        try:
            prediction = run_one(
                sample,
                search_client=search_client,
                openai_client=openai_client,
                openai_settings=openai_settings,
                args=args,
                options=options,
                venues=venues,
                failure_writer=failure_writer,
            )
            return prediction, False
        except Exception as exc:  # noqa: BLE001 - emit a valid empty row.
            print(f"FAIL {query_id}: {exc}", file=sys.stderr)
            prediction = {
                "query_id": query_id,
                "gold_papers": [],
                "evidence": [],
                "answer": build_answer(sample, {}, options),
            }
            return prediction, True

    failed = 0
    completed = 0
    write_lock = threading.Lock()
    workers = max(1, args.workers)
    with args.output.open(mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_sample, sample): sample for sample in pending}
            for future in as_completed(futures):
                prediction, did_fail = future.result()
                with write_lock:
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
                    if did_fail:
                        failed += 1
                    print(
                        f"[{completed}/{len(pending)}] {prediction.get('query_id')}",
                        file=sys.stderr,
                    )

    write_run_meta(args, search_settings, openai_settings, search_client)
    print(
        f"wrote {len(pending)} predictions to {args.output}; "
        f"failed={failed}; skipped={skipped}",
        file=sys.stderr,
    )
    return 1 if pending and failed == len(pending) else 0


if __name__ == "__main__":
    sys.exit(main())
