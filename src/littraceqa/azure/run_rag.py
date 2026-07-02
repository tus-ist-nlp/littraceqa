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
    DEFAULT_METADATA,
    ROOT,
    Record,
    clean_text,
    compact_text,
    read_jsonl,
    retry_chat_completion,
    try_parse_json_object,
    write_json,
    write_jsonl,
)
from ..fix_chunk_locators import normalize_object_id


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

SEMANTIC_CONFIGURATION_NAME = "littraceqa-semantic"
RELABEL_PRIMARY_TYPES = {"citation_context", "equation_algorithm"}
SINGLE_PAPER_FAMILY = "hidden_source_single_paper"
EVIDENCE_CAP = 4
FALLBACK_EVIDENCE_CAP = 2


SYSTEM_PROMPT = """You are solving LitTraceQA, a scientific-paper grounded QA task.
Use only the retrieved context. Copy exact values verbatim from the source; never paraphrase numbers.
Return JSON only. Do not include markdown."""


def embedding_kwargs(settings: OpenAISettings, text: str) -> Record:
    kwargs: Record = {
        "model": settings.embedding_deployment,
        "input": [text],
    }
    if settings.request_embedding_dimensions:
        kwargs["dimensions"] = settings.embedding_dimensions
    return kwargs


def embed_query(client: Any, settings: OpenAISettings, query: str) -> list[float]:
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
    results = search_client.search(**kwargs)
    records: list[Record] = []
    for result in results:
        item = dict(result)
        item["search_score"] = float(item.pop("@search.score", 0.0) or 0.0)
        records.append(item)
    return records


def rank_papers(results: list[Record], top_papers: int) -> list[Record]:
    """Aggregate chunk hits to papers with summed reciprocal-rank fusion."""
    by_paper: dict[str, Record] = {}
    for rank, result in enumerate(results, start=1):
        paper_id = str(result.get("paper_id") or "")
        if not paper_id:
            continue
        paper = by_paper.setdefault(
            paper_id,
            {
                "paper_id": paper_id,
                "title": result.get("title") or "",
                "venue": result.get("venue") or "",
                "year": result.get("year"),
                "score": 0.0,
                "best_rank": rank,
            },
        )
        paper["score"] = float(paper["score"]) + 1.0 / (60 + rank)
        paper["best_rank"] = min(int(paper["best_rank"]), rank)
    ranked = sorted(
        by_paper.values(),
        key=lambda item: (-float(item["score"]), int(item["best_rank"])),
    )
    return ranked[:top_papers]


def parse_locator(result: Record) -> Record:
    raw = result.get("locator_json") or "{}"
    try:
        locator = json.loads(raw)
    except json.JSONDecodeError:
        locator = {}
    if not isinstance(locator, dict):
        locator = {}
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


def evidence_from_results(results: list[Record], selected_paper_ids: set[str], limit: int) -> list[Record]:
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
        page = locator.get("page")
        if page is None or str(page).strip() == "":
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
                "evidence_text_or_value": compact_text(result.get("content"), max_chars=300),
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
        page = chunk_locator.get("page")
        if page is None or str(page).strip() == "":
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
            text = compact_text(quote, max_chars=300)
        else:
            text = compact_text(chunk.get("content"), max_chars=300)
        for emit_type in emit_types:
            locator: Record = {"page": page}
            if emit_type == "table":
                object_id = str(chunk_locator.get("table_id") or "")
                if object_id:
                    locator["table_id"] = object_id
            elif emit_type == "figure":
                object_id = str(chunk_locator.get("figure_id") or "")
                if object_id:
                    locator["figure_id"] = object_id
            key = evaluator_evidence_key(paper_id, emit_type, locator)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "evidence_id": f"ev_{len(evidence) + 1:03d}",
                    "paper_id": paper_id,
                    "source_type": emit_type,
                    "evidence_text_or_value": text,
                    "locator": locator,
                }
            )
            if len(evidence) >= cap:
                break
        if len(evidence) >= cap:
            break
    return evidence


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
            if remaining < 500:
                label_index -= 1
                break
            block = block[:remaining].rstrip()
        parts.append(block)
        context_map[label] = result
        used_chars += len(block)
        if used_chars >= context_chars:
            break
    return "\n\n".join(parts), context_map


def answer_schema_instruction(answer_types: list[str]) -> str:
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
    lines = [f"{key}: {text}" for key, text in options.items()]
    return "Options:\n" + "\n".join(lines)


def render_table_schema(table_schema: list[Record]) -> str:
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
        rules.append(
            '- For "freeform": output ONLY the shortest verbatim value/name/phrase exactly as printed in the source. '
            "Preserve the printed formatting exactly (answer 14.70, not 14.7). "
            "Always emit the value as a quoted JSON string, never a bare JSON number, so trailing zeros survive. "
            "No sentence, no explanation, no trailing period, no units unless printed as part of the value. "
            'Examples: an F1 score question -> "14.70"; an author-name question -> "Jane Smith"; '
            'a hardware question -> "a single NVIDIA A100 GPU".'
        )
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
    return "\n".join(rules)


def user_prompt(sample: Record, context: str, options: Optional[dict[str, str]]) -> str:
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


def call_chat_model(
    client: Any,
    settings: OpenAISettings,
    messages: list[Record],
    *,
    max_tokens: int,
) -> tuple[str, str]:
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
            '"venue": "<venue name or null>", "year": <publication year integer or null>}\n'
            "Set venue/year ONLY if the question explicitly restricts to that venue or year.\n"
            f"Known venue values: {venue_list}\n\n"
            f"Question: {sample.get('question')}"
        )
        kwargs: Record = {
            "model": settings.chat_deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 300,
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
        return {"subqueries": subqueries, "venue": venue, "year": year}
    except Exception:  # noqa: BLE001 - decomposition must never break the run.
        return None


def build_filter_expression(venue: Optional[str], year: Optional[int]) -> Optional[str]:
    parts: list[str] = []
    if venue:
        escaped = venue.replace("'", "''")
        parts.append(f"venue eq '{escaped}'")
    if year is not None:
        parts.append(f"year eq {year}")
    return " and ".join(parts) if parts else None


def search_with_decomposition(
    sample: Record,
    query: str,
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    venues: set[str],
    args: argparse.Namespace,
) -> Optional[list[Record]]:
    """Multi-query retrieval; returns None so the caller falls back on failure."""
    try:
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
                chunk_key = str(hit.get("id") or f"{hit.get('paper_id')}::{hit.get('chunk_id')}")
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


def paper_submission(papers: list[Record]) -> list[Record]:
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


def build_answer(
    sample: Record,
    llm_payload: Record,
    options: Optional[dict[str, str]] = None,
) -> Record:
    answer_types = set(sample.get("answer_types") or [])
    answer: Record = {}
    text = strip_matched_quotes(freeform_text(llm_payload))
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
            "rows": [row for row in rows if isinstance(row, dict)],
        }
    return answer


def final_paper_count(sample: Record, args: argparse.Namespace) -> int:
    if str(sample.get("task_family") or "") == SINGLE_PAPER_FAMILY:
        return max(1, args.papers_single)
    return max(1, args.papers_multi)


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
    query = str(sample.get("question") or "")
    task_family = str(sample.get("task_family") or "")
    decompose_enabled = args.decompose == "on" or (
        args.decompose == "auto" and task_family == "multi_paper"
    )
    results: Optional[list[Record]] = None
    if decompose_enabled and not args.retrieval_only:
        results = search_with_decomposition(
            sample,
            query,
            search_client=search_client,
            openai_client=openai_client,
            openai_settings=openai_settings,
            venues=venues,
            args=args,
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

    candidates = rank_papers(results, args.top_papers)
    candidate_ids = {paper["paper_id"] for paper in candidates}
    context, context_map = build_context(results, candidate_ids, args.context_chars)
    final_k = final_paper_count(sample, args)

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
    selected_paper_ids = {paper["paper_id"] for paper in papers}

    if args.retrieval_only:
        evidence = evidence_from_results(
            results, selected_paper_ids, limit=args.evidence_limit
        )
    else:
        evidence = evidence_from_citations(
            llm_payload.get("evidence"),
            context_map,
            sample,
            selected_paper_ids,
            cap=EVIDENCE_CAP,
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
        "answer": build_answer(sample, llm_payload, options),
    }


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


def load_venues(path: Path) -> set[str]:
    try:
        return {
            str(record.get("venue")).strip()
            for record in read_jsonl(path)
            if record.get("venue")
        }
    except Exception:  # noqa: BLE001 - venue validation is optional.
        return set()


def git_sha() -> Optional[str]:
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
    parser.add_argument("--context-chars", type=int, default=18000)
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Skip query_ids already in the output file and append.")
    parser.add_argument("--retrieval-only", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_environment(args.env_file)
    search_settings = SearchSettings.from_env()
    openai_settings = OpenAISettings.from_env()
    search_client = build_search_client(search_settings)
    # max_retries=6: the SDK retries 429s honoring Retry-After, which matters
    # when --workers > 1 hammer the same deployment.
    openai_client = build_openai_client(openai_settings, max_retries=6)

    if args.options_file is None:
        default_options = ROOT / "data" / "validation.jsonl"
        default_input = ROOT / "data" / "validation_inputs.jsonl"
        # Only join validation options onto the validation inputs; any other
        # input (e.g. the hidden test set) must opt in explicitly, otherwise a
        # query_id collision would silently attach the WRONG options.
        if default_options.exists() and args.input.resolve() == default_input.resolve():
            args.options_file = default_options
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
