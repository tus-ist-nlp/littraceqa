#!/usr/bin/env python3
"""Run LitTraceQA RAG over Azure AI Search and Azure OpenAI."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

from azure_config import (
    OpenAISettings,
    SearchSettings,
    build_openai_client,
    build_search_client,
    load_environment,
)
from littrace_common import (
    ROOT,
    Record,
    clean_text,
    compact_text,
    parse_json_object,
    read_jsonl,
    retry_chat_completion,
)


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


SYSTEM_PROMPT = """You are solving LitTraceQA, a scientific-paper grounded QA task.
Use only the retrieved context. Prefer exact numeric/table values over paraphrase.
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


def search_chunks(
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    *,
    query: str,
    top_chunks: int,
    vector_k: int,
) -> list[Record]:
    from azure.search.documents.models import VectorizedQuery

    query_vector = embed_query(openai_client, openai_settings, query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=vector_k,
        fields="content_vector",
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=top_chunks,
        select=SEARCH_SELECT,
    )
    records: list[Record] = []
    for result in results:
        item = dict(result)
        item["search_score"] = float(item.pop("@search.score", 0.0) or 0.0)
        records.append(item)
    return records


def rank_papers(results: list[Record], top_papers: int) -> list[Record]:
    by_paper: dict[str, Record] = {}
    for rank, result in enumerate(results, start=1):
        paper_id = str(result.get("paper_id") or "")
        if not paper_id:
            continue
        score = float(result.get("search_score") or 0.0) + 1.0 / (rank + 60)
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
        paper["score"] = max(float(paper["score"]), score)
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


def evidence_from_results(results: list[Record], selected_paper_ids: set[str], limit: int) -> list[Record]:
    evidence: list[Record] = []
    seen: set[tuple[str, str, str]] = set()
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
        locator = parse_locator(result)
        key = (paper_id, source_type, json.dumps(locator, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "evidence_id": f"ev_{len(evidence) + 1:03d}",
                "paper_id": paper_id,
                "source_type": source_type if source_type != "metadata" else "text_span",
                "evidence_text_or_value": compact_text(result.get("content"), max_chars=1000),
                "locator": locator,
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


def build_context(results: list[Record], selected_paper_ids: set[str], context_chars: int) -> str:
    parts: list[str] = []
    used_chars = 0
    for index, result in enumerate(results, start=1):
        if result.get("paper_id") not in selected_paper_ids:
            continue
        content = clean_text(result.get("content") or "")
        if not content:
            continue
        locator = parse_locator(result)
        header = (
            f"[C{index}] paper_id={result.get('paper_id')} "
            f"title={result.get('title')} source_type={result.get('source_type')} "
            f"locator={json.dumps(locator, ensure_ascii=False)} "
            f"score={result.get('search_score')}"
        )
        block = f"{header}\n{content}"
        if used_chars + len(block) > context_chars:
            remaining = context_chars - used_chars
            if remaining < 500:
                break
            block = block[:remaining].rstrip()
        parts.append(block)
        used_chars += len(block)
        if used_chars >= context_chars:
            break
    return "\n\n".join(parts)


def answer_schema_instruction(answer_types: list[str]) -> str:
    pieces = [
        '"paper_ids": array of relevant paper_id strings',
        '"evidence": array of objects with paper_id, source_type, page, short_quote',
    ]
    if "freeform" in answer_types:
        pieces.append('"freeform": concise answer string')
    if "multiple_choice" in answer_types:
        pieces.append('"multiple_choice": one option key such as "A"')
    if "table" in answer_types:
        pieces.append('"table": {"schema": [...], "rows": [...]}')
    return "{ " + ", ".join(pieces) + " }"


def user_prompt(sample: Record, context: str) -> str:
    answer_types = sample.get("answer_types") or []
    return f"""Question:
{sample.get("question")}

Task family: {sample.get("task_family")}
Primary evidence type: {sample.get("primary_evidence_type")}
Answer types: {answer_types}

Return a JSON object shaped like:
{answer_schema_instruction(answer_types)}

Retrieved context:
{context}
"""


def chat_json(
    client: Any,
    settings: OpenAISettings,
    sample: Record,
    context: str,
    *,
    max_tokens: int,
) -> Record:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(sample, context)},
    ]
    kwargs: Record = {
        "model": settings.chat_deployment,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = retry_chat_completion(client, kwargs)
    content = response.choices[0].message.content or "{}"
    return parse_json_object(content)


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


def normalize_choice(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"\b([A-Z])\b", text)
    return match.group(1) if match else text[:1]


def normalize_paper_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def build_answer(sample: Record, llm_payload: Record) -> Record:
    answer_types = set(sample.get("answer_types") or [])
    answer: Record = {}
    if "freeform" in answer_types:
        answer["freeform"] = {
            "text": compact_text(llm_payload.get("freeform") or llm_payload.get("answer"))
        }
    if "multiple_choice" in answer_types:
        answer["multiple_choice"] = {
            "gold": normalize_choice(
                llm_payload.get("multiple_choice")
                or llm_payload.get("choice")
                or llm_payload.get("gold")
            )
        }
    if "table" in answer_types:
        table = llm_payload.get("table") if isinstance(llm_payload.get("table"), dict) else {}
        answer["table"] = {
            "schema": table.get("schema") if isinstance(table.get("schema"), list) else [],
            "rows": table.get("rows") if isinstance(table.get("rows"), list) else [],
        }
    return answer


def run_one(
    sample: Record,
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    args: argparse.Namespace,
) -> Record:
    query = str(sample.get("question") or "")
    results = search_chunks(
        search_client,
        openai_client,
        openai_settings,
        query=query,
        top_chunks=args.top_chunks,
        vector_k=args.vector_k,
    )
    papers = rank_papers(results, args.top_papers)
    selected_paper_ids = {paper["paper_id"] for paper in papers}
    context = build_context(results, selected_paper_ids, args.context_chars)

    llm_payload: Record = {}
    if not args.retrieval_only:
        llm_payload = chat_json(
            openai_client,
            openai_settings,
            sample,
            context,
            max_tokens=args.max_answer_tokens,
        )

    llm_paper_ids = normalize_paper_ids(llm_payload.get("paper_ids"))
    if llm_paper_ids:
        paper_lookup = {paper["paper_id"]: paper for paper in papers}
        reordered = [paper_lookup[pid] for pid in llm_paper_ids if pid in paper_lookup]
        papers = (reordered + [paper for paper in papers if paper["paper_id"] not in llm_paper_ids])[
            : args.top_papers
        ]
        selected_paper_ids = {paper["paper_id"] for paper in papers}

    return {
        "query_id": sample.get("query_id"),
        "gold_papers": paper_submission(papers),
        "evidence": evidence_from_results(
            results,
            selected_paper_ids,
            limit=args.evidence_limit,
        ),
        "answer": build_answer(sample, llm_payload),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate LitTraceQA predictions with Azure Search + AOAI RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "validation_inputs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "submission.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-chunks", type=int, default=24)
    parser.add_argument("--vector-k", type=int, default=50)
    parser.add_argument("--top-papers", type=int, default=5)
    parser.add_argument("--evidence-limit", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=18000)
    parser.add_argument("--max-answer-tokens", type=int, default=1200)
    parser.add_argument("--retrieval-only", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_environment(args.env_file)
    search_settings = SearchSettings.from_env()
    openai_settings = OpenAISettings.from_env()
    search_client = build_search_client(search_settings)
    openai_client = build_openai_client(openai_settings)

    samples = read_jsonl(args.input)
    if args.limit:
        samples = samples[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            query_id = sample.get("query_id")
            print(f"[{index}/{len(samples)}] {query_id}", file=sys.stderr)
            try:
                prediction = run_one(
                    sample,
                    search_client=search_client,
                    openai_client=openai_client,
                    openai_settings=openai_settings,
                    args=args,
                )
            except Exception as exc:  # noqa: BLE001 - emit a valid empty row.
                failed += 1
                print(f"FAIL {query_id}: {exc}", file=sys.stderr)
                prediction = {
                    "query_id": query_id,
                    "gold_papers": [],
                    "evidence": [],
                    "answer": build_answer(sample, {}),
                }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"wrote {len(samples)} predictions to {args.output}; failed={failed}", file=sys.stderr)
    return 1 if failed == len(samples) and samples else 0


if __name__ == "__main__":
    sys.exit(main())
