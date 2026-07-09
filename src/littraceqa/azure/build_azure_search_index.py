#!/usr/bin/env python3
"""Create and populate the Azure AI Search index for LitTraceQA chunks.

The chunk corpus is ~1.9M records / 6+ GB, so this script streams input
(consolidated JSONL or a per-paper directory), embeds with bounded
concurrency, retries throttled calls, and checkpoints progress at paper
boundaries so an interrupted upload can be resumed with --resume.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .azure_config import (
    OpenAISettings,
    SearchSettings,
    build_openai_client,
    build_search_client,
    build_search_index_client,
    load_environment,
)
from ..common import ROOT, Record, _retry_after_seconds, batched, read_json, write_json


VECTOR_FIELD = "content_vector"
VECTOR_PROFILE = "littraceqa-vector-profile"
VECTOR_ALGORITHM = "littraceqa-hnsw"
SEMANTIC_CONFIGURATION = "littraceqa-semantic"

# The embedding model rejects inputs above 8,192 tokens. ~30k chars is safe
# for English prose (~4 chars/token) but NOT for digit/pipe-dense table
# chunks or mojibake OCR (~2 chars/token or worse), so a context-length 400
# triggers adaptive re-truncation in embed_texts_with_retry: the failing
# batch's texts are halved (down to MIN_EMBED_CHARS) without consuming
# transient retry attempts.
MAX_EMBED_CHARS = 30000
MIN_EMBED_CHARS = 512
EMBED_RETRY_ATTEMPTS = 8
UPLOAD_RETRY_ROUNDS = 3
# Facet bucket page size for --skip-indexed. Azure AI Search caps facet
# buckets per request (default 10, service-side hard cap well above 1000),
# so we page in value order with 1000 buckets per call.
FACET_PAGE_SIZE = 1000

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "docint" / "index_upload_checkpoint.json"


def _index_fields(embedding_dimensions: int) -> list[Any]:
    """Declarative field schema for the chunk index."""
    from azure.search.documents.indexes.models import (
        SearchField,
        SearchFieldDataType,
        SearchableField,
        SimpleField,
    )

    return [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(
            name="paper_id",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            sortable=True,
        ),
        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SearchableField(name="authors", type=SearchFieldDataType.String),
        SearchableField(name="abstract", type=SearchFieldDataType.String),
        SimpleField(
            name="venue",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="year",
            type=SearchFieldDataType.Int32,
            filterable=True,
            facetable=True,
            sortable=True,
        ),
        SimpleField(
            name="track",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(name="award", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="source_type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(name="section", type=SearchFieldDataType.String),
        SearchField(
            name="page_numbers",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Int32),
            filterable=True,
        ),
        SimpleField(name="source_url", type=SearchFieldDataType.String),
        SimpleField(name="pdf_url", type=SearchFieldDataType.String),
        SimpleField(name="anthology_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="locator_json", type=SearchFieldDataType.String),
        SimpleField(name="table_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="figure_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            stored=False,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]


def _vector_search_config() -> Any:
    """HNSW/cosine vector search configuration."""
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        VectorSearch,
        VectorSearchProfile,
    )

    return VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=VECTOR_ALGORITHM,
                parameters=HnswParameters(metric="cosine"),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE,
                algorithm_configuration_name=VECTOR_ALGORITHM,
            )
        ],
    )


def _semantic_search_config() -> Any:
    """Semantic ranking configuration (title/content/section fields)."""
    from azure.search.documents.indexes.models import (
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
    )

    return SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIGURATION,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="section")],
                ),
            )
        ],
        default_configuration_name=SEMANTIC_CONFIGURATION,
    )


def create_index(index_client: Any, settings: SearchSettings, embedding_dimensions: int) -> None:
    from azure.core.exceptions import HttpResponseError
    from azure.search.documents.indexes.models import SearchIndex

    index = SearchIndex(
        name=settings.index_name,
        fields=_index_fields(embedding_dimensions),
        vector_search=_vector_search_config(),
        semantic_search=_semantic_search_config(),
    )
    try:
        index_client.create_or_update_index(index)
    except HttpResponseError:
        print(
            "ERROR: create_or_update_index failed. If the index already exists, "
            "Azure rejects in-place schema changes (HTTP 400) such as adding the "
            "table_id/figure_id fields, the semantic configuration, or "
            f"stored=False on {VECTOR_FIELD}. Rerun with --recreate to drop and "
            "rebuild the index (existing documents are lost).",
            file=sys.stderr,
        )
        raise


def delete_index_if_exists(index_client: Any, index_name: str) -> None:
    from azure.core.exceptions import ResourceNotFoundError

    try:
        index_client.delete_index(index_name)
        print(f"deleted existing index: {index_name}", file=sys.stderr)
        time.sleep(2.0)
    except ResourceNotFoundError:
        pass


def embedding_kwargs(settings: OpenAISettings, texts: list[str]) -> Record:
    kwargs: Record = {
        "model": settings.embedding_deployment,
        "input": texts,
    }
    if settings.request_embedding_dimensions:
        kwargs["dimensions"] = settings.embedding_dimensions
    return kwargs


def embed_texts(client: Any, settings: OpenAISettings, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(**embedding_kwargs(settings, texts))
    data = sorted(response.data, key=lambda item: item.index)
    vectors = [list(item.embedding) for item in data]
    for vector in vectors:
        if len(vector) != settings.embedding_dimensions:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"index expects {settings.embedding_dimensions}, got {len(vector)}. "
                "Fix AZURE_OPENAI_EMBEDDING_DIMENSIONS or your embedding deployment."
            )
    return vectors


def request_status_code(exc: Exception) -> Optional[int]:
    """Extract an HTTP status code from an SDK exception, if present."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_permanent_request_error(exc: Exception) -> bool:
    """True for non-retryable 4xx errors (400 invalid request etc.).

    429 (throttling) and 408 (timeout) are transient and keep the full retry
    budget; anything else in the 4xx range will fail identically on every
    retry, so callers should truncate/skip instead of burning attempts.
    """
    status = request_status_code(exc)
    return status is not None and 400 <= status < 500 and status not in (408, 429)


def is_context_length_error(exc: Exception) -> bool:
    """True when the SDK error is a 400 caused by an over-long embedding input."""
    if request_status_code(exc) != 400:
        return False
    message = str(exc).lower()
    return (
        "context length" in message
        or "context_length_exceeded" in message
        or "maximum context" in message
        or "too many tokens" in message
        or "max input" in message
        or "too long" in message
    )


def embed_texts_with_retry(
    client: Any,
    settings: OpenAISettings,
    texts: list[str],
    *,
    attempts: int = EMBED_RETRY_ATTEMPTS,
) -> Optional[list[list[float]]]:
    """Embed one batch with retries.

    Transient errors (429/5xx/network) use the full ``attempts`` retry budget
    and raise RuntimeError when exhausted. Permanent 4xx errors (400 invalid
    request etc.) never abort a multi-hour run: the batch is retried once with
    each text truncated to half its length, then the batch is SKIPPED by
    returning None so the caller can record the affected keys and continue.
    """
    texts = list(texts)
    delay = 1.0
    last_error: Optional[Exception] = None
    max_chars = MAX_EMBED_CHARS
    permanent_retried = False
    # Counts TRANSIENT failures only; context-length re-truncations and the
    # one permanent-4xx halving retry do not consume the ``attempts`` budget.
    transient_failures = 0
    while transient_failures < attempts:
        try:
            return embed_texts(client, settings, texts)
        except ValueError:
            raise  # Dimension mismatch is a configuration error, not transient.
        except Exception as exc:  # noqa: BLE001 - SDK error classes vary.
            last_error = exc
            if is_context_length_error(exc) and max_chars > MIN_EMBED_CHARS:
                # Token-dense text (tables, OCR mojibake) can exceed the
                # 8,192-token limit well under MAX_EMBED_CHARS. A 400 is not
                # transient, so halve the batch's texts and retry (bounded by
                # MIN_EMBED_CHARS).
                max_chars = max(max_chars // 2, MIN_EMBED_CHARS)
                texts = [text[:max_chars] for text in texts]
                print(
                    f"WARN embedding input over token limit "
                    f"({type(exc).__name__}); re-truncating batch to "
                    f"{max_chars} chars and retrying",
                    file=sys.stderr,
                )
                continue
            if is_permanent_request_error(exc):
                # A permanent 4xx fails identically on every retry. Retry once
                # with halved content (covers borderline payload issues), then
                # skip the batch.
                if not permanent_retried:
                    permanent_retried = True
                    texts = [text[: max(len(text) // 2, 1)] or " " for text in texts]
                    print(
                        f"WARN embedding request rejected with permanent "
                        f"{request_status_code(exc)} ({type(exc).__name__}: {exc}); "
                        f"retrying once with content halved",
                        file=sys.stderr,
                    )
                    continue
                print(
                    f"WARN embedding batch of {len(texts)} text(s) still rejected "
                    f"with permanent {request_status_code(exc)} "
                    f"({type(exc).__name__}: {exc}); skipping these documents",
                    file=sys.stderr,
                )
                return None
            transient_failures += 1
            if transient_failures >= attempts:
                break
            wait = _retry_after_seconds(exc)  # already capped at 120s
            if wait is None:
                wait = delay
            print(
                f"WARN embedding attempt {transient_failures}/{attempts} failed "
                f"({type(exc).__name__}: {exc}); retrying in {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay = min(delay * 2.0, 60.0)
    raise RuntimeError(f"embedding failed after {attempts} attempts") from last_error


def embed_chunks(
    client: Any,
    settings: OpenAISettings,
    chunks: list[Record],
    *,
    batch_size: int,
    workers: int,
) -> list[Optional[list[float]]]:
    """Embed chunk contents, preserving chunk order across concurrent batches.

    Entries are None for chunks whose batch was skipped after a permanent 4xx
    embedding error (see embed_texts_with_retry); callers must skip those
    chunks rather than uploading them.
    """
    texts: list[str] = []
    for chunk in chunks:
        text = str(chunk.get("content") or "")
        if len(text) > MAX_EMBED_CHARS:
            print(
                f"WARN truncating chunk id={chunk.get('id')} from {len(text)} "
                f"to {MAX_EMBED_CHARS} chars for embedding (8192-token model limit)",
                file=sys.stderr,
            )
            text = text[:MAX_EMBED_CHARS]
        texts.append(text or " ")
    batches = list(batched(texts, batch_size))
    if workers <= 1 or len(batches) <= 1:
        results = [embed_texts_with_retry(client, settings, batch) for batch in batches]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(lambda batch: embed_texts_with_retry(client, settings, batch), batches)
            )
    vectors: list[Optional[list[float]]] = []
    for batch, batch_vectors in zip(batches, results):
        if batch_vectors is None:
            vectors.extend([None] * len(batch))
        else:
            vectors.extend(batch_vectors)
    return vectors


def parse_locator_ids(chunk: Record) -> tuple[str, str]:
    """Extract (table_id, figure_id) from the chunk's locator_json string."""
    raw = str(chunk.get("locator_json") or "{}")
    try:
        locator = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        locator = {}
    if not isinstance(locator, dict):
        locator = {}
    return str(locator.get("table_id") or ""), str(locator.get("figure_id") or "")


def search_document(
    chunk: Record,
    vector: list[float],
    *,
    omit_fields: Optional[set[str]] = None,
) -> Record:
    table_id, figure_id = parse_locator_ids(chunk)
    doc = {
        "id": chunk["id"],
        "paper_id": str(chunk.get("paper_id") or ""),
        "chunk_id": int(chunk.get("chunk_id") or 0),
        "title": str(chunk.get("title") or ""),
        "authors": str(chunk.get("authors") or ""),
        "abstract": str(chunk.get("abstract") or ""),
        "venue": str(chunk.get("venue") or ""),
        "year": int(chunk["year"]) if chunk.get("year") is not None else None,
        "track": str(chunk.get("track") or ""),
        "award": str(chunk.get("award") or "") if chunk.get("award") is not None else None,
        "source_type": str(chunk.get("source_type") or ""),
        "section": str(chunk.get("section") or ""),
        "page_numbers": [int(page) for page in chunk.get("page_numbers") or []],
        "source_url": str(chunk.get("source_url") or ""),
        "pdf_url": str(chunk.get("pdf_url") or ""),
        "anthology_id": str(chunk.get("anthology_id") or ""),
        "locator_json": str(chunk.get("locator_json") or "{}"),
        "table_id": table_id,
        "figure_id": figure_id,
        "content": str(chunk.get("content") or ""),
        VECTOR_FIELD: vector,
    }
    for field in omit_fields or ():
        doc.pop(field, None)
    return doc


def fields_missing_from_live_index(
    index_client: Any,
    index_name: str,
    field_names: tuple[str, ...] = ("table_id", "figure_id"),
) -> set[str]:
    """Return which of ``field_names`` the deployed index does NOT define.

    Used with --skip-index-create so appends into an older, still-deployed
    schema (without table_id/figure_id) keep working: unknown document keys
    are rejected by Azure AI Search, so missing fields must be omitted from
    uploaded documents. Any failure to fetch the definition (permissions,
    transient errors) is tolerated and treated as "new schema" (omit nothing).
    """
    try:
        index = index_client.get_index(index_name)
    except Exception as exc:  # noqa: BLE001 - tolerate and assume new schema.
        print(
            f"WARN could not fetch live index definition for '{index_name}' "
            f"({type(exc).__name__}: {exc}); assuming current schema",
            file=sys.stderr,
        )
        return set()
    existing: set[str] = set()
    for field in getattr(index, "fields", None) or []:
        name = getattr(field, "name", None)
        if name is None and isinstance(field, dict):
            name = field.get("name")
        if name:
            existing.add(str(name))
    if not existing:
        return set()
    return {name for name in field_names if name not in existing}


def iter_jsonl_stream(path: Path) -> Iterator[Record]:
    """Stream a JSONL file one record at a time without materializing it."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield record


def filter_records(
    records: Iterable[Record],
    include_paper_ids: set[str],
    skip_paper_ids: set[str],
) -> Iterator[Record]:
    for record in records:
        paper_id = str(record.get("paper_id") or "")
        if include_paper_ids and paper_id not in include_paper_ids:
            continue
        if paper_id in skip_paper_ids:
            continue
        yield record


def iter_input_records(
    *,
    input_path: Path,
    input_dir: Optional[Path],
    include_paper_ids: set[str],
    skip_paper_ids: set[str],
) -> Iterator[Record]:
    """Stream chunk records from --input-dir (per-paper files) or --input."""
    if input_dir is not None:
        for file_path in sorted(input_dir.glob("*.jsonl")):
            if file_path.name.startswith("."):
                continue  # Temp files from atomic writers.
            stem = file_path.stem  # Per-paper files are named {paper_id}.jsonl.
            if include_paper_ids and stem not in include_paper_ids:
                continue
            if stem in skip_paper_ids:
                continue
            yield from filter_records(
                iter_jsonl_stream(file_path), include_paper_ids, skip_paper_ids
            )
        return
    yield from filter_records(iter_jsonl_stream(input_path), include_paper_ids, skip_paper_ids)


def iter_paper_groups(
    records: Iterable[Record],
    *,
    limit: int = 0,
) -> Iterator[tuple[str, list[Record], bool]]:
    """Group a chunk stream by contiguous paper_id runs.

    Yields (paper_id, chunks, complete). ``complete`` is False only when
    ``limit`` truncated the paper mid-stream; such papers are uploaded but
    never checkpointed as done.
    """
    remaining = limit if limit > 0 else None
    for paper_id, group in itertools.groupby(
        records, key=lambda record: str(record.get("paper_id") or "")
    ):
        chunks: list[Record] = []
        truncated = False
        for record in group:
            if remaining is not None and remaining <= 0:
                truncated = True
                break
            chunks.append(record)
            if remaining is not None:
                remaining -= 1
        if chunks:
            yield paper_id, chunks, not truncated
        if remaining is not None and remaining <= 0:
            return


def read_paper_id_file(path: Path) -> set[str]:
    paper_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                paper_ids.add(value)
    return paper_ids


def fetch_indexed_paper_ids(search_client: Any, *, page_size: int = FACET_PAGE_SIZE) -> set[str]:
    """Collect distinct paper_id values from the live index via paginated facets.

    paper_id is facetable; a single facet request is capped at ``page_size``
    buckets (FACET_PAGE_SIZE=1000 per request), so we sort buckets by value
    and page with a ``paper_id gt '<last seen>'`` filter until a short page
    is returned.
    """
    paper_ids: set[str] = set()
    last_value = ""
    while True:
        kwargs: Record = {
            "search_text": "*",
            "top": 0,
            "facets": [f"paper_id,count:{page_size},sort:value"],
        }
        if last_value:
            escaped = last_value.replace("'", "''")
            kwargs["filter"] = f"paper_id gt '{escaped}'"
        results = search_client.search(**kwargs)
        facets = results.get_facets() or {}
        values = [
            str(bucket.get("value"))
            for bucket in facets.get("paper_id") or []
            if bucket.get("value")
        ]
        if not values:
            break
        paper_ids.update(values)
        last_value = values[-1]
        if len(values) < page_size:
            break
    return paper_ids


def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = read_json(path)
    if not isinstance(payload, dict):
        return set()
    return {str(paper_id) for paper_id in payload.get("completed_paper_ids") or []}


def save_checkpoint(path: Path, completed_paper_ids: set[str]) -> None:
    ordered = sorted(completed_paper_ids)
    write_json(
        path,
        {
            "completed_paper_ids": ordered,
            "count": len(ordered),
            "last_paper_id": ordered[-1] if ordered else "",
        },
    )


def upload_documents_with_retry(
    search_client: Any,
    docs: list[Record],
    *,
    batch_size: int,
    retry_rounds: int = UPLOAD_RETRY_ROUNDS,
) -> tuple[int, dict[str, str]]:
    """Upload documents in batches, retrying per-document failures.

    Returns (uploaded_count, failures) where failures maps document key to
    the last error message after ``retry_rounds`` retry rounds. Request-level
    exceptions raised by upload_documents itself (HttpResponseError, network
    errors) consume a retry round for the whole batch; if the batch still
    fails after all rounds, every key in it is recorded as failed and the
    next batch proceeds -- a request-level error never aborts the run.
    """
    uploaded = 0
    failures: dict[str, str] = {}
    for doc_batch in batched(docs, batch_size):
        pending = list(doc_batch)
        round_failures: dict[str, str] = {}
        delay = 2.0
        for round_index in range(retry_rounds + 1):
            if round_index:
                print(
                    f"retrying {len(pending)} failed documents "
                    f"(round {round_index}/{retry_rounds}) in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                delay = min(delay * 2.0, 30.0)
            try:
                results = search_client.upload_documents(documents=pending)
            except Exception as exc:  # noqa: BLE001 - HttpResponseError/network.
                message = f"{type(exc).__name__}: {exc}"
                round_failures = {str(doc["id"]): message for doc in pending}
                print(
                    f"WARN upload_documents raised {message} for a batch of "
                    f"{len(pending)} documents",
                    file=sys.stderr,
                )
                continue  # Retry the whole batch on the next round.
            round_failures = {
                str(result.key): str(result.error_message or "")
                for result in results
                if not result.succeeded
            }
            uploaded += len(pending) - len(round_failures)
            if not round_failures:
                break
            for key, message in round_failures.items():
                print(f"UPLOAD FAIL key={key} error={message}", file=sys.stderr)
            pending = [doc for doc in pending if doc["id"] in round_failures]
        failures.update(round_failures)
    return uploaded, failures


def _flush_paper_buffer(
    buffer: list[tuple[str, list[Record], bool]],
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    embedding_batch_size: int,
    embed_workers: int,
    upload_batch_size: int,
    checkpoint_path: Path,
    completed_paper_ids: set[str],
    omit_fields: Optional[set[str]],
    uploaded_before: int,
) -> tuple[int, dict[str, str]]:
    """Embed and upload one buffered group of papers, then checkpoint.

    Returns (uploaded_count, failures) for this buffer. ``completed_paper_ids``
    is updated in place with the papers that fully uploaded; ``uploaded_before``
    is only used for the cumulative progress line.
    """
    chunks = [chunk for _, paper_chunks, _ in buffer for chunk in paper_chunks]
    vectors = embed_chunks(
        openai_client,
        openai_settings,
        chunks,
        batch_size=embedding_batch_size,
        workers=embed_workers,
    )
    docs: list[Record] = []
    embed_failures: dict[str, str] = {}
    for chunk, vector in zip(chunks, vectors):
        if vector is None:
            key = str(chunk.get("id") or "")
            embed_failures[key] = "embedding skipped after permanent 4xx error"
            print(f"EMBED SKIP key={key}", file=sys.stderr)
            continue
        docs.append(search_document(chunk, vector, omit_fields=omit_fields))
    batch_uploaded, failures = upload_documents_with_retry(
        search_client, docs, batch_size=upload_batch_size
    )
    key_to_paper = {doc["id"]: doc["paper_id"] for doc in docs}
    # Only upload failures block a paper's checkpoint; embed-skipped
    # chunks are permanent, so re-running them on --resume is pointless.
    failed_papers = {key_to_paper[key] for key in failures if key in key_to_paper}
    for paper_id, _, complete in buffer:
        if complete and paper_id not in failed_papers:
            completed_paper_ids.add(paper_id)
    save_checkpoint(checkpoint_path, completed_paper_ids)
    print(
        f"progress: uploaded={uploaded_before + batch_uploaded} "
        f"papers_checkpointed={len(completed_paper_ids)}",
        file=sys.stderr,
    )
    failures.update(embed_failures)
    return batch_uploaded, failures


def upload_stream(
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    records: Iterable[Record],
    embedding_batch_size: int,
    embed_workers: int,
    upload_batch_size: int,
    limit: int,
    checkpoint_path: Path,
    completed_paper_ids: set[str],
    omit_fields: Optional[set[str]] = None,
) -> tuple[int, dict[str, str]]:
    """Embed and upload a chunk stream, checkpointing at paper boundaries.

    Papers are buffered until roughly ``embedding_batch_size * embed_workers``
    chunks are pending, then embedded concurrently, uploaded, and the fully
    uploaded papers are recorded in the checkpoint (atomic write). Failures
    are accumulated for the failed-keys report and the stream continues: a
    single poison chunk must never abort a multi-hour run. Chunks skipped by
    embedding (permanent 4xx) are recorded as failed but do NOT block their
    paper's checkpoint, so --resume never re-hits them; persistent upload
    failures do block the checkpoint so --resume retries those papers.
    """
    uploaded = 0
    all_failures: dict[str, str] = {}
    buffer: list[tuple[str, list[Record], bool]] = []
    buffer_chunks = 0
    flush_threshold = embedding_batch_size * max(embed_workers, 1)

    def flush() -> None:
        nonlocal uploaded, buffer, buffer_chunks
        if not buffer:
            return
        batch_uploaded, failures = _flush_paper_buffer(
            buffer,
            search_client=search_client,
            openai_client=openai_client,
            openai_settings=openai_settings,
            embedding_batch_size=embedding_batch_size,
            embed_workers=embed_workers,
            upload_batch_size=upload_batch_size,
            checkpoint_path=checkpoint_path,
            completed_paper_ids=completed_paper_ids,
            omit_fields=omit_fields,
            uploaded_before=uploaded,
        )
        uploaded += batch_uploaded
        all_failures.update(failures)
        buffer = []
        buffer_chunks = 0

    for paper_id, paper_chunks, complete in iter_paper_groups(records, limit=limit):
        buffer.append((paper_id, paper_chunks, complete))
        buffer_chunks += len(paper_chunks)
        if buffer_chunks >= flush_threshold:
            flush()
    flush()
    return uploaded, all_failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and populate an Azure AI Search vector/hybrid index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--input",
        "--chunks",
        dest="input",
        type=Path,
        default=ROOT / "artifacts" / "docint" / "chunks.jsonl",
        help="Consolidated chunks JSONL, streamed line by line.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory of per-paper *.jsonl chunk files "
            "(e.g. artifacts/docint/chunks); overrides --input."
        ),
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Only upload these paper ids (repeatable).",
    )
    parser.add_argument(
        "--paper-id-file",
        type=Path,
        default=None,
        help="Text file with one paper_id per line to upload (merged with --paper-id).",
    )
    parser.add_argument(
        "--skip-indexed",
        action="store_true",
        help=(
            "Skip papers whose paper_id already appears in the live index "
            f"(paginated facets, {FACET_PAGE_SIZE} buckets per request)."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many chunks.")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=4,
        help="Concurrent embedding requests (bounded thread pool).",
    )
    parser.add_argument("--upload-batch-size", type=int, default=256)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="JSON file recording fully uploaded paper_ids (written atomically).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip papers already recorded in --checkpoint (otherwise it is overwritten).",
    )
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--skip-index-create", action="store_true")
    parser.add_argument("--create-only", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_environment(args.env_file)
    search_settings = SearchSettings.from_env()
    openai_settings = OpenAISettings.from_env()

    index_client = build_search_index_client(search_settings)
    if args.recreate:
        delete_index_if_exists(index_client, search_settings.index_name)
    if not args.skip_index_create:
        create_index(
            index_client,
            search_settings,
            embedding_dimensions=openai_settings.embedding_dimensions,
        )
        print(f"index ready: {search_settings.index_name}", file=sys.stderr)

    if args.create_only:
        return 0

    omit_fields: set[str] = set()
    if args.skip_index_create:
        # The deployed index may predate the table_id/figure_id fields; Azure
        # rejects documents carrying unknown keys, so probe the live schema
        # once and omit missing fields from every uploaded document.
        omit_fields = fields_missing_from_live_index(
            index_client, search_settings.index_name
        )
        if omit_fields:
            print(
                f"WARN live index '{search_settings.index_name}' lacks fields "
                f"{sorted(omit_fields)}; omitting them from uploaded documents "
                "(old-schema compatibility for --skip-index-create)",
                file=sys.stderr,
            )

    include_paper_ids = {str(paper_id) for paper_id in args.paper_id}
    if args.paper_id_file:
        include_paper_ids |= read_paper_id_file(args.paper_id_file)

    completed_paper_ids = load_checkpoint(args.checkpoint) if args.resume else set()
    skip_paper_ids = set(completed_paper_ids)

    search_client = build_search_client(search_settings)
    if args.skip_indexed:
        indexed_paper_ids = fetch_indexed_paper_ids(search_client)
        print(
            f"skip-indexed: {len(indexed_paper_ids)} paper_ids already in the index",
            file=sys.stderr,
        )
        skip_paper_ids |= indexed_paper_ids

    records = iter_input_records(
        input_path=args.input,
        input_dir=args.input_dir,
        include_paper_ids=include_paper_ids,
        skip_paper_ids=skip_paper_ids,
    )
    openai_client = build_openai_client(openai_settings, max_retries=8, timeout=120.0)
    uploaded, failures = upload_stream(
        search_client=search_client,
        openai_client=openai_client,
        openai_settings=openai_settings,
        records=records,
        embedding_batch_size=args.embedding_batch_size,
        embed_workers=args.embed_workers,
        upload_batch_size=args.upload_batch_size,
        limit=args.limit,
        checkpoint_path=args.checkpoint,
        completed_paper_ids=completed_paper_ids,
        omit_fields=omit_fields,
    )

    if failures:
        failed_keys_path = args.checkpoint.with_name(args.checkpoint.stem + "_failed_keys.txt")
        failed_keys_path.parent.mkdir(parents=True, exist_ok=True)
        with failed_keys_path.open("w", encoding="utf-8") as handle:
            for key in sorted(failures):
                handle.write(f"{key}\n")
        print(
            f"finished: uploaded={uploaded} failed={len(failures)} "
            f"failed_keys={failed_keys_path}",
            file=sys.stderr,
        )
        return 1
    if not uploaded:
        source = args.input_dir if args.input_dir is not None else args.input
        print(f"no chunks to upload from {source}", file=sys.stderr)
        return 0
    print(f"finished: uploaded={uploaded} failed=0", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
