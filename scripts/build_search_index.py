#!/usr/bin/env python3
"""Create and populate the Azure AI Search index for LitTraceQA chunks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

from azure_config import (
    OpenAISettings,
    SearchSettings,
    build_openai_client,
    build_search_client,
    build_search_index_client,
    load_environment,
)
from littrace_common import ROOT, Record, batched, read_jsonl


VECTOR_FIELD = "content_vector"
VECTOR_PROFILE = "littraceqa-vector-profile"
VECTOR_ALGORITHM = "littraceqa-hnsw"


def create_index(index_client: Any, settings: SearchSettings, embedding_dimensions: int) -> None:
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    fields = [
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
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name=VECTOR_FIELD,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name=VECTOR_PROFILE,
        ),
    ]

    vector_search = VectorSearch(
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
    index = SearchIndex(
        name=settings.index_name,
        fields=fields,
        vector_search=vector_search,
    )
    index_client.create_or_update_index(index)


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


def search_document(chunk: Record, vector: list[float]) -> Record:
    return {
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
        "content": str(chunk.get("content") or ""),
        VECTOR_FIELD: vector,
    }


def load_chunks(path: Path, *, paper_ids: set[str], limit: int) -> list[Record]:
    chunks = read_jsonl(path)
    if paper_ids:
        chunks = [chunk for chunk in chunks if chunk.get("paper_id") in paper_ids]
    if limit:
        chunks = chunks[:limit]
    return chunks


def upload_chunks(
    *,
    search_client: Any,
    openai_client: Any,
    openai_settings: OpenAISettings,
    chunks: list[Record],
    embedding_batch_size: int,
    upload_batch_size: int,
) -> tuple[int, int]:
    uploaded = 0
    failed = 0
    pending_docs: list[Record] = []

    def flush() -> None:
        nonlocal uploaded, failed, pending_docs
        if not pending_docs:
            return
        results = search_client.upload_documents(documents=pending_docs)
        for result in results:
            if result.succeeded:
                uploaded += 1
            else:
                failed += 1
                print(
                    f"UPLOAD FAIL key={result.key} error={result.error_message}",
                    file=sys.stderr,
                )
        print(f"uploaded={uploaded} failed={failed}", file=sys.stderr)
        pending_docs = []

    for batch_index, chunk_batch in enumerate(
        batched(chunks, embedding_batch_size), start=1
    ):
        texts = [str(chunk.get("content") or "") for chunk in chunk_batch]
        vectors = embed_texts(openai_client, openai_settings, texts)
        for chunk, vector in zip(chunk_batch, vectors):
            pending_docs.append(search_document(chunk, vector))
            if len(pending_docs) >= upload_batch_size:
                flush()
        print(
            f"embedded batch {batch_index}: {len(chunk_batch)} chunks",
            file=sys.stderr,
        )
    flush()
    return uploaded, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and populate an Azure AI Search vector/hybrid index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=ROOT / "artifacts" / "docint" / "chunks.jsonl",
    )
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--upload-batch-size", type=int, default=512)
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

    chunks = load_chunks(args.chunks, paper_ids=set(args.paper_id), limit=args.limit)
    if not chunks:
        print(f"no chunks to upload from {args.chunks}", file=sys.stderr)
        return 0

    search_client = build_search_client(search_settings)
    openai_client = build_openai_client(openai_settings)
    uploaded, failed = upload_chunks(
        search_client=search_client,
        openai_client=openai_client,
        openai_settings=openai_settings,
        chunks=chunks,
        embedding_batch_size=args.embedding_batch_size,
        upload_batch_size=args.upload_batch_size,
    )
    print(f"finished: uploaded={uploaded} failed={failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
