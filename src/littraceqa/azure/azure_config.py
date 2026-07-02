"""Environment and client configuration for Azure-based LitTraceQA scripts."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from ..common import ROOT


def load_environment(env_file: Optional[Path] = None) -> None:
    load_dotenv(env_file or (ROOT / ".env"), override=False)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _required(name: str) -> str:
    value = _env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


@dataclasses.dataclass(frozen=True)
class DocumentIntelligenceSettings:
    endpoint: str
    key: str
    model: str = "prebuilt-layout"
    api_version: str = ""

    @classmethod
    def from_env(cls) -> "DocumentIntelligenceSettings":
        return cls(
            endpoint=_required("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            key=_required("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
            model=_env("AZURE_DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-layout"),
            api_version=_env("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION"),
        )


@dataclasses.dataclass(frozen=True)
class SearchSettings:
    endpoint: str
    admin_key: str
    index_name: str

    @classmethod
    def from_env(cls) -> "SearchSettings":
        return cls(
            endpoint=_required("AZURE_SEARCH_ENDPOINT"),
            admin_key=_required("AZURE_SEARCH_ADMIN_KEY"),
            index_name=_env("AZURE_SEARCH_INDEX_NAME", "littraceqa-papers"),
        )


@dataclasses.dataclass(frozen=True)
class OpenAISettings:
    endpoint: str
    api_key: str
    api_version: str
    use_v1: bool
    chat_deployment: str
    embedding_deployment: str
    embedding_dimensions: int
    request_embedding_dimensions: bool

    @classmethod
    def from_env(cls) -> "OpenAISettings":
        return cls(
            endpoint=_required("AZURE_OPENAI_ENDPOINT"),
            api_key=_required("AZURE_OPENAI_API_KEY"),
            api_version=_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            use_v1=_bool("AZURE_OPENAI_USE_V1", False),
            chat_deployment=_required("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            embedding_deployment=_required("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            embedding_dimensions=_int("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1536),
            request_embedding_dimensions=_bool(
                "AZURE_OPENAI_EMBEDDING_REQUEST_DIMENSIONS", False
            ),
        )


def build_document_intelligence_client(settings: DocumentIntelligenceSettings):
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    kwargs = {}
    if settings.api_version:
        kwargs["api_version"] = settings.api_version
    return DocumentIntelligenceClient(
        endpoint=settings.endpoint,
        credential=AzureKeyCredential(settings.key),
        **kwargs,
    )


def build_search_index_client(settings: SearchSettings):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient

    return SearchIndexClient(
        endpoint=settings.endpoint,
        credential=AzureKeyCredential(settings.admin_key),
    )


def build_search_client(settings: SearchSettings):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    return SearchClient(
        endpoint=settings.endpoint,
        index_name=settings.index_name,
        credential=AzureKeyCredential(settings.admin_key),
    )


def build_openai_client(
    settings: OpenAISettings,
    *,
    max_retries: Optional[int] = None,
    timeout: Optional[float] = None,
):
    """Build an OpenAI client; SDK retry/timeout defaults apply unless overridden."""
    client_kwargs: dict[str, Any] = {}
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    if settings.use_v1:
        from openai import OpenAI

        base_url = settings.endpoint.rstrip("/")
        if not base_url.endswith("/openai/v1"):
            base_url = f"{base_url}/openai/v1"
        return OpenAI(api_key=settings.api_key, base_url=base_url, **client_kwargs)

    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        api_version=settings.api_version,
        **client_kwargs,
    )
