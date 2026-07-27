"""Paper-level BM25 index built from deterministic aggregates of common chunks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult
from littraceqa.di_pipeline.index.bm25_index import BM25Index
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.method_aliases import (
    GENERIC_METHOD_ALIASES,
    extract_self_owned_method_aliases,
    has_standalone_exact_alias,
    method_aliases_equal,
    text_before_references,
)


_REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*)|[A-Z])?[.):\-]?\s*"
    r"(?:references?|bibliography)\s*$",
    re.IGNORECASE,
)
_PAPERS_FILENAME = "papers.jsonl"
_LEGACY_RECORDS_FILENAME = "chunks.jsonl"
_METHOD_GRAPH_FILENAME = "method_alias_graph.json"
_METHOD_GRAPH_SCHEMA_VERSION = 3
_METHOD_CONTEXT_CHARS = 200
_METHOD_FIRST_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_METHOD_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?![A-Za-z0-9_])"
)
_METHOD_CONTEXT_WORD_RE = re.compile(r"[a-z][a-z0-9]*")
_GENERIC_METHOD_CONTEXT_WORDS = frozenset(
    {
        "algorithm",
        "algorithms",
        "approach",
        "approaches",
        "based",
        "effective",
        "efficient",
        "framework",
        "frameworks",
        "general",
        "learning",
        "loss",
        "losses",
        "method",
        "methods",
        "model",
        "models",
        "module",
        "modules",
        "new",
        "novel",
        "objective",
        "objectives",
        "paper",
        "proposed",
        "simple",
        "strategy",
        "strategies",
        "system",
        "systems",
        "task",
        "tasks",
        "technique",
        "techniques",
        "training",
        "unified",
        "using",
        "with",
        "without",
        "work",
    }
)


def _body_text(chunk: Chunk) -> str:
    """Remove the repeated paper prefix used by the common Chunk contract."""
    _, separator, body = chunk.text.partition("\n")
    return (body if separator else chunk.text).strip()


def _paper_chunk(
    paper_id: str,
    chunks: list[Chunk],
    exclude_references: bool,
    extract_method_names: bool = False,
) -> Chunk:
    first = chunks[0]
    metadata = dict(first.metadata)
    title = str(metadata.get("title") or "").strip()
    parts: list[str] = []
    heading = ""
    inside_references = False

    for chunk in chunks:
        body = _body_text(chunk)
        if not body:
            continue
        if chunk.chunk_type == "title_abstract":
            parts.append(body)
            continue

        if chunk.metadata.get("text_level") is not None:
            heading = body
            inside_references = bool(_REFERENCE_HEADING_RE.fullmatch(heading))
            if not (exclude_references and inside_references):
                parts.append(body)
            continue

        if exclude_references and inside_references:
            continue
        parts.append(f"{heading}\n{body}" if heading else body)

    text = "\n".join(part for part in (title, *parts) if part)
    metadata.update(
        {
            "paper_aggregation": "whole_paper",
            "source_chunk_count": len(chunks),
            "indexed_char_count": len(text),
        }
    )
    if extract_method_names:
        evidence = extract_self_owned_method_aliases(title, text)
        metadata["method_names"] = [item.alias for item in evidence]
        metadata["method_alias_evidence"] = [
            item.to_dict() for item in evidence
        ]
    return Chunk(
        chunk_id=f"{paper_id}#paper",
        paper_id=paper_id,
        text=text,
        chunk_type="paper",
        metadata=metadata,
    )


def aggregate_papers(
    chunks: Iterable[Chunk],
    exclude_references: bool = True,
    extract_method_names: bool = False,
) -> Iterator[Chunk]:
    """Yield one paper document at a time from paper-contiguous input chunks."""
    current_paper_id: str | None = None
    current_chunks: list[Chunk] = []
    completed: set[str] = set()

    for chunk in chunks:
        if chunk.paper_id != current_paper_id:
            if current_paper_id is not None:
                yield _paper_chunk(
                    current_paper_id,
                    current_chunks,
                    exclude_references=exclude_references,
                    extract_method_names=extract_method_names,
                )
                completed.add(current_paper_id)
            if chunk.paper_id in completed:
                raise ValueError(
                    "paper_bm25 requires chunks for each paper to be contiguous: "
                    f"{chunk.paper_id} appeared more than once"
                )
            current_paper_id = chunk.paper_id
            current_chunks = []
        current_chunks.append(chunk)

    if current_paper_id is not None:
        yield _paper_chunk(
            current_paper_id,
            current_chunks,
            exclude_references=exclude_references,
            extract_method_names=extract_method_names,
        )


def _require_positive_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _generic_alias_key(alias: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", alias.upper())


def _first_method_token(alias: str) -> str | None:
    match = _METHOD_FIRST_TOKEN_RE.search(alias)
    return match.group(0) if match is not None else None


def _standalone_alias_pattern(alias: str) -> re.Pattern[str]:
    body = r"\s+".join(
        re.escape(part) for part in alias.split()
    )
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])")


def _owner_literal_alias_pattern(alias: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in alias.split())
    return re.compile(
        rf"(?<![A-Za-z0-9_+./-]){body}(?![A-Za-z0-9_])"
    )


def _normalized_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _case_preserving_lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(re.findall(r"[A-Za-z0-9]+", normalized))


def _is_mixed_case_alias(alias: str) -> bool:
    letters = [character for character in alias if character.isalpha()]
    return (
        any(character.islower() for character in letters)
        and any(character.isupper() for character in letters)
    )


def _alias_alnum_length(alias: str) -> int:
    return sum(character.isalnum() for character in alias)


def _distinctive_context_words(value: object) -> frozenset[str]:
    if not isinstance(value, str):
        return frozenset()
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return frozenset(
        word
        for word in _METHOD_CONTEXT_WORD_RE.findall(normalized)
        if len(word) >= 4 and word not in _GENERIC_METHOD_CONTEXT_WORDS
    )


def _mention_has_method_context(
    text: str,
    start: int,
    end: int,
    context_pattern: re.Pattern[str] | None,
    required_word_count: int,
) -> bool:
    if context_pattern is None:
        return True
    observed: set[str] = set()
    for match in context_pattern.finditer(
        text,
        max(0, start - _METHOD_CONTEXT_CHARS),
        min(len(text), end + _METHOD_CONTEXT_CHARS),
    ):
        observed.add(match.group(0).casefold())
        if len(observed) >= required_word_count:
            return True
    return False


def _method_context_pattern(
    context_words: frozenset[str],
) -> re.Pattern[str] | None:
    if not context_words:
        return None
    alternatives = "|".join(
        re.escape(word)
        for word in sorted(context_words, key=lambda word: (-len(word), word))
    )
    return re.compile(
        rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


def _method_corpus_signature(documents: list[Chunk]) -> str:
    """Return a deterministic signature for validating the graph sidecar."""
    digest = hashlib.sha256()
    for document in documents:
        metadata = document.metadata
        record = {
            "paper_id": document.paper_id,
            "chunk_id": document.chunk_id,
            "method_names": metadata.get("method_names") or [],
            "method_alias_evidence": (
                metadata.get("method_alias_evidence") or []
            ),
        }
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@register("indexer", "paper_bm25")
class PaperBM25Index:
    """Search one aggregated BM25 document per paper as a coarse retrieval stage."""

    name = "paper_bm25"
    checkpoint_dependencies = (BM25Index, extract_self_owned_method_aliases)

    def __init__(
        self,
        index_dir: str,
        exclude_references: bool = True,
        result_text_chars: int = 2000,
        extract_method_names: bool = False,
        method_max_degree: int = 10,
    ):
        if result_text_chars <= 0:
            raise ValueError("result_text_chars must be positive")
        if not isinstance(extract_method_names, bool):
            raise TypeError("extract_method_names must be a boolean")
        _require_positive_limit(method_max_degree, "method_max_degree")
        self.exclude_references = exclude_references
        self.result_text_chars = result_text_chars
        self.extract_method_names = extract_method_names
        self.method_max_degree = method_max_degree
        self._delegate = BM25Index(
            index_dir=index_dir,
            records_filename=_PAPERS_FILENAME,
        )
        self._document_by_paper_id: dict[str, Chunk] | None = None
        self._method_owner_by_alias: dict[str, str] | None = None
        self._method_neighbors_by_paper_id: dict[
            str, tuple[dict[str, Any], ...]
        ] | None = None

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._reset_caches()
        self._delegate.build(
            aggregate_papers(
                chunks,
                exclude_references=self.exclude_references,
                extract_method_names=self.extract_method_names,
            )
        )
        if self.extract_method_names:
            self._build_method_index_from_documents()
            self._save_method_sidecar()

    def load(self) -> None:
        self._reset_caches()
        canonical_path = self._delegate.index_dir / _PAPERS_FILENAME
        legacy_path = self._delegate.index_dir / _LEGACY_RECORDS_FILENAME
        if not canonical_path.is_file() and legacy_path.is_file():
            self._delegate.records_filename = _LEGACY_RECORDS_FILENAME
        self._delegate.load()

    def get_document(self, paper_id: str) -> Chunk | None:
        """Return the indexed full-paper document for ``paper_id``, if present."""
        self._ensure_method_index()
        if self._document_by_paper_id is None:
            self._document_by_paper_id = {
                chunk.paper_id: chunk for chunk in self._delegate._chunks
            }
        return self._document_by_paper_id.get(paper_id)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self._ensure_method_index()
        results = self._delegate.search(query, top_k)
        return [
            dataclasses.replace(
                result,
                text=result.text[: self.result_text_chars],
                source=self.name,
            )
            for result in results
        ]

    def get_method_neighbors(
        self,
        paper_id: str,
        limit: int = 10,
    ) -> tuple[dict[str, Any], ...]:
        """Return deterministic explicit-alias neighbors of one paper."""
        _require_positive_limit(limit, "limit")
        self._ensure_method_index()
        if self._method_neighbors_by_paper_id is None:
            return ()
        return self._method_neighbors_by_paper_id.get(paper_id, ())[:limit]

    def find_method_owners(
        self,
        methods: Iterable[str] | str,
        limit: int = 10,
    ) -> tuple[dict[str, Any], ...]:
        """Find unique paper owners for literal or normalized method hints."""
        _require_positive_limit(limit, "limit")
        self._ensure_method_index()
        if not self._method_owner_by_alias:
            return ()

        values = (methods,) if isinstance(methods, str) else tuple(methods)
        literal_values = [
            unicodedata.normalize("NFKC", value)
            for value in values
            if isinstance(value, str) and value.strip()
        ]
        normalized_values = [
            _normalized_lookup_text(value)
            for value in literal_values
        ]
        case_preserving_values = [
            _case_preserving_lookup_text(value)
            for value in literal_values
        ]
        matches_by_paper: dict[str, dict[str, int]] = defaultdict(dict)
        alias_records = [
            (
                alias,
                paper_id,
                _owner_literal_alias_pattern(alias),
                _normalized_lookup_text(alias),
                _case_preserving_lookup_text(alias),
                _is_mixed_case_alias(alias),
            )
            for alias, paper_id in self._method_owner_by_alias.items()
        ]

        for literal_value, normalized_value, case_preserving_value in zip(
            literal_values,
            normalized_values,
            case_preserving_values,
        ):
            value_is_mixed_case = _is_mixed_case_alias(literal_value)
            value_matches: list[
                tuple[str, str, str, int, bool]
            ] = []
            for (
                alias,
                paper_id,
                literal_pattern,
                normalized_alias,
                case_preserving_alias,
                mixed_case,
            ) in alias_records:
                strength = 0
                if literal_pattern.search(literal_value):
                    strength = 2
                normalized_equal = bool(normalized_alias) and (
                    case_preserving_alias == case_preserving_value
                    if mixed_case or value_is_mixed_case
                    else normalized_alias == normalized_value
                )
                if normalized_equal and not strength:
                    strength = 1
                if strength:
                    value_matches.append(
                        (
                            alias,
                            paper_id,
                            normalized_alias,
                            strength,
                            normalized_equal,
                        )
                    )

            if any(match[-1] for match in value_matches):
                value_matches = [
                    match for match in value_matches if match[-1]
                ]

            # Prefer the longest explicit alias when normalized punctuation
            # makes a shorter name look like a suffix (D-FINE versus FiNE).
            for (
                alias,
                paper_id,
                normalized_alias,
                strength,
                _,
            ) in value_matches:
                if strength == 1 and any(
                    normalized_alias != other_normalized
                    and f" {normalized_alias} "
                    in f" {other_normalized} "
                    for _, _, other_normalized, _, _ in value_matches
                ):
                    continue
                matches_by_paper[paper_id][alias] = max(
                    strength,
                    matches_by_paper[paper_id].get(alias, 0),
                )

        results = [
            {
                "paper_id": paper_id,
                "aliases": sorted(alias_strengths),
                "strength": sum(alias_strengths.values()),
            }
            for paper_id, alias_strengths in matches_by_paper.items()
        ]
        results.sort(key=lambda item: (-item["strength"], item["paper_id"]))
        return tuple(results[:limit])

    def find_method_owners_in_papers(
        self,
        methods: Iterable[str] | str,
        paper_ids: Iterable[str],
        limit: int = 10,
    ) -> tuple[dict[str, Any], ...]:
        """Resolve explicit method owners within a bounded candidate set.

        This applies the current extractor directly to the requested papers.
        It therefore remains useful when a read-only prebuilt method sidecar
        was created before a newly supported ownership expression was added.
        An alias is accepted only when it identifies one candidate paper.
        """

        _require_positive_limit(limit, "limit")
        values = (methods,) if isinstance(methods, str) else tuple(methods)
        method_values = tuple(
            value
            for value in values
            if isinstance(value, str) and value.strip()
        )
        if not method_values:
            return ()

        self._ensure_method_index()
        if self._document_by_paper_id is None:
            self._document_by_paper_id = {
                chunk.paper_id: chunk for chunk in self._delegate._chunks
            }

        requested_ids = tuple(
            dict.fromkeys(
                paper_id
                for paper_id in paper_ids
                if isinstance(paper_id, str) and paper_id
            )
        )
        evidence_by_paper: dict[str, tuple] = {}
        for paper_id in requested_ids:
            document = self._document_by_paper_id.get(paper_id)
            if document is None:
                continue
            if not any(
                has_standalone_exact_alias(document.text, method)
                for method in method_values
            ):
                continue
            title = str(document.metadata.get("title") or "").strip()
            evidence_by_paper[paper_id] = extract_self_owned_method_aliases(
                title,
                document.text,
            )

        matches_by_paper: dict[str, set[str]] = defaultdict(set)
        for method in method_values:
            matches = [
                (paper_id, evidence.alias)
                for paper_id, evidence_items in evidence_by_paper.items()
                for evidence in evidence_items
                if method_aliases_equal(method, evidence.alias)
            ]
            matching_papers = {paper_id for paper_id, _ in matches}
            if len(matching_papers) != 1:
                continue
            for paper_id, alias in matches:
                matches_by_paper[paper_id].add(alias)

        results = [
            {
                "paper_id": paper_id,
                "aliases": sorted(aliases),
                "strength": 2 * len(aliases),
            }
            for paper_id, aliases in matches_by_paper.items()
        ]
        results.sort(key=lambda item: (-item["strength"], item["paper_id"]))
        return tuple(results[:limit])

    def _reset_caches(self) -> None:
        self._document_by_paper_id = None
        self._method_owner_by_alias = None
        self._method_neighbors_by_paper_id = None

    def _ensure_method_index(self) -> None:
        if (
            not self.extract_method_names
            or self._method_owner_by_alias is not None
        ):
            return
        self._enrich_document_method_metadata()
        if not self._load_method_sidecar():
            # Loading an old or damaged index must remain read-only.  The
            # rebuilt graph therefore stays in memory until a future build.
            self._build_method_index_from_documents(enrich=False)

    def _enrich_document_method_metadata(self) -> None:
        for document in self._delegate._chunks:
            metadata = document.metadata
            if (
                isinstance(metadata.get("method_names"), list)
                and isinstance(metadata.get("method_alias_evidence"), list)
            ):
                continue
            title = str(metadata.get("title") or "").strip()
            evidence = extract_self_owned_method_aliases(
                title,
                text_before_references(document.text),
            )
            metadata["method_names"] = [item.alias for item in evidence]
            metadata["method_alias_evidence"] = [
                item.to_dict() for item in evidence
            ]

    def _build_method_index_from_documents(
        self,
        *,
        enrich: bool = True,
    ) -> None:
        documents = self._delegate._chunks
        if enrich:
            self._enrich_document_method_metadata()

        owners_by_alias: dict[str, set[str]] = defaultdict(set)
        for document in documents:
            for value in document.metadata.get("method_names") or ():
                if not isinstance(value, str) or not value:
                    continue
                if _generic_alias_key(value) in GENERIC_METHOD_ALIASES:
                    continue
                owners_by_alias[value].add(document.paper_id)

        unique_owners = {
            alias: next(iter(paper_ids))
            for alias, paper_ids in owners_by_alias.items()
            if len(paper_ids) == 1
        }
        documents_by_paper_id = {
            document.paper_id: document for document in documents
        }
        context_words_by_alias: dict[str, frozenset[str]] = {}
        for alias, owner in unique_owners.items():
            evidence_records = documents_by_paper_id[
                owner
            ].metadata.get("method_alias_evidence")
            context_words: set[str] = set()
            if isinstance(evidence_records, list):
                for evidence in evidence_records:
                    if (
                        isinstance(evidence, dict)
                        and evidence.get("alias") == alias
                    ):
                        context_words.update(
                            _distinctive_context_words(
                                evidence.get("long_name")
                            )
                        )
            context_words.difference_update(
                _distinctive_context_words(alias)
            )
            if context_words:
                context_words_by_alias[alias] = frozenset(context_words)
        context_patterns_by_alias = {
            alias: _method_context_pattern(context_words)
            for alias, context_words in context_words_by_alias.items()
        }
        context_word_requirements = {
            alias: (
                min(2, len(context_words))
                if _alias_alnum_length(alias) <= 4
                else 1
            )
            for alias, context_words in context_words_by_alias.items()
        }

        aliases_by_first_token: dict[str, list[str]] = defaultdict(list)
        alias_patterns: dict[str, re.Pattern[str]] = {}
        for alias in sorted(unique_owners):
            first_token = _first_method_token(alias)
            if first_token is None:
                continue
            aliases_by_first_token[first_token].append(alias)
            alias_patterns[alias] = _standalone_alias_pattern(alias)
        owned_aliases_by_paper: dict[str, set[str]] = defaultdict(set)
        for alias, paper_id in unique_owners.items():
            owned_aliases_by_paper[paper_id].add(alias)

        mentioning_papers_by_alias: dict[str, set[str]] = defaultdict(set)
        for document in documents:
            body = text_before_references(document.text)
            mentioned_aliases = set(
                owned_aliases_by_paper.get(document.paper_id, ())
            )
            for token_match in _METHOD_TOKEN_RE.finditer(body):
                candidates = aliases_by_first_token.get(token_match.group(0))
                if not candidates:
                    continue
                for alias in candidates:
                    if alias in mentioned_aliases:
                        continue
                    alias_match = alias_patterns[alias].match(
                        body,
                        token_match.start(),
                    )
                    if (
                        alias_match is not None
                        and _mention_has_method_context(
                            body,
                            alias_match.start(),
                            alias_match.end(),
                            context_patterns_by_alias.get(alias),
                            context_word_requirements.get(alias, 0),
                        )
                    ):
                        mentioning_papers_by_alias[alias].add(
                            document.paper_id
                        )
                        mentioned_aliases.add(alias)

        accepted_owners = {
            alias: paper_id
            for alias, paper_id in unique_owners.items()
            if len(mentioning_papers_by_alias.get(alias, ()))
            <= self.method_max_degree
        }
        pair_aliases: dict[tuple[str, str], set[str]] = defaultdict(set)
        for alias, owner in accepted_owners.items():
            for mentioner in mentioning_papers_by_alias.get(alias, ()):
                pair = tuple(sorted((owner, mentioner)))
                pair_aliases[pair].add(alias)

        neighbors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (left, right), aliases in sorted(pair_aliases.items()):
            aliases_list = sorted(aliases)
            strength = len(aliases_list)
            neighbors[left].append(
                {
                    "paper_id": right,
                    "aliases": aliases_list,
                    "strength": strength,
                }
            )
            neighbors[right].append(
                {
                    "paper_id": left,
                    "aliases": aliases_list,
                    "strength": strength,
                }
            )
        self._method_owner_by_alias = dict(sorted(accepted_owners.items()))
        self._method_neighbors_by_paper_id = {
            paper_id: tuple(
                sorted(
                    items,
                    key=lambda item: (
                        -item["strength"],
                        item["paper_id"],
                        item["aliases"],
                    ),
                )
            )
            for paper_id, items in sorted(neighbors.items())
        }

    @property
    def _method_sidecar_path(self) -> Path:
        return self._delegate.index_dir / _METHOD_GRAPH_FILENAME

    def _save_method_sidecar(self) -> None:
        if (
            self._method_owner_by_alias is None
            or self._method_neighbors_by_paper_id is None
        ):
            return
        payload = {
            "schema_version": _METHOD_GRAPH_SCHEMA_VERSION,
            "method_max_degree": self.method_max_degree,
            "corpus_signature": _method_corpus_signature(
                self._delegate._chunks
            ),
            "owners": self._method_owner_by_alias,
            "neighbors": {
                paper_id: list(items)
                for paper_id, items
                in self._method_neighbors_by_paper_id.items()
            },
        }
        self._delegate.index_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._delegate.index_dir,
                prefix=f".{_METHOD_GRAPH_FILENAME}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._method_sidecar_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def _load_method_sidecar(self) -> bool:
        path = self._method_sidecar_path
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            owners, neighbors = self._validate_method_sidecar(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False
        self._method_owner_by_alias = owners
        self._method_neighbors_by_paper_id = neighbors
        return True

    def _validate_method_sidecar(
        self,
        payload: object,
    ) -> tuple[
        dict[str, str],
        dict[str, tuple[dict[str, Any], ...]],
    ]:
        if not isinstance(payload, dict):
            raise ValueError("method graph sidecar must be an object")
        if payload.get("schema_version") != _METHOD_GRAPH_SCHEMA_VERSION:
            raise ValueError("method graph sidecar schema is incompatible")
        if payload.get("method_max_degree") != self.method_max_degree:
            raise ValueError("method graph sidecar degree limit differs")
        if payload.get("corpus_signature") != _method_corpus_signature(
            self._delegate._chunks
        ):
            raise ValueError("method graph sidecar corpus differs")

        paper_ids = {chunk.paper_id for chunk in self._delegate._chunks}
        raw_owners = payload.get("owners")
        raw_neighbors = payload.get("neighbors")
        if not isinstance(raw_owners, dict) or not isinstance(
            raw_neighbors,
            dict,
        ):
            raise ValueError("method graph sidecar fields are invalid")

        owners: dict[str, str] = {}
        for alias, paper_id in raw_owners.items():
            if (
                not isinstance(alias, str)
                or not alias
                or not isinstance(paper_id, str)
                or paper_id not in paper_ids
            ):
                raise ValueError("method graph owner is invalid")
            owners[alias] = paper_id

        neighbors: dict[str, tuple[dict[str, Any], ...]] = {}
        directed_relations: set[
            tuple[str, str, tuple[str, ...], int]
        ] = set()
        for paper_id, items in raw_neighbors.items():
            if paper_id not in paper_ids or not isinstance(items, list):
                raise ValueError("method graph neighbor list is invalid")
            checked_items: list[dict[str, Any]] = []
            seen_neighbors: set[str] = set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("method graph neighbor is invalid")
                neighbor_id = item.get("paper_id")
                aliases = item.get("aliases")
                strength = item.get("strength")
                if (
                    not isinstance(neighbor_id, str)
                    or neighbor_id not in paper_ids
                    or neighbor_id == paper_id
                    or not isinstance(aliases, list)
                    or not aliases
                    or aliases != sorted(set(aliases))
                    or any(
                        not isinstance(alias, str)
                        or alias not in owners
                        or owners[alias] not in {paper_id, neighbor_id}
                        for alias in aliases
                    )
                    or isinstance(strength, bool)
                    or not isinstance(strength, int)
                    or strength != len(aliases)
                    or neighbor_id in seen_neighbors
                ):
                    raise ValueError("method graph neighbor is invalid")
                seen_neighbors.add(neighbor_id)
                checked_items.append(
                    {
                        "paper_id": neighbor_id,
                        "aliases": list(aliases),
                        "strength": strength,
                    }
                )
                directed_relations.add(
                    (
                        paper_id,
                        neighbor_id,
                        tuple(aliases),
                        strength,
                    )
                )
            if checked_items != sorted(
                checked_items,
                key=lambda item: (
                    -item["strength"],
                    item["paper_id"],
                    item["aliases"],
                ),
            ):
                raise ValueError("method graph neighbors are not deterministic")
            neighbors[paper_id] = tuple(checked_items)
        for paper_id, neighbor_id, aliases, strength in directed_relations:
            if (
                neighbor_id,
                paper_id,
                aliases,
                strength,
            ) not in directed_relations:
                raise ValueError("method graph relation is not undirected")
        return dict(sorted(owners.items())), neighbors
