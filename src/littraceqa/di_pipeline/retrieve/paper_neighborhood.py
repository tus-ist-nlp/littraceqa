"""Gold-free reranking from explicit title mentions between candidate papers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from numbers import Real
from typing import Pattern

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult


_TITLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_GENERIC_TITLE_ALIASES = frozenset(
    {
        "ACL",
        "AI",
        "API",
        "AUC",
        "BERT",
        "BLEU",
        "CNN",
        "COCO",
        "CPU",
        "CV",
        "CVPR",
        "DNN",
        "DPO",
        "ECCV",
        "EMNLP",
        "FID",
        "GAN",
        "GPT",
        "GPU",
        "HTML",
        "ICCV",
        "ICLR",
        "ICML",
        "IOU",
        "JSON",
        "LLM",
        "LORA",
        "LSTM",
        "MAE",
        "ML",
        "MLP",
        "MSE",
        "NAACL",
        "NEURIPS",
        "NIPS",
        "NLP",
        "OCR",
        "PDF",
        "QA",
        "RAG",
        "RAM",
        "RL",
        "SOTA",
        "VAE",
        "VIT",
        "VLM",
        "VQA",
    }
)
_FULL_TITLE_MIN_ALNUM_CHARS = 20
_ALIAS_MIN_ALNUM_CHARS = 3

GetDocument = Callable[[str], Chunk | None]


def _normalized_title(value: object) -> str | None:
    """Return a compact NFKC title or ``None`` for an unavailable title."""

    if not isinstance(value, str):
        return None
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    return normalized or None


def _title_from(
    result: RetrievalResult,
    document: Chunk,
) -> str | None:
    """Read the canonical title without guessing one from document text."""

    for metadata in (result.metadata, document.metadata):
        if not isinstance(metadata, dict):
            continue
        title = _normalized_title(metadata.get("title"))
        if title is not None:
            return title
    return None


@lru_cache(maxsize=4096)
def _title_pattern(
    title: str,
    *,
    min_alnum_chars: int,
) -> Pattern[str] | None:
    """Compile a punctuation-tolerant, boundary-aware ASCII title pattern."""

    tokens = _TITLE_TOKEN_RE.findall(unicodedata.normalize("NFKC", title))
    if sum(len(token) for token in tokens) < min_alnum_chars:
        return None
    body = r"[^A-Za-z0-9]+".join(re.escape(token) for token in tokens)
    return re.compile(
        rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


@lru_cache(maxsize=4096)
def _conservative_title_alias(title: str) -> str | None:
    """Return an identifier-like prefix before a colon when unambiguous."""

    normalized = unicodedata.normalize("NFKC", title)
    prefix, separator, _ = normalized.partition(":")
    if not separator:
        return None

    alias = " ".join(prefix.split())
    words = _TITLE_TOKEN_RE.findall(alias)
    alnum_chars = sum(len(word) for word in words)
    if not (3 <= len(alias) <= 40) or not (1 <= len(words) <= 4):
        return None
    if alnum_chars < _ALIAS_MIN_ALNUM_CHARS:
        return None

    generic_key = re.sub(r"[^A-Z0-9]+", "", alias.upper())
    if generic_key in _GENERIC_TITLE_ALIASES:
        return None

    letters = [character for character in alias if character.isalpha()]
    is_camel_case = bool(re.search(r"[a-z0-9][A-Z]", alias))
    is_all_caps = bool(letters) and all(character.isupper() for character in letters)
    is_identifier_like = (
        is_camel_case
        or is_all_caps
        or any(character.isdigit() for character in alias)
        or "-" in alias
    )
    return alias if is_identifier_like else None


def _mentions(pattern: Pattern[str] | None, document_text: str) -> bool:
    """Return whether a compiled title pattern occurs in normalized text."""

    if pattern is None:
        return False
    return pattern.search(document_text) is not None


def _mention_strength(
    full_pattern: Pattern[str] | None,
    alias_pattern: Pattern[str] | None,
    document_text: str,
) -> int:
    """Score one direction while avoiding a redundant long-title scan."""

    alias_mentioned = _mentions(alias_pattern, document_text)
    # A full colon title begins with its alias, so a missing alias proves that
    # the full title is also absent. Titles without an alias still need a scan.
    full_mentioned = (
        _mentions(full_pattern, document_text)
        if alias_pattern is None or alias_mentioned
        else False
    )
    return (2 if full_mentioned else 0) + (1 if alias_mentioned else 0)


class PaperNeighborhoodReranker:
    """Fuse baseline ranks with strict bidirectional paper-title mentions.

    The highest-ranked candidate is the seed.  Other candidates receive a
    relation-lane rank only when an exact full title or conservative colon
    alias is present in either direction.  This component uses no gold labels,
    validation-only query fields, model inference, or external service.

    An optional two-hop lane can recover a non-direct paper through a strict,
    low-degree direct neighbor.  It is disabled by default so existing direct
    reranking output remains byte-for-byte compatible.

    The document provider is expected to reuse already loaded paper-level
    documents.  If any candidate document or title is unavailable, the method
    returns the original ranking unchanged instead of applying partial scores.
    """

    def __init__(
        self,
        get_document: GetDocument,
        *,
        rrf_k: float = 60,
        relation_weight: float = 0.2,
        two_hop_weight: float = 0.0,
        max_hub_degree: int = 4,
    ) -> None:
        if not callable(get_document):
            raise TypeError("get_document must be callable")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, Real):
            raise TypeError("rrf_k must be a number")
        if not math.isfinite(rrf_k) or rrf_k < 0:
            raise ValueError("rrf_k must be a finite non-negative number")
        if isinstance(relation_weight, bool) or not isinstance(
            relation_weight,
            Real,
        ):
            raise TypeError("relation_weight must be a number")
        if not math.isfinite(relation_weight) or relation_weight < 0:
            raise ValueError(
                "relation_weight must be a finite non-negative number"
            )
        if isinstance(two_hop_weight, bool) or not isinstance(
            two_hop_weight,
            Real,
        ):
            raise TypeError("two_hop_weight must be a number")
        if not math.isfinite(two_hop_weight) or two_hop_weight < 0:
            raise ValueError(
                "two_hop_weight must be a finite non-negative number"
            )
        if isinstance(max_hub_degree, bool) or not isinstance(
            max_hub_degree,
            int,
        ):
            raise TypeError("max_hub_degree must be an integer")
        if max_hub_degree <= 0:
            raise ValueError("max_hub_degree must be a positive integer")

        self._get_document = get_document
        self.rrf_k = float(rrf_k)
        self.relation_weight = float(relation_weight)
        self.two_hop_weight = float(two_hop_weight)
        self.max_hub_degree = max_hub_degree

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank one bounded candidate pool while preserving result payloads."""

        del query
        if top_k <= 0 or not candidates:
            return []

        unique_candidates: list[RetrievalResult] = []
        seen_paper_ids: set[str] = set()
        for candidate in candidates:
            if candidate.paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(candidate.paper_id)
            unique_candidates.append(candidate)
        candidates = unique_candidates
        fallback = list(candidates[:top_k])
        documents: dict[str, Chunk] = {}
        titles: list[str] = []
        normalized_texts: list[str] = []
        try:
            for candidate in candidates:
                document = documents.get(candidate.paper_id)
                if document is None:
                    document = self._get_document(candidate.paper_id)
                    if not isinstance(document, Chunk) or not document.text.strip():
                        return fallback
                    documents[candidate.paper_id] = document
                title = _title_from(candidate, document)
                if title is None:
                    return fallback
                titles.append(title)
                normalized_texts.append(
                    unicodedata.normalize("NFKC", document.text)
                )
        except Exception:
            return fallback

        title_patterns: list[
            tuple[Pattern[str] | None, Pattern[str] | None]
        ] = []
        for title in titles:
            full_pattern = _title_pattern(
                title,
                min_alnum_chars=_FULL_TITLE_MIN_ALNUM_CHARS,
            )
            alias = _conservative_title_alias(title)
            alias_pattern = (
                _title_pattern(
                    alias,
                    min_alnum_chars=_ALIAS_MIN_ALNUM_CHARS,
                )
                if alias is not None
                else None
            )
            title_patterns.append((full_pattern, alias_pattern))

        edge_cache: dict[tuple[int, int], int] = {}

        def edge_strength(left: int, right: int) -> int:
            """Return one symmetric edge score, computing each pair once."""

            key = (left, right) if left < right else (right, left)
            cached = edge_cache.get(key)
            if cached is not None:
                return cached

            left_full, left_alias = title_patterns[left]
            right_full, right_alias = title_patterns[right]
            strength = _mention_strength(
                right_full,
                right_alias,
                normalized_texts[left],
            )
            strength += _mention_strength(
                left_full,
                left_alias,
                normalized_texts[right],
            )
            edge_cache[key] = strength
            return strength

        strengths: dict[int, int] = {}
        for index in range(1, len(candidates)):
            strengths[index] = edge_strength(0, index)

        related = sorted(
            (index for index, strength in strengths.items() if strength > 0),
            key=lambda index: (
                -strengths[index],
                index + 1,
                candidates[index].paper_id,
            ),
        )
        relation_ranks = {
            candidate_index: rank
            for rank, candidate_index in enumerate(related, start=1)
        }

        two_hop_stats: dict[int, tuple[int, int]] = {}
        two_hop_ranks: dict[int, int] = {}
        if self.two_hop_weight > 0:
            direct = {
                index for index, strength in strengths.items() if strength > 0
            }
            for hub in range(1, len(candidates)):
                seed_to_hub = strengths[hub]
                if seed_to_hub < 2:
                    continue

                neighbors: list[tuple[int, int]] = []
                for neighbor in range(len(candidates)):
                    if neighbor == hub:
                        continue
                    strength = edge_strength(hub, neighbor)
                    if strength <= 0:
                        continue
                    neighbors.append((neighbor, strength))
                    # High-degree papers are likely surveys or bibliography
                    # hubs. Stop scanning as soon as they exceed the cap.
                    if len(neighbors) > self.max_hub_degree:
                        break
                if len(neighbors) > self.max_hub_degree:
                    continue

                for target, hub_to_target in neighbors:
                    if target == 0 or target in direct or hub_to_target < 2:
                        continue
                    bottleneck = min(seed_to_hub, hub_to_target)
                    best, path_count = two_hop_stats.get(target, (0, 0))
                    two_hop_stats[target] = (
                        max(best, bottleneck),
                        path_count + 1,
                    )

            two_hop_order = sorted(
                two_hop_stats,
                key=lambda index: (
                    -two_hop_stats[index][0],
                    -two_hop_stats[index][1],
                    index + 1,
                    candidates[index].paper_id,
                ),
            )
            two_hop_ranks = {
                candidate_index: rank
                for rank, candidate_index in enumerate(
                    two_hop_order,
                    start=1,
                )
            }

        scored: list[tuple[float, int, RetrievalResult]] = []
        for index, candidate in enumerate(candidates):
            baseline_rank = index + 1
            relation_rank = relation_ranks.get(index)
            two_hop_rank = two_hop_ranks.get(index)
            score = 1.0 / (self.rrf_k + baseline_rank)
            if relation_rank is not None:
                score += self.relation_weight / (
                    self.rrf_k + relation_rank
                )
            if two_hop_rank is not None:
                score += self.two_hop_weight / (
                    self.rrf_k + two_hop_rank
                )

            metadata = (
                dict(candidate.metadata)
                if isinstance(candidate.metadata, dict)
                else {}
            )
            metadata.update(
                {
                    "paper_neighborhood_baseline_rank": baseline_rank,
                    "paper_neighborhood_relation_rank": relation_rank,
                    "paper_neighborhood_relation_strength": strengths.get(
                        index,
                        0,
                    ),
                }
            )
            if self.two_hop_weight > 0:
                two_hop_strength, path_count = two_hop_stats.get(
                    index,
                    (0, 0),
                )
                metadata.update(
                    {
                        "paper_neighborhood_two_hop_rank": two_hop_rank,
                        "paper_neighborhood_two_hop_strength": (
                            two_hop_strength
                        ),
                        "paper_neighborhood_two_hop_path_count": path_count,
                    }
                )
            scored.append(
                (
                    score,
                    baseline_rank,
                    replace(
                        candidate,
                        score=score,
                        metadata=metadata,
                        source="paper_neighborhood_rrf",
                    ),
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2].paper_id,
            )
        )
        return [result for _, _, result in scored[:top_k]]
