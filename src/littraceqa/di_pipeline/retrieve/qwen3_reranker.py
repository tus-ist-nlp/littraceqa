"""Qwen3 causal-language-model reranking for fused retrieval candidates."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.registry import register

_DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)
_SUPPORTED_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})


def _format_pair(instruction: str, query: str, document: str) -> str:
    """Format one query-document pair using the model's official template."""
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def _normalize_dtype(dtype: str | None) -> str | None:
    """Validate and normalize a user-facing model dtype."""

    if dtype is None:
        return None
    if not isinstance(dtype, str):
        raise TypeError("dtype must be a string or None")
    normalized = dtype.strip().lower()
    if normalized not in _SUPPORTED_DTYPES:
        raise ValueError(
            "dtype must be one of: auto, float32, float16, bfloat16"
        )
    return normalized


def _resolve_torch_dtype(
    torch_module: Any,
    dtype: str | None,
    *,
    device: str,
    model_config: Any,
) -> Any:
    """Resolve a validated dtype without relying on Transformers defaults."""

    dtype = _normalize_dtype(dtype)
    if dtype is None:
        return None

    if dtype == "auto":
        resolved = getattr(model_config, "dtype", None)
        if resolved is None:
            resolved = getattr(model_config, "torch_dtype", None)
        if isinstance(resolved, str):
            resolved = getattr(torch_module, resolved, "auto")
        valid_config_dtypes = (
            torch_module.float32,
            torch_module.float16,
            torch_module.bfloat16,
        )
        if resolved not in valid_config_dtypes:
            resolved = "auto"
    else:
        resolved = getattr(torch_module, dtype)

    half_dtypes = (torch_module.float16, torch_module.bfloat16)
    if device.startswith("cpu") and resolved in half_dtypes:
        raise ValueError(
            f"dtype={dtype!r} resolves to {resolved} on CPU; "
            "use float32 or leave dtype unset for legacy behavior"
        )

    if (
        device.startswith("cuda")
        and resolved is torch_module.bfloat16
        and callable(getattr(torch_module.cuda, "get_device_capability", None))
    ):
        major, _ = torch_module.cuda.get_device_capability(device)
        if major < 8:
            raise ValueError(
                "bfloat16 requires a CUDA device with compute capability 8.0 "
                "or newer"
            )
    return resolved


class _Qwen3CausalLMScorer:
    """Expose Qwen3's official inputs and monotonic yes/no logit margin."""

    def __init__(
        self,
        *,
        torch_module: Any,
        tokenizer: Any,
        model: Any,
        device: str,
        max_tokens: int,
        instruction: str | None,
    ) -> None:
        self._torch = torch_module
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._instruction = instruction or _DEFAULT_INSTRUCTION

        self._prefix_tokens = tokenizer.encode(_PREFIX, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(_SUFFIX, add_special_tokens=False)
        self._content_tokens = (
            max_tokens - len(self._prefix_tokens) - len(self._suffix_tokens)
        )
        if self._content_tokens <= 0:
            raise ValueError(
                "max_tokens is too small for the Qwen3 reranker prompt template"
            )

        self._false_token_id = tokenizer.convert_tokens_to_ids("no")
        self._true_token_id = tokenizer.convert_tokens_to_ids("yes")
        if (
            self._false_token_id is None
            or self._true_token_id is None
            or self._false_token_id == self._true_token_id
        ):
            raise ValueError("Qwen3 tokenizer does not define distinct yes/no tokens")

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> list[float]:
        """Score pairs in bounded batches without loading all token tensors."""
        del show_progress_bar, convert_to_numpy
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts = [
                _format_pair(self._instruction, query, document)
                for query, document in batch
            ]
            encoded = self._tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self._content_tokens,
            )
            encoded["input_ids"] = [
                self._prefix_tokens + token_ids + self._suffix_tokens
                for token_ids in encoded["input_ids"]
            ]
            inputs = self._tokenizer.pad(
                encoded,
                padding=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self._device) for key, value in inputs.items()
            }

            with self._torch.inference_mode():
                logits = self._model(
                    **inputs,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                true_logits = logits[:, self._true_token_id]
                false_logits = logits[:, self._false_token_id]
                margins = true_logits.float() - false_logits.float()
            scores.extend(margins.cpu().tolist())
        return scores


def _load_qwen3_scorer(
    model_name: str,
    *,
    device: str,
    max_tokens: int,
    local_files_only: bool,
    instruction: str | None,
    revision: str | None,
    dtype: str | None = None,
) -> Any:
    """Load Qwen3 with the model author's causal-LM scoring procedure."""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3 reranking requires torch and transformers>=4.51. "
            "Install the project's di_pipeline dependencies before enabling it."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Qwen3 reranker requested unavailable device: {device}")

    load_kwargs = {
        "local_files_only": local_files_only,
        "revision": revision,
    }
    source = "the local cache" if local_files_only else "the model source"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            truncation_side="right",
            **load_kwargs,
        )
        model_config = AutoConfig.from_pretrained(
            model_name,
            **load_kwargs,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to load Qwen3 reranker metadata {model_name!r} from {source}"
        ) from exc

    prompt_tokens = len(tokenizer.encode(_PREFIX, add_special_tokens=False))
    prompt_tokens += len(tokenizer.encode(_SUFFIX, add_special_tokens=False))
    if max_tokens <= prompt_tokens:
        raise ValueError(
            "max_tokens is too small for the Qwen3 reranker prompt template"
        )
    context_limit = getattr(model_config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and max_tokens > context_limit:
        raise ValueError(
            f"max_tokens ({max_tokens}) exceeds model context limit ({context_limit})"
        )

    resolved_dtype = _resolve_torch_dtype(
        torch,
        dtype,
        device=device,
        model_config=model_config,
    )
    model_load_kwargs = dict(load_kwargs)
    if resolved_dtype is not None:
        # ``dtype`` is supported by both Transformers 4.57 and 5.x.
        model_load_kwargs["dtype"] = resolved_dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            **model_load_kwargs,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to load Qwen3 reranker {model_name!r} from {source}"
        ) from exc
    model = model.to(device).eval()
    return _Qwen3CausalLMScorer(
        torch_module=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_tokens=max_tokens,
        instruction=instruction,
    )


@register("reranker", "qwen3")
class Qwen3Reranker:
    """Rerank ``(query, chunk text)`` pairs with a local Qwen3 causal LM.

    Model loading is lazy and local-only by default, so selecting another
    reranker never initializes this model and a cache miss cannot download it.
    Set ``local_files_only=False`` explicitly only in a controlled setup step.
    Explicit dtypes are validated before model weights are loaded.
    ``base_rank_weight`` can conservatively blend the original and Qwen ranks;
    its zero default preserves pure Qwen ordering.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: str = "cpu",
        dtype: str | None = None,
        batch_size: int = 1,
        max_tokens: int = 1024,
        local_files_only: bool = True,
        instruction: str | None = None,
        revision: str | None = None,
        base_rank_weight: float = 0.0,
        rank_fusion_k: float = 0.0,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= base_rank_weight <= 1.0:
            raise ValueError("base_rank_weight must be between 0 and 1")
        if rank_fusion_k < 0:
            raise ValueError("rank_fusion_k must be non-negative")

        self.model_name = model
        self.device = device
        self.dtype = _normalize_dtype(dtype)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.local_files_only = local_files_only
        self.instruction = instruction.strip() if instruction else None
        self.revision = revision
        self.base_rank_weight = base_rank_weight
        self.rank_fusion_k = rank_fusion_k
        self._model_loader = model_loader or _load_qwen3_scorer
        self._model: Any | None = None
        self._score_cache_query: str | None = None
        self._score_cache: dict[str, float] = {}

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(
                self.model_name,
                device=self.device,
                max_tokens=self.max_tokens,
                local_files_only=self.local_files_only,
                instruction=self.instruction,
                revision=self.revision,
                dtype=self.dtype,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if top_k <= 0 or not candidates:
            return []

        scores = self.score_candidates(query, candidates)
        return self.rerank_scored(candidates, scores, top_k)

    def score_candidates(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[float]:
        """Return raw Qwen scores, reusing exact texts for the latest query."""

        if not candidates:
            return []

        if query != self._score_cache_query:
            self._score_cache_query = query
            self._score_cache.clear()

        missing_texts: list[str] = []
        missing_indices: list[int] = []
        pending: set[str] = set()
        for index, candidate in enumerate(candidates):
            text = candidate.text
            if text in self._score_cache or text in pending:
                continue
            pending.add(text)
            missing_texts.append(text)
            missing_indices.append(index)

        if missing_texts:
            pairs = [(query, text) for text in missing_texts]
            raw_scores = self._get_model().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            new_scores = list(raw_scores)
            if len(new_scores) != len(missing_texts):
                raise ValueError(
                    "reranker returned "
                    f"{len(new_scores)} scores for {len(missing_texts)} candidates"
                )

            validated: dict[str, float] = {}
            for text, raw_score, candidate_index in zip(
                missing_texts,
                new_scores,
                missing_indices,
                strict=True,
            ):
                score = float(raw_score)
                if not math.isfinite(score):
                    raise ValueError(
                        "reranker returned a non-finite score at index "
                        f"{candidate_index}"
                    )
                validated[text] = score
            self._score_cache.update(validated)

        return [self._score_cache[candidate.text] for candidate in candidates]

    def rerank_scored(
        self,
        candidates: list[RetrievalResult],
        scores: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rank candidates from raw Qwen scores without running the model."""

        if top_k <= 0 or not candidates:
            return []

        if len(scores) != len(candidates):
            raise ValueError(
                "reranker returned "
                f"{len(scores)} scores for {len(candidates)} candidates"
            )

        ranked: list[tuple[float, int, RetrievalResult]] = []
        for original_index, (candidate, raw_score) in enumerate(
            zip(candidates, scores, strict=True)
        ):
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(
                    f"reranker returned a non-finite score at index {original_index}"
                )
            metadata = dict(candidate.metadata)
            metadata["pre_rerank_score"] = candidate.score
            metadata["pre_rerank_rank"] = original_index + 1
            ranked.append(
                (
                    score,
                    original_index,
                    dataclasses.replace(candidate, score=score, metadata=metadata),
                )
            )

        # Original rank is the secondary key, making equal-score ordering stable.
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if self.base_rank_weight == 0:
            return [result for _, _, result in ranked[:top_k]]

        blended: list[tuple[float, int, RetrievalResult]] = []
        qwen_weight = 1.0 - self.base_rank_weight
        for qwen_rank, (_, original_index, result) in enumerate(ranked, start=1):
            original_rank = original_index + 1
            fusion_score = self.base_rank_weight / (
                self.rank_fusion_k + original_rank
            )
            fusion_score += qwen_weight / (
                self.rank_fusion_k + qwen_rank
            )
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "qwen3_score": result.score,
                    "qwen3_rank": qwen_rank,
                    "rank_fusion_base_weight": self.base_rank_weight,
                    "rank_fusion_k": self.rank_fusion_k,
                }
            )
            blended.append(
                (
                    fusion_score,
                    original_index,
                    dataclasses.replace(
                        result,
                        score=fusion_score,
                        metadata=metadata,
                    ),
                )
            )
        blended.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in blended[:top_k]]
