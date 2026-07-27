"""Unit tests for local-only Qwen3 candidate reranking."""

from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.qwen3_reranker import (
    Qwen3Reranker,
    _Qwen3CausalLMScorer,
    _format_pair,
    _load_qwen3_scorer,
    _normalize_dtype,
    _resolve_torch_dtype,
)


def _candidate(index: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"paper-{index}#c0000",
        paper_id=f"paper-{index}",
        score=score,
        text=f"candidate text {index}",
        chunk_type="text_span",
        metadata={"marker": index},
        source="rrf",
    )


def test_rerank_uses_pair_api_and_preserves_original_ranking_metadata():
    load_calls: list[tuple[str, dict]] = []
    predict_calls: list[tuple[list[tuple[str, str]], dict]] = []

    class FakeScorer:
        def predict(self, pairs, **kwargs):
            predict_calls.append((pairs, kwargs))
            return [0.1, 0.9, 0.9]

    def fake_loader(model_name: str, **kwargs):
        load_calls.append((model_name, kwargs))
        return FakeScorer()

    candidates = [_candidate(1, 0.8), _candidate(2, 0.7), _candidate(3, 0.6)]
    reranker = Qwen3Reranker(
        device="cuda:1",
        dtype="bfloat16",
        batch_size=4,
        max_tokens=2048,
        instruction="Find scientific evidence.",
        model_loader=fake_loader,
    )

    results = reranker.rerank("question", candidates, top_k=3)

    assert load_calls == [
        (
            "Qwen/Qwen3-Reranker-0.6B",
            {
                "device": "cuda:1",
                "dtype": "bfloat16",
                "max_tokens": 2048,
                "local_files_only": True,
                "instruction": "Find scientific evidence.",
                "revision": None,
            },
        )
    ]
    assert predict_calls == [
        (
            [
                ("question", "candidate text 1"),
                ("question", "candidate text 2"),
                ("question", "candidate text 3"),
            ],
            {
                "batch_size": 4,
                "show_progress_bar": False,
                "convert_to_numpy": True,
            },
        )
    ]
    # Candidates 2 and 3 tie, so their original relative order is retained.
    assert [result.paper_id for result in results] == ["paper-2", "paper-3", "paper-1"]
    assert [result.score for result in results] == [0.9, 0.9, 0.1]
    assert results[0].metadata == {
        "marker": 2,
        "pre_rerank_score": 0.7,
        "pre_rerank_rank": 2,
    }
    assert candidates[1].metadata == {"marker": 2}
    assert candidates[1].score == 0.7


def test_empty_candidates_do_not_load_model():
    def fail_loader(*args, **kwargs):
        raise AssertionError("model must not be loaded")

    reranker = Qwen3Reranker(model_loader=fail_loader)

    assert reranker.rerank("question", [], top_k=10) == []


def test_non_finite_score_is_rejected():
    class FakeScorer:
        def predict(self, pairs, **kwargs):
            return [float("nan")]

    reranker = Qwen3Reranker(model_loader=lambda *args, **kwargs: FakeScorer())

    with pytest.raises(ValueError, match="non-finite"):
        reranker.rerank("question", [_candidate(1, 0.8)], top_k=1)


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        Qwen3Reranker(batch_size=0)
    with pytest.raises(ValueError, match="max_tokens"):
        Qwen3Reranker(max_tokens=0)
    with pytest.raises(ValueError, match="base_rank_weight"):
        Qwen3Reranker(base_rank_weight=-0.1)
    with pytest.raises(ValueError, match="base_rank_weight"):
        Qwen3Reranker(base_rank_weight=1.1)
    with pytest.raises(ValueError, match="rank_fusion_k"):
        Qwen3Reranker(rank_fusion_k=-1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("auto", "auto"),
        (" FLOAT32 ", "float32"),
        ("float16", "float16"),
        ("BFLOAT16", "bfloat16"),
    ],
)
def test_supported_dtype_values_are_normalized(value, expected):
    assert _normalize_dtype(value) == expected


@pytest.mark.parametrize("value", ["", "float64", "half", "automatic"])
def test_unsupported_dtype_values_are_rejected(value):
    with pytest.raises(ValueError, match="dtype must be one of"):
        Qwen3Reranker(dtype=value)


@pytest.mark.parametrize("value", [True, 16, object()])
def test_non_string_dtype_values_are_rejected(value):
    with pytest.raises(TypeError, match="dtype must be a string or None"):
        Qwen3Reranker(dtype=value)


def test_optional_rank_fusion_blends_original_and_qwen_ranks():
    class FakeScorer:
        def predict(self, pairs, **kwargs):
            return [0.1, 0.9, 0.5]

    candidates = [_candidate(1, 0.8), _candidate(2, 0.7), _candidate(3, 0.6)]
    reranker = Qwen3Reranker(
        base_rank_weight=0.25,
        rank_fusion_k=0,
        model_loader=lambda *args, **kwargs: FakeScorer(),
    )

    results = reranker.rerank("question", candidates, top_k=3)

    assert [result.paper_id for result in results] == [
        "paper-2",
        "paper-1",
        "paper-3",
    ]
    assert [result.score for result in results] == pytest.approx(
        [0.875, 0.5, 11 / 24]
    )
    assert results[0].metadata == {
        "marker": 2,
        "pre_rerank_score": 0.7,
        "pre_rerank_rank": 2,
        "qwen3_score": 0.9,
        "qwen3_rank": 1,
        "rank_fusion_base_weight": 0.25,
        "rank_fusion_k": 0,
    }


def test_official_pair_format_is_used():
    assert _format_pair(
        "Find scientific evidence.",
        "Which paper introduced the method?",
        "This paper introduces Method X.",
    ) == (
        "<Instruct>: Find scientific evidence.\n"
        "<Query>: Which paper introduced the method?\n"
        "<Document>: This paper introduces Method X."
    )


def test_model_loader_is_local_only_and_moves_model_to_requested_device(monkeypatch):
    calls: list[tuple[str, str, dict]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("tokenizer", model_name, kwargs))
            return cls()

        def encode(self, text, **kwargs):
            return [1]

        def convert_tokens_to_ids(self, token):
            return {"no": 2, "yes": 3}[token]

    class FakeConfig:
        max_position_embeddings = 4096

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("config", model_name, kwargs))
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("model", model_name, kwargs))
            return cls()

        def to(self, device):
            calls.append(("device", device, {}))
            return self

        def eval(self):
            calls.append(("eval", "", {}))
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = FakeConfig
    transformers.AutoModelForCausalLM = FakeModel
    transformers.AutoTokenizer = FakeTokenizer
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    scorer = _load_qwen3_scorer(
        "model",
        device="cpu",
        max_tokens=512,
        local_files_only=True,
        instruction="Find scientific evidence.",
        revision="revision-1",
    )

    assert calls[0] == (
        "tokenizer",
        "model",
        {
            "padding_side": "left",
            "truncation_side": "right",
            "local_files_only": True,
            "revision": "revision-1",
        },
    )
    assert calls[1] == (
        "config",
        "model",
        {"local_files_only": True, "revision": "revision-1"},
    )
    assert calls[2][:2] == ("model", "model")
    assert calls[2][2]["local_files_only"] is True
    assert calls[2][2]["revision"] == "revision-1"
    assert "dtype" not in calls[2][2]
    assert isinstance(calls[2][2]["config"], FakeConfig)
    assert calls[3:] == [("device", "cpu", {}), ("eval", "", {})]
    assert scorer._instruction == "Find scientific evidence."


def test_loader_passes_resolved_float32_to_transformers(monkeypatch):
    float32 = object()
    float16 = object()
    bfloat16 = object()
    model_kwargs: list[dict] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

        def encode(self, text, **kwargs):
            return [1]

        def convert_tokens_to_ids(self, token):
            return {"no": 2, "yes": 3}[token]

    class FakeConfig:
        max_position_embeddings = 4096
        dtype = bfloat16

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            model_kwargs.append(kwargs)
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = FakeConfig
    transformers.AutoModelForCausalLM = FakeModel
    transformers.AutoTokenizer = FakeTokenizer
    torch = types.ModuleType("torch")
    torch.float32 = float32
    torch.float16 = float16
    torch.bfloat16 = bfloat16
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    _load_qwen3_scorer(
        "model",
        device="cpu",
        dtype="float32",
        max_tokens=512,
        local_files_only=True,
        instruction=None,
        revision=None,
    )

    assert model_kwargs[0]["dtype"] is float32


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "auto"])
def test_loader_rejects_half_precision_on_cpu_before_loading_weights(
    monkeypatch,
    dtype,
):
    float32 = object()
    float16 = object()
    bfloat16 = object()

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

        def encode(self, text, **kwargs):
            return [1]

    class FakeConfig:
        max_position_embeddings = 4096

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            config = cls()
            config.dtype = bfloat16
            return config

    class FailIfLoaded:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            raise AssertionError("model weights must not be loaded")

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = FakeConfig
    transformers.AutoModelForCausalLM = FailIfLoaded
    transformers.AutoTokenizer = FakeTokenizer
    torch = types.ModuleType("torch")
    torch.float32 = float32
    torch.float16 = float16
    torch.bfloat16 = bfloat16
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(ValueError, match="on CPU"):
        _load_qwen3_scorer(
            "model",
            device="cpu",
            dtype=dtype,
            max_tokens=512,
            local_files_only=True,
            instruction=None,
            revision=None,
        )


def test_auto_dtype_uses_configured_bfloat16_on_supported_cuda():
    float32 = object()
    float16 = object()
    bfloat16 = object()
    torch = types.SimpleNamespace(
        float32=float32,
        float16=float16,
        bfloat16=bfloat16,
        cuda=types.SimpleNamespace(
            get_device_capability=lambda device: (8, 6),
        ),
    )
    config = types.SimpleNamespace(dtype=bfloat16)

    resolved = _resolve_torch_dtype(
        torch,
        "auto",
        device="cuda:1",
        model_config=config,
    )

    assert resolved is bfloat16


def test_bfloat16_rejects_unsupported_cuda_capability():
    torch = types.SimpleNamespace(
        float32=object(),
        float16=object(),
        bfloat16=object(),
        cuda=types.SimpleNamespace(
            get_device_capability=lambda device: (7, 5),
        ),
    )

    with pytest.raises(ValueError, match="compute capability 8.0"):
        _resolve_torch_dtype(
            torch,
            "bfloat16",
            device="cuda:0",
            model_config=types.SimpleNamespace(dtype=None),
        )


def test_loader_rejects_tiny_token_budget_before_loading_weights(monkeypatch):
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

        def encode(self, text, **kwargs):
            return [1, 2, 3, 4, 5]

    class FakeConfig:
        max_position_embeddings = 4096

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

    class FailIfLoaded:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            raise AssertionError("model weights must not be loaded")

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = FakeConfig
    transformers.AutoModelForCausalLM = FailIfLoaded
    transformers.AutoTokenizer = FakeTokenizer
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(ValueError, match="too small"):
        _load_qwen3_scorer(
            "model",
            device="cpu",
            max_tokens=10,
            local_files_only=True,
            instruction=None,
            revision=None,
        )


def test_causal_lm_scorer_batches_inputs_and_returns_unsaturated_logit_margins():
    tokenizer_calls: list[tuple[list[str], dict]] = []
    pad_calls: list[tuple[dict, dict]] = []
    model_calls: list[dict] = []

    class FakeInput:
        def to(self, device):
            assert device == "cpu"
            return self

    class FakeVector:
        def __init__(self, values):
            self.values = values

        def float(self):
            return self

        def __sub__(self, other):
            return FakeVector(
                [left - right for left, right in zip(self.values, other.values)]
            )

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeLastLogits:
        def __init__(self, no_values, yes_values):
            self.values = {2: no_values, 3: yes_values}

        def __getitem__(self, key):
            rows, token_id = key
            assert rows == slice(None)
            return FakeVector(self.values[token_id])

    class FakeLogits:
        def __init__(self, no_values, yes_values):
            self.last = FakeLastLogits(no_values, yes_values)

        def __getitem__(self, key):
            rows, position, vocabulary = key
            assert rows == slice(None)
            assert position == -1
            assert vocabulary == slice(None)
            return self.last

    class FakeTokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return [10]

        def convert_tokens_to_ids(self, token):
            return {"no": 2, "yes": 3}[token]

        def __call__(self, texts, **kwargs):
            tokenizer_calls.append((texts, kwargs))
            return {"input_ids": [[20] for _ in texts]}

        def pad(self, encoded, **kwargs):
            pad_calls.append((encoded, kwargs))
            return {"input_ids": FakeInput()}

    class FakeModel:
        def __call__(self, **kwargs):
            model_calls.append(kwargs)
            batch_size = len(pad_calls[-1][0]["input_ids"])
            if len(model_calls) == 1:
                assert batch_size == 2
                return types.SimpleNamespace(
                    logits=FakeLogits([1.0, 3.0], [5.0, 2.0])
                )
            assert batch_size == 1
            return types.SimpleNamespace(logits=FakeLogits([0.0], [0.5]))

    torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    scorer = _Qwen3CausalLMScorer(
        torch_module=torch,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device="cpu",
        max_tokens=512,
        instruction="Find scientific evidence.",
    )

    scores = scorer.predict(
        [("q1", "d1"), ("q2", "d2"), ("q3", "d3")],
        batch_size=2,
    )

    assert scores == [4.0, -1.0, 0.5]
    assert [len(texts) for texts, _ in tokenizer_calls] == [2, 1]
    assert tokenizer_calls[0][1] == {
        "padding": False,
        "truncation": "longest_first",
        "return_attention_mask": False,
        "max_length": 510,
    }
    assert all(
        call["use_cache"] is False and call["logits_to_keep"] == 1
        for call in model_calls
    )
    assert all(
        kwargs == {"padding": True, "return_tensors": "pt"}
        for _, kwargs in pad_calls
    )


@pytest.mark.skipif(
    os.environ.get("LITTRACEQA_RUN_QWEN3_SMOKE") != "1",
    reason="set LITTRACEQA_RUN_QWEN3_SMOKE=1 with a local model cache",
)
def test_cached_qwen3_model_ranks_relevant_document_first():
    candidates = [
        RetrievalResult(
            chunk_id="irrelevant#c0000",
            paper_id="irrelevant",
            score=0.9,
            text="Gravity attracts bodies and governs planetary motion.",
            chunk_type="text_span",
            metadata={},
        ),
        RetrievalResult(
            chunk_id="relevant#c0000",
            paper_id="relevant",
            score=0.4,
            text="The capital of China is Beijing.",
            chunk_type="text_span",
            metadata={},
        ),
    ]
    reranker = Qwen3Reranker(
        device="cpu",
        batch_size=1,
        max_tokens=512,
        local_files_only=True,
    )

    results = reranker.rerank(
        "What is the capital of China?",
        candidates,
        top_k=2,
    )

    assert [result.paper_id for result in results] == ["relevant", "irrelevant"]
    assert results[0].score > results[1].score
