"""The accel helpers: choosing Flash Attention and torch.compile."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from littraceqa.search.accel import (
    best_attn_implementation,
    load_with_best_attn,
    maybe_compile,
)


def test_cpu_uses_eager():
    assert best_attn_implementation("cpu") == "eager"


def test_cuda_uses_sdpa_or_flash():
    # Without the flash-attn package, PyTorch's built-in SDPA (which contains
    # Flash v2) is the choice.
    assert best_attn_implementation("cuda") in ("sdpa", "flash_attention_2")
    assert best_attn_implementation(["cuda:0", "cuda:1"]) in (
        "sdpa",
        "flash_attention_2",
    )


def test_mixed_devices_pick_flash_if_any_cuda():
    assert best_attn_implementation(["cpu", "cuda:3"]) in (
        "sdpa",
        "flash_attention_2",
    )


def test_maybe_compile_is_noop_when_disabled():
    sentinel = object()
    assert maybe_compile(sentinel, enabled=False) is sentinel


def test_load_falls_back_when_attn_unsupported():
    """A model that will not take attn_implementation is loaded again without it."""
    calls: list[dict] = []

    def loader(name, **kwargs):
        calls.append(kwargs)
        if "attn_implementation" in kwargs:
            raise ValueError("unsupported")
        return "loaded"

    assert load_with_best_attn(loader, "some-model", "cuda") == "loaded"
    # Two calls: the first with attn fails, the second without it succeeds.
    assert len(calls) == 2
    assert "attn_implementation" in calls[0]
    assert "attn_implementation" not in calls[1]
