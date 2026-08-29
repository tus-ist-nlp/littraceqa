"""Speeding up transformer inference: Flash Attention and torch.compile.

The standalone `flash-attn` package cannot be installed on these machines (no
nvcc, and torch is new enough — cu13 — that no prebuilt wheel exists). It does not
have to be: **PyTorch 2.x ships the Flash Attention v2 kernel inside SDPA**, and on
Ampere or later with fp16 `scaled_dot_product_attention` dispatches straight to it.
So `attn_implementation="sdpa"` buys real Flash Attention with no extra dependency.
Where `flash-attn` does exist, `"flash_attention_2"` is preferred.

**torch.compile only pays off in the index-building hot loop**, where batches keep
coming. A single embedding at query time never earns back the tens of seconds of
first-call warmup, so the caller decides with `enabled`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import torch


def best_attn_implementation(device: str | list[str]) -> str:
    """Name the fastest attention implementation available here.

    On CUDA (Ampere+) a Flash kernel is available: the `flash-attn` package if it is
    installed, otherwise PyTorch's built-in SDPA (which contains Flash v2). On CPU,
    eager.
    """
    devices = device if isinstance(device, list) else [device]
    on_cuda = any(str(d).startswith("cuda") for d in devices)
    if not on_cuda:
        return "eager"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def load_with_best_attn(
    loader: Callable[..., Any], model_name: str, device: str | list[str], **kwargs: Any
) -> Any:
    """Load a model with `attn_implementation` set.

    If the model or the library does not accept it, load again without it — an
    unsupported attention kernel must never be the reason a build fails.
    """
    attn = best_attn_implementation(device)
    try:
        return loader(model_name, attn_implementation=attn, **kwargs)
    except (ValueError, TypeError, ImportError, KeyError):
        return loader(model_name, **kwargs)


def maybe_compile(model: Any, enabled: bool) -> Any:
    """torch.compile the model when `enabled`.

    `dynamic=True` because the inputs are variable-length; without it dynamo
    recompiles on nearly every new shape. A compile failure returns the eager model
    rather than raising.
    """
    if not enabled:
        return model
    try:
        return torch.compile(model, dynamic=True)
    except Exception:
        return model
