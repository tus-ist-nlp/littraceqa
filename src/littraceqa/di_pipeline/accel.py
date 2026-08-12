"""Transformer 推論の高速化ヘルパー（Flash Attention / torch.compile）。

このリポジトリの環境では standalone の `flash-attn` パッケージは導入できない
（nvcc が無く、torch も cu13 と新しいためビルド済みホイールも無い）。しかし
**PyTorch 2.x の SDPA には Flash Attention v2 カーネルが内蔵**されており、
Ampere 以降 + fp16 なら `scaled_dot_product_attention` がそのまま Flash カーネルに
ディスパッチされる。したがって `attn_implementation="sdpa"` を指定すれば追加依存
ゼロで実質的な Flash Attention が効く。`flash-attn` が入っている環境なら
`"flash_attention_2"` を優先する。

torch.compile は**大量のバッチを流す索引構築のホットループでのみ**効く。
クエリ時の単発推論では初回コンパイルの warmup（数十秒）が回収できず逆に遅く
なるので、呼び出し側で `enabled` を切り替える前提。
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import torch


def best_attn_implementation(device: str | list[str]) -> str:
    """その環境で最速の attention 実装名を返す。

    CUDA(Ampere+) では Flash カーネルが使える。`flash-attn` パッケージがあれば
    それを、無ければ PyTorch 内蔵の SDPA(Flash v2 を含む)を選ぶ。CPU では eager。
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
    """`attn_implementation` を付けてモデルをロードする。

    モデルやライブラリが未対応で例外になったら、指定なしで素直に読み直す
    （既存パイプラインを壊さないためのフォールバック）。
    """
    attn = best_attn_implementation(device)
    try:
        return loader(model_name, attn_implementation=attn, **kwargs)
    except (ValueError, TypeError, ImportError, KeyError):
        return loader(model_name, **kwargs)


def maybe_compile(model: Any, enabled: bool) -> Any:
    """`enabled` なら torch.compile する。可変長入力で再コンパイルが頻発しない
    よう `dynamic=True`。コンパイルに失敗しても eager のまま返す。"""
    if not enabled:
        return model
    try:
        return torch.compile(model, dynamic=True)
    except Exception:
        return model
