"""名前（文字列） → クラス の対応表を管理するレジストリ。

config.yaml で `name: bm25s` のように指定された文字列から、対応する実装クラスを
if/elif の分岐なしにインスタンス化できるようにする。実装クラス側は @register で
自分自身を登録し、呼び出し側は build(kind, name, **kwargs) でインスタンス化する。
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[tuple[str, str], type[Any]] = {}


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    """クラスを (kind, name) で登録するデコレータ。"""

    def decorator(cls: type[T]) -> type[T]:
        _REGISTRY[(kind, name)] = cls
        return cls

    return decorator


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """登録済みのクラスを kwargs でインスタンス化する。未登録なら KeyError。"""
    key = (kind, name)
    if key not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(f"'{kind}:{name}' is not registered. Registered: {available}")
    return _REGISTRY[key](**kwargs)


def list_registered() -> list[tuple[str, str]]:
    """登録済みの (kind, name) 一覧を返す。デバッグ用。"""
    return list(_REGISTRY.keys())
