"""indexer ごとに「どの chunk_type を索引に入れるか」を絞るヘルパ。

同じチャンク集合を全 indexer に流しつつ、indexer ごとに粒度を変えられるようにする。
モデルには設計上の想定粒度があり、そこから外すと性能が出ない:

* SPECTER2 の proximity アダプタは title+abstract（論文まるごと）で学習された
  **論文単位**のモデル。本文の断片・表・数式を個別に埋め込むのは学習時の入力分布から
  外れる。chunk_types: [title_abstract] にすると設計どおりの使い方になる。
* BM25 は語彙一致なので粒度を選ばない。LitTraceQA の質問は手法名・データセット名・
  指標名（PointLoRA, ModelNet40, ECM-XL …）だらけで、稀語の完全一致に強い BM25 とは
  相性が良い。全チャンクに掛けてよい。
* passage 単位の密ベクトル（Qwen3 等）は本文チャンクに掛ける。
"""

from __future__ import annotations

from collections.abc import Iterable

from litqa.contracts import Chunk

# MinerUChunker が出す chunk_type の観測値。
ALL_CHUNK_TYPES = (
    "title_abstract",
    "text_span",
    "table",
    "figure",
    "equation_algorithm",
    "citation_context",
)


def filter_chunk_types(
    chunks: Iterable[Chunk], chunk_types: list[str] | None
) -> list[Chunk]:
    """chunk_types に含まれる chunk_type のチャンクだけを返す。None なら全部。"""
    if not chunk_types:
        return list(chunks)

    unknown = set(chunk_types) - set(ALL_CHUNK_TYPES)
    if unknown:
        raise ValueError(
            f"unknown chunk_types: {sorted(unknown)} "
            f"(expected a subset of {list(ALL_CHUNK_TYPES)})"
        )

    wanted = set(chunk_types)
    return [chunk for chunk in chunks if chunk.chunk_type in wanted]
