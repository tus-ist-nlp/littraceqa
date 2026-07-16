"""Qwen3-Embedding-8B 索引の設定まわりのテスト（GPU不要）。

実測した制約:
* fp32 だと重みが約32GBになり 24GB の RTX 3090 に載らない → fp16 が既定
* 4096次元 x 256万件 = 42GB。RAM に貯めるとピーク126GBで実RAM 125GB を超える
  → memmap 経由にする
* 5.7 chunks/s (1GPU) なので 256万件は約124時間 → 複数GPUへのデータ並列が必須
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from litqa.index.faiss_qwen3 import _ADD_ROWS, Qwen3FAISSIndex


def test_fp16_is_the_default(tmp_path):
    """fp32 では 8B の重み(約32GB)が 24GB の 3090 に載らない。"""
    assert Qwen3FAISSIndex(index_dir=str(tmp_path), device="cuda").fp16 is True


def test_fp16_is_disabled_on_cpu(tmp_path):
    assert Qwen3FAISSIndex(index_dir=str(tmp_path), device="cpu").fp16 is False


def test_devices_are_parsed_for_data_parallel(tmp_path):
    """複数GPUをカンマ区切りで受け取り、データ並列で分散する。"""
    index = Qwen3FAISSIndex(
        index_dir=str(tmp_path), devices="cuda:0, cuda:1 ,cuda:2,cuda:3"
    )
    assert index.devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def test_devices_defaults_to_the_single_device(tmp_path):
    index = Qwen3FAISSIndex(index_dir=str(tmp_path), device="cuda:2")
    assert index.devices == ["cuda:2"]
    assert index.fp16 is True


def test_shard_boundaries_cover_every_row_exactly_once():
    """担当範囲の切り方に穴も重複も無いこと。

    ここがずれると memmap の一部が未初期化（ゼロ）のまま faiss に入り、
    検索結果が静かに壊れる。
    """
    for n in (1, 7, 100, 2_564_545):
        for world in (1, 2, 3, 4):
            bounds = [(n * r // world, n * (r + 1) // world) for r in range(world)]
            assert bounds[0][0] == 0
            assert bounds[-1][1] == n
            for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:]):
                assert prev_end == next_start
            assert sum(end - start for start, end in bounds) == n


def test_faiss_add_is_sliced_to_avoid_touching_42gb_at_once():
    assert _ADD_ROWS <= 200_000
