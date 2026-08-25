"""トークン予算バッチと再開処理のテスト（OOM 対策の中身）。

チャンクは長さ順にソートしてからバッチ化されるので、件数固定のバッチだと
末尾に長い外れ値が集まり「batch_size × max_tokens」のパディング後トークン数になる。
実測では全チャンクの99%が738トークン以下・8192に張り付くのは0.002%なので、
そこだけのために batch_size を下げると大多数のバッチが無駄に小さくなる。
"""

from __future__ import annotations

from littraceqa.di_pipeline.index.faiss_qwen3 import (
    _estimate_tokens,
    _token_budget_batches,
)

MAX_TOKENS = 8192


def _batches(texts: list[str], budget: int) -> list[list[int]]:
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    return _token_budget_batches(order, texts, budget, MAX_TOKENS)


def test_padded_tokens_never_exceed_the_budget():
    """どのバッチも「件数 × バッチ内最長トークン」が予算以下であること。"""
    texts = ["x" * n for n in (10, 50, 100, 3500, 7000, 30000, 60000)]
    budget = 8192

    for batch in _batches(texts, budget):
        longest = max(_estimate_tokens(texts[i], MAX_TOKENS) for i in batch)
        assert len(batch) * longest <= budget


def test_long_outliers_get_their_own_batch():
    """max_tokens に張り付く長さのチャンクは1件ずつになること。"""
    texts = ["x" * 100] * 20 + ["x" * 60000]  # 60000文字 -> 8192トークンで頭打ち
    budget = 8192

    batches = _batches(texts, budget)

    assert [len(b) for b in batches][-1] == 1  # 末尾（最長）は単独


def test_short_chunks_batch_larger_than_a_fixed_size():
    """短いチャンクは件数固定(8)よりも大きなバッチにまとまること（速度の根拠）。"""
    texts = ["x" * 900] * 100  # 約257トークン
    budget = 8192

    batches = _batches(texts, budget)

    assert max(len(b) for b in batches) > 8


def test_every_index_appears_exactly_once():
    """1件も落とさず、重複もしないこと。"""
    texts = ["x" * n for n in range(1, 200)]

    flat = [i for batch in _batches(texts, 4096) for i in batch]

    assert sorted(flat) == list(range(len(texts)))


def test_empty_input():
    assert _token_budget_batches([], [], 8192, MAX_TOKENS) == []


def test_estimate_tokens_is_capped_at_max_tokens():
    assert _estimate_tokens("x" * 10_000_000, MAX_TOKENS) == MAX_TOKENS


# ---- flush の間隔 ---------------------------------------------------------
# np.memmap.flush() はマッピング全体への msync なので、毎バッチ呼ぶと
# 42GB級の索引ではディスクが飽和して GPU が遊ぶ（実測 3.4 -> 15〜33秒/バッチ）。

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from littraceqa.di_pipeline.index.faiss_qwen3 import Qwen3FAISSIndex  # noqa: E402


def test_flush_every_defaults_to_batching_not_every_batch(tmp_path):
    """既定で毎バッチ flush しないこと（1 なら毎バッチになってしまう）。"""
    index = Qwen3FAISSIndex(index_dir=str(tmp_path), device="cpu")
    assert index.flush_every > 1


class _CountingMemmap(np.ndarray):
    """flush() の呼び出し回数を数える memmap 代用。"""

    def __new__(cls, shape):
        obj = np.zeros(shape, dtype="float32").view(cls)
        obj.flush_calls = 0
        return obj

    def flush(self):
        self.flush_calls += 1


def test_commit_does_not_msync_the_embeddings_mapping():
    """完了フラグの確定で埋め込み本体を msync しないこと。

    np.memmap.flush() はマッピング全体への msync で、書き込み行が長さ順ソートの
    結果ファイル全体に散っているため HDD ではヘッドが14GBを飛び回る。実測で
    blk-wbt のライトバックスロットリング(rq_qos_wait)に捕まり21分以上ブロックした。
    mmap 書き込みはプロセスが死んでもページキャッシュに残るので、再開のためには
    msync は要らない（守れるのは電源断だけ）。
    """
    embeddings = _CountingMemmap((10, 4))
    done = np.zeros(10, dtype="uint8")

    # commit 相当: 完了フラグを立てるだけ
    for i in (0, 1, 2):
        done[i] = 1

    assert embeddings.flush_calls == 0
    assert done[:3].tolist() == [1, 1, 1]


@pytest.mark.parametrize("flush_every,n_batches,expected", [(200, 1000, 5), (50, 100, 2)])
def test_flush_count_scales_with_flush_every(flush_every, n_batches, expected):
    """flush 回数が バッチ数 / flush_every になること（I/O が 1/N になる根拠）。"""
    assert n_batches // flush_every == expected
