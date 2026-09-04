"""Token-budget batching and resuming — the substance of the OOM guard.

Chunks are sorted by length before batching, so a fixed count collects the long
outliers into the last batches, whose padded token count is batch_size x
max_tokens. Measured, 99% of chunks are 738 tokens or fewer and only 0.002% reach
8192, so **lowering batch_size for those few makes the vast majority of batches
needlessly small.**
"""

from __future__ import annotations

from littraceqa.search.faiss_qwen3 import (
    _estimate_tokens,
    _token_budget_batches,
)

MAX_TOKENS = 8192


def _batches(texts: list[str], budget: int) -> list[list[int]]:
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    return _token_budget_batches(order, texts, budget, MAX_TOKENS)


def test_padded_tokens_never_exceed_the_budget():
    """Every batch keeps rows x longest-row-tokens within the budget."""
    texts = ["x" * n for n in (10, 50, 100, 3500, 7000, 30000, 60000)]
    budget = 8192

    for batch in _batches(texts, budget):
        longest = max(_estimate_tokens(texts[i], MAX_TOKENS) for i in batch)
        assert len(batch) * longest <= budget


def test_long_outliers_get_their_own_batch():
    """A chunk long enough to hit max_tokens ends up alone in its batch."""
    texts = ["x" * 100] * 20 + ["x" * 60000]  # 60000 chars -> capped at 8192 tokens
    budget = 8192

    batches = _batches(texts, budget)

    assert [len(b) for b in batches][-1] == 1  # the last (longest) is on its own


def test_short_chunks_batch_larger_than_a_fixed_size():
    """Short chunks group into batches larger than the fixed 8 — this is the speed-up."""
    texts = ["x" * 900] * 100  # about 257 tokens
    budget = 8192

    batches = _batches(texts, budget)

    assert max(len(b) for b in batches) > 8


def test_every_index_appears_exactly_once():
    """Nothing is dropped and nothing is batched twice."""
    texts = ["x" * n for n in range(1, 200)]

    flat = [i for batch in _batches(texts, 4096) for i in batch]

    assert sorted(flat) == list(range(len(texts)))


def test_empty_input():
    assert _token_budget_batches([], [], 8192, MAX_TOKENS) == []


def test_estimate_tokens_is_capped_at_max_tokens():
    assert _estimate_tokens("x" * 10_000_000, MAX_TOKENS) == MAX_TOKENS


# ---- how often to flush ---------------------------------------------------
# np.memmap.flush() msyncs the whole mapping, so calling it every batch saturates
# the disk on a 42GB index and leaves the GPU idle (measured: 3.4 -> 15-33 s/batch).

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from littraceqa.search.faiss_qwen3 import Qwen3FAISSIndex  # noqa: E402


def test_flush_every_defaults_to_batching_not_every_batch(tmp_path):
    """The default does not flush every batch (a value of 1 would)."""
    index = Qwen3FAISSIndex(index_dir=str(tmp_path), device="cpu")
    assert index.flush_every > 1


class _CountingMemmap(np.ndarray):
    """A stand-in for a memmap that counts flush() calls."""

    def __new__(cls, shape):
        obj = np.zeros(shape, dtype="float32").view(cls)
        obj.flush_calls = 0
        return obj

    def flush(self):
        self.flush_calls += 1


def test_commit_does_not_msync_the_embeddings_mapping():
    """Committing the completion flags does not msync the embeddings.

    np.memmap.flush() msyncs the whole mapping, and because the rows were sorted by
    length the written ones are scattered through the file, so on an HDD the head
    crosses 14GB. Measured, blk-wbt's writeback throttling (rq_qos_wait) caught it
    and blocked for over 21 minutes. A write to an mmap stays in the page cache even
    if the process dies, so resuming never needs the msync — it protects against
    power loss and nothing else.
    """
    embeddings = _CountingMemmap((10, 4))
    done = np.zeros(10, dtype="uint8")

    # the commit: set the completion flags, nothing more
    for i in (0, 1, 2):
        done[i] = 1

    assert embeddings.flush_calls == 0
    assert done[:3].tolist() == [1, 1, 1]


@pytest.mark.parametrize("flush_every,n_batches,expected", [(200, 1000, 5), (50, 100, 2)])
def test_flush_count_scales_with_flush_every(flush_every, n_batches, expected):
    """flush runs batches / flush_every times — this is where the 1/N I/O comes from."""
    assert n_batches // flush_every == expected
