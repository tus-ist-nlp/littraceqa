"""The Qwen3-Embedding-8B index's configuration (no GPU required).

The measured constraints behind it:
* At fp32 the weights are about 32GB and do not fit a 24GB RTX 3090 -> fp16 is the
  default
* 4096 dims x 2.56M chunks = 42GB; held in RAM the peak is 126GB, past the 125GB
  the machine has -> it goes through a memmap
* At 5.7 chunks/s on one GPU, 2.56M chunks take about 124 hours -> data parallelism
  across GPUs is not optional
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from littraceqa.search.faiss_qwen3 import _ADD_ROWS, Qwen3FAISSIndex


def test_fp16_is_the_default(tmp_path):
    """At fp32 the 8B weights (about 32GB) do not fit a 24GB 3090."""
    assert Qwen3FAISSIndex(index_dir=str(tmp_path), device="cuda").fp16 is True


def test_fp16_is_disabled_on_cpu(tmp_path):
    assert Qwen3FAISSIndex(index_dir=str(tmp_path), device="cpu").fp16 is False


def test_devices_are_parsed_for_data_parallel(tmp_path):
    """Several GPUs arrive comma-separated and the work is split data-parallel."""
    index = Qwen3FAISSIndex(
        index_dir=str(tmp_path), devices="cuda:0, cuda:1 ,cuda:2,cuda:3"
    )
    assert index.devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def test_devices_defaults_to_the_single_device(tmp_path):
    index = Qwen3FAISSIndex(index_dir=str(tmp_path), device="cuda:2")
    assert index.devices == ["cuda:2"]
    assert index.fp16 is True


def test_shard_boundaries_cover_every_row_exactly_once():
    """The shard ranges leave no gap and no overlap.

    **A mistake here puts uninitialised (zero) rows of the memmap into faiss** and
    the search results break silently.
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


# ---- resuming -------------------------------------------------------------
# What keeps a 30-hour build from starting over when it dies partway.

import numpy as np  # noqa: E402

from littraceqa.search.faiss_qwen3 import (  # noqa: E402
    _DONE_FILENAME,
    _EMBEDDINGS_FILENAME,
)


def _index(tmp_path, **kwargs) -> Qwen3FAISSIndex:
    return Qwen3FAISSIndex(index_dir=str(tmp_path), device="cpu", **kwargs)


def test_first_build_creates_both_intermediate_files(tmp_path):
    index = _index(tmp_path)
    memmap_path = tmp_path / _EMBEDDINGS_FILENAME
    done_path = tmp_path / _DONE_FILENAME

    assert index._prepare_memmap(memmap_path, done_path, n=10, dim=4) is False
    assert memmap_path.stat().st_size == 10 * 4 * 4
    assert done_path.stat().st_size == 10
    assert np.memmap(done_path, dtype="uint8", mode="r", shape=(10,)).sum() == 0


def test_resume_keeps_the_previous_progress(tmp_path):
    """Last run's progress is picked up when the row count and dimension match."""
    index = _index(tmp_path)
    memmap_path = tmp_path / _EMBEDDINGS_FILENAME
    done_path = tmp_path / _DONE_FILENAME
    index._prepare_memmap(memmap_path, done_path, n=10, dim=4)

    done = np.memmap(done_path, dtype="uint8", mode="r+", shape=(10,))
    done[:6] = 1
    done.flush()
    del done

    assert index._prepare_memmap(memmap_path, done_path, n=10, dim=4) is True
    assert np.memmap(done_path, dtype="uint8", mode="r", shape=(10,)).sum() == 6


def test_mismatched_size_starts_over(tmp_path):
    """A changed chunk count rebuilds, so nothing stale is mixed in."""
    index = _index(tmp_path)
    memmap_path = tmp_path / _EMBEDDINGS_FILENAME
    done_path = tmp_path / _DONE_FILENAME
    index._prepare_memmap(memmap_path, done_path, n=10, dim=4)
    done = np.memmap(done_path, dtype="uint8", mode="r+", shape=(10,))
    done[:] = 1
    done.flush()
    del done

    # the count changed, 10 -> 12
    assert index._prepare_memmap(memmap_path, done_path, n=12, dim=4) is False
    assert np.memmap(done_path, dtype="uint8", mode="r", shape=(12,)).sum() == 0


def test_resume_can_be_disabled(tmp_path):
    index = _index(tmp_path, resume=False)
    memmap_path = tmp_path / _EMBEDDINGS_FILENAME
    done_path = tmp_path / _DONE_FILENAME
    index._prepare_memmap(memmap_path, done_path, n=10, dim=4)

    assert index._prepare_memmap(memmap_path, done_path, n=10, dim=4) is False
