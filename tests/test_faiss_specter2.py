"""Guards the SPECTER2 index against regressing to "dies past 512 tokens".

SPECTER2 is a BERT with max_position_embeddings=512, but its tokenizer leaves
model_max_length unset (1e19). A max_length above 512 is therefore not truncated
even with truncation=True, and forward runs past the position embeddings and dies:

    RuntimeError: The size of tensor a (1002) must match the size of tensor b (512)

About 8% of MinerU's chunks are longer than 512 tokens, so hitting this kills any
build partway through — it is not a rare edge case.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from littraceqa.di_pipeline.indexes import _MAX_TOKENS, Specter2FAISSIndex


class _RecordingTokenizer:
    """A stub tokenizer that records the max_length it was given."""

    def __init__(self):
        self.max_lengths: list[int] = []

    def __call__(self, batch, padding, truncation, max_length, return_tensors):
        self.max_lengths.append(max_length)
        # Cap at max_length, as the real tokenizer does
        seq_len = min(max_length, 1000)
        return _Encoded(len(batch), seq_len)


class _Encoded(dict):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__(input_ids=torch.ones(batch_size, seq_len, dtype=torch.long))

    def to(self, device):
        return self


class _StubModel:
    """A stub BERT with 512 position embeddings; past 512 it dies, as the real one does."""

    max_positions = 512

    def __init__(self):
        self.active: str | None = None

    def set_active_adapters(self, adapter: str) -> None:
        self.active = adapter

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]
        if seq_len > self.max_positions:
            raise RuntimeError(
                f"The size of tensor a ({seq_len}) must match "
                f"the size of tensor b ({self.max_positions}) at non-singleton dimension 1"
            )
        return _Output(torch.ones(input_ids.shape[0], seq_len, 768))


class _Output:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


@pytest.fixture
def index(tmp_path):
    ix = Specter2FAISSIndex(index_dir=str(tmp_path), device="cpu")
    ix._tokenizer = _RecordingTokenizer()
    ix._model = _StubModel()
    return ix


def test_max_tokens_does_not_exceed_specter2_position_embeddings():
    """SPECTER2 is a BERT(512). Raising this makes forward die on long chunks."""
    assert _MAX_TOKENS == 512


def test_embed_truncates_to_512(index):
    """Long text is truncated to 512 before tokenising."""
    embeddings = index._embed(["word " * 4000], adapter="proximity")
    assert index._tokenizer.max_lengths == [512]
    assert embeddings.shape == (1, 768)


def test_embed_survives_chunks_longer_than_512_tokens(index):
    """A chunk of 8000 characters (>512 tokens) in the mix raises nothing."""
    texts = ["short", "word " * 4000, "medium " * 200]
    embeddings = index._embed(texts, adapter="proximity")
    assert embeddings.shape == (3, 768)


def test_embed_switches_adapter(index):
    """proximity for documents, adhoc_query for queries, through the adapters API."""
    index._embed(["doc"], adapter="proximity")
    assert index._model.active == "proximity"
    index._embed(["query"], adapter="adhoc_query")
    assert index._model.active == "adhoc_query"


def test_embed_returns_l2_normalized_float32(index):
    """Returns L2-normalised float32, so the inner product is cosine similarity."""
    embeddings = index._embed(["a", "b"], adapter="proximity")
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_fp16_disabled_on_cpu(tmp_path):
    """half() is unavailable on CPU, so fp16 turns itself off."""
    assert Specter2FAISSIndex(index_dir=str(tmp_path), device="cpu").fp16 is False
    assert Specter2FAISSIndex(index_dir=str(tmp_path), device="cuda").fp16 is True
