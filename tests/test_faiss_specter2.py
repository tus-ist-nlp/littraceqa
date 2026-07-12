"""SPECTER2 索引が「512トークン超えで落ちる」バグに戻らないことを守るテスト。

SPECTER2 は max_position_embeddings=512 の BERT だが、tokenizer の model_max_length は
未設定（1e19）。そのため max_length に 512 より大きい値を渡すと truncation=True でも
切り詰められず、forward が位置埋め込みの範囲外で落ちる:

    RuntimeError: The size of tensor a (1002) must match the size of tensor b (512)

MinerU のチャンクは約8%が512トークンを超えるので、これを踏むと索引構築が必ず途中で死ぬ。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from litqa.index.faiss_specter2 import _MAX_TOKENS, Specter2FAISSIndex


class _RecordingTokenizer:
    """tokenizer に渡された max_length を記録するスタブ。"""

    def __init__(self):
        self.max_lengths: list[int] = []

    def __call__(self, batch, padding, truncation, max_length, return_tensors):
        self.max_lengths.append(max_length)
        # 実際の tokenizer と同じく max_length で頭打ちにする
        seq_len = min(max_length, 1000)
        return _Encoded(len(batch), seq_len)


class _Encoded(dict):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__(input_ids=torch.ones(batch_size, seq_len, dtype=torch.long))

    def to(self, device):
        return self


class _StubModel:
    """位置埋め込み512本の BERT を模したスタブ。512を超えたら本物と同じように落ちる。"""

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
    """SPECTER2 は BERT(512)。ここを増やすと長いチャンクで forward が落ちる。"""
    assert _MAX_TOKENS == 512


def test_embed_truncates_to_512(index):
    """長文を渡しても 512 に切り詰めてトークナイズする。"""
    embeddings = index._embed(["word " * 4000], adapter="proximity")
    assert index._tokenizer.max_lengths == [512]
    assert embeddings.shape == (1, 768)


def test_embed_survives_chunks_longer_than_512_tokens(index):
    """8000文字級のチャンク（>512トークン）が混ざっても例外にならない。"""
    texts = ["short", "word " * 4000, "medium " * 200]
    embeddings = index._embed(texts, adapter="proximity")
    assert embeddings.shape == (3, 768)


def test_embed_switches_adapter(index):
    """文書側 proximity / クエリ側 adhoc_query を adapters の API で切り替える。"""
    index._embed(["doc"], adapter="proximity")
    assert index._model.active == "proximity"
    index._embed(["query"], adapter="adhoc_query")
    assert index._model.active == "adhoc_query"


def test_embed_returns_l2_normalized_float32(index):
    """内積がコサイン類似度になるよう L2 正規化した float32 を返す。"""
    embeddings = index._embed(["a", "b"], adapter="proximity")
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_fp16_disabled_on_cpu(tmp_path):
    """CPU では half() が使えないので fp16 は自動的に無効になる。"""
    assert Specter2FAISSIndex(index_dir=str(tmp_path), device="cpu").fp16 is False
    assert Specter2FAISSIndex(index_dir=str(tmp_path), device="cuda").fp16 is True
