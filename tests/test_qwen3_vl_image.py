"""Qwen3VLImageIndex の索引対象の絞り込みロジックのテスト。

モデル本体（16GB・.venv-vl でしか動かない）は使わず、_ensure_model をスタブに
差し替えて「どのチャンクを索引に入れるか」だけを検証する。実際の埋め込み品質では
なく、image_path の扱いを間違えないことが壊れやすい部分なので。
"""

from __future__ import annotations

import numpy as np
import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.qwen3_vl_image import Qwen3VLImageIndex


def _chunk(chunk_id: str, image_path: str | None, chunk_type: str = "figure") -> Chunk:
    metadata: dict = {"page": 1}
    if image_path is not None:
        metadata["image_path"] = image_path
    return Chunk(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("#")[0],
        text=f"caption of {chunk_id}",
        chunk_type=chunk_type,
        metadata=metadata,
    )


class _StubModel:
    """encode() が呼ばれた入力数だけ固定次元のベクトルを返すスタブ。"""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, items, batch_size=None, show_progress_bar=False):
        return np.ones((len(items), self.dim), dtype="float32")


@pytest.fixture
def index(tmp_path, monkeypatch):
    idx = Qwen3VLImageIndex(index_dir=str(tmp_path / "idx"), device="cpu", fp16=False)
    monkeypatch.setattr(idx, "_ensure_model", lambda: None)
    idx._model = _StubModel()
    return idx


def test_only_chunks_with_existing_image_are_indexed(index, tmp_path):
    """image_path があり、かつファイルが実在するチャンクだけを索引に入れる。

    MinerU が失敗した論文で image_path だけ残ることがあるため、存在チェックまでする。
    """
    real = tmp_path / "real.jpg"
    real.write_bytes(b"dummy")

    chunks = [
        _chunk("p1#c0", str(real)),                       # 実在 -> 入る
        _chunk("p1#c1", str(tmp_path / "missing.jpg")),   # パスはあるが実体なし -> 落とす
        _chunk("p1#c2", None, chunk_type="text_span"),    # 本文 -> 落とす
    ]

    # PIL.Image.open をスタブ（ダミーバイトは実画像ではないため）
    import littraceqa.di_pipeline.index.qwen3_vl_image as mod

    class _FakeImage:
        @staticmethod
        def open(path):
            class _Img:
                def convert(self, mode):
                    return self
            return _Img()

    import sys
    sys.modules.setdefault("PIL", type(sys)("PIL"))
    sys.modules["PIL"].Image = _FakeImage
    index.build(chunks)

    assert [c.chunk_id for c in index._chunks] == ["p1#c0"]


def test_no_indexable_chunk_leaves_index_empty(index):
    """索引対象が1件も無ければ index は None のまま（例外にしない）。

    figure_vlm を通していない process_style と組み合わせたときに、
    パイプライン全体を落とさず「この索引だけ空」にするため。
    """
    index.build([_chunk("p1#c0", None, chunk_type="text_span")])

    assert index._chunks == []
    assert index._index is None
    assert index.search("any query", top_k=5) == []


def test_search_returns_empty_when_not_built(tmp_path):
    """build も load もしていなければ空を返す（他の索引の足を引っ張らない）。"""
    idx = Qwen3VLImageIndex(index_dir=str(tmp_path / "idx"), device="cpu", fp16=False)
    assert idx.search("query", top_k=5) == []


def test_registered_in_registry():
    """registry に登録されていないと search_style から呼び出せない。"""
    from littraceqa.di_pipeline import registry

    assert registry.build is not None
    # 登録済みなら例外なくクラスを引ける（インスタンス化はモデルを触らない）
    obj = registry.build("indexer", "qwen3_vl_image", index_dir="/tmp/_vl_test", device="cpu")
    assert obj.name == "qwen3_vl_image"
