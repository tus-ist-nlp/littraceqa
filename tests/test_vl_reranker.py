"""Qwen3VLReranker が候補を CrossEncoder にどう渡すかのテスト。

モデル本体（17.6GB・.venv-vl でしか動かない）は使わず、_ensure_model をスタブに
差し替えて「図表チャンクに画像を付け、本文チャンクはテキストのみで渡す」という
出し分けだけを検証する。ここを間違えると、画像を見られる利点が消えたり、
存在しない画像パスで実行が止まったりする。
"""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.vl_reranker import Qwen3VLReranker


def _result(chunk_id: str, score: float, image_path: str | None = None) -> RetrievalResult:
    metadata: dict = {"page": 1}
    if image_path is not None:
        metadata["image_path"] = image_path
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("#")[0],
        score=score,
        text=f"text of {chunk_id}",
        chunk_type="figure" if image_path else "text_span",
        metadata=metadata,
    )


class _StubCrossEncoder:
    """predict() に渡された pairs を記録し、逆順のスコアを返すスタブ。"""

    def __init__(self):
        self.seen_pairs = None
        self.seen_prompt = None

    def predict(self, pairs, batch_size=None, prompt=None, show_progress_bar=False):
        self.seen_pairs = pairs
        self.seen_prompt = prompt
        # 後ろの候補ほど高スコア -> 並べ替えが実際に効いたか確認できる
        return [float(i) for i in range(len(pairs))]


@pytest.fixture
def reranker(monkeypatch):
    r = Qwen3VLReranker(device="cpu", fp16=False)
    stub = _StubCrossEncoder()
    monkeypatch.setattr(r, "_ensure_model", lambda: None)
    r._model = stub
    return r, stub


def test_figure_chunks_get_image_and_text(reranker, tmp_path):
    """画像が実在する図表チャンクは {"text":..., "image":...} で渡す。"""
    img = tmp_path / "fig.jpg"
    img.write_bytes(b"dummy")
    r, stub = reranker

    r.rerank("q", [_result("p#c0", 1.0, str(img)), _result("p#c1", 0.9)], top_k=2)

    docs = [pair[1] for pair in stub.seen_pairs]
    assert docs[0] == {"text": "text of p#c0", "image": str(img)}
    assert docs[1] == "text of p#c1"  # 本文チャンクは素のテキスト


def test_missing_image_falls_back_to_text(reranker, tmp_path):
    """image_path はあるが実体が無い場合はテキストだけにする（実行を止めない）。

    MinerU が失敗した論文で image_path だけ残ることがあるため。
    """
    r, stub = reranker
    r.rerank("q", [_result("p#c0", 1.0, str(tmp_path / "missing.jpg"))], top_k=1)

    assert stub.seen_pairs[0][1] == "text of p#c0"


def test_use_images_false_disables_images(tmp_path, monkeypatch):
    """use_images=False ならテキスト版と同じ扱い（比較実験用）。"""
    img = tmp_path / "fig.jpg"
    img.write_bytes(b"dummy")
    r = Qwen3VLReranker(device="cpu", fp16=False, use_images=False)
    stub = _StubCrossEncoder()
    monkeypatch.setattr(r, "_ensure_model", lambda: None)
    r._model = stub

    r.rerank("q", [_result("p#c0", 1.0, str(img))], top_k=1)

    assert stub.seen_pairs[0][1] == "text of p#c0"


def test_max_image_docs_limits_images(tmp_path, monkeypatch):
    """max_image_docs で画像込みにする上位件数を絞れる（コスト対策）。"""
    img = tmp_path / "fig.jpg"
    img.write_bytes(b"dummy")
    r = Qwen3VLReranker(device="cpu", fp16=False, max_image_docs=1)
    stub = _StubCrossEncoder()
    monkeypatch.setattr(r, "_ensure_model", lambda: None)
    r._model = stub

    r.rerank("q", [_result("p#c0", 1.0, str(img)), _result("p#c1", 0.9, str(img))], top_k=2)

    docs = [pair[1] for pair in stub.seen_pairs]
    assert isinstance(docs[0], dict)  # 1件目は画像込み
    assert docs[1] == "text of p#c1"  # 2件目以降はテキストのみ


def test_reorders_by_score_and_truncates(reranker):
    """スコア降順に並べ替えて top_k で切る（rerankerの本体機能）。"""
    r, stub = reranker
    candidates = [_result(f"p#c{i}", 1.0) for i in range(4)]

    ranked = r.rerank("q", candidates, top_k=2)

    # スタブは後ろほど高スコアを返すので、逆順の先頭2件になる
    assert [c.chunk_id for c in ranked] == ["p#c3", "p#c2"]


def test_empty_candidates_returns_empty(reranker):
    r, _ = reranker
    assert r.rerank("q", [], top_k=5) == []


def test_registered_in_registry():
    """registry に登録されていないと search_style から呼び出せない。"""
    from littraceqa.di_pipeline import registry

    obj = registry.build("reranker", "qwen3_vl", device="cpu", fp16=False)
    assert obj.name == "qwen3_vl"
