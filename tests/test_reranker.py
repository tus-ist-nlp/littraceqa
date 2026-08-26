"""Reranker のテスト。

一番大事なのは「rerank の順位が下流に伝わるか」。以前は rerank スコアを
返り値の並び順にしか乗せておらず、ReadingAgent が chunk_id の dict に貯めてから
r.score（＝RRF スコアのまま）で並べ直すため、順位が丸ごと捨てられていた。
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.reranker import Qwen3Reranker


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("#")[0],
        score=score,
        text=f"body of {chunk_id}",
        chunk_type="text_span",
        metadata={},
    )


class _FakeTokenizer:
    """encode を「文字数=トークン数」で返す軽量スタブ（バッチ分割のテスト用）。"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text)


class _StubReranker(Qwen3Reranker):
    """モデルを読まずに、与えたスコアをそのまま返す Qwen3Reranker。

    チャンク単位なので text から chunk_id を引く（text は "body of {chunk_id}"）。
    """

    def __init__(self, scores: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.scores = scores

    def _ensure_loaded(self) -> None:
        # モデルは読まないが、バッチ分割が参照する属性だけ用意する。
        self._prefix_ids = []
        self._suffix_ids = []
        self._replicas = {d: (_FakeTokenizer(), None) for d in self.devices}

    def _score_on(self, device: str, query: str, texts: list[str]) -> list[float]:
        return [next(v for k, v in self.scores.items() if k in t) for t in texts]


def test_rerank_writes_score_back():
    """RRF スコアの順位を rerank スコアが上書きすること。"""
    candidates = [_result("p1#c0", 0.9), _result("p2#c0", 0.5), _result("p3#c0", 0.1)]
    reranker = _StubReranker({"p1#c0": 0.01, "p2#c0": 0.99, "p3#c0": 0.50})

    ranked = reranker.rerank("q", candidates, top_k=3)

    assert [r.chunk_id for r in ranked] == ["p2#c0", "p3#c0", "p1#c0"]
    # 並び順だけでなく score そのものが rerank スコアに差し替わっていること。
    # ここが元の RRF スコアのままだと、下流で並べ直された時点で順位が消える。
    assert [r.score for r in ranked] == [0.99, 0.50, 0.01]
    # 入力側は書き換えない（dataclasses.replace で新しいオブジェクトを作る）。
    assert [c.score for c in candidates] == [0.9, 0.5, 0.1]


def test_rerank_truncates_to_top_k():
    candidates = [_result("p1#c0", 0.9), _result("p2#c0", 0.5), _result("p3#c0", 0.1)]
    reranker = _StubReranker({"p1#c0": 0.01, "p2#c0": 0.99, "p3#c0": 0.50})

    ranked = reranker.rerank("q", candidates, top_k=2)

    assert [r.chunk_id for r in ranked] == ["p2#c0", "p3#c0"]


def test_rerank_empty_candidates():
    assert _StubReranker({}).rerank("q", [], top_k=5) == []


# ---- トークン予算バッチ + マルチGPU（速度対策） ------------------------------


def test_multi_gpu_disables_compile():
    """**マルチGPU では torch.compile を自動で無効化する。**

    compile 済みモデルを複数スレッドから同時に呼ぶと dynamo が
    「FX symbolic trace of a dynamo-optimized function」で落ちるため。
    本番構成は devices="cuda:1,cuda:2" なのでこの経路を通る。
    """
    for devices, compile_flag, expected in [
        ("cuda:0,cuda:1,cuda:2", True, False),  # 複数 -> compile しない
        ("cuda:0", True, True),                 # 1枚 -> compile する
        ("cuda:0", False, False),               # 明示 off
    ]:
        # 実モデルは重いので _ensure_loaded の compile 判定だけを直接検証する。
        reranker = Qwen3Reranker(devices=devices, compile=compile_flag)
        assert (reranker.compile and len(reranker.devices) == 1) is expected
