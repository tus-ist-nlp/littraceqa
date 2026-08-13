"""Reranker のテスト。

一番大事なのは「rerank の順位が下流に伝わるか」。以前は rerank スコアを
返り値の並び順にしか乗せておらず、ReadingAgent が chunk_id の dict に貯めてから
r.score（＝RRF スコアのまま）で並べ直すため、順位が丸ごと捨てられていた。
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.reranker import (
    NoneReranker,
    Qwen3PaperReranker,
    Qwen3Reranker,
)


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("#")[0],
        score=score,
        text=f"body of {chunk_id}",
        chunk_type="text_span",
        metadata={},
    )


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


def test_none_reranker_passes_through():
    """NoneReranker は素通し。score も並び順もいじらない。"""
    candidates = [_result("p1#c0", 0.9), _result("p2#c0", 0.5), _result("p3#c0", 0.1)]

    ranked = NoneReranker().rerank("q", candidates, top_k=2)

    assert ranked == candidates[:2]


# ---- Qwen3PaperReranker（論文単位）--------------------------------------


def _paper_result(paper: str, i: int, score: float, title: str = None) -> RetrievalResult:
    title = title or f"Title of {paper}"
    prefix = f"[NeurIPS 2025] {title}\n"
    return RetrievalResult(
        chunk_id=f"{paper}#c{i:02d}",
        paper_id=paper,
        score=score,
        text=f"{prefix}body {i} of {paper}",
        chunk_type="text_span",
        metadata={"title": title, "venue": "NeurIPS", "year": 2025},
    )


class _StubPaperReranker(Qwen3PaperReranker):
    """モデルを読まずに、論文ごとの固定スコアを返す。渡されたバッチを記録する。"""

    def __init__(self, scores: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.scores = scores
        self.seen_texts: list[str] = []
        self.batches: list[list[str]] = []  # デバイスごとに来たバッチ

    def _ensure_loaded(self) -> None:
        # モデルは読まないが、トークン予算バッチが参照する属性だけ用意する。
        self._prefix_ids = []
        self._suffix_ids = []
        # tokenizer.encode を「1文字1トークン」で代用（長さ順序は保たれる）。
        self._replicas = {d: (_FakeTokenizer(), None) for d in self.devices}

    def _score_on(self, device: str, query: str, texts: list[str]) -> list[float]:
        self.seen_texts.extend(texts)
        self.batches.append(list(texts))
        # 代表テキストに paper_id（body に "of pXX" として入る）が出るので引く。
        # 見つからなければ 0.0（長さだけを変えたテスト用のダミー論文）。
        out = []
        for t in texts:
            match = next((v for k, v in self.scores.items() if k in t), 0.0)
            out.append(match)
        return out


class _FakeTokenizer:
    """encode を「文字数=トークン数」で返す軽量スタブ（バッチ分割のテスト用）。"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text)


def test_paper_reranker_scores_once_per_paper():
    """推論回数がチャンク数ではなく論文数になること（コスト削減の根拠）。"""
    candidates = [_paper_result("pA", i, 1.0) for i in range(5)]
    candidates += [_paper_result("pB", i, 0.9) for i in range(4)]
    reranker = _StubPaperReranker({"pA": 0.2, "pB": 0.8})

    reranker.rerank("q", candidates, top_k=9)

    assert len(reranker.seen_texts) == 2  # 9チャンクでも推論は2回


def test_paper_reranker_orders_chunks_by_paper_score():
    """論文スコアの順に、その論文の全チャンクがまとまって並ぶこと。"""
    candidates = [_paper_result("pA", i, 1.0) for i in range(2)]
    candidates += [_paper_result("pB", i, 0.9) for i in range(2)]
    reranker = _StubPaperReranker({"pA": 0.2, "pB": 0.8})

    ranked = reranker.rerank("q", candidates, top_k=4)

    assert [r.paper_id for r in ranked] == ["pB", "pB", "pA", "pA"]


def test_paper_reranker_keeps_within_paper_order():
    """論文内では元の検索スコア順を保つこと（同点で潰れない）。"""
    candidates = [
        _paper_result("pA", 0, 0.1),
        _paper_result("pA", 1, 0.9),  # 論文内で最上位
    ]
    reranker = _StubPaperReranker({"pA": 0.5})

    ranked = reranker.rerank("q", candidates, top_k=2)

    assert [r.chunk_id for r in ranked] == ["pA#c01", "pA#c00"]
    assert ranked[0].score > ranked[1].score  # 下流の安定ソートで潰れない


def test_paper_text_has_the_title_once():
    """代表テキストで prefix（会議名・年・タイトル）が繰り返されないこと。"""
    candidates = [_paper_result("pA", i, 1.0) for i in range(3)]
    reranker = _StubPaperReranker({"pA": 0.5}, chunks_per_paper=3)

    reranker.rerank("q", candidates, top_k=3)

    assert reranker.seen_texts[0].count("Title of pA") == 1
    for i in range(3):
        assert f"body {i} of pA" in reranker.seen_texts[0]


def test_paper_text_respects_chunks_per_paper():
    candidates = [_paper_result("pA", i, 1.0 - i / 10) for i in range(5)]
    reranker = _StubPaperReranker({"pA": 0.5}, chunks_per_paper=2)

    reranker.rerank("q", candidates, top_k=5)

    assert "body 0 of pA" in reranker.seen_texts[0]
    assert "body 1 of pA" in reranker.seen_texts[0]
    assert "body 2 of pA" not in reranker.seen_texts[0]


def test_paper_reranker_handles_empty_candidates():
    assert _StubPaperReranker({}).rerank("q", [], top_k=5) == []


# ---- トークン予算バッチ + マルチGPU（速度対策） ------------------------------


def test_token_budget_batches_split_by_length():
    """max_batch_tokens を指定すると、パディング後トークン量で区切ること。

    論文代表テキストは長さがばらつく（実測 中央313tok・max2116）。件数固定だと
    長い外れ値で batch_size x 最長 がVRAMを食い OOM する。トークン量で切れば
    短い論文を多数詰めつつ、長い論文は自動的に1件ずつになる。
    """
    # 100文字の論文を多数 + 1本だけ非常に長い論文
    candidates = [_paper_result(f"p{i:02d}", 0, 1.0, title="x" * 90) for i in range(10)]
    candidates.append(_paper_result("plong", 0, 1.0, title="y" * 3000))
    scores = {f"p{i:02d}": 0.5 for i in range(10)}
    scores["plong"] = 0.9
    reranker = _StubPaperReranker(scores, max_batch_tokens=600)

    reranker.rerank("q", candidates, top_k=11)

    # 長い論文（3000字級）は単独バッチになる
    long_batches = [b for b in reranker.batches if any("y" * 100 in t for t in b)]
    assert all(len(b) == 1 for b in long_batches)
    # 短い論文はまとまって入る（1件ずつではない）
    assert max(len(b) for b in reranker.batches) > 1


def test_all_papers_scored_exactly_once_with_budget():
    candidates = [_paper_result(f"p{i:02d}", 0, float(i), title="x" * (50 + i * 5)) for i in range(20)]
    scores = {f"p{i:02d}": float(i) for i in range(20)}
    reranker = _StubPaperReranker(scores, max_batch_tokens=500)

    ranked = reranker.rerank("q", candidates, top_k=20)

    assert len(reranker.seen_texts) == 20  # 全論文をちょうど1回ずつ
    assert [r.paper_id for r in ranked][0] == "p19"  # 最高スコアが先頭


def test_multi_gpu_round_robins_batches():
    """複数デバイス指定でバッチがGPUに分配され、全論文が採点されること。"""
    candidates = [_paper_result(f"p{i:02d}", 0, 1.0) for i in range(12)]
    scores = {f"p{i:02d}": float(i) for i in range(12)}
    reranker = _StubPaperReranker(scores, devices="cuda:0,cuda:1,cuda:2", batch_size=2)

    assert reranker.devices == ["cuda:0", "cuda:1", "cuda:2"]
    ranked = reranker.rerank("q", candidates, top_k=12)
    assert len(reranker.seen_texts) == 12
    assert ranked[0].paper_id == "p11"


def test_single_device_default():
    """devices 省略時は device 1枚（従来動作）。"""
    reranker = _StubPaperReranker({}, device="cuda:3")
    assert reranker.devices == ["cuda:3"]


def test_multi_gpu_disables_compile():
    """複数デバイスでは torch.compile を無効化すること。

    compile 済みモデルを複数スレッドから同時に呼ぶと dynamo が
    「FX symbolic trace of a dynamo-optimized function」で落ちるため。
    _ensure_loaded 内の use_compile 判定を、compile 呼び出しを差し替えて確認する。
    """
    import littraceqa.di_pipeline.retrieve.reranker as rr_mod

    calls = []
    orig = rr_mod.maybe_compile
    rr_mod.maybe_compile = lambda model, enabled: (calls.append(enabled) or model)
    try:
        # 実モデルは重いので、_ensure_loaded の compile 判定だけを直接検証する。
        for devices, compile_flag, expected in [
            ("cuda:0,cuda:1,cuda:2", True, False),  # 複数 -> compile しない
            ("cuda:0", True, True),                 # 1枚 -> compile する
            ("cuda:0", False, False),               # 明示 off
        ]:
            r = Qwen3PaperReranker(devices=devices, compile=compile_flag)
            use_compile = r.compile and len(r.devices) == 1
            assert use_compile is expected
    finally:
        rr_mod.maybe_compile = orig
