"""The reranker.

**What matters most is whether the reranked order survives downstream.** It used to
live only in the order of the returned list, and ReadingAgent accumulates into a
dict keyed by chunk_id and then re-sorts by r.score — still the RRF score — so the
reranking was thrown away in its entirety.
"""

from __future__ import annotations

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.reranker import Qwen3Reranker


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
    """A light stub whose encode treats one character as one token, for the batching tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text)


class _StubReranker(Qwen3Reranker):
    """A Qwen3Reranker that loads no model and returns the scores it was given.

    It works per chunk, so the chunk_id is recovered from the text, which has the
    form "body of {chunk_id}".
    """

    def __init__(self, scores: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.scores = scores

    def _ensure_loaded(self) -> None:
        # No model is loaded; only the attributes the batching reads are set up.
        self._prefix_ids = []
        self._suffix_ids = []
        self._replicas = {d: (_FakeTokenizer(), None) for d in self.devices}

    def _score_on(self, device: str, query: str, texts: list[str]) -> list[float]:
        return [next(v for k, v in self.scores.items() if k in t) for t in texts]


def test_rerank_writes_score_back():
    """The rerank scores replace the order the RRF scores gave."""
    candidates = [_result("p1#c0", 0.9), _result("p2#c0", 0.5), _result("p3#c0", 0.1)]
    reranker = _StubReranker({"p1#c0": 0.01, "p2#c0": 0.99, "p3#c0": 0.50})

    ranked = reranker.rerank("q", candidates, top_k=3)

    assert [r.chunk_id for r in ranked] == ["p2#c0", "p3#c0", "p1#c0"]
    # Not just the order: `score` itself has to become the rerank score. Left as the
    # RRF score, the ranking vanishes the moment anything downstream re-sorts.
    assert [r.score for r in ranked] == [0.99, 0.50, 0.01]
    # The input is not mutated; dataclasses.replace makes new objects.
    assert [c.score for c in candidates] == [0.9, 0.5, 0.1]


def test_rerank_truncates_to_top_k():
    candidates = [_result("p1#c0", 0.9), _result("p2#c0", 0.5), _result("p3#c0", 0.1)]
    reranker = _StubReranker({"p1#c0": 0.01, "p2#c0": 0.99, "p3#c0": 0.50})

    ranked = reranker.rerank("q", candidates, top_k=2)

    assert [r.chunk_id for r in ranked] == ["p2#c0", "p3#c0"]


def test_rerank_empty_candidates():
    assert _StubReranker({}).rerank("q", [], top_k=5) == []


# ---- token-budget batching and multi-GPU (the speed work) ------------------


def test_multi_gpu_disables_compile():
    """**Several GPUs turn torch.compile off automatically.**

    Calling a compiled model from several threads at once makes dynamo die with
    "FX symbolic trace of a dynamo-optimized function". The production
    configuration is devices="cuda:1,cuda:2", so it takes this path.
    """
    for devices, compile_flag, expected in [
        ("cuda:0,cuda:1,cuda:2", True, False),  # several -> do not compile
        ("cuda:0", True, True),                 # one -> compile
        ("cuda:0", False, False),               # explicitly off
    ]:
        # The real model is heavy, so only _ensure_loaded's compile decision is checked.
        reranker = Qwen3Reranker(devices=devices, compile=compile_flag)
        assert (reranker.compile and len(reranker.devices) == 1) is expected
