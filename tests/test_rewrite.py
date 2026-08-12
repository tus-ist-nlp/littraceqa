"""検索結果に接地したクエリ書き換え（agent/rewrite.py）のテスト。

仕様は docs/search_agent2_spec.md。ここで守りたいのは3つ:

  1. `rewrite` を書かなければ挙動が1ビットも変わらない（従来の `_refine()` 経路）
  2. 書き換えの材料に「ヒットしたチャンク」と「展開論文の自己記述」が入る
  3. 重複除去は文字列ではなく**引いてくる論文**で判定する
"""

from __future__ import annotations

import json

from littraceqa.di_pipeline.agent.reading import ReadingAgent
from littraceqa.di_pipeline.agent.rewrite import QueryRewriter, SubqueryDeduper
from littraceqa.di_pipeline.contracts import Query, RetrievalResult

from test_reading_agent import FakeLLM, _judge, _query, _result, _StubRetriever, _subqueries


class _StubRelated:
    """関連ランキング（B）のスタブ。expand() は既存候補を除いた近傍を返す。"""

    combine = "rrf"
    combine_rrf_k = 60
    related_weight = 1.0
    related_offset = 0
    anchors = 1

    def __init__(self, related: list[str]):
        self.related = related
        self.expand_calls: list[list[str]] = []

    def rank(self, ranked: list[str]) -> list[str]:
        return [p for p in self.related if p != ranked[0]]

    def expand(self, ranked: list[str]) -> list[str]:
        self.expand_calls.append(list(ranked))
        return [p for p in self.related if p not in set(ranked)]


class _StubStore:
    """ChunkStore のスタブ。paper_id から本文チャンクを返す。"""

    def __init__(self, by_paper: dict[str, list[dict]]):
        self.by_paper = by_paper

    def __contains__(self, paper_id: str) -> bool:
        return paper_id in self.by_paper

    def load_paper(self, paper_id: str) -> list[dict]:
        return self.by_paper.get(paper_id, [])


class _StubBM25:
    """重複除去のプローブ用。クエリごとに返す論文集合を決め打ちする。"""

    name = "bm25s"

    def __init__(self, by_query: dict[str, list[str]]):
        self.by_query = by_query
        self.calls: list[str] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append(query)
        papers = self.by_query.get(query, [])[:top_k]
        return [
            RetrievalResult(
                chunk_id=f"{p}#c0",
                paper_id=p,
                score=1.0,
                text="",
                chunk_type="text_span",
                metadata={},
            )
            for p in papers
        ]


class _RetrieverWithBM25(_StubRetriever):
    def __init__(self, by_query, indexer):
        super().__init__(by_query)
        self.indexers = [indexer]


# ---- 1. 書かなければ変わらない --------------------------------------------


def test_without_rewrite_the_refine_path_is_unchanged():
    """`rewrite` を渡さなければ `_refine()` の従来プロンプトがそのまま出る。"""
    retriever = _StubRetriever({"sq": [_result(0, "p0", 0.9)]})
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=False, missing="need the batch size"),
            _subqueries("sq2"),
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=2, retrieve_top_k=5)
    agent.run(_query())

    refine_prompt = llm.calls[2]
    assert "Search subqueries already tried" in refine_prompt
    assert "matched chunk" not in refine_prompt
    assert agent.rewriter is None
    assert agent.deduper is None


# ---- 2. 材料が入る ---------------------------------------------------------


def test_rewrite_replaces_refine_and_carries_hit_chunks_and_related_papers():
    """書き換えプロンプトに「ヒットしたチャンク」と「展開論文」が入る。

    `_refine()` は候補の本文を一度も見ないので、サブクエリが質問文の言い換えに
    収束する。書き換えは検索が実際に当てたチャンクと、展開で拾った
    （＝まだ候補に入っていない）論文の自己記述を材料にする。
    """
    hits = [_result(i, f"p{i}", 0.9 - 0.01 * i) for i in range(3)]
    retriever = _StubRetriever({"sq": hits})
    expander = _StubRelated(["pR1", "pR2"])
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=False, missing="need the reward shape"),
            _subqueries("rewritten"),
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(
        retriever,
        llm=llm,
        max_steps=2,
        retrieve_top_k=5,
        paper_expander=expander,
        rewrite={"enabled": True, "at_step": 1},
    )
    # ChunkStore は 3.8GB の実コーパスを開くのでスタブに差し替える。
    agent.rewriter._store = _StubStore(
        {
            "p0": [{"chunk_type": "title_abstract", "text": "abstract of p0"}],
            "pR1": [{"chunk_type": "title_abstract", "text": "we introduce AlphaPO"}],
        }
    )
    agent.run(_query())

    prompt = llm.calls[2]
    # 材料A 詳細: ヒットしたチャンクが本文つきで入る
    assert "matched chunk" in prompt
    assert "body of p0 chunk 0" in prompt
    assert "abstract of p0" in prompt
    # 材料A 俯瞰: タイトルだけの並び（軸ズレ検出用）
    assert "1. [NeurIPS 2025] Paper p0" in prompt
    # 材料B: 展開で拾った論文の**自己記述**が入る（paper_id だけでは語彙にならない）
    assert "we introduce AlphaPO" in prompt
    # 指示の核: 質問の言い換えを書くな
    assert "Do not paraphrase the question" in prompt
    # _refine() のプロンプトではない
    assert "Propose at most" not in prompt
    # 展開は候補列を渡して呼ばれている（ループの中で走る）
    assert expander.expand_calls and expander.expand_calls[0][0] == "p0"


def test_rewrite_starts_at_the_configured_step():
    """`at_step` より前のステップは `_refine()` のまま。"""
    retriever = _StubRetriever({"sq": [_result(0, "p0", 0.9)]})
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=False, missing="m1"),
            _subqueries("sq2"),
            _judge(["p0"], sufficient=False, missing="m2"),
            _subqueries("sq3"),
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(
        retriever,
        llm=llm,
        max_steps=3,
        retrieve_top_k=5,
        rewrite={"enabled": True, "at_step": 2},
    )
    agent.run(_query())

    assert "Propose at most" in llm.calls[2]  # step0 -> step1 は従来の _refine()
    assert "Do not paraphrase the question" in llm.calls[4]  # step1 -> step2 で切り替わる


def test_material_a_skips_whole_paper_pseudo_chunks():
    """`bm25s_paper` 由来の擬似チャンクは「ヒットしたチャンク」に使わない。

    論文単位索引の text は論文全文で chunk_id も擬似 ID なので、
    snippet で切ると abstract と重複するうえ「質問のどこに当たったか」を持たない。
    """
    paper_hit = RetrievalResult(
        chunk_id="p0#paper",
        paper_id="p0",
        score=0.9,
        text="WHOLE PAPER TEXT",
        chunk_type="paper",
        metadata={"title": "Paper p0", "venue": "ICML", "year": 2025},
        source="bm25s_paper",
    )
    rewriter = QueryRewriter(from_a={"include_abstract": False})
    text = rewriter.material_a([("p0", [paper_hit])], snippet_chars=200)

    assert "WHOLE PAPER TEXT" not in text
    assert "matched at whole-paper level only" in text


# ---- 3. 重複除去は「引いてくる論文」で判定する -----------------------------


def test_dedup_drops_queries_that_retrieve_the_same_papers():
    """文字列が違っても同じ論文しか引かないクエリは捨てる。"""
    indexer = _StubBM25(
        {
            "reference-free preference optimization": ["pA", "pB", "pC"],
            "preference optimization without a reference model": ["pA", "pB", "pC"],
            "direct alignment algorithm reward shape": ["pX", "pY", "pZ"],
        }
    )
    deduper = SubqueryDeduper(indexer, probe_k=20, max_overlap=0.7, max_queries=4)
    kept = deduper.filter(
        [
            "reference-free preference optimization",
            "preference optimization without a reference model",
            "direct alignment algorithm reward shape",
        ]
    )

    assert kept == [
        "reference-free preference optimization",
        "direct alignment algorithm reward shape",
    ]


def test_dedup_compares_against_already_tried_subqueries():
    """すでに投げたクエリと同じ論文しか引かないなら、投げる意味がない。"""
    indexer = _StubBM25({"old": ["pA", "pB"], "new but same": ["pA", "pB"], "fresh": ["pZ"]})
    deduper = SubqueryDeduper(indexer, max_overlap=0.7)
    kept = deduper.filter(["new but same", "fresh"], already=["old"])

    assert kept == ["fresh"]


def test_dedup_falls_back_to_a_count_cap_without_a_bm25_index():
    """BM25 索引が無い構成では上限だけを効かせる（埋め込みや reranker は回さない）。"""
    deduper = SubqueryDeduper(None, max_queries=2)
    assert deduper.filter(["a", "b", "c"]) == ["a", "b"]


def test_dedup_is_off_unless_configured():
    """`subquery_dedup` を書かなければ None（既存構成の挙動を変えない）。"""
    assert SubqueryDeduper.from_retriever(_StubRetriever(), None) is None
    assert SubqueryDeduper.from_retriever(_StubRetriever(), {"method": "none"}) is None
    assert SubqueryDeduper.from_retriever(_StubRetriever(), {"probe_k": 5}) is not None


def test_dedup_runs_inside_the_agent_and_skips_the_expensive_search():
    """重複したサブクエリは retriever に一度も渡らない。"""
    indexer = _StubBM25({"sq": ["p0"], "same": ["p0"], "other": ["p9"]})
    retriever = _RetrieverWithBM25({"sq": [_result(0, "p0", 0.9)]}, indexer)
    llm = FakeLLM(
        responses=[_subqueries("sq", "same", "other"), _judge(["p0"], sufficient=True)]
    )
    agent = ReadingAgent(
        retriever,
        llm=llm,
        max_steps=1,
        retrieve_top_k=5,
        subquery_dedup={"probe_k": 20, "max_overlap": 0.7},
    )
    agent.run(_query())

    assert retriever.calls == ["sq", "other"]
