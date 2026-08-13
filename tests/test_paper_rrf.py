"""論文単位 RRF（1論文1票）と Seed Expansion のテスト。"""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever, to_gold_papers
from littraceqa.di_pipeline.retrieve.paper_rrf import PaperRRFFuser, paper_rrf_fuse
from littraceqa.di_pipeline.retrieve.rrf import RRFFuser


def result(chunk_id, paper_id, source="bm25s", text="body", chunk_type="text_span"):
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=paper_id,
        score=0.0,
        text=text,
        chunk_type=chunk_type,
        metadata={"title": f"title of {paper_id}"},
        source=source,
    )


class TestPaperRRFFuse:
    def test_one_vote_per_paper_ignores_chunk_count(self):
        """チャンクを大量に持つ論文が、上位1本の論文を抜かないこと。

        チャンク単位 RRF では p_many が 1/61 + 1/62 + 1/63 = 0.0484 を集めて
        p_top の 1/61 = 0.0164 を抜いてしまう。論文単位なら p_top が勝つ。
        """
        run = [
            result("p_top#c0", "p_top"),
            result("p_many#c0", "p_many"),
            result("p_many#c1", "p_many"),
            result("p_many#c2", "p_many"),
        ]
        chunk_order = to_gold_papers(RRFFuser().fuse([run], top_k=10))
        paper_order = to_gold_papers(paper_rrf_fuse([run], top_k=10))
        # チャンク単位でも1位は p_top（1本の run では順位がそのまま出る）。
        assert chunk_order[0] == "p_top"
        assert paper_order == ["p_top", "p_many"]

    def test_papers_in_both_runs_win(self):
        """両方の run の上位に居る論文が、片方だけ1位の論文に勝つこと。"""
        run_a = [result("only_a#c0", "only_a"), result("both#c0", "both")]
        run_b = [result("only_b#c0", "only_b", source="faiss"), result("both#c1", "both", source="faiss")]
        papers = to_gold_papers(paper_rrf_fuse([run_a, run_b], top_k=10))
        assert papers[0] == "both"

    def test_chunks_per_paper_caps_one_paper(self):
        run = [result(f"p#c{i}", "p") for i in range(10)]
        fused = paper_rrf_fuse([run], top_k=100, chunks_per_paper=3)
        assert len(fused) == 3
        # 論文内の順位は元の順位を保つ（先頭が最上位）。
        assert [r.chunk_id for r in fused] == ["p#c0", "p#c1", "p#c2"]

    def test_within_paper_order_is_written_to_score(self):
        """下流は score で並べ直すので、返り値の並び順だけでは足りない。"""
        run = [result(f"p#c{i}", "p") for i in range(3)]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=3)
        assert [r.score for r in fused] == sorted((r.score for r in fused), reverse=True)
        assert len({r.score for r in fused}) == 3

    def test_within_paper_offset_never_reorders_papers(self):
        """論文内オフセットが論文間のスコア差を超えないこと。"""
        run = [result(f"p1#c{i}", "p1") for i in range(3)] + [result("p2#c0", "p2")]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=3)
        assert to_gold_papers(fused) == ["p1", "p2"]

    def test_paper_level_pseudo_chunk_is_not_a_representative(self):
        """`{paper_id}#paper` は chunk_id が実在しないので evidence に使えない。"""
        run = [
            result("p#paper", "p", source="bm25s_paper", text="論文全文"),
            result("p#c7", "p", source="bm25s"),
        ]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=1)
        assert [r.chunk_id for r in fused] == ["p#c7"]

    def test_paper_level_pseudo_chunk_is_kept_when_alone(self):
        """実チャンクが無い論文を候補から消してはいけない（順位付けには使う）。"""
        run = [result("p#paper", "p", source="bm25s_paper")]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=1)
        assert [r.chunk_id for r in fused] == ["p#paper"]

    def test_source_weights_apply_to_paper_votes(self):
        run_a = [result("a#c0", "a", source="bm25s")]
        run_b = [result("b#c0", "b", source="faiss")]
        fused = paper_rrf_fuse([run_a, run_b], top_k=10, weights={"bm25s": 2.0, "faiss": 1.0})
        assert to_gold_papers(fused) == ["a", "b"]

    def test_ties_are_broken_deterministically(self):
        """実行のたびに並びが変わらないこと（bib_coupling で踏んだ事故の予防）。"""
        run = [result("b#c0", "b"), result("a#c0", "a")]
        first = paper_rrf_fuse([run, run], top_k=10)
        second = paper_rrf_fuse([run, run], top_k=10)
        assert [r.chunk_id for r in first] == [r.chunk_id for r in second]

    def test_top_k_truncates(self):
        run = [result(f"p{i}#c0", f"p{i}") for i in range(10)]
        assert len(paper_rrf_fuse([run], top_k=4)) == 4

    def test_empty_runs(self):
        assert paper_rrf_fuse([], top_k=10) == []
        assert paper_rrf_fuse([[]], top_k=10) == []

    def test_usable_as_a_fuser(self):
        run = [result("p#c0", "p")]
        assert PaperRRFFuser(k=60).fuse([run], top_k=5)[0].chunk_id == "p#c0"


class StubIndexer:
    """クエリ文字列ごとに返す結果を決め打ちするスタブ。"""

    name = "bm25s"

    def __init__(self, by_query):
        self.by_query = by_query
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        # 拡張クエリは「元の質問 + anchor 本文」なので両方のキーに当たる。
        # 長いキー（= より具体的な語彙）を優先して、拡張後だけ別の結果を返す。
        for key in sorted(self.by_query, key=len, reverse=True):
            if key in query:
                return self.by_query[key][:top_k]
        return []

    def load(self):
        pass


class StubStore:
    def __init__(self, texts):
        self.texts = texts

    def load_paper(self, paper_id):
        return [
            {"chunk_id": f"{paper_id}#c0000", "chunk_type": "title_abstract", "text": self.texts[paper_id]},
            {"chunk_id": f"{paper_id}#c0001", "chunk_type": "text_span", "text": "本文"},
        ]


class TestSeedExpansion:
    def _retriever(self, **kwargs):
        indexer = StubIndexer(
            {
                # 元の質問では p_seed が1位、gold は引けない。
                "元の質問": [result("p_seed#c0", "p_seed"), result("p_other#c0", "p_other")],
                # anchor の語彙（reward shape）を足すと gold が引ける。
                "reward shape": [result("p_gold#c0", "p_gold"), result("p_seed#c0", "p_seed")],
            }
        )
        return HybridRetriever(
            indexers=[indexer],
            fuser=RRFFuser(),
            per_index_k=10,
            **kwargs,
        ), indexer

    def test_disabled_by_default_runs_one_search(self):
        retriever, indexer = self._retriever()
        papers = to_gold_papers(retriever.retrieve("元の質問", 10))
        assert indexer.queries == ["元の質問"]
        assert "p_gold" not in papers

    def test_expands_with_the_top_paper_vocabulary(self):
        retriever, indexer = self._retriever(
            seed_expansion={"enabled": True, "query_chars": 512},
            anchor_store=StubStore({"p_seed": "[ICML 2025] Seed Paper\nWe study reward shape."}),
        )
        papers = to_gold_papers(retriever.retrieve("元の質問", 10))
        assert len(indexer.queries) == 2
        # 2本目のクエリは「元の質問 + anchor の title+abstract」。
        assert indexer.queries[1].startswith("元の質問")
        assert "reward shape" in indexer.queries[1]
        # 拡張でしか引けない論文が候補に入る。
        assert "p_gold" in papers
        # 元の質問の1位は融合後も残る（両方の run に居るので上位）。
        assert papers[0] == "p_seed"

    def test_query_chars_truncates_the_anchor_text(self):
        retriever, indexer = self._retriever(
            seed_expansion={"enabled": True, "query_chars": 10},
            anchor_store=StubStore({"p_seed": "0123456789reward shape"}),
        )
        retriever.retrieve("元の質問", 10)
        assert "reward shape" not in indexer.queries[1]
        assert indexer.queries[1].endswith("0123456789")

    def test_falls_back_to_the_hit_chunk_without_a_store(self):
        retriever, indexer = self._retriever(seed_expansion={"enabled": True})
        retriever.retrieve("元の質問", 10)
        assert len(indexer.queries) == 2
        assert "title of p_seed" in indexer.queries[1]

    def test_no_op_when_the_first_search_is_empty(self):
        retriever, indexer = self._retriever(seed_expansion={"enabled": True})
        assert retriever.retrieve("当たらない質問", 10) == []
        assert len(indexer.queries) == 1

    def test_reranker_runs_once_with_the_original_query(self):
        """Seed Expansion は reranker の前に置くので推論回数が増えないこと。"""
        calls = []

        class StubReranker:
            def rerank(self, query, candidates, top_k):
                calls.append(query)
                return candidates[:top_k]

        retriever, _ = self._retriever(
            seed_expansion={"enabled": True},
            anchor_store=StubStore({"p_seed": "reward shape"}),
        )
        retriever.reranker = StubReranker()
        retriever.pool_k = 20
        retriever.retrieve("元の質問", 10)
        assert calls == ["元の質問"]


class TestComposeConfig:
    def test_compose_without_seed_expansion(self):
        from littraceqa.di_pipeline.config import compose_config

        cfg = compose_config(
            paths={"pdf_dir": "/p", "chunks_dir": "/c", "index_dir": "/i", "paper_metadata": "/m.jsonl"},
            process={"name": "mineru", "params": {}},
            search={
                "per_index_k": 100,
                "indexers": [{"name": "bm25s", "params": {}}],
                "fuser": {"name": "rrf", "params": {}},
                "reranker": {"name": "none", "params": {}},
            },
            agent={"name": "reading", "params": {}},
        )
        assert cfg["retriever"]["seed_expansion"] is None

    def test_compose_passes_seed_expansion(self):
        from littraceqa.di_pipeline.config import compose_config

        cfg = compose_config(
            paths={"pdf_dir": "/p", "chunks_dir": "/c", "index_dir": "/i", "paper_metadata": "/m.jsonl"},
            process={"name": "mineru", "params": {}},
            search={
                "per_index_k": 100,
                "indexers": [{"name": "bm25s", "params": {}}],
                "fuser": {"name": "paper_rrf", "params": {"chunks_per_paper": 3}},
                "reranker": {"name": "none", "params": {}},
                "seed_expansion": {"enabled": True, "query_chars": 512},
            },
            agent={"name": "reading", "params": {}},
        )
        assert cfg["retriever"]["seed_expansion"]["query_chars"] == 512
        assert cfg["retriever"]["fuser"]["name"] == "paper_rrf"


@pytest.mark.parametrize(
    "path",
    [
        "configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_paperbm25.yaml",
        "configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_paperrrf.yaml",
        "configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_seed.yaml",
        "configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/k100_paperrrf_seed.yaml",
    ],
)
def test_new_search_styles_load(path):
    from littraceqa.common import ROOT
    from littraceqa.di_pipeline.config import load_config

    cfg = load_config(str(ROOT / path))
    assert cfg["per_index_k"] == 100
    assert cfg["fuser"]["name"] in ("rrf", "paper_rrf")
    # 索引パスは compose_config が導出するので yaml には書かない。
    for indexer in cfg["indexers"]:
        assert "index_dir" not in indexer.get("params", {})
