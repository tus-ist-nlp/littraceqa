"""Paper-level RRF (one vote per paper) and Seed Expansion."""

from __future__ import annotations


from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve import (
    HybridRetriever,
    PaperRRFFuser,
    SeedExpansion,
    paper_rrf_fuse,
    to_gold_papers,
)


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
        """A paper with many chunks does not rise on chunk count instead of rank.

        Fused per chunk, p_many collects 1/62 + 1/63 + 1/64 = 0.0473 and passes
        p_top's 1/61 = 0.0164 on sheer number of votes. Per paper it is one vote per
        run, so p_top, which appears first in the run, wins.
        """
        run = [
            result("p_top#c0", "p_top"),
            result("p_many#c0", "p_many"),
            result("p_many#c1", "p_many"),
            result("p_many#c2", "p_many"),
        ]
        # Chunk-level RRF for contrast; the production fuser is paper_rrf alone, so
        # this lives here rather than in the source.
        chunk_scores: dict[str, float] = {}
        for rank, r in enumerate(run):
            chunk_scores[r.paper_id] = chunk_scores.get(r.paper_id, 0.0) + 1 / (60 + rank + 1)
        assert chunk_scores["p_many"] > chunk_scores["p_top"]

        assert to_gold_papers(paper_rrf_fuse([run], top_k=10)) == ["p_top", "p_many"]

    def test_papers_in_both_runs_win(self):
        """A paper high in both runs beats one that is first in only one of them."""
        run_a = [result("only_a#c0", "only_a"), result("both#c0", "both")]
        run_b = [result("only_b#c0", "only_b", source="faiss"), result("both#c1", "both", source="faiss")]
        papers = to_gold_papers(paper_rrf_fuse([run_a, run_b], top_k=10))
        assert papers[0] == "both"

    def test_chunks_per_paper_caps_one_paper(self):
        run = [result(f"p#c{i}", "p") for i in range(10)]
        fused = paper_rrf_fuse([run], top_k=100, chunks_per_paper=3)
        assert len(fused) == 3
        # Within a paper the original order is kept, best first.
        assert [r.chunk_id for r in fused] == ["p#c0", "p#c1", "p#c2"]

    def test_within_paper_order_is_written_to_score(self):
        """Downstream re-sorts by score, so the returned list order is not enough."""
        run = [result(f"p#c{i}", "p") for i in range(3)]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=3)
        assert [r.score for r in fused] == sorted((r.score for r in fused), reverse=True)
        assert len({r.score for r in fused}) == 3

    def test_within_paper_offset_never_reorders_papers(self):
        """The within-paper offset never exceeds the gap between papers."""
        run = [result(f"p1#c{i}", "p1") for i in range(3)] + [result("p2#c0", "p2")]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=3)
        assert to_gold_papers(fused) == ["p1", "p2"]

    def test_paper_level_pseudo_chunk_is_not_a_representative(self):
        """`{paper_id}#paper` is not a real chunk_id, so it cannot serve as evidence."""
        run = [
            result("p#paper", "p", source="bm25s_paper", text="the whole paper"),
            result("p#c7", "p", source="bm25s"),
        ]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=1)
        assert [r.chunk_id for r in fused] == ["p#c7"]

    def test_paper_level_pseudo_chunk_is_kept_when_alone(self):
        """A paper with no real chunk stays a candidate; it is still ranked."""
        run = [result("p#paper", "p", source="bm25s_paper")]
        fused = paper_rrf_fuse([run], top_k=10, chunks_per_paper=1)
        assert [r.chunk_id for r in fused] == ["p#paper"]

    def test_ties_are_broken_deterministically(self):
        """The order does not change between runs — the accident bib_coupling hit."""
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


# The question the seed-expansion tests search with. **It has to be shorter than
# the anchor-vocabulary key ("reward shape")**: StubIndexer matches substrings
# longest key first, and the expanded query contains both, so the anchor vocabulary
# only wins while it is the longer of the two.
QUESTION = "the query"


class StubIndexer:
    """A stub whose results are fixed per query string."""

    name = "bm25s"

    def __init__(self, by_query):
        self.by_query = by_query
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        # The expanded query is "the question + the anchor's text", so it matches
        # both keys. The longer key (the more specific vocabulary) wins, so only the
        # expanded query gets the other results.
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
            {"chunk_id": f"{paper_id}#c0001", "chunk_type": "text_span", "text": "body"},
        ]


class TestSeedExpansion:
    def _retriever(self, **kwargs):
        indexer = StubIndexer(
            {
                # The question alone puts p_seed first and never reaches gold.
                QUESTION: [result("p_seed#c0", "p_seed"), result("p_other#c0", "p_other")],
                # Adding the anchor's vocabulary (reward shape) reaches gold.
                "reward shape": [result("p_gold#c0", "p_gold"), result("p_seed#c0", "p_seed")],
            }
        )
        return HybridRetriever(
            indexers=[indexer],
            fuser=PaperRRFFuser(),
            per_index_k=10,
            **kwargs,
        ), indexer

    def test_disabled_by_default_runs_one_search(self):
        retriever, indexer = self._retriever()
        papers = to_gold_papers(retriever.retrieve(QUESTION, 10))
        assert indexer.queries == [QUESTION]
        assert "p_gold" not in papers

    def test_expands_with_the_top_paper_vocabulary(self):
        retriever, indexer = self._retriever(
            seed_expansion=SeedExpansion(query_chars=512),
            anchor_store=StubStore({"p_seed": "[ICML 2025] Seed Paper\nWe study reward shape."}),
        )
        papers = to_gold_papers(retriever.retrieve(QUESTION, 10))
        assert len(indexer.queries) == 2
        # The second query is the question plus the anchor's title+abstract.
        assert indexer.queries[1].startswith(QUESTION)
        assert "reward shape" in indexer.queries[1]
        # The paper only the expansion reaches is now a candidate.
        assert "p_gold" in papers
        # The question's own top hit survives the fusion, being in both runs.
        assert papers[0] == "p_seed"

    def test_query_chars_truncates_the_anchor_text(self):
        retriever, indexer = self._retriever(
            seed_expansion=SeedExpansion(query_chars=10),
            anchor_store=StubStore({"p_seed": "0123456789reward shape"}),
        )
        retriever.retrieve(QUESTION, 10)
        assert "reward shape" not in indexer.queries[1]
        assert indexer.queries[1].endswith("0123456789")

    def test_falls_back_to_the_hit_chunk_without_a_store(self):
        retriever, indexer = self._retriever(seed_expansion=SeedExpansion())
        retriever.retrieve(QUESTION, 10)
        assert len(indexer.queries) == 2
        assert "title of p_seed" in indexer.queries[1]

    def test_no_op_when_the_first_search_is_empty(self):
        retriever, indexer = self._retriever(seed_expansion=SeedExpansion())
        assert retriever.retrieve("a query that matches nothing", 10) == []
        assert len(indexer.queries) == 1

    def test_reranker_runs_once_with_the_original_query(self):
        """Seed Expansion runs before the reranker, so inference count does not rise."""
        calls = []

        class StubReranker:
            def rerank(self, query, candidates, top_k):
                calls.append(query)
                return candidates[:top_k]

        retriever, _ = self._retriever(
            seed_expansion=SeedExpansion(),
            anchor_store=StubStore({"p_seed": "reward shape"}),
        )
        retriever.reranker = StubReranker()
        retriever.pool_k = 20
        retriever.retrieve(QUESTION, 10)
        assert calls == [QUESTION]


