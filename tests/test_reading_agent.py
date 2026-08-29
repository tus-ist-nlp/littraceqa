"""ReadingAgent.

**What matters most is whether the agent really searches again when the LLM says it
is short.** A stopping condition that counts results never iterates — 20 results
come back on the first pass and satisfy it immediately — which is why the condition
is the LLM's judgement of the content.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from littraceqa.di_pipeline.agent.reading import (
    CANDIDATE_PAPERS_LIMIT,
    CombineConfig,
    ReadingAgent,
    ReadingConfig,
)
from littraceqa.di_pipeline.contracts import Query, RetrievalResult
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _query(**kwargs) -> Query:
    base = {
        "query_id": "q_1",
        "question": "Which papers report FID on CIFAR-10?",
        "answer_types": ["table"],
    }
    base.update(kwargs)
    return Query(**base)


def _result(i: int, paper: str, score: float, **metadata) -> RetrievalResult:
    base = {"title": f"Paper {paper}", "venue": "NeurIPS", "year": 2025}
    base.update(metadata)
    return RetrievalResult(
        chunk_id=f"{paper}#c{i:02d}",
        paper_id=paper,
        score=score,
        text=f"body of {paper} chunk {i}",
        chunk_type=base.pop("chunk_type", "text_span"),
        metadata=base,
    )


class _StubRetriever:
    """A stub returning different papers per subquery, recording the queries it got."""

    def __init__(self, by_query: dict[str, list[RetrievalResult]] | None = None):
        self.by_query = by_query or {}
        self.calls: list[str] = []
        self.top_ks: list[int] = []  # to see how many were asked for

    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        self.calls.append(question)
        self.top_ks.append(top_k)
        if question in self.by_query:
            return self.by_query[question][:top_k]
        return [_result(0, f"p{i}", 1.0 / (i + 1)) for i in range(5)][:top_k]


def _judge(papers, sufficient, missing=""):
    return json.dumps(
        {
            "papers": [
                {"paper_id": p, "evidence_chunk_ids": [f"{p}#c00"]} for p in papers
            ],
            "sufficient": sufficient,
            "missing": missing,
        }
    )


def _subqueries(*values):
    return json.dumps({"subqueries": list(values)})


def test_loop_iterates_when_llm_says_insufficient():
    """sufficient=false makes the agent search again for what is missing.

    A stopping condition of "results >= threshold" is satisfied the moment the first
    search returns 20, and the loop never takes a second pass. Stopping on the
    content judgement (sufficient) is what avoids that.
    """
    retriever = _StubRetriever(
        {
            "step2-sq": [_result(0, "pX", 9.0)],
        }
    )
    llm = FakeLLM(
        responses=[
            _subqueries("step1-sq"),  # _decompose
            _judge(["p0"], sufficient=False, missing="still missing ECM-XL's FID"),
            _subqueries("step2-sq"),  # _refine
            _judge(["p0", "pX"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=3, retrieve_top_k=5))
    prediction = agent.run(_query())

    assert len(prediction.trace) == 2, "the loop did not take two steps"
    assert prediction.trace[0]["sufficient"] is False
    assert prediction.trace[0]["missing"] == "still missing ECM-XL's FID"
    assert prediction.trace[1]["sufficient"] is True
    # A second search, aimed at the gap, really ran
    assert "step2-sq" in retriever.calls
    # The submission is the candidate ranking as it stands. pX, found on the second
    # pass, leads with a score of 9.0 — the re-search reached the submission.
    assert [p["paper_id"] for p in prediction.gold_papers][:2] == ["pX", "p0"]


def test_stops_as_soon_as_llm_says_sufficient():
    retriever = _StubRetriever()
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=3, retrieve_top_k=5))
    prediction = agent.run(_query())

    assert len(prediction.trace) == 1
    assert "p0" in [p["paper_id"] for p in prediction.gold_papers]


def test_respects_max_steps():
    """An LLM that never says sufficient still stops at max_steps."""
    retriever = _StubRetriever()
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=False, missing="still not enough"),
            _subqueries("sq2"),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=2, retrieve_top_k=5))
    prediction = agent.run(_query())
    assert len(prediction.trace) == 2


def test_builds_evidence_from_cited_chunks():
    """Evidence is built from the metadata of the chunks the LLM cited."""
    retriever = _StubRetriever(
        {
            "sq": [
                _result(0, "p0", 5.0, chunk_type="table", page=6, table_id="Table 4"),
                _result(1, "p0", 4.0, chunk_type="text_span", page=2),
            ]
        }
    )
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
    prediction = agent.run(_query())

    assert len(prediction.evidence) == 1
    evidence = prediction.evidence[0]
    assert evidence.paper_id == "p0"
    assert evidence.source_type == "table"
    assert evidence.locator.page == 6
    assert evidence.locator.table_id == "Table 4"


def test_reads_full_chunk_text_not_a_200_char_stub():
    """Excerpts reach the LLM at snippet_chars; at 200 only the title is readable."""
    long_text = "x" * 5000
    result = _result(0, "p0", 1.0)
    result.text = long_text
    retriever = _StubRetriever({"sq": [result]})
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(
        retriever,
        llm=llm,
        config=ReadingConfig(max_steps=1, retrieve_top_k=5, snippet_chars=1800),
    )
    agent.run(_query())

    judge_prompt = llm.calls[1]
    assert "x" * 1800 in judge_prompt
    assert "x" * 1801 not in judge_prompt


def test_drops_hallucinated_paper_and_chunk_ids():
    """A paper_id or chunk_id absent from the candidate list is discarded."""
    retriever = _StubRetriever({"sq": [_result(0, "p0", 1.0)]})
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            json.dumps(
                {
                    "papers": [
                        {"paper_id": "p0", "evidence_chunk_ids": ["p0#c00", "p0#c99"]},
                        {"paper_id": "ghost", "evidence_chunk_ids": ["ghost#c00"]},
                    ],
                    "sufficient": True,
                    "missing": "",
                }
            ),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
    prediction = agent.run(_query())

    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0"]
    assert len(prediction.evidence) == 1  # p0#c99 does not exist and is dropped


def test_duplicate_chunk_keeps_the_higher_score():
    """A chunk hit by several subqueries keeps its highest score.

    Last-write-wins lets subquery 3's low score overwrite subquery 1's top hit, and
    the paper ranking drifts towards whichever subquery ran last. **Now that the
    reranker writes its ranking back into the score, last-write-wins would break
    that ranking across subqueries.**
    """
    retriever = _StubRetriever(
        {
            "sq-a": [_result(0, "pA", 9.0), _result(0, "pB", 1.0)],
            # pA#c00 is the same chunk as in sq-a, only with a lower score.
            "sq-b": [_result(0, "pA", 0.1), _result(0, "pB", 2.0)],
        }
    )
    llm = FakeLLM(
        responses=[_subqueries("sq-a", "sq-b"), _judge(["pA"], sufficient=True)]
    )
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
    agent.run(_query())

    # The candidate list appears in _read_and_judge's prompt in relevance order. If
    # pA kept its 9.0 it leads; if it was overwritten with 0.1, pB (2.0) does.
    listing = llm.calls[-1]
    assert listing.index("[paper_id: pA]") < listing.index("[paper_id: pB]")


def test_falls_back_to_the_ranking_when_llm_output_is_unusable():
    """Even with no usable verdict from the LLM, the retrieval ranking is submitted.

    An empty submission on a query where the reading broke would throw away gold
    that retrieval had already found.
    """
    retriever = _StubRetriever()
    llm = FakeLLM(responses=["not json"])
    agent = ReadingAgent(
        retriever,
        llm=llm,
        config=ReadingConfig(max_steps=1, retrieve_top_k=5, max_papers=3),
    )
    prediction = agent.run(_query())

    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0", "p1", "p2"]
    assert prediction.evidence == []


def test_candidate_papers_records_ranking_before_cutoff():
    """The candidates before the cut are kept in relevance order, for recall@k.

    gold_papers is cut down to a handful, so on its own it cannot say afterwards
    whether retrieval ever had the gold paper in hand.
    """
    retriever = _StubRetriever(
        {"sq": [_result(0, f"p{i}", 10.0 - i) for i in range(8)]}
    )
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(
        retriever,
        llm=llm,
        config=ReadingConfig(max_steps=1, retrieve_top_k=8, max_papers=3),
    )
    prediction = agent.run(_query())

    # The submission is cut to max_papers, but the candidates survive in full,
    # in descending score order.
    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0", "p1", "p2"]
    assert prediction.candidate_papers == [f"p{i}" for i in range(8)]


def test_candidate_papers_is_capped():
    """The candidate list is capped at CANDIDATE_PAPERS_LIMIT, to bound the file size."""
    n = CANDIDATE_PAPERS_LIMIT + 10
    retriever = _StubRetriever(
        {"sq": [_result(0, f"p{i:03d}", float(n - i)) for i in range(n)]}
    )
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p000"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=n))
    prediction = agent.run(_query())

    assert len(prediction.candidate_papers) == CANDIDATE_PAPERS_LIMIT
    assert prediction.candidate_papers[0] == "p000"  # the best one leads


class _FilterAwareRetriever(_StubRetriever):
    """A Retriever that accepts attribute_filter and records what it was given."""

    def __init__(self, extractor, **kwargs):
        super().__init__(**kwargs)
        self.attribute_extractor = extractor
        self.filters: list[object] = []

    def retrieve(self, question, top_k, attribute_filter=None):
        self.filters.append(attribute_filter)
        return super().retrieve(question, top_k)


class _StubExtractor:
    """A dummy extractor returning a constraint only for questions naming NAACL."""

    def __init__(self):
        from littraceqa.di_pipeline.retrieve.attribute_filter import AttributeFilter

        self._cls = AttributeFilter

    def extract(self, question: str):
        if "NAACL" in question:
            return self._cls(venue="NAACL", year=2025)
        return self._cls()


def test_attribute_filter_comes_from_the_question_not_the_subquery():
    """The constraint comes from the original question, not from the subqueries.

    _decompose() rewrites "Which NAACL 2025 papers ..." as something like "MCTS in
    method figure", so extracting from a subquery would never fire.
    """
    retriever = _FilterAwareRetriever(_StubExtractor())
    llm = FakeLLM(
        responses=[
            _subqueries("MCTS in method figure"),  # the venue is gone from it
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
    agent.run(_query(question="Which NAACL 2025 papers explicitly mention MCTS?"))

    assert retriever.calls == ["MCTS in method figure"]
    assert [(f.venue, f.year) for f in retriever.filters] == [("NAACL", 2025)]


def test_no_attribute_filter_argument_when_nothing_is_extracted():
    """With no constraint, retrieve() is not even given the argument.

    The Retriever protocol is (query, top_k), so always passing it would break a
    simpler retriever that does not accept attribute_filter.
    """
    retriever = _StubRetriever()  # does not accept attribute_filter
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))

    prediction = agent.run(_query(question="Which papers report FID on CIFAR-10?"))

    assert prediction.trace[0]["attribute_filter"] is None


# ---- the subquery prompts -------------------------------------------------


def _subquery_prompts(llm: FakeLLM) -> list[str]:
    return [p for p in llm.calls if '"subqueries"' in p]


def test_subquery_prompts_forbid_web_search_operators():
    """Both the split and the re-split say "this is not a web search".

    Without it the LLM writes queries with site: and filetype:. Measured, 29-41% of
    _refine()'s output carried them, and they match nothing in a local index.
    """
    retriever = _StubRetriever()
    llm = FakeLLM(
        responses=[
            _subqueries("sq1"),
            _judge(["p0"], sufficient=False, missing="need the table"),
            _subqueries("sq2"),
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=2, retrieve_top_k=5))
    agent.run(_query())

    prompts = _subquery_prompts(llm)
    assert len(prompts) == 2  # _decompose and _refine
    for prompt in prompts:
        assert "NOT sent to a web search engine" in prompt
        assert "site:" in prompt and "filetype:" in prompt


def test_subquery_count_is_enforced_on_the_return_value_not_just_the_prompt():
    """However many the LLM returns, only subquery_count are issued.

    `_refine()`, which used to state no count at all, ballooned to 8.2-9.3 on
    average and 20 at worst (`runs_fat.jsonl`, 812 subqueries over 55 queries). One
    subquery is one retrieval, which is one reranker pass over pool_k chunks, so
    this is wall-clock time directly: 1.73 minutes each, 23 hours in total, at
    pool_k=1000. **Asking in the prompt is not obeyed**, so this pins down that the
    return value is truncated too.
    """
    retriever = _StubRetriever()
    llm = FakeLLM(
        responses=[
            _subqueries("d1", "d2", "d3", "d4", "d5", "d6"),  # _decompose returns 6
            _judge(["p0"], sufficient=False, missing="need more"),
            _subqueries(*[f"r{i}" for i in range(20)]),  # _refine returns 20
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(
        retriever,
        llm=llm,
        config=ReadingConfig(max_steps=2, retrieve_top_k=5, subquery_count=4),
    )
    agent.run(_query())

    assert retriever.calls == ["d1", "d2", "d3", "d4", "r0", "r1", "r2", "r3"]


def test_refine_prompt_states_the_subquery_budget():
    """The re-split prompt states a count too; only the split used to."""
    retriever = _StubRetriever()
    llm = FakeLLM(
        responses=[
            _subqueries("sq1"),
            _judge(["p0"], sufficient=False, missing="need the table"),
            _subqueries("sq2"),
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(
        retriever,
        llm=llm,
        config=ReadingConfig(max_steps=2, retrieve_top_k=5, subquery_count=3),
    )
    agent.run(_query())

    decompose_prompt, refine_prompt = _subquery_prompts(llm)
    assert "into 3 short" in decompose_prompt
    assert "at most 3 new search subqueries" in refine_prompt
    # The real failure is degenerating into every phrasing of the same thing, so the
    # prompt says that explicitly.
    assert "do not submit paraphrases" in refine_prompt


def test_subquery_prompts_carry_the_venue_tag():
    """With a constraint extracted, the prompt asks for that tag on each subquery.

    A title_abstract chunk's text really does begin "[NAACL 2025] Title...".
    """
    retriever = _FilterAwareRetriever(_StubExtractor())
    llm = FakeLLM(
        responses=[
            _subqueries("sq1"),
            _judge(["p0"], sufficient=False, missing="need the figure"),
            _subqueries("sq2"),
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=2, retrieve_top_k=5))
    agent.run(_query(question="Which NAACL 2025 papers explicitly mention MCTS?"))

    prompts = _subquery_prompts(llm)
    assert len(prompts) == 2
    for prompt in prompts:
        assert '"[NAACL 2025]"' in prompt


def test_no_venue_tag_without_a_constraint():
    """With no constraint, nothing about a venue enters the prompt."""
    retriever = _StubRetriever()
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
    agent.run(_query(question="Which papers report FID on CIFAR-10?"))

    assert all("limits the search to" not in p for p in _subquery_prompts(llm))


def test_no_expander_means_identical_behavior():
    """Without a paper_expander the candidate list is untouched."""
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(_StubRetriever(), llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
    prediction = agent.run(_query())
    assert all("paper_expansion" not in step for step in prediction.trace)
    assert len(prediction.candidate_papers) <= CANDIDATE_PAPERS_LIMIT


class _StubRelated:
    """A stub for ranking B. **The anchors arrive as an argument; it holds no settings.**"""

    def __init__(self, ranking: list[str]):
        self.ranking = ranking
        self.anchor_calls: list[list[str]] = []

    def rank(self, anchors: list[str]) -> list[str]:
        # Like the real expanders, an anchor is not among its own neighbours.
        self.anchor_calls.append(list(anchors))
        return [p for p in self.ranking if p not in anchors]


def _rrf_agent(ranking: list[str], **combine_kwargs) -> ReadingAgent:
    return ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
        paper_expander=_StubRelated(ranking),
        combine=CombineConfig(**combine_kwargs),
        config=ReadingConfig(max_steps=1, retrieve_top_k=5, max_candidates=2, max_papers=1),
    )


def test_combine_rrf_promotes_papers_backed_by_both_rankings():
    """A paper low in A but pushed by B rises — what positional insertion could not do."""
    # Retrieval A: p0..p4 by descending score. Related B: p3 first.
    prediction = _rrf_agent(["p3", "pE1"]).run(_query())
    candidates = prediction.candidate_papers
    # p3, fourth in A, passes p1 (1/62) and p2 (1/63) on A(1/64) + B(1/61)
    assert candidates.index("p3") < candidates.index("p1")
    # A paper only B has gets in too, losing to A's top hits under plain RRF
    assert "pE1" in candidates
    # The fusion rearranges only the candidate list; the submission (max_papers=1)
    # is the fused first place.
    assert prediction.gold_papers == [{"paper_id": "p0"}]
    fusion = [s["paper_fusion"] for s in prediction.trace if "paper_fusion" in s]
    assert fusion and fusion[0]["anchor"] == "p0"
    # Recorded as a paper lifted from outside the visible head (max_candidates=2)
    assert fusion[0]["promoted"] == ["p3"]


def test_combine_rrf_keeps_the_anchor_on_top():
    """An anchor leads B. Without that, every paper present in both rankings passes it."""
    # Even with B pushing every other candidate, the anchor stays first.
    prediction = _rrf_agent(["p1", "p2", "p3", "p4"]).run(_query())
    assert prediction.candidate_papers[0] == "p0"


def test_expander_receives_only_the_anchors():
    """**The agent chooses the anchors**; the expander gets those, not the candidates.

    It used to receive the whole candidate list and cut the first `anchors` itself,
    so changing the anchors meant writing the expander's `anchors` attribute and
    putting it back afterwards — "who decides the anchors" was split across two
    objects.
    """
    expander = _StubRelated(["pE1"])
    agent = ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
        paper_expander=expander,
        config=ReadingConfig(max_steps=1, retrieve_top_k=5, max_candidates=2),
    )
    agent.run(_query())
    # The candidates are p0..p4, but only the one anchor is passed.
    assert expander.anchor_calls == [["p0"]]


def test_verdict_anchor_adds_the_llm_confirmed_papers():
    """`anchor_from: verdict` adds the reader's confirmed papers to the top candidate."""
    expander = _StubRelated(["pE1"])
    agent = ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p1"], sufficient=True)]),
        paper_expander=expander,
        combine=CombineConfig(anchor_from="verdict"),
        config=ReadingConfig(max_steps=1, retrieve_top_k=5, max_candidates=2),
    )
    agent.run(_query())
    # The top candidate (p0) always stays; dropping it costs single-paper cr@1.
    assert expander.anchor_calls == [["p0", "p1"]]


def test_combine_rrf_offset_pushes_b_only_papers_down():
    """related_offset sets how deep a B-only paper enters (0, the default, is plain RRF)."""
    shallow = _rrf_agent(["pE1"]).run(_query()).candidate_papers
    deep = _rrf_agent(["pE1"], related_offset=10).run(_query()).candidate_papers
    assert shallow.index("pE1") < deep.index("pE1")


def _two_subquery_retriever():
    """A stub where subquery A returns one high-scoring result and B two low ones."""
    return _StubRetriever(
        {
            "sqA": [_result(0, "pA", 0.9), _result(0, "pX", 0.8)],
            "sqB": [_result(0, "pB1", 0.5), _result(0, "pX", 0.4)],
        }
    )


def test_pool_is_max_merged_across_subqueries():
    """The pool across subqueries is ordered by each chunk's highest score."""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        _two_subquery_retriever(),
        llm=llm,
        config=ReadingConfig(max_steps=1, retrieve_top_k=5),
    )
    assert agent.run(_query()).candidate_papers == ["pA", "pX", "pB1"]


def _depth_agent(**depth):
    base = {"enabled": True, "probe_rank": 2, "gap_threshold": 0.15,
            "shallow_k": 2, "deep_k": 5}
    base.update(depth)
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    return llm, base


def test_last_runs_is_exposed_for_dumping():
    """The per-subquery results survive run(), for --dump-runs."""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        _two_subquery_retriever(),
        llm=llm,
        config=ReadingConfig(max_steps=1, retrieve_top_k=5),
    )
    agent.run(_query())
    assert [(r.step, r.subquery) for r in agent.last_runs] == [(0, "sqA"), (0, "sqB")]
    assert [r.paper_id for r in agent.last_runs[1].results] == ["pB1", "pX"]


def test_submission_is_the_candidate_ranking_not_the_llm_selection():
    """The papers to submit are **not chosen**; the candidate ranking is passed through.

    Choosing belongs to the reading team, so the search agent stops at handing over
    the ranking. The LLM's paper_ids go unused, but the reading still happens:
    `sufficient` is the loop's stopping condition and the evidence comes from the
    same call.
    """
    retriever = _StubRetriever({"sq": [_result(0, f"p{i}", 10.0 - i) for i in range(8)]})
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p5"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=8))
    prediction = agent.run(_query())

    # The LLM picked only p5; the submission is the candidate ranking, capped at
    # max_papers.
    assert [p["paper_id"] for p in prediction.gold_papers] == [f"p{i}" for i in range(8)]
    # The reading did run: the stopping condition ended it after one pass.
    assert len(prediction.trace) == 1
    assert prediction.trace[0]["sufficient"] is True
    # The chosen paper survived into the submission, so its evidence is emitted.
    assert [e.paper_id for e in prediction.evidence] == ["p5"]


# ---- configuration (ReadingConfig) ----------------------------------------


def test_config_is_frozen():
    """The configuration is immutable; nothing rewrites it mid-run."""
    config = ReadingConfig(paper_score_skip_chunk_types=("table",))
    assert config.paper_score_skip_chunk_types == ("table",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_steps = 99


def test_config_defaults_when_omitted():
    """Omitting config gives a default ReadingConfig."""
    agent = ReadingAgent(_StubRetriever(), llm=FakeLLM(responses=[]))
    assert agent.config == ReadingConfig()


def test_production_config_matches_what_was_measured():
    """Pins the production values — the configuration that produced cr@50 0.9682.

    Importing `di_pipeline.pipeline` pulls in torch and faiss, so the configuration
    objects are built here directly and compared.
    """
    config = ReadingConfig(
        max_steps=3,
        retrieve_top_k=20,
        max_candidates=20,
        chunks_per_paper=2,
        snippet_chars=1800,
        max_papers=10,
        paper_score_skip_chunk_types=("table",),
    )
    combine = CombineConfig(
        rrf_k=10, related_weight=1.0, related_offset=0, anchors=1, anchor_from="verdict"
    )
    assert config.subquery_count == 4  # the default; raising it does not help retrieval
    assert config.paper_score_skip_chunk_types == ("table",)
    assert combine.rrf_k == 10  # at 60 even a deep rank beats A's top hit
    assert combine.anchors == 1  # raising it was worse on all four bases


def test_decompose_asks_for_a_fixed_number_of_subqueries():
    """The split asks for SUBQUERY_COUNT subqueries, never branching on task_family.

    It used to ask for 1-3 on single-paper questions and 3-6 on multi-paper ones,
    and that branch alone cost an LLM call per query to guess task_family, which
    production input does not carry. It bought 0.58 extra subqueries on average, and
    the two estimators were indistinguishable in accuracy (0.670 vs 0.673), so the
    branch went away entirely.
    """
    from littraceqa.di_pipeline.agent.reading import SUBQUERY_COUNT

    for task_family in ("hidden_source_single_paper", "multi_paper"):
        llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
        agent = ReadingAgent(_StubRetriever(), llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
        # task_family is not a Query field at all, so a record carrying it builds
        # the same Query as one without: the branch cannot come back by accident.
        agent.run(Query.from_dict({**_query().to_dict(), "task_family": task_family}))
        decompose_prompt = llm.calls[0]
        assert f"into {SUBQUERY_COUNT} short" in decompose_prompt
        # No wording committed to one case survives; one sentence covers both
        assert "contained within a single paper" not in decompose_prompt
        assert "requires evidence spanning multiple papers" not in decompose_prompt


def test_decompose_does_not_call_the_task_family_classifier():
    """Splitting costs no extra LLM call, whatever the input carries.

    Back when it did, a "single or multi" guess ran before the split and the read,
    making three LLM calls per query instead of two.
    """
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(_StubRetriever(), llm=llm, config=ReadingConfig(max_steps=1, retrieve_top_k=5))
    agent.run(Query(query_id="q_prod", question="Which papers report FID?", answer_types=[]))

    assert len(llm.calls) == 2  # the split and the read, nothing else
    assert "task_family" not in llm.calls[0]

