"""ReadingAgent のテスト。

一番大事なのは「LLM が足りないと言ったら本当に再検索するか」。停止条件を本数カウントに
すると反復が回らない（初回で20件返った時点で常に条件を満たす）ため、ここを内容判定にした。
"""

from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.agent.reading import CANDIDATE_PAPERS_LIMIT, ReadingAgent
from littraceqa.di_pipeline.contracts import Query, RetrievalResult
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _query(**kwargs) -> Query:
    # task_family を明示して TaskFamilyClassifier が LLM を呼ばないようにする。
    # そうしないと分類用のプロンプトが FakeLLM の応答を1つ消費してしまい、
    # エージェント本体のロジックを見るテストにならない。
    base = {
        "query_id": "q_1",
        "question": "Which papers report FID on CIFAR-10?",
        "answer_types": ["table"],
        "task_family": "multi_paper",
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
    """サブクエリごとに違う論文を返すスタブ。呼ばれたクエリを記録する。"""

    def __init__(self, by_query: dict[str, list[RetrievalResult]] | None = None):
        self.by_query = by_query or {}
        self.calls: list[str] = []
        self.top_ks: list[int] = []  # adaptive_depth が要求した件数を見るため

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
    """LLM が sufficient=false を返したら、不足分で再検索して2周目に入る。

    停止条件を「本数 >= しきい値」にすると、初回検索で20件返った時点で常に打ち切られ、
    再検索が一度も走らない。それを避けるため内容判定（sufficient）で止める。
    """
    retriever = _StubRetriever(
        {
            "step2-sq": [_result(0, "pX", 9.0)],
        }
    )
    llm = FakeLLM(
        responses=[
            _subqueries("step1-sq"),  # _decompose
            _judge(["p0"], sufficient=False, missing="ECM-XL の FID がまだ無い"),
            _subqueries("step2-sq"),  # _refine
            _judge(["p0", "pX"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=3, retrieve_top_k=5)
    prediction = agent.run(_query())

    assert len(prediction.trace) == 2, "2ステップ回っていない"
    assert prediction.trace[0]["sufficient"] is False
    assert prediction.trace[0]["missing"] == "ECM-XL の FID がまだ無い"
    assert prediction.trace[1]["sufficient"] is True
    # 不足を埋めるための再検索が実際に走っている
    assert "step2-sq" in retriever.calls
    # 提出は候補列の順位そのまま（submit_from の既定）。2周目で拾った pX が
    # スコア9.0で先頭に来ている = 再検索の結果が提出に反映されている。
    assert [p["paper_id"] for p in prediction.gold_papers][:2] == ["pX", "p0"]


def test_stops_as_soon_as_llm_says_sufficient():
    retriever = _StubRetriever()
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=3, retrieve_top_k=5)
    prediction = agent.run(_query())

    assert len(prediction.trace) == 1
    assert "p0" in [p["paper_id"] for p in prediction.gold_papers]


def test_respects_max_steps():
    """LLM がずっと sufficient=false でも max_steps で止まる。"""
    retriever = _StubRetriever()
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=False, missing="まだ足りない"),
            _subqueries("sq2"),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=2, retrieve_top_k=5)
    prediction = agent.run(_query())
    assert len(prediction.trace) == 2


def test_builds_evidence_from_cited_chunks():
    """LLM が根拠として挙げたチャンクの metadata から Evidence を組む。"""
    retriever = _StubRetriever(
        {
            "sq": [
                _result(0, "p0", 5.0, chunk_type="table", page=6, table_id="Table 4"),
                _result(1, "p0", 4.0, chunk_type="text_span", page=2),
            ]
        }
    )
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5)
    prediction = agent.run(_query())

    assert len(prediction.evidence) == 1
    evidence = prediction.evidence[0]
    assert evidence.paper_id == "p0"
    assert evidence.source_type == "table"
    assert evidence.locator.page == 6
    assert evidence.locator.table_id == "Table 4"


def test_reads_full_chunk_text_not_a_200_char_stub():
    """LLM に渡す抜粋は snippet_chars まで。200文字ではタイトルしか読めない。"""
    long_text = "x" * 5000
    result = _result(0, "p0", 1.0)
    result.text = long_text
    retriever = _StubRetriever({"sq": [result]})
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5, snippet_chars=1800)
    agent.run(_query())

    judge_prompt = llm.calls[1]
    assert "x" * 1800 in judge_prompt
    assert "x" * 1801 not in judge_prompt


def test_drops_hallucinated_paper_and_chunk_ids():
    """候補一覧に無い paper_id / chunk_id は捨てる。"""
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
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5)
    prediction = agent.run(_query())

    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0"]
    assert len(prediction.evidence) == 1  # 実在しない p0#c99 は落ちる


def test_duplicate_chunk_keeps_the_higher_score():
    """同じチャンクが複数のサブクエリで当たったら、スコアが高いほうを残す。

    後勝ちにすると、サブクエリ1で最上位だったチャンクが後のサブクエリの低い
    スコアで上書きされ、候補論文の順位が「最後に投げたサブクエリ」に引きずられる。
    reranker がスコアに順位を書き戻すようになった以上、ここが後勝ちだと
    rerank の順位がサブクエリ間で壊れる。
    """
    retriever = _StubRetriever(
        {
            "sq-a": [_result(0, "pA", 9.0), _result(0, "pB", 1.0)],
            # pA#c00 は sq-a と同じチャンク。スコアだけが低い。
            "sq-b": [_result(0, "pA", 0.1), _result(0, "pB", 2.0)],
        }
    )
    llm = FakeLLM(
        responses=[_subqueries("sq-a", "sq-b"), _judge(["pA"], sufficient=True)]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5)
    agent.run(_query())

    # 候補一覧は _read_and_judge のプロンプトに関連度順で並ぶ。pA が 9.0 を
    # 保っていれば pA が先頭、0.1 に上書きされていれば pB (2.0) が先頭になる。
    listing = llm.calls[-1]
    assert listing.index("[paper_id: pA]") < listing.index("[paper_id: pB]")


@pytest.mark.parametrize(
    "task_family,expected", [("hidden_source_single_paper", 2), ("multi_paper", 5)]
)
def test_falls_back_to_cutoff_when_llm_output_is_unusable(task_family, expected):
    """LLM が一度も使える判定を返さなければ、順位カットオフで出す。"""
    retriever = _StubRetriever()
    llm = FakeLLM(responses=["not json"])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5)
    prediction = agent.run(_query(task_family=task_family))

    assert len(prediction.gold_papers) == expected
    assert prediction.evidence == []


def test_candidate_papers_records_ranking_before_cutoff():
    """打ち切り前の候補論文を関連度順で残す（recall@k の分析に使う）。

    gold_papers は LLM の選定と cutoff で数本に絞られるので、これだけでは
    「検索がそもそも gold を候補に拾えていたか」を後から測れない。
    """
    retriever = _StubRetriever(
        {"sq": [_result(0, f"p{i}", 10.0 - i) for i in range(8)]}
    )
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=True),  # 提出は1本に絞られる
        ]
    )
    agent = ReadingAgent(
        retriever, llm=llm, max_steps=1, retrieve_top_k=8, paper_cutoff="llm",
        submit_from="llm",  # 選定込みで「提出は絞られる」ことを見るテストなので明示する
    )
    prediction = agent.run(_query())

    # 提出は絞られても、候補はスコア降順で丸ごと残っている。
    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0"]
    assert prediction.candidate_papers == [f"p{i}" for i in range(8)]


def test_candidate_papers_is_capped():
    """候補列は CANDIDATE_PAPERS_LIMIT 本で頭打ちにする（予測ファイルの肥大化を防ぐ）。"""
    n = CANDIDATE_PAPERS_LIMIT + 10
    retriever = _StubRetriever(
        {"sq": [_result(0, f"p{i:03d}", float(n - i)) for i in range(n)]}
    )
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p000"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=n, paper_cutoff="llm")
    prediction = agent.run(_query())

    assert len(prediction.candidate_papers) == CANDIDATE_PAPERS_LIMIT
    assert prediction.candidate_papers[0] == "p000"  # 最上位が先頭


class _FilterAwareRetriever(_StubRetriever):
    """attribute_filter を受け取れる Retriever。渡された制約を記録する。"""

    def __init__(self, extractor, **kwargs):
        super().__init__(**kwargs)
        self.attribute_extractor = extractor
        self.filters: list[object] = []

    def retrieve(self, question, top_k, attribute_filter=None):
        self.filters.append(attribute_filter)
        return super().retrieve(question, top_k)


class _StubExtractor:
    """「NAACL」を含む質問だけ制約を返すダミー抽出器。"""

    def __init__(self):
        from littraceqa.di_pipeline.retrieve.attribute_filter import AttributeFilter

        self._cls = AttributeFilter

    def extract(self, question: str):
        if "NAACL" in question:
            return self._cls(venue="NAACL", year=2025)
        return self._cls()


def test_attribute_filter_comes_from_the_question_not_the_subquery():
    """制約は元の質問から取る。サブクエリが会議名を落としても効くこと。

    _decompose() は「Which NAACL 2025 papers ...」を「MCTS in method figure」の
    ように言い換えるので、サブクエリから抽出すると発火しない。
    """
    retriever = _FilterAwareRetriever(_StubExtractor())
    llm = FakeLLM(
        responses=[
            _subqueries("MCTS in method figure"),  # 会議名が落ちたサブクエリ
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5, paper_cutoff="llm")
    agent.run(_query(question="Which NAACL 2025 papers explicitly mention MCTS?"))

    assert retriever.calls == ["MCTS in method figure"]
    assert [(f.venue, f.year) for f in retriever.filters] == [("NAACL", 2025)]


def test_no_attribute_filter_argument_when_nothing_is_extracted():
    """制約が取れない質問では retrieve() に引数自体を渡さないこと。

    Retriever Protocol は (query, top_k) の2引数なので、常に渡すと
    attribute_filter を受け取らない自作 Retriever を壊す。
    """
    retriever = _StubRetriever()  # attribute_filter を受け取れない
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5, paper_cutoff="llm")

    prediction = agent.run(_query(question="Which papers report FID on CIFAR-10?"))

    assert prediction.trace[0]["attribute_filter"] is None


class _LLMStubExtractor(_StubExtractor):
    """extract_with_llm() を持つ抽出器（LLM 抽出が有効な構成に相当）。"""

    def __init__(self):
        super().__init__()
        self.llm_calls: list[str] = []

    def extract_with_llm(self, question: str):
        self.llm_calls.append(question)
        return self._cls(venue="NeurIPS", year=None)


def test_llm_extractor_is_used_when_available():
    """extract_with_llm() を持つ抽出器ではそちらを使うこと。"""
    extractor = _LLMStubExtractor()
    retriever = _FilterAwareRetriever(extractor)
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5, paper_cutoff="llm")

    agent.run(_query(question="Among NIPS papers, which one uses MCTS?"))

    # 元の質問に対して1回だけ。サブクエリごとには呼ばない。
    assert extractor.llm_calls == ["Among NIPS papers, which one uses MCTS?"]
    assert [(f.venue, f.year) for f in retriever.filters] == [("NeurIPS", None)]


# ---- サブクエリ生成プロンプト ---------------------------------------------


def _subquery_prompts(llm: FakeLLM) -> list[str]:
    return [p for p in llm.calls if '"subqueries"' in p]


def test_subquery_prompts_forbid_web_search_operators():
    """分解・再分解の両方で「Web検索ではない」と伝えること。

    これが無いと LLM は site: / filetype: 付きのクエリを書く。実測では
    _refine() の出力の 29〜41% がそれで、ローカル索引には1件もヒットしない。
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
    agent = ReadingAgent(retriever, llm=llm, max_steps=2, retrieve_top_k=5, paper_cutoff="llm")
    agent.run(_query())

    prompts = _subquery_prompts(llm)
    assert len(prompts) == 2  # _decompose と _refine
    for prompt in prompts:
        assert "NOT sent to a web search engine" in prompt
        assert "site:" in prompt and "filetype:" in prompt


def test_subquery_count_is_enforced_on_the_return_value_not_just_the_prompt():
    """LLM が上限を超えて返しても、投げる本数は subquery_count で切ること。

    本数指定を書いていなかった `_refine()` は実測で平均8.2〜9.3本・最大20本まで
    膨らんでいた（`runs_fat.jsonl`, 55件812本）。サブクエリ1本は reranker が
    pool_k 件を推論する検索1回ぶんなので、これがそのまま走行時間になる
    （pool_k=1000 の構成で 1.73分/本 = 23時間）。プロンプトで頼むだけでは
    守られないので、返り値を切っていることをここで固定する。
    """
    retriever = _StubRetriever()
    llm = FakeLLM(
        responses=[
            _subqueries("d1", "d2", "d3", "d4", "d5", "d6"),  # _decompose が6本返す
            _judge(["p0"], sufficient=False, missing="need more"),
            _subqueries(*[f"r{i}" for i in range(20)]),  # _refine が20本返す
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(
        retriever, llm=llm, max_steps=2, retrieve_top_k=5, paper_cutoff="llm", subquery_count=4
    )
    agent.run(_query())

    assert retriever.calls == ["d1", "d2", "d3", "d4", "r0", "r1", "r2", "r3"]


def test_refine_prompt_states_the_subquery_budget():
    """再分解のプロンプトにも本数を書くこと（分解側にしか書いていなかった）。"""
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
        retriever, llm=llm, max_steps=2, retrieve_top_k=5, paper_cutoff="llm", subquery_count=3
    )
    agent.run(_query())

    decompose_prompt, refine_prompt = _subquery_prompts(llm)
    assert "into 3 short" in decompose_prompt
    assert "at most 3 new search subqueries" in refine_prompt
    # 言い換えの総当たりに流れるのが実際の失敗形なので、そこも明示する。
    assert "do not submit paraphrases" in refine_prompt


def test_subquery_prompts_carry_the_venue_tag():
    """制約が取れたら、その表記をサブクエリの先頭に付けるよう指示すること。

    索引の title_abstract チャンクは本文が「[NAACL 2025] タイトル…」で始まる。
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
    agent = ReadingAgent(retriever, llm=llm, max_steps=2, retrieve_top_k=5, paper_cutoff="llm")
    agent.run(_query(question="Which NAACL 2025 papers explicitly mention MCTS?"))

    prompts = _subquery_prompts(llm)
    assert len(prompts) == 2
    for prompt in prompts:
        assert '"[NAACL 2025]"' in prompt


def test_no_venue_tag_without_a_constraint():
    """制約が無い質問ではプロンプトに会議名の指示を混ぜないこと。"""
    retriever = _StubRetriever()
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=5, paper_cutoff="llm")
    agent.run(_query(question="Which papers report FID on CIFAR-10?"))

    assert all("limits the search to" not in p for p in _subquery_prompts(llm))


def test_no_expander_means_identical_behavior():
    """paper_expander を渡さなければ候補列に一切手を付けない（既定経路の保全）。"""
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(_StubRetriever(), llm=llm, max_steps=1, retrieve_top_k=5)
    prediction = agent.run(_query())
    assert all("paper_expansion" not in step for step in prediction.trace)
    assert len(prediction.candidate_papers) <= CANDIDATE_PAPERS_LIMIT


class _StubRelated:
    """関連ランキング（B）のスタブ。rank() は既存候補も含めた順位を返す。"""

    combine = "rrf"
    combine_rrf_k = 60
    related_weight = 1.0
    related_offset = 0
    anchors = 1

    def __init__(self, ranking: list[str]):
        self.ranking = ranking

    def rank(self, ranked: list[str]) -> list[str]:
        # 本物の expander と同じく anchor 自身は近傍に含めない。
        return [p for p in self.ranking if p != ranked[0]]


def _rrf_agent(ranking: list[str], **kwargs) -> ReadingAgent:
    expander = _StubRelated(ranking)
    for key, value in kwargs.items():
        setattr(expander, key, value)
    return ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
        max_steps=1,
        retrieve_top_k=5,
        max_candidates=2,
        paper_expander=expander,
        submit_from="llm",
    )


def test_combine_rrf_promotes_papers_backed_by_both_rankings():
    """A の下位でも B が推す論文は押し上がる。位置挿入では作れなかった動き。"""
    # 検索A: p0..p4（スコア降順）。関連B: p3 を先頭に推す。
    prediction = _rrf_agent(["p3", "pE1"]).run(_query())
    candidates = prediction.candidate_papers
    # A で4位だった p3 が、A(1/64) + B(1/61) で p1(1/62) / p2(1/63) を抜く
    assert candidates.index("p3") < candidates.index("p1")
    # B にしか居ない論文も入る（素の RRF なので A の上位とは競り負ける）
    assert "pE1" in candidates
    # 統合は候補列だけを組み替える。LLM 選定（submit_from="llm"）には触らない。
    assert prediction.gold_papers == [{"paper_id": "p0"}]
    fusion = [s["paper_fusion"] for s in prediction.trace if "paper_fusion" in s]
    assert fusion and fusion[0]["anchor"] == "p0"
    # 可視域(max_candidates=2)の外から押し上がった論文として記録される
    assert fusion[0]["promoted"] == ["p3"]


def test_combine_rrf_keeps_the_anchor_on_top():
    """anchor は B の1位として扱う。外すと「A にも B にも居る」論文に軒並み抜かれる。"""
    # B が候補列の他の論文を軒並み推しても、anchor は先頭のまま。
    prediction = _rrf_agent(["p1", "p2", "p3", "p4"]).run(_query())
    assert prediction.candidate_papers[0] == "p0"


def test_combine_rrf_offset_pushes_b_only_papers_down():
    """related_offset は B 単独の論文が入る深さを決める（既定0 = 素の RRF）。"""
    shallow = _rrf_agent(["pE1"]).run(_query()).candidate_papers
    deep = _rrf_agent(["pE1"], related_offset=10).run(_query()).candidate_papers
    assert shallow.index("pE1") < deep.index("pE1")


class _StubPools(_StubRelated):
    """anchor ごとのランキングを潰さずに返す expander（consensus: true 用）。"""

    consensus = True
    anchors = 2

    def __init__(self, pools: list[list[str]]):
        super().__init__(pools[0])
        self.pools = pools

    def rank_pools(self, ranked: list[str]) -> list[list[str]]:
        anchors = set(ranked[: self.anchors])
        return [[p for p in pool if p not in anchors] for pool in self.pools]


def test_consensus_prefers_papers_backed_by_several_pools():
    """複数の pool が揃って推した論文が、1つの pool だけの論文より上に来る。

    _interleave() で1本に潰していたときは作れなかった動き（重複が消えるため）。
    """
    # pE1 は両方の pool に居る（2項）。pE2 は片方の1位にしか居ない（1項）。
    expander = _StubPools([["pE1", "pE2"], ["pE1", "pE3"]])
    agent = ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
        max_steps=1,
        retrieve_top_k=5,
        max_candidates=2,
        paper_expander=expander,
    )
    candidates = agent.run(_query()).candidate_papers
    assert candidates.index("pE1") < candidates.index("pE2")
    assert candidates.index("pE1") < candidates.index("pE3")


def test_consensus_off_is_the_existing_path():
    """consensus を書かなければ rank_pools があっても使わない（既定経路の保全）。"""
    pools = [["pE1", "pE2"], ["pE1", "pE3"]]
    on = _StubPools(pools)
    off = _StubPools(pools)
    off.consensus = False
    # consensus オフ側は rank()（= pools[0] 相当）だけを見るので pE3 が入らない。
    def _run(expander):
        return ReadingAgent(
            _StubRetriever(),
            llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
            max_steps=1,
            retrieve_top_k=5,
            max_candidates=2,
            paper_expander=expander,
        ).run(_query()).candidate_papers

    assert "pE3" in _run(on)
    assert "pE3" not in _run(off)


class _ScoreReranker:
    """paper_id -> スコアの対応表で並べ替える偽 reranker。"""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[str] = []

    def rerank(self, query, candidates, top_k):
        self.calls.append(query)
        out = []
        for c in candidates:
            out.append(
                RetrievalResult(
                    chunk_id=c.chunk_id, paper_id=c.paper_id,
                    score=self.scores.get(c.paper_id, 0.0), text=c.text,
                    chunk_type=c.chunk_type, metadata=c.metadata, source=c.source,
                )
            )
        return sorted(out, key=lambda r: r.score, reverse=True)[:top_k]


class _RerankingRetriever(_StubRetriever):
    def __init__(self, reranker, **kwargs):
        super().__init__(**kwargs)
        self.reranker = reranker


# ---------------------------------------------------------------------------
# 反復ループの拡張（docs/search_agent_spec.md 3〜7節）。
# 共通の約束: **新 params を書かなければ既存の経路と完全に同一**。
# ---------------------------------------------------------------------------


def test_new_params_are_all_opt_in():
    """新 params を一切書かない構成が、従来と同じ候補列を出す（後方互換）。

    上のテスト24件が丸ごとこの保証を兼ねているが、既定値そのものが
    「従来と同一」であることを明示的に固定しておく。
    """
    agent = ReadingAgent(_StubRetriever(), llm=FakeLLM(responses=[]))
    assert agent.subquery_merge == "max"
    assert agent.grounded_refine is False
    assert agent.pool_rescore is False and agent.pool_prune_to is None
    assert agent.adaptive_depth is None


def test_unknown_subquery_merge_is_rejected():
    """yaml の綴り間違いを黙って max に落とさない。"""
    with pytest.raises(ValueError, match="subquery_merge"):
        ReadingAgent(_StubRetriever(), llm=FakeLLM(responses=[]), subquery_merge="rff")


def _two_subquery_retriever():
    """サブクエリAが高スコアで1本、Bが低スコアで2本返すスタブ。

    max マージだと pB1(0.5) は pA(0.9) に勝てないが、RRF なら pB1 は
    「サブクエリBの1位」なので pA と同じ 1/(k+1) を得る。multi の gold が
    「1本のサブクエリだけが見つける論文」であるときの再現。
    """
    return _StubRetriever(
        {
            "sqA": [_result(0, "pA", 0.9), _result(0, "pX", 0.8)],
            "sqB": [_result(0, "pB1", 0.5), _result(0, "pX", 0.4)],
        }
    )


def test_subquery_merge_rrf_ranks_by_rank_not_absolute_score():
    """RRF では「別サブクエリの1位」が「あるサブクエリの2位」より上に来る。

    max マージでは pX(0.8) が pB1(0.5) に勝つ。異なるサブクエリに対する
    reranker の yes 確率を突き合わせているためで、この比較には意味が無い。
    """
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        _two_subquery_retriever(), llm=llm, max_steps=1, retrieve_top_k=5,
        subquery_merge="rrf",
    )
    cands = agent.run(_query()).candidate_papers
    # pX は両サブクエリの2位なので2項ぶん入り首位。pB1（B の1位）は pA と同点で、
    # 同点は挿入順（A が先）で決まるため pA の直後。
    assert cands[0] == "pX", cands
    assert cands.index("pB1") < cands.index("pA") + 2, cands


def test_subquery_merge_max_is_unchanged():
    """既定（max）では従来どおり絶対スコア順。上のテストとの対比。"""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        _two_subquery_retriever(), llm=llm, max_steps=1, retrieve_top_k=5,
    )
    assert agent.run(_query()).candidate_papers == ["pA", "pX", "pB1"]


def test_grounded_refine_shows_candidates_and_dead_subqueries():
    """接地版の _refine が「候補上位」と「効かなかったサブクエリ」をプロンプトに含む。"""
    retriever = _StubRetriever(
        {
            "hit": [_result(0, "pTop", 0.9, title="Speculative Decoding for LLMs")],
            "dud": [_result(0, "pLow", 0.1, title="Unrelated Paper")],
        }
    )
    llm = FakeLLM(
        responses=[
            _subqueries("hit", "dud"),
            _judge(["pTop"], sufficient=False, missing="need the batch size"),
            _subqueries("next"),
            _judge(["pTop"], sufficient=True),
        ]
    )
    agent = ReadingAgent(
        retriever, llm=llm, max_steps=2, retrieve_top_k=5,
        max_candidates=1, grounded_refine=True, grounded_refine_top_n=1,
    )
    agent.run(_query())

    refine_prompt = llm.calls[2]
    assert "Speculative Decoding for LLMs" in refine_prompt
    # 上位1本に1件も残らなかった "dud" が名指しされ、残った "hit" はされない。
    dead = refine_prompt.split("contributed nothing")[1]
    assert "- dud" in dead and "- hit" not in dead
    # 誤解の訂正を促す一文（q_003 の型）が入る。
    assert "correct the assumption" in refine_prompt


def test_refine_prompt_is_unchanged_without_grounding():
    """grounded_refine を書かなければ _refine のプロンプトは従来のまま。"""
    llm = FakeLLM(
        responses=[
            _subqueries("sq"),
            _judge(["p0"], sufficient=False, missing="more"),
            _subqueries("next"),
            _judge(["p0"], sufficient=True),
        ]
    )
    agent = ReadingAgent(_StubRetriever(), llm=llm, max_steps=2, retrieve_top_k=5)
    agent.run(_query())
    assert "contributed nothing" not in llm.calls[2]
    assert "corpus actually returned" not in llm.calls[2]


def test_pool_rescore_reranks_the_pool_with_the_original_question():
    """pool_rescore はプール全体を**元の質問**で1回だけリランクする。

    サブクエリで測り直すと、解消したかったスコア非可換性がそのまま残る。
    """
    reranker = _ScoreReranker({"pA": 0.1, "pB1": 0.9, "pX": 0.5})
    retriever = _RerankingRetriever(reranker, by_query=_two_subquery_retriever().by_query)
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        retriever, llm=llm, max_steps=1, retrieve_top_k=5, pool_rescore=True,
    )
    prediction = agent.run(_query())

    assert reranker.calls == ["Which papers report FID on CIFAR-10?"]
    # 全チャンクが同じ尺度で測り直されるので、素の max マージの順（pA, pX, pB1）が覆る。
    assert prediction.candidate_papers == ["pB1", "pX", "pA"]
    assert [s for s in prediction.trace if "pool_rescore" in s]


def test_pool_rescore_is_skipped_without_a_reranker():
    """reranker を持たない構成では黙って skip する（NoneReranker も同じ）。"""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        _two_subquery_retriever(), llm=llm, max_steps=1, retrieve_top_k=5,
        pool_rescore=True,
    )
    prediction = agent.run(_query())
    assert prediction.candidate_papers == ["pA", "pX", "pB1"]
    assert not [s for s in prediction.trace if "pool_rescore" in s]


def test_pool_prune_to_cuts_the_pool():
    """pool_prune_to は再スコア後（無ければマージ後）に上位 N 件へ切る。"""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        _two_subquery_retriever(), llm=llm, max_steps=1, retrieve_top_k=5,
        pool_prune_to=2,
    )
    assert agent.run(_query()).candidate_papers == ["pA", "pX"]


def _depth_agent(**depth):
    base = {"enabled": True, "probe_rank": 2, "gap_threshold": 0.15,
            "shallow_k": 2, "deep_k": 5}
    base.update(depth)
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    return llm, base


def test_adaptive_depth_takes_few_results_when_the_winner_is_clear():
    """1位が飛び抜けていれば shallow_k 件しか採らない（single_paper の型）。"""
    steep = [_result(0, "p0", 1.0), _result(0, "p1", 0.9), _result(0, "p2", 0.2),
             _result(0, "p3", 0.1), _result(0, "p4", 0.05)]
    llm, depth = _depth_agent()
    agent = ReadingAgent(
        _StubRetriever({"sq": steep}), llm=llm, max_steps=1, adaptive_depth=depth,
    )
    assert agent.run(_query()).candidate_papers == ["p0", "p1"]


def test_adaptive_depth_takes_many_results_when_scores_are_flat():
    """スコアが平坦なら deep_k 件まで採る（multi_paper の型）。"""
    flat = [_result(0, f"p{i}", 1.0 - i * 0.01) for i in range(5)]
    llm, depth = _depth_agent()
    agent = ReadingAgent(
        _StubRetriever({"sq": flat}), llm=llm, max_steps=1, adaptive_depth=depth,
    )
    assert agent.run(_query()).candidate_papers == ["p0", "p1", "p2", "p3", "p4"]


def test_adaptive_depth_asks_the_retriever_for_deep_k():
    """retriever には常に deep_k を渡し、切るのはエージェント側。

    reranker の推論件数は search_style の pool_k で決まるので、深く受け取っても
    推論コストは増えない。浅く要求してしまうと拾い直せない。
    """
    seen: list[int] = []

    class _Recording(_StubRetriever):
        def retrieve(self, question, top_k):
            seen.append(top_k)
            return super().retrieve(question, top_k)

    llm, depth = _depth_agent()
    agent = ReadingAgent(
        _Recording(), llm=llm, max_steps=1, retrieve_top_k=20, adaptive_depth=depth,
    )
    agent.run(_query())
    assert seen == [5]


def test_adaptive_depth_is_off_when_not_enabled():
    """enabled: false は「書いていない」と同じ（従来の retrieve_top_k を使う）。"""
    llm, depth = _depth_agent(enabled=False)
    agent = ReadingAgent(
        _StubRetriever(), llm=llm, max_steps=1, retrieve_top_k=3, adaptive_depth=depth,
    )
    assert agent.adaptive_depth is None
    assert len(agent.run(_query()).candidate_papers) == 3


def test_last_runs_is_exposed_for_dumping():
    """--dump-runs 用に、サブクエリ単位の結果が run() の後に残る。"""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(_two_subquery_retriever(), llm=llm, max_steps=1, retrieve_top_k=5)
    agent.run(_query())
    assert [(r.step, r.subquery) for r in agent.last_runs] == [(0, "sqA"), (0, "sqB")]
    assert [r.paper_id for r in agent.last_runs[1].results] == ["pB1", "pX"]


def test_stacked_loop_config_runs_with_every_extension_active():
    """反復ループの拡張キーを同時に有効にしても反復が最後まで回る。

    4つの拡張は別々の地点に効くので併用できる、というのがこの組み合わせの前提。

    以前は `configs/agent_style/reading_loop/stacked.yaml` を実ファイルとして
    読んでいたが、そのフォルダは削除した（拡張キー自体は reading.py に残る）。
    params はここに直接置く——テストの対象は yaml ではなくコードの側なので、
    プリセットが無くなっても併用可能性は保証し続ける必要がある。

    `pool_rescore` は入れない。`_rescore_pool()` は `_merged_results()` の後に走り、
    reranker がプール全件を再スコアして並びを置き換えるので、`subquery_merge` が
    作った順位が candidate_papers では消える（同じ問題への別々の答えなので、
    足すのではなく選ぶ）。
    """
    params = {
        "max_steps": 3,
        "retrieve_top_k": 20,  # adaptive_depth 有効時は使われない
        "max_candidates": 20,
        "chunks_per_paper": 2,
        "snippet_chars": 1800,
        "paper_cutoff": "llm",
        "max_papers": 10,
        "subquery_merge": "rrf",
        "subquery_rrf_k": 60,
        "grounded_refine": True,
        "grounded_refine_top_n": 10,
        "adaptive_depth": {
            "enabled": True,
            "probe_rank": 4,
            "gap_threshold": 0.15,
            "shallow_k": 10,
            "deep_k": 40,
        },
        "pool_rescore": False,
        "pool_prune_to": None,
    }

    # adaptive_depth の probe_rank(4) を超える件数を返す（落差を測れる長さ）。
    flat = [_result(i, f"p{i}", 0.90 - 0.001 * i) for i in range(8)]
    retriever = _StubRetriever({"hit": flat, "dud": [_result(0, "pLow", 0.1)]})
    llm = FakeLLM(
        responses=[
            _subqueries("hit", "dud"),
            _judge(["p0"], sufficient=False, missing="need the batch size"),
            _subqueries("next"),
            _judge(["p0", "p1"], sufficient=True),
        ]
    )
    agent = ReadingAgent(retriever, llm=llm, **params)
    prediction = agent.run(_query())

    submitted = [p["paper_id"] for p in prediction.gold_papers]
    assert submitted[0] == "p0"
    # subquery_merge: rrf が効いている——"dud" の中では1位だった pLow が、
    # 絶対スコア(0.1)では最下位なのに順位融合で上位に来る（max マージだと沈む）。
    assert "pLow" in submitted[:3]
    # grounded_refine が効いている（_refine のプロンプトに候補上位の接地情報が入る）。
    assert "corpus actually returned" in llm.calls[2]
    # adaptive_depth が「retrieve_top_k(20) ではなく deep_k(40)」で retriever を呼ぶ。
    assert retriever.top_ks == [40, 40, 40]
    assert prediction.candidate_papers


def test_submit_from_candidates_is_the_default_and_ignores_llm_selection():
    """既定では提出論文を**選ばない**。候補列の順位をそのまま渡す。

    どれを提出するかの選定は読解チーム側の担当なので、検索エージェントは
    順位を渡すところで止める。LLM の paper_ids は使わないが、`sufficient`
    （反復の停止条件）と evidence は選定とは別の役割なので読解自体は残る。
    """
    retriever = _StubRetriever({"sq": [_result(0, f"p{i}", 10.0 - i) for i in range(8)]})
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p5"], sufficient=True)])
    agent = ReadingAgent(retriever, llm=llm, max_steps=1, retrieve_top_k=8, paper_cutoff="llm")
    prediction = agent.run(_query())

    # LLM は p5 だけを選んだが、提出は候補列の順位（max_papers で頭打ち）。
    assert [p["paper_id"] for p in prediction.gold_papers] == [f"p{i}" for i in range(8)]
    # 読解は走っている（停止条件が効いて1周で止まった）。
    assert len(prediction.trace) == 1
    assert prediction.trace[0]["sufficient"] is True
    # 選んだ論文が提出に残っているので evidence も出る。
    assert [e.paper_id for e in prediction.evidence] == ["p5"]


def test_submit_from_llm_restores_the_selection():
    """`submit_from: llm` にすると従来どおり LLM の選定結果を提出する。"""
    retriever = _StubRetriever({"sq": [_result(0, f"p{i}", 10.0 - i) for i in range(8)]})
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p5"], sufficient=True)])
    agent = ReadingAgent(
        retriever, llm=llm, max_steps=1, retrieve_top_k=8,
        paper_cutoff="llm", submit_from="llm",
    )
    prediction = agent.run(_query())
    assert [p["paper_id"] for p in prediction.gold_papers] == ["p5"]


def test_unknown_submit_from_is_rejected():
    with pytest.raises(ValueError, match="submit_from"):
        ReadingAgent(_StubRetriever(), llm=FakeLLM(responses=[]), submit_from="reading_team")


def test_decompose_asks_for_a_fixed_number_of_subqueries():
    """分解の件数は task_family で振り分けず SUBQUERY_COUNT 本に固定する。

    以前は single「1〜3個」/ multi「3〜6個」と分けていたが、その分岐のためだけに
    TaskFamilyClassifier が LLM を1回呼んでいた（本番入力に task_family が無いため）。
    実測で買えていたのは平均0.58本、推定精度も LLM 0.67 / ヒューリスティック 0.673 と
    差が無かったので、分岐ごと外した。
    """
    from littraceqa.di_pipeline.agent.reading import SUBQUERY_COUNT

    for task_family in ("hidden_source_single_paper", "multi_paper"):
        llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
        agent = ReadingAgent(_StubRetriever(), llm=llm, max_steps=1, retrieve_top_k=5)
        agent.run(_query(task_family=task_family))
        decompose_prompt = llm.calls[0]
        assert f"into {SUBQUERY_COUNT} short" in decompose_prompt
        # 片方に決め打ちした文面が残っていない（single/multi の両方を扱う1文に統一）
        assert "contained within a single paper" not in decompose_prompt
        assert "requires evidence spanning multiple papers" not in decompose_prompt


def test_decompose_does_not_call_the_task_family_classifier():
    """本番入力（task_family 無し）でも分解のために LLM を余分に呼ばない。

    呼んでいた頃は 分解1 + 読解1 の前に「single か multi か」の推定が1回入り、
    クエリ1件につき LLM が3回になっていた。
    """
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(_StubRetriever(), llm=llm, max_steps=1, retrieve_top_k=5,
                         paper_cutoff="llm")
    # task_family を持たない（= --production-input 相当の）クエリ
    agent.run(Query(query_id="q_prod", question="Which papers report FID?", answer_types=[]))

    assert len(llm.calls) == 2  # 分解1 + 読解1 だけ
    assert "task_family" not in llm.calls[0]


# ---- 生質問1位のピン留め（rawq_pin） --------------------------------------


def _rawq_retriever() -> _StubRetriever:
    """サブクエリと元の質問で違う1位を返すスタブ。

    `_decompose()` が作るサブクエリは元の質問の語の組み合わせを保たないので、
    融合すると1位が薄まる——という実測の構図をそのまま写している。
    """
    return _StubRetriever(
        {
            "sq": [_result(0, "pSUB", 9.0), _result(1, "pB", 1.0)],
            _query().question: [_result(0, "pRAW", 8.0), _result(1, "pB", 0.5)],
        }
    )


def _rawq_agent(retriever: _StubRetriever, expander=None, **params) -> ReadingAgent:
    return ReadingAgent(
        retriever,
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["pSUB"], sufficient=True)]),
        max_steps=1,
        retrieve_top_k=5,
        max_candidates=2,
        paper_expander=expander,
        **params,
    )


def test_rawq_pin_is_off_by_default():
    """`rawq_pin` を書かなければ元の質問は検索にも順位にも一切現れない。"""
    retriever = _rawq_retriever()
    prediction = _rawq_agent(retriever).run(_query())

    assert retriever.calls == ["sq"], "既定で検索が増えている"
    assert "pRAW" not in prediction.candidate_papers
    assert prediction.candidate_papers[0] == "pSUB"


def test_rawq_pin_searches_the_original_question_and_pins_its_top_paper():
    """元の質問を step0 に足し、その1位を候補列の先頭に固定する。

    **足すだけでは効かない**（max マージは chunk 単位の最高スコアを残すだけで、
    実測でも cr / ecr が全 k で不変だった）ので、固定まで含めて1つの打ち手。
    """
    retriever = _rawq_retriever()
    prediction = _rawq_agent(retriever, rawq_pin=1).run(_query())

    # 元の質問が5本目のサブクエリとして実際に投げられている
    assert retriever.calls == ["sq", _query().question]
    # スコアでは pSUB(9.0) が上でも、固定した pRAW(8.0) が先頭に来る
    assert prediction.candidate_papers[0] == "pRAW"
    assert prediction.candidate_papers[1] == "pSUB"


def test_rawq_pin_only_pins_the_requested_number_of_papers():
    """`rawq_pin: 1` なら生質問の2位は固定しない（2本以上は @5 を崩す）。"""
    prediction = _rawq_agent(_rawq_retriever(), rawq_pin=1).run(_query())
    # 生質問の2位 pB はスコア順のまま（先頭2枠は pRAW / pSUB）
    assert prediction.candidate_papers.index("pB") >= 2


def test_rawq_pin_moves_the_anchor_of_ranking_b():
    """固定は統合の**前**なので、ランキングB の起点も生質問1位に変わる。

    統合後の候補列に置くだけだと @1 しか動かないが、A の先頭に置くと
    @5 以降にも効く（土台 notable の ecr@5: 統合後 0.8465 / A の先頭 0.8556）。
    """
    expander = _StubRelated(["pE1"])
    prediction = _rawq_agent(_rawq_retriever(), expander=expander, rawq_pin=1).run(_query())

    fusion = [s["paper_fusion"] for s in prediction.trace if "paper_fusion" in s]
    assert fusion and fusion[0]["anchor"] == "pRAW"
    assert prediction.candidate_papers[0] == "pRAW"


def test_rawq_pin_keeps_the_original_question_chunks_in_the_pool():
    """足した run は普通のサブクエリなので、生質問が引いたチャンクも evidence に出せる。"""
    retriever = _rawq_retriever()
    agent = ReadingAgent(
        retriever,
        llm=FakeLLM(
            responses=[
                _subqueries("sq"),
                # 生質問だけが引いた論文のチャンクを根拠に指名する
                _judge(["pRAW"], sufficient=True),
            ]
        ),
        max_steps=1,
        retrieve_top_k=5,
        max_candidates=2,
        rawq_pin=1,
    )
    prediction = agent.run(_query())

    assert [e.paper_id for e in prediction.evidence] == ["pRAW"]


# ---- LLM 不要の起点（anchor_from: score） ----------------------------------


def _anchor_agent(**expander_attrs) -> ReadingAgent:
    """`_anchor_papers()` だけを見るための最小構成。"""
    expander = _StubRelated([])
    for key, value in expander_attrs.items():
        setattr(expander, key, value)
    return ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
        max_steps=1,
        paper_expander=expander,
    )


def test_score_anchor_picks_papers_above_the_threshold():
    """読解を走らせない構成では verdict が無いので、リランカのスコアで代替する。"""
    agent = _anchor_agent(anchor_from="score", anchor_score_min=0.99, anchor_max=None)
    anchors = agent._anchor_papers(
        ["p0", "p1", "p2", "p3"],
        verdict_papers=None,
        paper_scores={"p0": 0.999, "p1": 0.995, "p2": 0.5, "p3": 0.9999},
    )
    # しきい値を超えた p1 / p3 が起点に加わる。順序は候補列の順のまま。
    assert anchors == ["p0", "p1", "p3"]


def test_score_anchor_always_keeps_the_top_candidate():
    """候補1位を外すと single_paper の cr@1 が落ちる（verdict 版と同じ事故）。"""
    agent = _anchor_agent(anchor_from="score", anchor_score_min=0.99, anchor_max=None)
    anchors = agent._anchor_papers(
        ["p0", "p1"], verdict_papers=None, paper_scores={"p0": 0.1, "p1": 0.999}
    )
    assert anchors[0] == "p0"


def test_score_anchor_respects_anchor_max():
    agent = _anchor_agent(anchor_from="score", anchor_score_min=0.0, anchor_max=2)
    anchors = agent._anchor_papers(
        ["p0", "p1", "p2"],
        verdict_papers=None,
        paper_scores={"p0": 1.0, "p1": 1.0, "p2": 1.0},
    )
    assert anchors == ["p0", "p1"]


def test_score_anchor_needs_scores():
    """スコアが渡らない経路（位置挿入など）では従来の起点に落ちる。"""
    agent = _anchor_agent(anchor_from="score", anchor_score_min=0.99)
    assert agent._anchor_papers(["p0"], verdict_papers=None, paper_scores=None) is None


def test_anchor_max_defaults_to_no_cap_so_verdict_is_unchanged():
    """`anchor_max` の既定で verdict 版の挙動を変えてはいけない（無制限で検証済み）。"""
    agent = _anchor_agent(anchor_from="verdict")
    anchors = agent._anchor_papers(["p0"], verdict_papers=["p1", "p2", "p3", "p4", "p5"], paper_scores=None)
    assert anchors == ["p0", "p1", "p2", "p3", "p4", "p5"]
