"""ReadingAgent のテスト。

一番大事なのは「LLM が足りないと言ったら本当に再検索するか」。停止条件を本数カウントに
すると反復が回らない（初回で20件返った時点で常に条件を満たす）ため、ここを内容判定にした。
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import yaml

from littraceqa.di_pipeline.agent.reading import (
    CANDIDATE_PAPERS_LIMIT,
    CombineConfig,
    ReadingAgent,
    ReadingConfig,
)
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
        self.top_ks: list[int] = []  # 何件を要求されたかを見るため

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
    # 提出は候補列の順位そのまま。2周目で拾った pX が
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

    gold_papers は cutoff で数本に絞られるので、これだけでは
    「検索がそもそも gold を候補に拾えていたか」を後から測れない。
    """
    retriever = _StubRetriever(
        {"sq": [_result(0, f"p{i}", 10.0 - i) for i in range(8)]}
    )
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    agent = ReadingAgent(
        retriever, llm=llm, max_steps=1, retrieve_top_k=8,
        paper_cutoff="llm", max_papers=3,
    )
    prediction = agent.run(_query())

    # 提出は max_papers で切られても、候補はスコア降順で丸ごと残っている。
    assert [p["paper_id"] for p in prediction.gold_papers] == ["p0", "p1", "p2"]
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
    """関連ランキング（B）のスタブ。**起点は引数で受け取り、設定は持たない。**"""

    def __init__(self, ranking: list[str]):
        self.ranking = ranking
        self.anchor_calls: list[list[str]] = []

    def rank(self, anchors: list[str]) -> list[str]:
        # 本物の expander と同じく anchor 自身は近傍に含めない。
        self.anchor_calls.append(list(anchors))
        return [p for p in self.ranking if p not in anchors]


def _rrf_agent(ranking: list[str], **combine_kwargs) -> ReadingAgent:
    return ReadingAgent(
        _StubRetriever(),
        llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
        max_steps=1,
        retrieve_top_k=5,
        max_candidates=2,
        paper_expander=_StubRelated(ranking),
        combine=CombineConfig(**combine_kwargs),
        paper_cutoff="llm",
        max_papers=1,
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
    # 統合は候補列だけを組み替える。提出（max_papers=1）は融合後の1位のまま。
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


def test_expander_receives_only_the_anchors():
    """**起点を決めるのは agent 側。** expander には候補列ではなく起点だけを渡す。

    以前は候補列をまるごと渡し、expander が先頭 `anchors` 本を自分で切っていた。
    起点を差し替えるたびに expander の `anchors` 属性を書き換えて戻す必要があり、
    「誰が起点を決めるのか」が2つのオブジェクトに割れていた。
    """
    expander = _StubRelated(["pE1"])
    agent = ReadingAgent(
        _StubRetriever(), llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)]),
        max_steps=1, retrieve_top_k=5, max_candidates=2, paper_expander=expander,
    )
    agent.run(_query())
    # 候補列は p0..p4 だが、渡るのは起点1本だけ。
    assert expander.anchor_calls == [["p0"]]


def test_verdict_anchor_adds_the_llm_confirmed_papers():
    """`anchor_from: verdict` は候補1位に読解 LLM の確認済み論文を足す。"""
    expander = _StubRelated(["pE1"])
    agent = ReadingAgent(
        _StubRetriever(), llm=FakeLLM(responses=[_subqueries("sq"), _judge(["p1"], sufficient=True)]),
        max_steps=1, retrieve_top_k=5, max_candidates=2, paper_expander=expander,
        combine=CombineConfig(anchor_from="verdict"),
    )
    agent.run(_query())
    # 候補1位（p0）は必ず残す。外すと single の cr@1 が落ちる。
    assert expander.anchor_calls == [["p0", "p1"]]


def test_combine_rrf_offset_pushes_b_only_papers_down():
    """related_offset は B 単独の論文が入る深さを決める（既定0 = 素の RRF）。"""
    shallow = _rrf_agent(["pE1"]).run(_query()).candidate_papers
    deep = _rrf_agent(["pE1"], related_offset=10).run(_query()).candidate_papers
    assert shallow.index("pE1") < deep.index("pE1")


def _two_subquery_retriever():
    """サブクエリAが高スコアで1本、Bが低スコアで2本返すスタブ。"""
    return _StubRetriever(
        {
            "sqA": [_result(0, "pA", 0.9), _result(0, "pX", 0.8)],
            "sqB": [_result(0, "pB1", 0.5), _result(0, "pX", 0.4)],
        }
    )


def test_pool_is_max_merged_across_subqueries():
    """サブクエリをまたいだプールは chunk ごとの最高スコア順に並ぶ。"""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(
        _two_subquery_retriever(), llm=llm, max_steps=1, retrieve_top_k=5,
    )
    assert agent.run(_query()).candidate_papers == ["pA", "pX", "pB1"]


def _depth_agent(**depth):
    base = {"enabled": True, "probe_rank": 2, "gap_threshold": 0.15,
            "shallow_k": 2, "deep_k": 5}
    base.update(depth)
    llm = FakeLLM(responses=[_subqueries("sq"), _judge(["p0"], sufficient=True)])
    return llm, base


def test_last_runs_is_exposed_for_dumping():
    """--dump-runs 用に、サブクエリ単位の結果が run() の後に残る。"""
    llm = FakeLLM(responses=[_subqueries("sqA", "sqB"), _judge(["pA"], sufficient=True)])
    agent = ReadingAgent(_two_subquery_retriever(), llm=llm, max_steps=1, retrieve_top_k=5)
    agent.run(_query())
    assert [(r.step, r.subquery) for r in agent.last_runs] == [(0, "sqA"), (0, "sqB")]
    assert [r.paper_id for r in agent.last_runs[1].results] == ["pB1", "pX"]


def test_submission_is_the_candidate_ranking_not_the_llm_selection():
    """提出論文を**選ばない**。候補列の順位をそのまま渡す。

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


# ---- 設定（ReadingConfig）--------------------------------------------------


def test_config_rejects_unknown_params_by_name():
    """yaml の綴り間違いを既定値のまま黙って走らせない。

    `agent.params` は registry.build が **kwargs で展開してくるので、キーの正しさを
    見張れるのは ReadingConfig だけ。削除済みの ablation フラグを書いた古い yaml も
    ここで止まる（黙って無視されると、効いていない設定で実験してしまう）。
    """
    with pytest.raises(ValueError, match=r"unknown agent params: \['retrive_top_k'\]"):
        ReadingConfig.from_params({"retrive_top_k": 20})
    # エラーには正しいキーの一覧が入る。
    with pytest.raises(ValueError, match="valid params:.*retrieve_top_k"):
        ReadingAgent(_StubRetriever(), llm=FakeLLM(responses=[]), retrive_top_k=20)


def test_config_is_frozen_and_normalizes_list_params():
    """設定は不変。yaml がリストで書く項目はタプルに寄せる。"""
    config = ReadingConfig.from_params({"paper_score_skip_chunk_types": ["table"]})
    assert config.paper_score_skip_chunk_types == ("table",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_steps = 99


def test_config_can_be_passed_as_an_object():
    """設定オブジェクトを直接渡せる（個別 params との併用は弾く）。"""
    agent = ReadingAgent(
        _StubRetriever(), llm=FakeLLM(responses=[]), config=ReadingConfig(max_steps=1)
    )
    assert agent.config.max_steps == 1
    with pytest.raises(ValueError, match="同時に渡せない"):
        ReadingAgent(
            _StubRetriever(), llm=FakeLLM(responses=[]),
            config=ReadingConfig(), max_steps=1,
        )


def test_production_config_params_all_resolve():
    """最終構成の yaml が書く params が1つ残らず ReadingConfig に載る。"""
    params = yaml.safe_load(
        open("configs/agent_style/reading_expand_rrf/notable.yaml", encoding="utf-8")
    )["params"]
    config = ReadingConfig.from_params(params)
    assert config.paper_cutoff == "llm"
    assert config.paper_score_skip_chunk_types == ("table",)


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

