"""検索 → 読解 → 不足分の再検索 を繰り返す、本命の反復エージェント。

過去に試した2つのベースラインは片翼ずつしか持っていなかった（どちらも削除済み）:

* 「反復するが中身を検証しない」型。停止条件を
  「検索が返した論文の本数 >= しきい値」にすると、retrieve_top_k=20 で検索した時点で
  初回から満たされてしまい、2周目に入らず反復が空回りした。
* 「LLM に候補を読ませて選ばせるが1回で終わる」型。足りなくても検索し直さない。

ReadingAgent はこの2つを合体させる。停止条件を「LLM が根拠として確認できた論文が
質問に答えるのに足りているか」に置き換えることで、初めて反復が意味を持つ。

    1. 質問をサブクエリに分解する
    2. 各サブクエリで検索する
    3. 上位候補のチャンクを**全文**（既定1800文字）LLM に読ませ、
       本当に根拠になる論文と、その根拠チャンクを選ばせる
    4. LLM が「まだ足りない」と言えば、何が欠けているかを聞いて再分解し 2 に戻る
    5. 足りたか、max_steps に達したら終了

選ばれた根拠チャンクから Evidence を組み立てる（src/littraceqa/di_pipeline/agent/evidence.py 参照）。
**Answer（freeform / multiple_choice / table）は埋めない**——回答生成も提出論文の選定も
読解チーム側の担当なので、このエージェントが渡すのは候補列（candidate_papers）と
evidence まで。提出論文も選ばない（`submit_from: candidates`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.agent.rewrite import QueryRewriter, SubqueryDeduper
from littraceqa.di_pipeline.agent.task_family import TaskFamilyClassifier, apply_paper_cutoff
from littraceqa.di_pipeline.contracts import Answer, Evidence, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.hybrid import (
    HybridRetriever,
    paper_scores,
    to_gold_papers,
)
from littraceqa.di_pipeline.retrieve.rrf import RRFFuser


# 予測ファイルに残す候補論文の本数。看板指標は recall@20 だが、再実行が高コストなので
# あとから recall@50 まで測れるよう多めに持たせる。
CANDIDATE_PAPERS_LIMIT = 50

# `_decompose()` が作らせるサブクエリの本数。**task_family で振り分けない。**
#
# 以前は single なら「1〜3個」、multi なら「3〜6個」と指示を分けていたが、その分岐の
# ためだけに TaskFamilyClassifier が**クエリ1件につき LLM を1回**呼んでいた
# （本番入力に task_family が無いため）。買えていたものは実測で平均0.58本しかない:
#
#   predictions_8b_chunk_b_merged.jsonl の trace（55件）
#     single 26件: step0 のサブクエリ数 平均 3.08（26件中25件が上限の3本）
#     multi  29件: 平均 3.66（29件中17件が下限の3本）
#
# 推定精度も LLM 0.67 / ヒューリスティック 0.673（55件実測）と差が無く、分岐する
# 根拠がない。両者の実測値を挟む4本に固定して、LLM 呼び出しを1回減らした。
#
# **本数はプロンプトで頼むだけでなく、返り値も機械的に切る。** LLM は
# `_decompose()` では概ね守る（実測 平均3.3本）が上限を超えることがあり（最大6本）、
# **本数を書いていなかった `_refine()` では平均8.2〜9.3本・最大20本まで膨らんでいた**。
# サブクエリ1本 = 検索1回 = reranker が pool_k 件を推論する量なので、ここが
# そのまま走行時間になる（`pool_k: 1000` の構成で実測 1.73分/本）。
SUBQUERY_COUNT = 4

# サブクエリを作らせる全プロンプトの先頭に置く注意書き。
#
# **これが無いと LLM は Google 検索クエリを書く。** 実測（予測ファイルの trace を
# 集計）では `_refine()` が作るサブクエリの 29〜41% が `site:arxiv.org` /
# `filetype:pdf` / `site:openaccess.thecvf.com/content/CVPR2025/html` といった
# Web検索演算子付きで、55件中30〜39件のクエリが影響を受けていた。投げ先は
# ローカルの BM25 と faiss なので、これらは1件もヒットしない純粋なノイズであり、
# max_steps=3 の2周目・3周目の検索が丸ごと空振りしていた。
# （step0 の `_decompose()` では 0% だったが、同じ誤解が起きうるので両方に置く。）
CORPUS_NOTE = (
    "The subqueries are sent to a local search index built over the full text of the "
    "papers (BM25 + dense embeddings). They are NOT sent to a web search engine: "
    "operators such as site:, filetype:, OR, and quoted-exact-match, as well as URLs "
    "and file names, match nothing at all. Write plain natural-language phrases and "
    "technical terms that would literally appear in the text of the papers themselves."
)


@dataclass
class SubqueryRun:
    """1本のサブクエリが返した検索結果を、**検索が返した順のまま**保持する。

    反復ループの `chunks: dict[chunk_id -> RetrievalResult]` は「同じチャンクは
    スコアが高いほうを残す」max マージで、どのサブクエリの何位だったかを潰してしまう。
    サブクエリ間 RRF（`subquery_merge: rrf`）も、`_refine()` の接地（どのサブクエリが
    効かなかったか）も、順位が要る。**ランキングの出所だけをここに移し、
    `chunks` は chunk_id からの引き当て用に残す**（捏造チェックと evidence 引きが
    chunk_id で引いているため）。
    """

    step: int
    subquery: str
    results: list[RetrievalResult] = field(default_factory=list)


@register("agent", "reading")
class ReadingAgent:
    """候補を読んで根拠を確定し、足りなければ検索し直す反復エージェント。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        max_steps: int = 3,
        # 1ステップで投げるサブクエリの上限。**プロンプトの本数指定と返り値の
        # 切り詰めの両方に使う**（LLM は守らないことがある。SUBQUERY_COUNT 参照）。
        #
        # runs_fat.jsonl（55件812本）を1本ずつ抜いて候補列を組み直した実測では、
        # **抜くと ecr@50 の gold が減るのは812本中5本**（step0 4/180・step1 0/305・
        # step2 1/327）しかない。各ステップを先頭N本に絞った再生（replay と同じ経路、
        # subquery_merge=rrf）でも、増やすほど良くなる関係にはなっていない:
        #
        #     N      本数   削減   cr@20  ecr@20  cr@50  ecr@50
        #     2       253   -69%   0.796   0.878  0.836   0.912
        #     4       461   -43%   0.774   0.852  0.836   0.914
        #     6       611   -25%   0.770   0.841  0.841   0.912
        #     上限なし 812     0%   0.779   0.847  0.841   0.914
        #
        # N=3〜8 の差は 55件では誤差幅なので、`_decompose()` の指示と同じ4本に
        # 揃えてある。**「4より上に価値が無い」ほうが確かな結論**で、2 が両マージ方式で
        # @20 最良だったのはノイズと区別が付かない。
        subquery_count: int = SUBQUERY_COUNT,
        # サブクエリ1本の検索で retriever から受け取る**チャンク**数（論文数ではない）。
        # search_style 側の per_index_k / pool_k とは別物で、reranker が pool_k 件を
        # 全件スコアリングし終えたあとの「上位何件を受け取るか」。増やしても
        # reranker の推論は増えない（捨てていた分を拾うだけ）。
        retrieve_top_k: int = 20,
        max_candidates: int = 15,
        chunks_per_paper: int = 2,
        snippet_chars: int = 1800,
        paper_cutoff: str = "task_family",
        max_papers: int = 10,
        # 提出論文（gold_papers）をどのランキングから作るか。
        #   "candidates"（既定）: 候補列の順位そのまま。**どれを提出するかの選定は
        #     読解チーム側の担当**なので、検索エージェントは順位を渡すだけにする。
        #   "llm": 読解 LLM が選んだ paper_ids を使う（選定込みで測りたい ablation 用）。
        # どちらでも `_read_and_judge()` は呼ぶ。停止条件（sufficient）と evidence は
        # 選定とは別の役割なので、選定を外しても要るため。
        submit_from: str = "candidates",
        # 候補列を論文→論文類似で後ろに拡張するコンポーネント
        # （retrieve/paper_expander.py の Specter2PaperExpander）。
        # 質問文が名指ししないピア gold を拾うためのもので、既存候補の順位も
        # 提出（gold_papers）も変えない。None なら従来と完全に同一。
        paper_expander: object | None = None,
        # ---- 以下は反復ループの拡張。**書かなければ挙動は1ビットも変わらない。** ----
        # サブクエリ間のマージ方法。"max"（既定 = 従来）はチャンクごとに最大スコアを
        # 残すが、これは**異なるサブクエリに対する reranker の yes 確率**という
        # 比較可能でない値を突き合わせている。"rrf" は順位だけを見るのでその問題が無い。
        subquery_merge: str = "max",
        subquery_rrf_k: int = 60,
        # _refine() のプロンプトに「いまの候補上位」と「効かなかったサブクエリ」を
        # 足す（ADORE 系の relevance feedback）。追加の LLM 呼び出しはゼロ。
        grounded_refine: bool = False,
        grounded_refine_top_n: int = 10,
        # プール全体を**元の質問**で1回リランクして単一スケールに揃える（CRAG 系）。
        # 既定オフ。reranker は「質問に答えるか」で判定するので、質問文が名指ししない
        # ピア gold を落としやすい（ecr が上がって cr が下がる向きに出る）。
        pool_rescore: bool = False,
        pool_prune_to: int | None = None,
        # 検索の深さをリランカのスコア分布で決める。
        # {enabled, probe_rank, gap_threshold, shallow_k, deep_k}
        adaptive_depth: dict | None = None,
        # **論文の代表スコアに使わないチャンク種別**（`["table"]` が実測での最良）。
        # 表チャンクは数値と短いラベルが密なので、論文が質問の主題でなくても
        # 表1枚で代表スコアが跳ね上がる。詳細は retrieve/hybrid.py の to_gold_papers。
        # **読解に渡すチャンクプールは変えない**ので evidence には従来どおり出せる。
        paper_score_skip_chunk_types: list[str] | None = None,
        # **元の質問1本を step0 に足し、その検索が返した論文順位の上位N本を
        # ランキングA の先頭に固定する。** 0（既定）で無効＝挙動は1ビットも変わらない。
        #
        # `_decompose()` が作るサブクエリは元の質問の**語の組み合わせ**を保たないので、
        # 融合すると1位が薄まる。実測では生の質問1本で引くほうが @1 が高い
        # （ecr@1 0.637 vs 0.605 / cr@1 0.557 vs 0.529）。土台 notable で
        # 1位が違うのは15件あり、内訳は **生質問だけが根拠付き gold 4件 /
        # 土台だけが gold 1件 / 両方 6件 / どちらも違う 4件＝正味 +3件**。
        #
        # **足すだけでは効かない。** 生質問の run をプールに入れても
        # `subquery_merge: "max"` では chunk 単位の最高スコアを残すだけなので、
        # 候補列は 34/55 件で並びが変わるのに **cr / ecr は全 k で小数4桁まで不変**
        # だった（生質問が引くチャンクは、すでにどれかのサブクエリが同等以上の
        # スコアで持っている）。順位を動かすには明示的に固定する必要がある。
        #
        # **固定先はランキングA（`_combine_rrf` の前）。** 統合後の候補列の先頭に
        # 置くだけでは @1 しか動かないが、A の先頭に置くと `_anchor_papers()` 経由で
        # **ランキングB の起点も生質問1位に変わる**ぶん @5 以降にも効く
        # （土台 notable の ecr@5: 統合後 0.8465 / A の先頭 0.8556、土台 0.8419）。
        #
        # 土台7本の実測（total_ecr の差分。上3本は runs から全長A を復元した版、
        # 下4本は候補列（50本で切られた後）を A にした版）:
        #
        #     土台                @1       @5       @10      @20      @50   土台の@1
        #     notable          +0.032   +0.014   +0.011   +0.006    0.000    0.605
        #     steps2_notable   +0.036   +0.015   +0.006   -0.006    0.000    0.601
        #     fat(08-03)       +0.041   +0.026   -0.006   -0.005    0.000    0.596
        #     k100_cand50       0.000   +0.015   +0.017   +0.006   +0.006    0.637
        #     fat(候補列)       +0.005   +0.023    0.000   +0.005   -0.005    0.632
        #     chunk_cand50     -0.032   +0.009   -0.003   +0.006   +0.006    0.669
        #     b_merged         -0.020   +0.009   -0.005    0.000   -0.005    0.657
        #
        # **@5 は7土台すべてで改善**（+0.009〜+0.026）。ここが効く本体で、
        # **@5 は外部チームに離されている位置**でもある。
        #
        # **@1 は土台次第。** 生質問1本の ecr@1 は 0.637 で固定なので、
        # **土台自身の @1 がそれを上回っていると負ける**（上表の右端と符号が対応する）。
        # 現行の既定構成（notable, @1 0.605）では勝つが、`cand50` 系のように
        # @1 が 0.66 を超える土台に足してはいけない。
        #
        # @10 / @20 は符号が割れるのでノイズ。@50 はほぼ不変（集合は変えず
        # 並べ替えるだけなので、動くのは A の裾が入れ替わるぶんだけ）。2本以上を固定すると
        # @5 が崩れる（notable の ecr@5: N=1 で 0.8556 / N=2 で 0.8247 / N=3 で 0.7975）
        # ので **N=1 以外を使わない**。
        #
        # コストは step0 の検索が1本増えるだけ（LLM 呼び出しは増えない）。足した run は
        # 普通のサブクエリとして `chunks` に積まれるので、**生質問が引いたチャンクも
        # evidence に出せる**。
        rawq_pin: int = 0,
        # ---- 検索結果を材料にしたクエリ書き換え（docs/search_agent2_spec.md） ----
        # {enabled, at_step, chunk_store, from_a: {...}, from_b: {...}}
        #
        # **`at_step` 以降は `_refine()` を置き換える。** _refine() の材料は
        # 読解 LLM の `missing` だけでコーパスの反応を見ないので、サブクエリが
        # ひとつの語彙ファミリーの言い換えに収束する。runs_fat.jsonl の
        # leave-one-out 実測でも **step1 の305本は1本抜いても ecr@50 の gold が
        # 減らない（0/305）**（step2 は 1/327、step0 は 4/180）。置き換え先として
        # 空いているのはそこ。詳細は agent/rewrite.py の docstring。
        rewrite: dict | None = None,
        # サブクエリの重複除去。**本数を固定せず「中身が重ならない数」にする。**
        # {method, probe_k, max_overlap, max_queries}
        subquery_dedup: dict | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.subquery_count = subquery_count
        self.retrieve_top_k = retrieve_top_k
        self.max_candidates = max_candidates
        self.chunks_per_paper = chunks_per_paper
        self.snippet_chars = snippet_chars
        self.paper_cutoff = paper_cutoff
        self.max_papers = max_papers
        self.submit_from = submit_from
        self.paper_expander = paper_expander
        self.subquery_merge = subquery_merge
        self.subquery_rrf_k = subquery_rrf_k
        self.grounded_refine = grounded_refine
        self.grounded_refine_top_n = grounded_refine_top_n
        self.pool_rescore = pool_rescore
        self.pool_prune_to = pool_prune_to
        self.paper_score_skip_chunk_types = tuple(paper_score_skip_chunk_types or ())
        self.rawq_pin = max(0, int(rawq_pin))
        # enabled: false と「そもそも書いていない」を同じ None に畳んでおく。
        # 以降は `self.adaptive_depth is None` だけを見れば従来経路になる。
        self.adaptive_depth = (
            dict(adaptive_depth)
            if isinstance(adaptive_depth, dict) and adaptive_depth.get("enabled")
            else None
        )
        # enabled: false と未指定を同じ None に畳む。None なら `_refine()` の
        # 従来経路をそのまま通る（既存の yaml は1ビットも挙動が変わらない）。
        self.rewriter = (
            QueryRewriter(**{k: v for k, v in rewrite.items() if k != "enabled"})
            if isinstance(rewrite, dict) and rewrite.get("enabled")
            else None
        )
        # 書き換えを使わない構成でも重複除去は効かせられる（独立したキー）。
        self.deduper = SubqueryDeduper.from_retriever(retriever, subquery_dedup)
        # scripts/run_search.py の --dump-runs が読む。Prediction.trace には入れない
        # （提出ファイルが膨らむため）。
        self.last_runs: list[SubqueryRun] = []
        self.task_family = TaskFamilyClassifier(llm)

        if subquery_merge not in ("max", "rrf"):
            raise ValueError(f"unknown subquery_merge: {subquery_merge!r} (expected 'max' or 'rrf')")
        if submit_from not in ("candidates", "llm"):
            raise ValueError(
                f"unknown submit_from: {submit_from!r} (expected 'candidates' or 'llm')"
            )

    def run(self, query: Query) -> Prediction:
        # 会議名・年の制約は**元の質問から1回だけ**取る。_decompose() が作る
        # サブクエリは「NAACL 2025」のような制約語を落とすことがあるので、
        # サブクエリから抽出すると発火しない。反復ステップをまたいで使い回す。
        attribute_filter = self._extract_attribute_filter(query)
        # 制約が取れたときだけ引数を足す。Retriever Protocol の retrieve() は
        # (query, top_k) の2引数なので、常に渡すと自作 Retriever を壊す。
        retrieve_kwargs = (
            {} if attribute_filter is None else {"attribute_filter": attribute_filter}
        )

        subqueries = self._decompose(query, attribute_filter)
        if self.deduper is not None:
            subqueries = self.deduper.filter(subqueries)
        # **元の質問そのものを step0 に足す**（`rawq_pin` が有効なときだけ）。
        # 重複除去のあとに足すのは、言い回しが近いサブクエリと重なって
        # 落とされると `_rawq_ranking()` が起点を見つけられなくなるため。
        if self.rawq_pin and query.question not in subqueries:
            subqueries = [*subqueries, query.question]
        tried: list[str] = []
        chunks: dict[str, RetrievalResult] = {}
        runs: list[SubqueryRun] = []
        verdict: dict | None = None
        trace: list[dict] = []

        for step in range(self.max_steps):
            tried.extend(subqueries)
            for subquery in subqueries:
                results = self._retrieve(subquery, retrieve_kwargs)
                runs.append(SubqueryRun(step=step, subquery=subquery, results=results))
                for result in results:
                    # 同じチャンクが複数のサブクエリで当たったら、スコアが高いほうを残す。
                    # 後勝ちにすると、サブクエリ1で最上位だったチャンクがサブクエリ3の
                    # 低いスコアで上書きされ、_candidate_papers の論文順位が
                    # 「最後に投げたサブクエリ」に引きずられる。
                    previous = chunks.get(result.chunk_id)
                    if previous is None or result.score > previous.score:
                        chunks[result.chunk_id] = result

            merged = self._merged_results(runs, chunks)
            candidates = self._candidate_papers(merged)
            new_verdict = self._read_and_judge(query, candidates, chunks)
            if new_verdict is not None:
                verdict = new_verdict

            trace.append(
                {
                    "step": step,
                    "subqueries": subqueries,
                    # 実際に絞り込みが効いたかを後から追えるように残す。
                    "attribute_filter": (
                        None
                        if attribute_filter is None
                        else {"venue": attribute_filter.venue, "year": attribute_filter.year}
                    ),
                    "n_chunks": len(chunks),
                    "n_candidates": len(candidates),
                    "selected": [] if verdict is None else verdict["paper_ids"],
                    "sufficient": None if verdict is None else verdict["sufficient"],
                    "missing": None if verdict is None else verdict["missing"],
                }
            )

            # ここが「本数で止める反復」との決定的な違い。検索が何本返したかではなく、
            # LLM が根拠として確認できた論文で足りているかどうかで打ち切る。
            if verdict is not None and verdict["sufficient"]:
                break
            if step == self.max_steps - 1:
                break

            missing = "" if verdict is None else verdict["missing"]
            # `at_step` 以降は検索結果に接地した書き換えが `_refine()` を置き換える。
            # rewrite を書いていなければ self.rewriter is None で従来経路のまま。
            if self.rewriter is not None and step + 1 >= self.rewriter.at_step:
                subqueries = self._rewrite_subqueries(
                    query, missing, tried, attribute_filter, runs, merged
                )
            else:
                subqueries = self._refine(query, missing, tried, attribute_filter, runs, merged)
            if self.deduper is not None:
                subqueries = self.deduper.filter(subqueries, already=tried)
            if not subqueries:
                break

        self.last_runs = runs
        return self._build_prediction(query, verdict, chunks, trace, runs)

    # ---- 検索1本ぶん（深さの決定を含む） -----------------------------------

    def _retrieve(self, subquery: str, retrieve_kwargs: dict) -> list[RetrievalResult]:
        """サブクエリ1本を検索し、採る件数を決めて返す。

        `adaptive_depth` が無効なら従来どおり `retrieve_top_k` 件をそのまま返す。

        有効なときは **retriever には常に `deep_k` を渡して取り、切るのはここ**。
        reranker の推論件数は search_style 側の `pool_k` で決まっているので、
        受け取る件数を増やしても**推論コストは1件も増えない**（捨てていた分を
        拾うだけ）。retrieve() の呼び出し回数も1回のまま。
        """
        depth = self.adaptive_depth
        top_k = self.retrieve_top_k if depth is None else depth.get("deep_k", 40)
        results = list(self.retriever.retrieve(subquery, top_k, **retrieve_kwargs))
        if depth is None:
            return results
        return results[: self._depth_for(results, depth)]

    def _depth_for(self, results: list[RetrievalResult], depth: dict) -> int:
        """スコア分布の落差から、このサブクエリを浅く採るか深く採るかを決める。

        1位と `probe_rank` 位の差が大きい = 勝者が明確なので浅く、平坦なら深く採る。
        reranker 有効時のスコアは yes 確率なので 0〜1 の解釈可能なスケールになる。

        **task_family の推定に依存しないのが要点。** single_paper は1位が飛び抜けるので
        自動的に浅くなり、multi_paper は平坦なので深くなる。推定精度0.67の分類器を
        経路に挟まずに同じ効果が得られる。
        """
        shallow_k = depth.get("shallow_k", 10)
        deep_k = depth.get("deep_k", 40)
        probe_rank = depth.get("probe_rank", 4)
        # 候補が probe_rank に届かないときは落差を測れない。浅く切ると取りこぼすので深く。
        if len(results) <= probe_rank:
            return deep_k
        gap = results[0].score - results[probe_rank].score
        return shallow_k if gap >= depth.get("gap_threshold", 0.15) else deep_k

    # ---- サブクエリ間のマージ ----------------------------------------------

    def _merged_results(
        self, runs: list[SubqueryRun], chunks: dict[str, RetrievalResult]
    ) -> list[RetrievalResult]:
        """全サブクエリの結果を1本のランキングに統合する。

        `"max"`（既定）は `chunks` をそのまま返す。`chunks` の構築自体が
        「同じ chunk_id はスコアが高いほうを残す」max マージそのものなので、
        **従来の `list(chunks.values())` と要素も順序も完全に同一**になる。

        `"rrf"` は既存の `RRFFuser` に**サブクエリ1本を1つの run として**渡す
        （indexer 間の融合に使っているのと同じクラス。新しい融合ロジックは書かない）。
        `weights` は `result.source` = indexer 名で引くが既定 1.0 なので素通しになる。

        **multi に効く理屈**: multi の gold は「1本のサブクエリだけが見つける論文」が
        多い。max マージではその論文が単一サブクエリの絶対スコアで他と競うが、
        RRF なら「そのサブクエリの中での順位」で評価される。
        """
        if self.subquery_merge != "rrf":
            return list(chunks.values())
        # top_k は切らない（プールの剪定は pool_prune_to の担当）。
        return RRFFuser(k=self.subquery_rrf_k).fuse(
            [run.results for run in runs], top_k=len(chunks)
        )

    # ---- 0. 属性制約の抽出 -------------------------------------------------

    def _extract_attribute_filter(self, query: Query):
        """質問が明示した会議名・年の制約を取る。retriever 側が無効なら None。

        抽出器は retriever が持っている（search_style の attribute_filter 設定で
        構築される）。無効な構成ではここが None を返し、retrieve() は従来どおりの
        コードパスを通る。
        """
        extractor = getattr(self.retriever, "attribute_extractor", None)
        if extractor is None:
            return None
        # LLM 抽出が有効な構成（search_style の attribute_filter.llm_extract）では
        # extract_with_llm() を持つ。正規表現で取れたときは LLM を呼ばないので、
        # 追加の API 呼び出しは「会議名が書かれていない質問」1件につき1回だけ。
        extract = getattr(extractor, "extract_with_llm", extractor.extract)
        attribute_filter = extract(query.question)
        # 空なら None にして、retrieve() に引数自体を渡さないようにする
        # （制約が無い質問の挙動を従来と完全に同一に保つため）。
        return None if attribute_filter.is_empty() else attribute_filter

    def _constraint_note(self, attribute_filter) -> str:
        """取れた制約をサブクエリ生成プロンプトに伝える文面。制約が無ければ空。

        絞り込み自体は attribute_filter が検索結果に対して行うので、これは
        **検索語としての**制約。title_abstract チャンクの本文は
        `[ACL 2025] タイトル…` と実際にこの表記で始まる（preprocess/mineru_chunker.py）
        ので、同じ表記を頭に付けると BM25 でその会議の論文に寄る。
        `_decompose()` は自発的に会議名を残すことがあるが、`_refine()` では
        落ちていたため両方に明示する。
        """
        if attribute_filter is None:
            return ""
        tag = " ".join(
            str(part)
            for part in (attribute_filter.venue, attribute_filter.year)
            if part is not None
        )
        if not tag:
            return ""
        return (
            f"The question limits the search to {tag}. Begin every subquery with the "
            f'tag "[{tag}]" and keep the rest of the subquery about the content: the '
            "title/abstract text of each paper in the index literally starts with that tag."
        )

    # ---- 1. 分解 ----------------------------------------------------------

    def _decompose(self, query: Query, attribute_filter=None) -> list[str]:
        """質問を検索用のサブクエリに分解する。

        multi_paper のときだけ分解する手もあるが、single_paper でも
        「どの論文か」と「その中のどの表か」は別々の検索語になりうるので常に分解する。

        **件数は task_family で振り分けず `subquery_count` 本に固定する**（下記）。
        指示した本数を超えることがある（実測 最大6本）ので、返り値も切る。
        """
        prompt = "\n".join(
            part
            for part in (
                "You are helping to decompose a research question into search "
                "subqueries against a scientific paper corpus.",
                f"Question: {query.question}",
                # 1本の論文で足りる質問も、複数論文にまたがる質問も同じ文面で扱う。
                # どちらかを決め打ちすると、外したときに「主役の論文を引き当てる
                # 言い換え」か「各論文を個別に取りに行く分解」のどちらかが欠ける。
                "The evidence may live in a single paper or be spread across several. "
                "Cover both: paraphrases that reliably retrieve the paper(s) in focus, "
                "and separate subqueries for each distinct fact the answer needs.",
                self._constraint_note(attribute_filter),
                CORPUS_NOTE,
                f"Decompose it into {self.subquery_count} short, self-contained search "
                "subqueries.",
                'Respond with JSON only, in the form {"subqueries": ["...", "..."]}.',
            )
            if part
        )
        subqueries = self._ask_for_list(prompt, "subqueries")[: self.subquery_count]
        return subqueries or [query.question]

    # ---- 2. 候補の組み立て ------------------------------------------------

    def _candidate_papers(
        self, results: list[RetrievalResult]
    ) -> list[tuple[str, list[RetrievalResult]]]:
        """統合済みランキングを論文単位にまとめ、スコア上位の論文を候補として返す。

        引数は `_merged_results()` の出力。既定（`subquery_merge: "max"`）では
        従来どおり `chunks.values()` がそのまま渡ってくる。
        """
        by_paper: dict[str, list[RetrievalResult]] = {}
        for result in results:
            by_paper.setdefault(result.paper_id, []).append(result)

        for results in by_paper.values():
            results.sort(key=lambda r: r.score, reverse=True)

        ranked = sorted(
            by_paper.items(), key=lambda item: item[1][0].score, reverse=True
        )
        return [
            (paper_id, results[: self.chunks_per_paper])
            for paper_id, results in ranked[: self.max_candidates]
        ]

    # ---- 3. 読解と判定 ----------------------------------------------------

    def _read_and_judge(
        self,
        query: Query,
        candidates: list[tuple[str, list[RetrievalResult]]],
        chunks: dict[str, RetrievalResult],
    ) -> dict | None:
        """候補チャンクを LLM に読ませ、根拠になる論文とチャンクを選ばせる。"""
        if not candidates:
            return None

        listing = "\n\n".join(
            self._format_paper(paper_id, results)
            for paper_id, results in candidates
        )
        prompt = (
            "You are reading excerpts from papers returned by a search and selecting "
            "only the papers that are truly needed as evidence to answer the "
            "question.\n\n"
            f"Question: {query.question}\n"
            f"{self._format_answer_spec(query)}\n\n"
            "Candidates (most relevant first; each chunk is an excerpt of a paper's "
            "body, table, or figure caption):\n"
            f"{listing}\n\n"
            "After reading the excerpts, determine the following.\n"
            "1. Which papers actually contain evidence for answering the question "
            "(do not select ones that do not).\n"
            "2. For each paper, which chunk_ids are the evidence.\n"
            "3. Whether this fully answers the question. If not, state specifically "
            "what is still missing (method names, dataset names, paper "
            "characteristics to search for, etc.).\n\n"
            "Do not invent any paper_id / chunk_id that is not in the candidate list.\n"
            "Respond with JSON only, in the following form:\n"
            '{"papers": [{"paper_id": "...", "evidence_chunk_ids": ["..."]}], '
            '"sufficient": true, "missing": ""}'
        )

        parsed = self._ask_for_json(prompt)
        if parsed is None:
            return None

        papers = parsed.get("papers")
        if not isinstance(papers, list):
            return None

        candidate_ids = {paper_id for paper_id, _ in candidates}
        paper_ids: list[str] = []
        evidence_chunk_ids: list[str] = []
        for item in papers:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id", ""))
            if paper_id not in candidate_ids or paper_id in paper_ids:
                continue
            paper_ids.append(paper_id)
            for chunk_id in item.get("evidence_chunk_ids") or []:
                chunk_id = str(chunk_id)
                # LLM の捏造を弾く。実在するチャンクで、かつその論文のものだけ通す。
                result = chunks.get(chunk_id)
                if result is not None and result.paper_id == paper_id:
                    evidence_chunk_ids.append(chunk_id)

        if not paper_ids:
            return None

        # 本数の打ち切りはここではやらない。paper_cutoff で一括して決める
        # （比較実験で本数の決め方を揃えられるようにするため）。
        return {
            "paper_ids": paper_ids,
            "evidence_chunk_ids": evidence_chunk_ids,
            "sufficient": bool(parsed.get("sufficient")),
            "missing": str(parsed.get("missing") or ""),
        }

    def _format_paper(self, paper_id: str, results: list[RetrievalResult]) -> str:
        head = results[0]
        title = (head.metadata or {}).get("title", "")
        venue = (head.metadata or {}).get("venue", "")
        year = (head.metadata or {}).get("year", "")
        lines = [f"[paper_id: {paper_id}] {title} ({venue} {year})"]
        for result in results:
            metadata = result.metadata or {}
            where = [f"type={result.chunk_type}"]
            for key in ("page", "section", "table_id", "figure_id", "equation_id"):
                if metadata.get(key) is not None:
                    where.append(f"{key}={metadata[key]}")
            lines.append(f"  - chunk_id: {result.chunk_id} ({', '.join(where)})")
            lines.append(f"    {result.text[: self.snippet_chars]}")
        return "\n".join(lines)

    def _format_answer_spec(self, query: Query) -> str:
        parts = [f"Answer format: {', '.join(query.answer_types) or '(unspecified)'}"]
        if query.table_schema:
            columns = ", ".join(str(c.get("name")) for c in query.table_schema)
            parts.append(f"Answer table columns: {columns}")
        return "\n".join(parts)

    # ---- 4. 不足分の再分解 ------------------------------------------------

    def _refine(
        self,
        query: Query,
        missing: str,
        tried: list[str],
        attribute_filter=None,
        runs: list[SubqueryRun] | None = None,
        merged: list[RetrievalResult] | None = None,
    ) -> list[str]:
        """「何が欠けているか」を踏まえて、次に投げる検索サブクエリを作る。"""
        tried_text = "\n".join(f"- {sq}" for sq in dict.fromkeys(tried))
        missing_text = missing or "(No specific note from the LLM. Search from a different angle.)"
        prompt = "\n\n".join(
            part
            for part in (
                f"Original question: {query.question}",
                "The search so far still lacks sufficient evidence. What is missing:\n"
                f"{missing_text}",
                "Search subqueries already tried (do not repeat the same or similar ones):\n"
                f"{tried_text}",
                # 検索結果への接地。既定では空文字なのでプロンプトは従来と同一。
                self._grounding_note(runs, merged),
                # ここを省くと LLM は「もっと絞り込んだ検索」として site: や
                # filetype: を付け始める（実測で 29〜41%）。CORPUS_NOTE 参照。
                CORPUS_NOTE,
                self._constraint_note(attribute_filter),
                # **本数を書かないと LLM は出したいだけ出す**（実測 平均8.2〜9.3本・
                # 最大20本）。中身は「手法名 × 言い回し」の総当たりに流れていて、
                # q_021 の step1 は `SimLingo trained on Bench2Drive Base split` /
                # `SimLingo only uses the Bench2Drive Base dataset` … を20本並べていた。
                # 検索1本ぶんのコストは reranker の pool_k 件推論なので、これが
                # そのまま走行時間を数倍にする。
                f"Propose at most {self.subquery_count} new search subqueries to fill "
                "this gap. Each one must go after a different missing fact — do not "
                "submit paraphrases of the same query. "
                "If further searching is unlikely to find anything, return an empty list.\n"
                'Respond with JSON only, in the form {"subqueries": ["...", "..."]}.',
            )
            if part
        )
        # プロンプトの上限は守られないことがあるので、ここでも必ず切る。
        return self._ask_for_list(prompt, "subqueries")[: self.subquery_count]

    def _rewrite_subqueries(
        self,
        query: Query,
        missing: str,
        tried: list[str],
        attribute_filter,
        runs: list[SubqueryRun],
        merged: list[RetrievalResult],
    ) -> list[str]:
        """検索結果を材料にして次のサブクエリを作る（`_refine()` の置き換え）。

        **材料は2系統**（詳細は agent/rewrite.py）:

          * A = 検索の上位論文。ヒットしたチャンクを見せるので「いま質問のどの側面に
            当たっているか」が分かる。加えてタイトルだけの俯瞰リストで軸ズレを見せる。
          * B = 論文→論文展開の上位論文。**候補にまだ入っていない論文**の自己記述で、
            質問文には出てこない語彙（手法族の名前、拡張元のベースライン名）を供給する。

        LLM 呼び出しは `_refine()` と同じ1回。展開はここで呼ぶので、
        `_build_prediction()` の `_combine_rrf()` とは別に走る（同じ expander を
        使い回すだけで、B の順位そのものには手を触れない）。
        """
        candidates = self._candidate_papers(merged)
        material_a = self.rewriter.material_a(candidates, self.snippet_chars)
        listing_a = self.rewriter.listing_a(merged)

        material_b = ""
        if self.paper_expander is not None:
            ranked = to_gold_papers(merged)
            # **既存候補を除いた**近傍だけを材料にする。書き換えの狙いは
            # 「まだ候補圏内に居ない論文の語彙」なので、既に居る論文を材料に
            # しても取りに行く先が無い。`rank()` は統合のために既存候補を
            # 落とさないので、ここで落とす（除外分だけ本数は減る）。
            seen = set(ranked)
            related = [p for p in self.paper_expander.rank(ranked) if p not in seen]
            material_b = self.rewriter.material_b(
                related, query.question, self.snippet_chars
            )

        prompt = self.rewriter.prompt(
            question=query.question,
            material_a=material_a,
            listing_a=listing_a,
            material_b=material_b,
            tried=tried,
            dead=self._dead_subqueries(runs, merged),
            corpus_note=CORPUS_NOTE,
            constraint_note=self._constraint_note(attribute_filter),
            missing=missing,
        )
        return self._ask_for_list(prompt, "subqueries")

    def _dead_subqueries(
        self,
        runs: list[SubqueryRun] | None,
        merged: list[RetrievalResult] | None,
        top_n: int | None = None,
    ) -> list[str]:
        """上位N本の論文に1本も残せなかったサブクエリ（＝効かなかったもの）。

        `_grounding_note()` と書き換えプロンプトの両方が使う。
        """
        if not runs or not merged:
            return []
        limit = top_n if top_n is not None else self.grounded_refine_top_n
        top: list[str] = []
        for result in merged:
            if result.paper_id not in top:
                top.append(result.paper_id)
            if len(top) >= limit:
                break
        top_set = set(top)
        dead = [
            run.subquery
            for run in runs
            if not any(result.paper_id in top_set for result in run.results)
        ]
        return list(dict.fromkeys(dead))

    def _grounding_note(
        self,
        runs: list[SubqueryRun] | None,
        merged: list[RetrievalResult] | None,
    ) -> str:
        """再分解のプロンプトに「コーパスが実際に何を返したか」を足す（ADORE 系）。

        `_refine()` の材料は本来「読解 LLM の `missing`」だけで、**コーパスの反応を
        一度も見ていない**。実測の症状:

        * `EasySpec` を分光解析ソフトと誤解したまま2ステップ暴走した（step0 では
          gold を1位で引けていたのに、上位候補を見ていないので誤解に気づけない）。
        * 空振りしたサブクエリの語をそのまま言い換え続けた（`Ours500→1` /
          `Ours 500→1` / `500:1` / `500 to 1`）。「similar を繰り返すな」という指示が
          意味の転換ではなく表記ゆれの総当たりを生んでいた。

        そこで2つ渡す。**追加の LLM 呼び出しはゼロ**（プロンプトが太るだけ）。

        1. いま候補上位 N 本が何なのか（`[venue year] title`）。上位が全部
           speculative decoding 系だと見えれば、誤った前提はその場で崩れる。
        2. 各サブクエリの寄与。上位 N 本に1本も残っていないサブクエリを
           「効かなかった」として名指しする。
        """
        if not self.grounded_refine or not merged or not runs:
            return ""

        # 統合済みランキングを論文順に潰す（同じ論文の複数チャンクは1本に数える）。
        top_papers: list[str] = []
        head: dict[str, RetrievalResult] = {}
        for result in merged:
            if result.paper_id not in head:
                head[result.paper_id] = result
                top_papers.append(result.paper_id)
            if len(top_papers) >= self.grounded_refine_top_n:
                break
        if not top_papers:
            return ""

        lines = []
        for rank, paper_id in enumerate(top_papers, 1):
            metadata = head[paper_id].metadata or {}
            venue = metadata.get("venue", "")
            year = metadata.get("year", "")
            lines.append(f"{rank}. [{venue} {year}] {metadata.get('title', '')}")
        listing = "\n".join(lines)

        top_set = set(top_papers)
        dead = [
            run.subquery
            for run in runs
            if not any(result.paper_id in top_set for result in run.results)
        ]
        # 同じサブクエリを複数ステップで投げていることがあるので重複を潰す。
        dead_text = "\n".join(f"- {sq}" for sq in dict.fromkeys(dead))

        parts = [
            "This is what the corpus actually returned. The current top "
            f"{len(top_papers)} candidate papers are:\n{listing}",
        ]
        if dead_text:
            parts.append(
                "These subqueries contributed nothing to the list above "
                f"(none of their hits are in it) — do not rephrase them:\n{dead_text}"
            )
        parts.append(
            "Use this feedback: build on the wording that did surface relevant papers, "
            "target the aspects that are still not covered by the list, and abandon the "
            "vocabulary that produced nothing. If the list shows that the question was "
            "misunderstood (the papers are about a different topic than you assumed), "
            "correct the assumption instead of rephrasing."
        )
        return "\n\n".join(parts)

    # ---- 5. 提出物の組み立て ----------------------------------------------

    def _build_prediction(
        self,
        query: Query,
        verdict: dict | None,
        chunks: dict[str, RetrievalResult],
        trace: list[dict],
        runs: list[SubqueryRun] | None = None,
    ) -> Prediction:
        # 反復中に溜めたチャンクを論文順位に直したもの。打ち切り前の「検索が拾えた候補」
        # なので、recall@k の分析はこちらを見る必要がある（gold_papers は LLM 選定と
        # cutoff の後なので、検索力と選定力が混ざってしまう）。
        merged = self._merged_results(runs or [], chunks)
        merged = self._rescore_pool(query, merged, trace)
        if self.paper_expander is not None:
            # 論文→論文展開は**A/B の RRF 統合のみ**（位置挿入は順位融合に全指標で
            # 負けたので実装ごと削除した。詳細は CLAUDE.md）。
            #
            # 統合では**50件で切る前の全長**を A のランキングとして使う。
            # 51位の論文を関連ランキングが強く推していても、先に切ってしまうと
            # そもそも押し上げようがないため。切るのは統合したあと。
            candidate_papers = to_gold_papers(
                merged, skip_chunk_types=self.paper_score_skip_chunk_types
            )
            # 生質問1位の固定は**統合の前**（A を差し替えるので B の起点も変わる）。
            candidate_papers = self._pin_front(
                candidate_papers, self._rawq_ranking(query, runs or [])
            )
            if candidate_papers:
                # `anchor_from: verdict` のときだけ使う。verdict が None のクエリでは
                # 空リストになり、`_anchor_papers()` が従来の起点に戻す。
                # スコアは `anchor_from: score`（LLM 不要の起点）でだけ読まれる。
                candidate_papers = self._combine_rrf(
                    candidate_papers,
                    trace,
                    [] if verdict is None else verdict["paper_ids"],
                    paper_scores(
                        merged, skip_chunk_types=self.paper_score_skip_chunk_types
                    ),
                )
            candidate_papers = candidate_papers[:CANDIDATE_PAPERS_LIMIT]
        else:
            candidate_papers = to_gold_papers(
                merged,
                max_papers=CANDIDATE_PAPERS_LIMIT,
                skip_chunk_types=self.paper_score_skip_chunk_types,
            )
            candidate_papers = self._pin_front(
                candidate_papers, self._rawq_ranking(query, runs or [])
            )

        # **どれを提出するかは選ばない**（`submit_from: candidates`、既定）。検索の
        # 順位をそのまま渡し、選定は読解チーム側に任せる。`submit_from: llm` にすると
        # 従来どおり読解 LLM の選定結果を使う（選定込みで測りたい ablation 用）。
        # verdict が None（LLM が一度も使える判定を返さなかった）ときも順位のまま出す。
        # 直後の apply_paper_cutoff が最大 max_papers 本に切るので、上限50でも等価。
        evidence: list[Evidence] = []
        if verdict is None or self.submit_from == "candidates":
            ranked = candidate_papers
        else:
            ranked = verdict["paper_ids"]

        paper_ids = apply_paper_cutoff(
            ranked, query, self.task_family, self.paper_cutoff, self.max_papers
        )

        evidence_results: list[RetrievalResult] = []
        if verdict is not None:
            # 打ち切りで落ちた論文の evidence は出さない。
            kept = set(paper_ids)
            evidence_results = [
                chunks[chunk_id]
                for chunk_id in dict.fromkeys(verdict["evidence_chunk_ids"])
                if chunks[chunk_id].paper_id in kept
            ]
            evidence = [evidence_from_result(r) for r in evidence_results]

        # **回答は生成しない**（freeform / multiple_choice / table は読解チーム側の担当）。
        # 空の Answer をそのまま置く。1クエリにつき LLM 呼び出しが1回減る。
        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=evidence,
            answer=Answer(),
            trace=trace,
            candidate_papers=candidate_papers,
        )

    def _rescore_pool(
        self, query: Query, results: list[RetrievalResult], trace: list[dict]
    ) -> list[RetrievalResult]:
        """プール全体を**元の質問**で1回リランクし、必要なら上位 N 件に剪定する。

        サブクエリ間のスコア非可換性の根治。`subquery_merge: rrf` が「順位しか見ない」
        ことで回避するのに対し、こちらは全チャンクを同じクエリで測り直して直接解消する。
        **サブクエリではなく元の質問で測る**のが要点で、そうしないと同じ問題が残る。

        **既定オフのまま出す。** reranker は「質問に答える論文」を選べているほど
        no_evidence gold（質問文が名指ししないピア論文）を落とすので、
        ecr が上がって cr が下がる向きに出やすい。評価は cr と ecr を必ず並べて読む。

        reranker が無い構成・`NoneReranker` の構成では黙って skip する
        （`_expansion_reranker()` と同じ流儀）。
        """
        if not self.pool_rescore and self.pool_prune_to is None:
            return results

        if self.pool_rescore:
            reranker = self._reranker()
            if reranker is not None and results:
                results = list(reranker.rerank(query.question, results, len(results)))
                trace.append({"pool_rescore": {"n_chunks": len(results)}})

        if self.pool_prune_to is not None:
            results = results[: self.pool_prune_to]
        return results

    # ---- 生質問1位のピン留め（`rawq_pin`） ---------------------------------

    def _rawq_ranking(self, query: Query, runs: list[SubqueryRun]) -> list[str]:
        """元の質問1本の検索が返した論文順位の上位 `rawq_pin` 本。

        `run()` が step0 の末尾に足した run を**サブクエリ文字列で引き当てる**
        （その run だけを論文順位に直すので、他のサブクエリの結果は混ざらない）。
        無効なとき・見つからないときは空を返し、`_pin_front()` が素通しになる。
        """
        if not self.rawq_pin:
            return []
        for run in runs:
            if run.step == 0 and run.subquery == query.question:
                ranked = to_gold_papers(
                    run.results, skip_chunk_types=self.paper_score_skip_chunk_types
                )
                return ranked[: self.rawq_pin]
        return []

    @staticmethod
    def _pin_front(ranked: list[str], pinned: list[str]) -> list[str]:
        """`pinned` を先頭に置き、残りは元の順序のまま詰める（重複は落とす）。"""
        if not pinned:
            return ranked
        head = list(dict.fromkeys(pinned))
        seen = set(head)
        return head + [p for p in ranked if p not in seen]

    def _anchor_papers(
        self,
        candidate_papers: list[str],
        verdict_papers: list[str] | None,
        paper_scores: dict[str, float] | None = None,
    ) -> list[str] | None:
        """ランキングB の起点。既定（`anchor_from` 未指定）では None を返す。

        `anchor_from: verdict` のとき、**読解 LLM が本文を読んで根拠を確認した論文**
        （`_read_and_judge()` の paper_ids）を候補1位と合わせて起点にする。

        `anchor_from: score` は**その LLM 不要版**。読解を走らせない構成
        （`eval_retrieval.py` の検索単体＝検索エージェント抜き）では verdict が
        存在せず、起点が候補1位だけに落ちて verdict 版との差がそのまま失われる。
        reranker のスコアは「質問に答えるか」の yes 確率なので、
        **しきい値を超えた論文を「確認済み」の代わりに使う**。
        判定に LLM を1回も呼ばない。

        **候補1位を必ず先頭に残す。** LLM の確認済みだけにすると、候補1位が B の
        先頭から外れて「A にも B にも居る」論文に抜かれ、**single_paper の cr@1 が
        0.923 -> 0.885 に落ちた**（`anchors: 3` を入れたときと同じ事故）。
        和集合にすれば single は完全に不変のまま伸びだけ残る。

        **効くのは精度ではなく本数。** LLM 確認済みの gold 率は 68本中52本 = 76% で、
        候補1位の 85%（47/55）より**低い**。それでも効くのは、1本の anchor では
        1つのトピッククラスタしか展開できないため。anchor が2本以上になるのは
        55件中16件で、**うち14件が multi_paper**（single に副作用が出ないのはこのため）。

        実測（土台4本 + 全長A、`rrfk10.yaml` 土台。single の cr@1 は5条件とも不変）:

            土台           multi@5          multi@10         multi@20
            k100_cand50   0.682 -> 0.717   0.767 -> 0.830   0.869 -> 0.885
            chunk_cand50  0.610 -> 0.686   0.762 -> 0.842   0.899 -> 0.908
            fat           0.666 -> 0.669   0.809 -> 0.848   0.914 -> 0.914
            b_merged      0.680 -> 0.726   0.786 -> 0.868   0.876 -> 0.899
            全長A          0.683 -> 0.677   0.814 -> 0.844   0.910 -> 0.914

        **伸びが @10 に集中する**のは、沈んでいるのがそこだから。multi の根拠あり
        gold をクエリ内で順位順に並べると、1本目は中央1位（@5 で 100%）なのに
        2本目が中央4位・3本目が中央8位・4本目が中央14位まで落ちる。
        """
        anchor_from = getattr(self.paper_expander, "anchor_from", None)
        # 既定は上限なし。verdict 版は無制限で検証したので、ここで切ると挙動が変わる。
        limit = getattr(self.paper_expander, "anchor_max", None)

        def capped(papers: list[str]) -> list[str]:
            return papers if limit is None else papers[:limit]

        if anchor_from == "score":
            if not paper_scores or not candidate_papers:
                return None
            minimum = getattr(self.paper_expander, "anchor_score_min", None)
            ratio = getattr(self.paper_expander, "anchor_score_ratio", None)
            top = paper_scores.get(candidate_papers[0], 0.0)
            picked = [
                paper
                for paper in candidate_papers
                if (minimum is None or paper_scores.get(paper, 0.0) >= minimum)
                and (ratio is None or paper_scores.get(paper, 0.0) >= top * ratio)
            ]
            # **候補1位は必ず先頭に残す**（verdict 版と同じ理由。外すと
            # single_paper の cr@1 が落ちる）。しきい値が厳しすぎて誰も残らない
            # ときは候補1位だけの起点になり、`anchor_from` 未指定と同じ挙動になる。
            return capped(list(dict.fromkeys(candidate_papers[:1] + picked)))

        if anchor_from != "verdict":
            return None
        if not verdict_papers:
            return None
        return capped(
            list(dict.fromkeys(candidate_papers[:1] + [p for p in verdict_papers if p]))
        )

    def _rank_related(self, rank_fn, seed: list[str], custom: list[str] | None):
        """関連ランキングB を作る。`custom` があいだに入るときだけ起点の本数を合わせる。

        各 expander は渡されたリストの先頭 `self.anchors` 本を起点にする
        （`retrieve/paper_expander.py` の `_pools()`）。起点を差し替えるときは
        本数もそれに合わせないと、集合の一部しか使われない。
        **呼び出しの前後で元の値に戻す**——`anchors` は yaml の設定値なので、
        クエリごとの都合で書き換えたままにしない。
        """
        if custom is None:
            return rank_fn(seed)
        expander = self.paper_expander
        sources = getattr(expander, "sources", None) or [expander]
        saved = [getattr(source, "anchors", 1) for source in sources]
        for source in sources:
            source.anchors = len(custom)
        try:
            return rank_fn(seed)
        finally:
            for source, previous in zip(sources, saved):
                source.anchors = previous

    def _combine_rrf(
        self,
        candidate_papers: list[str],
        trace: list[dict],
        verdict_papers: list[str] | None = None,
        paper_scores: dict[str, float] | None = None,
    ) -> list[str]:
        """ランキングを2本に分けて RRF 統合する（`combine: rrf`）。

        * **A（質問→論文）**: 検索そのもの。BM25 + 埋め込み → RRF → reranker。
        * **B（論文→論文）**: 論文間の近さ（SPECTER2 / 書誌結合 / 全文MLT の RRF 融合）。
          **reranker には通さない**——reranker は「質問に答えるか」で判定するので、
          質問文が名指ししないピア gold を必ず下げる。B をそこに晒さないのが要点。

            score(p) = w_A / (k + rank_A) + w_B / (k + offset + rank_B)

        かつてあった位置挿入（決まった位置に差し込む方式）との違いは2つ。
        **A にも B にも居る論文が加点される**こと（例: A 30位 × B 3位 → 実効1位相当）と、
        本数決め打ちが要らないこと。位置挿入は全指標で負けたので削除した。
        スコアで混ぜて壊れた（cr@20 0.822 -> 0.773）のは reranker の絶対スコアと
        展開の仮スコアを足していたからで、RRF は順位しか見ないのでその問題が無い。

        **素の RRF（`related_weight: 1.0` / `related_offset: 0`）が実測で最良**
        （オフライン55件: cr@20 0.789 -> 0.879、ecr@20 0.868 -> 0.926、
        multi@20 0.601 -> 0.770、cr@50 0.832 -> 0.917、ecr@50 0.889 -> 0.956。
        single_paper の cr@20 は 1.000 のまま）。重みを下げると B 単独の論文が A の裾より
        下に落ちて統合の意味が消え（w=0.5 で cr@20 0.817）、上げると B が候補列を
        占領する（w=2.0 で 0.830）。`related_offset` も 0 が最良で、15 にすると 0.839 まで
        下がる——位置挿入で効いていた「差し込み深さ」は、順位融合では下駄ではなく
        重なりの加点が担う。

        `consensus: true` にすると B を1本に潰さず、anchor × ソースごとの
        ランキングを別々の項として足す。複数の pool に居る論文が項の数だけ
        加点されるので「揃って推された」が信号になる（重みは pool 数で正規化する）。
        """
        expander = self.paper_expander
        # 起点をどこから取るか。既定（`anchor_from` 未指定）では候補列をそのまま渡し、
        # 各 expander が先頭 `anchors` 本を起点にする（従来と1ビットも変わらない）。
        custom = self._anchor_papers(candidate_papers, verdict_papers, paper_scores)
        seed = candidate_papers if custom is None else custom

        # `consensus: true` なら anchor（と融合ソース）ごとのランキングを潰さずに
        # 受け取り、1本ずつ RRF の項にする。**複数の pool に居る論文が項の数だけ
        # 加点される**ので、「2本の anchor が揃って推した」が信号になる
        # （既定の `rank()` は _interleave() で1本に潰すため、この情報が消える）。
        if getattr(expander, "consensus", False) and hasattr(expander, "rank_pools"):
            pools = [pool for pool in self._rank_related(expander.rank_pools, seed, custom) if pool]
        else:
            related = self._rank_related(expander.rank, seed, custom)
            pools = [related] if related else []
        if not pools:
            return candidate_papers

        # **anchor 自身をランキングB の先頭に置く。** 各 expander は anchor を自分の
        # 近傍から外すので、そのままだと anchor は A の 1/(k+1) しか持てず、
        # 「A にも B にも居る」論文（2項ぶん）に軒並み抜かれる。実測では
        # single_paper 2件で gold そのものだった候補1位が top20 から消えた。
        # 論文は自分自身に最も近いので、B の1位に置くのが定義どおりでもある。
        anchors = custom if custom is not None else candidate_papers[: getattr(expander, "anchors", 1)]
        pools = [anchors + [p for p in pool if p not in anchors] for pool in pools]

        k = getattr(expander, "combine_rrf_k", 60)
        weight = getattr(expander, "related_weight", 1.0)
        offset = getattr(expander, "related_offset", 15)
        # **B の重みは pool 数で割る。** consensus ではソース3 × anchor3 = 9本まで
        # 増えるので、割らないと B 側の項が9倍になり A（1項）を圧倒する
        # （related_weight: 2.0 の時点で既に候補列を占領するのが実測済み）。
        # 割っても「複数 pool に居る論文が単独より高い」関係は保たれるので、
        # A/B の釣り合いだけを元に戻すことになる。
        weight /= len(pools)

        scores = {paper_id: 1.0 / (k + rank + 1) for rank, paper_id in enumerate(candidate_papers)}
        for pool in pools:
            for rank, paper_id in enumerate(pool):
                scores[paper_id] = scores.get(paper_id, 0.0) + weight / (k + offset + rank + 1)
        # 同点は挿入順（= A の順位が先、次に B の順位）で決まる。sorted が安定なため。
        fused = sorted(scores, key=lambda paper_id: -scores[paper_id])

        before = {paper_id: rank for rank, paper_id in enumerate(candidate_papers)}
        head = fused[: self.max_candidates]
        trace.append(
            {
                "paper_fusion": {
                    "anchor": candidate_papers[0],
                    # 実際に使った起点。既定では [candidate_papers[0]] と同じだが、
                    # `anchor_from: verdict` では読解 LLM の確認済み論文が並ぶ。
                    "anchors": anchors,
                    "n_a": len(candidate_papers),
                    "n_b": len({p for pool in pools for p in pool}),
                    "n_pools": len(pools),
                    # B にしか居なかったのに上位圏へ入った論文（新規に拾えた分）。
                    "b_only_promoted": [p for p in head if p not in before],
                    # A では上位圏の外だったのに押し上がった論文（順位調整が効いた分）。
                    "promoted": [
                        p for p in head if before.get(p, 0) >= self.max_candidates
                    ],
                }
            }
        )
        return fused

    def _reranker(self):
        """retriever が持つ reranker。無い / NoneReranker なら None。

        `_expansion_reranker()`（展開論文の rerank）と `_rescore_pool()`（プールの
        再スコア）の両方が通る。検索と同じインスタンスなのでスコアの尺度が揃う。
        """
        reranker = getattr(self.retriever, "reranker", None)
        if reranker is None or type(reranker).__name__ == "NoneReranker":
            return None
        return reranker

    # ---- LLM 呼び出しの薄いラッパ ------------------------------------------

    def _ask_for_json(self, prompt: str) -> dict | None:
        try:
            return parse_json_object(self.llm(prompt))
        except Exception:
            return None

    def _ask_for_list(self, prompt: str, key: str) -> list[str]:
        parsed = self._ask_for_json(prompt)
        if not parsed:
            return []
        values = parsed.get(key)
        if not isinstance(values, list):
            return []
        return [str(v) for v in values if v]
