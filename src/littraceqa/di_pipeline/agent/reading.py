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
evidence まで。読解 LLM が返す `paper_ids` は反復の停止条件と evidence にだけ使い、
提出論文の選定には使わない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.agent.task_family import TaskFamilyClassifier, apply_paper_cutoff
from littraceqa.di_pipeline.contracts import Answer, Evidence, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.hybrid import (
    HybridRetriever,
    paper_scores,
    to_gold_papers,
)


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
    スコアが高いほうを残す」max マージなので、どのサブクエリの何位だったかを潰す。
    順位そのものを後から見たい用途（`scripts/run_search.py --dump-runs` が落とす
    オフライン再生の土台）のために、**ランキングの出所だけをここに残す**。
    `chunks` は chunk_id からの引き当て用（捏造チェックと evidence 引き）。
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
        # 切り詰めの両方に使う**（LLM は本数指定を守らないことがある）。
        # 増やしても検索力は上がらない — 実測は SUBQUERY_COUNT のコメント参照。
        subquery_count: int = SUBQUERY_COUNT,
        # サブクエリ1本の検索で retriever から受け取る**チャンク**数（論文数ではない）。
        # search_style 側の per_index_k / pool_k とは別物で、reranker が pool_k 件を
        # 全件スコアリングし終えたあとの「上位何件を受け取るか」。増やしても
        # reranker の推論は増えない（捨てていた分を拾うだけ）。
        retrieve_top_k: int = 20,
        # 1ステップで読解 LLM に読ませる論文の本数と、1論文あたりの見せ方。
        max_candidates: int = 15,
        chunks_per_paper: int = 2,
        snippet_chars: int = 1800,
        # 提出本数の決め方と上限。**どれを提出するかの選定は読解チーム側の担当**なので、
        # このエージェントは候補列の順位をそのまま渡して本数だけを切る。
        paper_cutoff: str = "task_family",
        max_papers: int = 10,
        # 論文→論文展開（retrieve/paper_expander.py）。質問文が名指ししないピア gold を
        # 拾うためのもので、None なら候補列は検索の順位そのまま。
        # **A/B の RRF 統合の設定もこのオブジェクトが持つ**（_combine_rrf 参照）。
        paper_expander: object | None = None,
        # **論文の代表スコアに使わないチャンク種別**（`["table"]` が実測での最良）。
        # 表チャンクは数値と短いラベルが密なので、論文が質問の主題でなくても
        # 表1枚で代表スコアが跳ね上がる。詳細は retrieve/hybrid.py の to_gold_papers。
        # **読解に渡すチャンクプールは変えない**ので evidence には従来どおり出せる。
        paper_score_skip_chunk_types: list[str] | None = None,
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
        self.paper_expander = paper_expander
        self.paper_score_skip_chunk_types = tuple(paper_score_skip_chunk_types or ())
        # scripts/run_search.py の --dump-runs が読む。Prediction.trace には入れない
        # （提出ファイルが膨らむため）。
        self.last_runs: list[SubqueryRun] = []
        self.task_family = TaskFamilyClassifier(llm)

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

            merged = self._merged_results(chunks)
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
            subqueries = self._refine(query, missing, tried, attribute_filter)
            if not subqueries:
                break

        self.last_runs = runs
        return self._build_prediction(query, verdict, chunks, trace)

    # ---- 検索1本ぶん --------------------------------------------------------

    def _retrieve(self, subquery: str, retrieve_kwargs: dict) -> list[RetrievalResult]:
        """サブクエリ1本を検索して上位 `retrieve_top_k` 件を返す。"""
        return list(self.retriever.retrieve(subquery, self.retrieve_top_k, **retrieve_kwargs))

    # ---- サブクエリ間のマージ ----------------------------------------------

    def _merged_results(self, chunks: dict[str, RetrievalResult]) -> list[RetrievalResult]:
        """全サブクエリの結果を1本のランキングに統合する。

        `chunks` の構築自体が「同じ chunk_id はスコアが高いほうを残す」max マージ
        なので、ここは貯め込んだものをそのまま列にするだけ。
        """
        return list(chunks.values())

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

        引数は `_merged_results()` の出力（貯め込んだチャンクをスコア順に見た列）。
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

    # ---- 5. 提出物の組み立て ----------------------------------------------

    def _build_prediction(
        self,
        query: Query,
        verdict: dict | None,
        chunks: dict[str, RetrievalResult],
        trace: list[dict],
    ) -> Prediction:
        # 反復中に溜めたチャンクを論文順位に直したもの。打ち切り前の「検索が拾えた候補」
        # なので、recall@k の分析はこちらを見る必要がある（gold_papers は LLM 選定と
        # cutoff の後なので、検索力と選定力が混ざってしまう）。
        merged = self._merged_results(chunks)
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
            if candidate_papers:
                # `anchor_from: verdict` のときだけ使う。verdict が None のクエリでは
                # 空リストになり、`_anchor_papers()` が従来の起点に戻す。
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

        # **どれを提出するかは選ばない。** 検索の順位をそのまま渡し、選定は読解チーム側に
        # 任せる（読解 LLM が返す `paper_ids` は停止条件と evidence にだけ使う）。
        # 直後の apply_paper_cutoff が最大 max_papers 本に切る。
        evidence: list[Evidence] = []
        paper_ids = apply_paper_cutoff(
            candidate_papers, query, self.task_family, self.paper_cutoff, self.max_papers
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

    # ---- 6. ランキング A/B の統合（論文→論文展開）--------------------------

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
