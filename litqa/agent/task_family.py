"""question から task_family（single/multi）を推定し、返す論文数の上限を決めるモジュール。

本番の入力には task_family が無い（query_id / question / answer_types / table_schema の
4つだけ）。一方 gold_papers の件数は task_family でほぼ決まっており、手元の検証データでは

    hidden_source_single_paper: 26件すべて gold_papers = 1本
    multi_paper:               29件中27件が gold_papers = 4本（範囲 3〜9）

となる。論文集合は F1 で採点されるので、件数を絞らず上位を全部返すと precision が落ちる。
そこで task_family が入力に無い場合は question から推定して従来の cutoff を復元する。

推定は LLM を第一手段とする。ヒューリスティック（下記）は手元の55件で正解率0.67程度しか
出ず（多数決ベースラインが0.53）、単独では当てにならないため、LLM が無い場合と
LLM 呼び出しが失敗した場合のフォールバックとしてのみ使う。
"""

from __future__ import annotations

import re

from litqa.agent.json_utils import parse_json_object
from litqa.contracts import Query, RetrievalResult
from litqa.llm.base import LLMClient

SINGLE = "hidden_source_single_paper"
MULTI = "multi_paper"

# 最終的に提出する論文数の上限。
CUTOFF_BY_TASK_FAMILY = {
    SINGLE: 2,
    MULTI: 5,
}

# 反復検索を打ち切ってよいと判断する、発見済み論文数のしきい値。
SUFFICIENT_COUNT_BY_TASK_FAMILY = {
    SINGLE: 1,
    MULTI: 4,
}

_ENUMERATION_PATTERNS = [
    re.compile(r"\bwhich papers\b", re.IGNORECASE),
    re.compile(r"\bwhat papers\b", re.IGNORECASE),
    re.compile(r"\blist (all|the)\b", re.IGNORECASE),
    re.compile(r"\ball (of the )?papers (that|which)\b", re.IGNORECASE),
    re.compile(r"\bevery paper\b", re.IGNORECASE),
    re.compile(r"\bhow many papers\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bcomparison (of|between)\b", re.IGNORECASE),
    re.compile(r"\bacross (the )?(papers|studies|works)\b", re.IGNORECASE),
    re.compile(r"\bboth .+ and .+\b", re.IGNORECASE),
]

_PAPERS_PLURAL_RE = re.compile(r"\bpapers\b", re.IGNORECASE)

# "TCM, sCT, ECM-XL, and ECM" のように3つ以上の対象を並べる書き方。
# 各対象がそれぞれ別論文で提案された手法であることが多く、multi_paper の手掛かりになる。
_ITEM_LIST_RE = re.compile(r",\s*[^,]+,?\s+and\s+", re.IGNORECASE)

# LLM に見せる候補論文の数。多すぎるとノイズになる。
_MAX_CANDIDATE_PAPERS = 10


def is_enumeration_or_comparison(question: str) -> bool:
    """質問文が列挙・比較型（複数論文にまたがりやすい）パターンかどうかを判定する。"""
    return any(pattern.search(question) for pattern in _ENUMERATION_PATTERNS)


def heuristic_task_family(query: Query) -> str:
    """LLM を使わずに task_family を推定する。フォールバック用で、精度は高くない。"""
    question = query.question
    if (
        is_enumeration_or_comparison(question)
        or _PAPERS_PLURAL_RE.search(question)
        or _ITEM_LIST_RE.search(question)
    ):
        return MULTI
    return SINGLE


_PROMPT = """あなたは、科学論文コーパスへの質問が「1本の論文だけで答えられるか」
「複数の論文から根拠を集める必要があるか」を判定しています。

判定の鍵は「その手法・数値が、**どの論文の成果として報告されたものか**」です。
手法名がいくつ並んでいるかではありません。

- {multi}: 並んでいる手法が**それぞれ別の論文で提案されたもの**で、各論文が自分の
  成果として報告した値を集める必要がある。手法が**対等に列挙**されていて主役がいない
  （例:「A, B, C, D の Tiny ImageNet 精度は？」— A〜D は各自の論文を持つ）。
  条件に当てはまる論文を列挙させる質問もこれ。

- {single}: **主役となる論文が1本あり**、他の手法はその論文が自分と比較するために
  表に載せたベースラインにすぎない。「X の（Y という条件での）性能は？」のように
  **1つの手法に焦点があり、他は引き立て役**（例:「DetAny3D の Omni3D での AP は？
  Cube R-CNN の 2D detections を使う条件で」— 主役は DetAny3D）。
  1本の論文の中の記述・図・参考文献について訊く質問もこれ。

迷ったら「対等な列挙か（→{multi}）／主役と脇役か（→{single}）」で判断してください。
安易に「1本の比較表に全部載っているかもしれない」と考えないこと。別々の論文で提案
された手法の値が1本の表に揃っていることは稀です。

質問: {question}
回答形式: {answer_types}{table_schema}

出力は JSON のみとし、{{"task_family": "{single}" または "{multi}"}} の形式で答えてください。"""


_CANDIDATES_PROMPT = """検索でヒットした候補論文のタイトルを見て、判定を確かめてください。

質問に出てくる手法名それぞれについて、**その手法を提案した論文が候補の中に別々に
存在するか**を見ます。
- A の論文、B の論文、C の論文…が**それぞれ別に**候補に挙がっている -> {multi}
- 1本の論文が明らかに主役で、他の手法名はその論文が比較に使ったベースライン -> {single}

候補論文（関連度が高い順）:
{candidates}
"""


class TaskFamilyClassifier:
    """Query の task_family を解決する。入力にあればそれを使い、無ければ推定する。

    LLM を渡すと LLM で推定し、渡さない（または LLM が失敗した）場合は
    heuristic_task_family() にフォールバックする。同じ query_id への再問い合わせは
    キャッシュして LLM の呼び出し回数を抑える（IterativeAgent は1クエリ中に複数回引く）。

    **検索してから判定する。** 質問文だけからの推定は実測で正解率0.673 が頭打ちだった
    （ヒューリスティックも LLM も同じ）。「NCFM, AP-BPTT, ATT, DEDA の精度は？」を見て、
    それが4本の別論文なのか1本の比較表なのかは、質問文には書かれていないため。
    候補論文のタイトルを見れば「4本とも別々の論文が主役だ」と分かる可能性がある。
    そのため infer() は candidates を受け取れる。

    候補なしで一度推定した結果は暫定値として扱い、後から候補付きで呼ばれたら
    推定し直してキャッシュを更新する（ReadingAgent は検索前の分解で先に引くため）。
    """

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm
        # query_id -> (task_family, 候補を見て判定したか)
        self._cache: dict[str, tuple[str, bool]] = {}

    def infer(
        self, query: Query, candidates: list[RetrievalResult] | None = None
    ) -> str:
        if query.task_family:
            return query.task_family

        cached = self._cache.get(query.query_id)
        if cached is not None:
            value, with_candidates = cached
            # 候補を見て出した判定は確定。候補なしの判定は暫定なので上書きを許す。
            if with_candidates or candidates is None:
                return value

        task_family = None
        if self.llm is not None:
            task_family = self._infer_with_llm(query, candidates)
        if task_family is None:
            task_family = heuristic_task_family(query)

        self._cache[query.query_id] = (task_family, candidates is not None)
        return task_family

    def _format_candidates(self, candidates: list[RetrievalResult]) -> str:
        """候補を論文単位に畳んで、タイトルを一覧にする。"""
        seen: dict[str, str] = {}
        for result in candidates:
            if result.paper_id in seen:
                continue
            metadata = result.metadata or {}
            title = metadata.get("title") or result.text[:80]
            venue = metadata.get("venue", "")
            year = metadata.get("year", "")
            seen[result.paper_id] = f"- {title} ({venue} {year})"
        return "\n".join(list(seen.values())[:_MAX_CANDIDATE_PAPERS])

    def _infer_with_llm(
        self, query: Query, candidates: list[RetrievalResult] | None = None
    ) -> str | None:
        if query.table_schema:
            columns = ", ".join(str(c.get("name")) for c in query.table_schema)
            table_schema = f"\n回答テーブルの列: {columns}"
        else:
            table_schema = ""
        prompt = _PROMPT.format(
            single=SINGLE,
            multi=MULTI,
            question=query.question,
            answer_types=", ".join(query.answer_types) or "（指定なし）",
            table_schema=table_schema,
        )
        if candidates:
            prompt += "\n\n" + _CANDIDATES_PROMPT.format(
                single=SINGLE,
                multi=MULTI,
                candidates=self._format_candidates(candidates),
            )
        try:
            parsed = parse_json_object(self.llm(prompt))
        except Exception:
            return None
        if not parsed:
            return None
        task_family = parsed.get("task_family")
        if task_family in (SINGLE, MULTI):
            return task_family
        return None


def cutoff_for(query: Query, classifier: TaskFamilyClassifier) -> int | None:
    """query に対して提出する論文数の上限を返す。"""
    return CUTOFF_BY_TASK_FAMILY.get(classifier.infer(query))


# 提出本数の決め方。比較実験では、これを揃えないと「エージェントの賢さ」ではなく
# 「本数の決め方」の差を測ってしまう（論文集合は F1 採点なので本数がスコアを支配する）。
#   "task_family": task_family の cutoff（single=2, multi=5）で機械的に切る
#   "llm":         LLM が選んだ本数をそのまま採用する（max_papers で頭打ち）
PAPER_CUTOFF_MODES = ("task_family", "llm")


def apply_paper_cutoff(
    paper_ids: list[str],
    query: Query,
    classifier: TaskFamilyClassifier,
    mode: str,
    max_papers: int,
) -> list[str]:
    """関連度順に並んだ paper_ids を、指定したモードで打ち切る。"""
    if mode not in PAPER_CUTOFF_MODES:
        raise ValueError(
            f"unknown paper_cutoff: {mode!r} (expected one of {PAPER_CUTOFF_MODES})"
        )
    if mode == "task_family":
        cutoff = cutoff_for(query, classifier)
        if cutoff is not None:
            return paper_ids[:cutoff]
        return paper_ids[:max_papers]
    return paper_ids[:max_papers]
