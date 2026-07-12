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
from litqa.contracts import Query
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

- {single}: 根拠が1本の論文の中に閉じている。複数の手法名が並んでいても、それらが
  1本の論文の比較表・実験表にまとめて載っているなら、これに当たる。
- {multi}: 根拠が複数の論文にまたがる。各手法・各値をそれぞれ別の論文から集める
  必要がある場合や、条件に当てはまる論文を列挙させる場合はこれに当たる。

質問: {question}
回答形式: {answer_types}{table_schema}

出力は JSON のみとし、{{"task_family": "{single}" または "{multi}"}} の形式で答えてください。"""


class TaskFamilyClassifier:
    """Query の task_family を解決する。入力にあればそれを使い、無ければ推定する。

    LLM を渡すと LLM で推定し、渡さない（または LLM が失敗した）場合は
    heuristic_task_family() にフォールバックする。同じ query_id への再問い合わせは
    キャッシュして LLM の呼び出し回数を抑える（IterativeAgent は1クエリ中に複数回引く）。
    """

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm
        self._cache: dict[str, str] = {}

    def infer(self, query: Query) -> str:
        if query.task_family:
            return query.task_family
        cached = self._cache.get(query.query_id)
        if cached is not None:
            return cached

        task_family = None
        if self.llm is not None:
            task_family = self._infer_with_llm(query)
        if task_family is None:
            task_family = heuristic_task_family(query)

        self._cache[query.query_id] = task_family
        return task_family

    def _infer_with_llm(self, query: Query) -> str | None:
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
