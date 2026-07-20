"""クエリを分解して検索するエージェント（ablation 用。実質 one-shot）。

注意: このエージェントの反復ループは事実上回らない。_is_sufficient() が
「見つかった論文の**本数** >= しきい値(single=1, multi=4)」でしか判定せず、found には
検索が返した論文が無条件に全部入るため、top_k=20 で引いた時点で初回から条件を満たし、
2周目に入らない（_refine は一度も呼ばれない）。「必要な論文が揃ったか」ではなく
「何本ヒットしたか」を数えているのが原因。

中身を読んで根拠を確認し、足りなければ本当に検索し直す実装は ReadingAgent（reading.py）。
こちらは「LLMによる分解あり・検証なし」のベースラインとして残してある。
"""

from __future__ import annotations

from littraceqa.di_pipeline.agent.json_utils import parse_json_object as _parse_json_object
from littraceqa.di_pipeline.agent.task_family import (
    CUTOFF_BY_TASK_FAMILY,
    MULTI,
    SUFFICIENT_COUNT_BY_TASK_FAMILY,
    TaskFamilyClassifier,
    is_enumeration_or_comparison as _is_enumeration_or_comparison,
)
from littraceqa.di_pipeline.contracts import Answer, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever, to_gold_papers


def _growth_rate(prev_count: int, new_count: int) -> float:
    """直前ステップからの新規発見論文数の相対成長率。prev_count が 0 の初回ステップは特別扱い。"""
    if prev_count == 0:
        if new_count > 0:
            return 1.0
        return 0.0
    return (new_count - prev_count) / prev_count


@register("agent", "iterative")
class IterativeAgent:
    """検索結果を見てから不足分を再分解し、最大 max_steps 回まで反復検索するエージェント。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        max_steps: int = 3,
        top_k: int = 20,
        use_hyde: bool = False,
        growth_threshold: float = 0.0,
        stagnation_patience: int = 1,
        dynamic_sufficiency: bool = False,
        enumeration_fanout: bool = False,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.top_k = top_k
        self.use_hyde = use_hyde
        self.growth_threshold = growth_threshold
        self.stagnation_patience = stagnation_patience
        self.dynamic_sufficiency = dynamic_sufficiency
        self.enumeration_fanout = enumeration_fanout
        self.task_family = TaskFamilyClassifier(llm)

    def run(self, query: Query) -> Prediction:
        subqueries = self._decompose(query)

        dynamic_threshold = None
        if self.dynamic_sufficiency and self.task_family.infer(query) == MULTI:
            dynamic_threshold = self._estimate_required_paper_count(query)

        found: dict[str, RetrievalResult] = {}
        trace: list[dict] = []
        stagnant_streak = 0
        tried_subqueries: list[str] = []

        for step in range(self.max_steps):
            prev_count = len(found)
            tried_subqueries.extend(subqueries)

            for sq in subqueries:
                if self.use_hyde:
                    sq_for_search = self._hyde(sq)
                else:
                    sq_for_search = sq
                results = self.retriever.retrieve(sq_for_search, self.top_k)
                for r in results:
                    if r.paper_id not in found:
                        found[r.paper_id] = r

            trace.append({"step": step, "subqueries": subqueries, "n_found": len(found)})

            if self._is_sufficient(query, found, override_threshold=dynamic_threshold):
                break

            growth_rate = _growth_rate(prev_count, len(found))
            if growth_rate <= self.growth_threshold:
                stagnant_streak += 1
            else:
                stagnant_streak = 0
            if stagnant_streak >= self.stagnation_patience:
                break

            new_subqueries = self._refine(query, found, tried_subqueries)
            if not new_subqueries:
                break
            subqueries = new_subqueries

        paper_ids = to_gold_papers(list(found.values()))
        cutoff = (
            dynamic_threshold
            if dynamic_threshold is not None
            else CUTOFF_BY_TASK_FAMILY.get(self.task_family.infer(query))
        )
        if cutoff is not None:
            paper_ids = paper_ids[:cutoff]

        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=[],
            answer=Answer(),
            trace=trace,
        )

    def _decompose(self, query: Query) -> list[str]:
        if self.task_family.infer(query) != MULTI:
            return [query.question]

        if self.enumeration_fanout and _is_enumeration_or_comparison(query.question):
            count_hint = "4〜6個"
        else:
            count_hint = "2〜4個"

        prompt = (
            "あなたは、研究に関する質問を、科学論文コーパスに対する検索用のサブクエリに"
            "分解する作業を手伝っています。\n"
            f"質問: {query.question}\n"
            "この質問は複数の論文にまたがる根拠を必要とすると考えられます。\n"
            f"回答に必要な論文をすべて検索できるように、{count_hint}の短く自己完結した"
            "検索サブクエリに分解してください。\n"
            '出力は JSON のみとし、{"subqueries": ["...", "..."]} の形式で答えてください。'
        )
        try:
            response = self.llm(prompt)
            parsed = _parse_json_object(response)
            if parsed:
                subqueries = parsed.get("subqueries")
            else:
                subqueries = None
            if isinstance(subqueries, list) and subqueries:
                return [str(sq) for sq in subqueries]
        except Exception:
            pass
        return [query.question]

    def _hyde(self, subquery: str) -> str:
        prompt = (
            "次の検索クエリに直接答える、あるいはその根拠を含むような、科学論文の一節を"
            f"想定して短く書いてください: {subquery}\n"
            "前置きは付けず、本文のテキストのみを出力してください。"
        )
        try:
            passage = self.llm(prompt)
            if passage and passage.strip():
                return passage.strip()
        except Exception:
            pass
        return subquery

    def _is_sufficient(
        self, query: Query, found: dict, override_threshold: int | None = None
    ) -> bool:
        if override_threshold is not None:
            threshold = override_threshold
        else:
            threshold = SUFFICIENT_COUNT_BY_TASK_FAMILY.get(self.task_family.infer(query))
        if threshold is None:
            return True
        return len(found) >= threshold

    def _estimate_required_paper_count(self, query: Query) -> int | None:
        prompt = (
            "あなたは、次の質問に完全に答えるために根拠として必要な、"
            "重複のない論文の本数を見積もる作業をしています。\n"
            f"質問: {query.question}\n"
            "この質問は複数の論文の列挙・比較を必要とする可能性があります。\n"
            "必要な論文数について、最善の整数の見積もりを答えてください（最小 1）。\n"
            '出力は JSON のみとし、{"n_papers": <整数>} の形式で答えてください。'
        )
        try:
            response = self.llm(prompt)
            parsed = _parse_json_object(response)
            if parsed:
                n_papers = parsed.get("n_papers")
            else:
                n_papers = None
            if isinstance(n_papers, bool):
                return None
            if isinstance(n_papers, int) and n_papers >= 1:
                return n_papers
            if isinstance(n_papers, float) and n_papers >= 1:
                return int(n_papers)
        except Exception:
            pass
        return None

    def _refine(
        self, query: Query, found: dict, tried_subqueries: list[str]
    ) -> list[str]:
        ranked = sorted(found.values(), key=lambda r: r.score, reverse=True)
        snippets = "\n".join(f"- paper {r.paper_id}: {r.text[:200]}" for r in ranked[:10])
        tried_text = "\n".join(f"- {sq}" for sq in dict.fromkeys(tried_subqueries))
        prompt = (
            f"元の質問: {query.question}\n"
            "これまでの検索で、以下の論文から根拠が見つかっています:\n"
            f"{snippets}\n"
            "これまでに試した検索サブクエリ（同じ、または似たものを繰り返さないでください）:\n"
            f"{tried_text}\n"
            "これだけでは質問に十分答えられない可能性があります。上記と重複しない、別の観点からの"
            "追加の検索サブクエリを提案してください。"
            "これ以上検索する価値がなければ、空のリストを返してください。\n"
            '出力は JSON のみとし、{"subqueries": ["...", "..."]} の形式で答えてください。'
        )
        try:
            response = self.llm(prompt)
            parsed = _parse_json_object(response)
            if parsed:
                subqueries = parsed.get("subqueries")
            else:
                subqueries = None
            if isinstance(subqueries, list):
                return [str(sq) for sq in subqueries if sq]
        except Exception:
            pass
        return []
