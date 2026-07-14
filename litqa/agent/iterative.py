"""見てから次を決める反復検索エージェント。one-shot クエリ分解を超えるための多段検索。"""

from __future__ import annotations

import json
import re

from litqa.contracts import Answer, Prediction, Query, RetrievalResult
from litqa.llm.base import LLMClient
from litqa.registry import register
from litqa.retrieve.hybrid import HybridRetriever, to_gold_papers

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

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


def _is_enumeration_or_comparison(question: str) -> bool:
    """質問文が列挙・比較型（複数論文にまたがりやすい）パターンかどうかを判定する。"""
    return any(pattern.search(question) for pattern in _ENUMERATION_PATTERNS)


def _growth_rate(prev_count: int, new_count: int) -> float:
    """直前ステップからの新規発見論文数の相対成長率。prev_count が 0 の初回ステップは特別扱い。"""
    if prev_count == 0:
        if new_count > 0:
            return 1.0
        return 0.0
    return (new_count - prev_count) / prev_count


def _parse_json_object(text: str) -> dict | None:
    """LLM の出力から JSON オブジェクトを安全に取り出す。失敗したら None。"""
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        candidate = text
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


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
        decompose_queries: bool = True,
        sufficient_papers: int | None = None,
        max_papers: int | None = None,
    ):
        if sufficient_papers is not None and sufficient_papers <= 0:
            raise ValueError("sufficient_papers must be positive or None")
        if max_papers is not None and max_papers <= 0:
            raise ValueError("max_papers must be positive or None")
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.top_k = top_k
        self.use_hyde = use_hyde
        self.growth_threshold = growth_threshold
        self.stagnation_patience = stagnation_patience
        self.dynamic_sufficiency = dynamic_sufficiency
        self.enumeration_fanout = enumeration_fanout
        self.decompose_queries = decompose_queries
        self.sufficient_papers = sufficient_papers
        self.max_papers = max_papers

    def run(self, query: Query) -> Prediction:
        subqueries = self._decompose(query)

        dynamic_threshold = None
        if self.dynamic_sufficiency:
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
        cutoff = dynamic_threshold if dynamic_threshold is not None else self.max_papers
        if cutoff is not None and self.max_papers is not None:
            cutoff = min(cutoff, self.max_papers)
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
        if not self.decompose_queries:
            return [query.question]

        if self.enumeration_fanout and _is_enumeration_or_comparison(query.question):
            count_hint = "4 to 6"
        else:
            count_hint = "2 to 4"

        prompt = (
            "Decompose the research question into short, self-contained search queries "
            "for a scientific-paper corpus.\n"
            f"Question: {query.question}\n"
            f"Produce {count_hint} queries that collectively retrieve the necessary evidence.\n"
            'Return JSON only: {"subqueries": ["...", "..."]}.'
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
            "Write a short hypothetical scientific-paper passage that directly answers "
            f"or contains evidence for this search query: {subquery}\n"
            "Return only the passage text."
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
        threshold = override_threshold if override_threshold is not None else self.sufficient_papers
        if threshold is None:
            return False
        return len(found) >= threshold

    def _estimate_required_paper_count(self, query: Query) -> int | None:
        prompt = (
            "Estimate the number of distinct papers needed as evidence to answer the question.\n"
            f"Question: {query.question}\n"
            "Return the best integer estimate, with a minimum of 1.\n"
            'Return JSON only: {"n_papers": <integer>}.'
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
            f"Original question: {query.question}\n"
            "Evidence has been found in these papers:\n"
            f"{snippets}\n"
            "Previously attempted search queries; do not repeat equivalent queries:\n"
            f"{tried_text}\n"
            "Propose non-overlapping search queries from a different angle. "
            "Return an empty list if further retrieval is not useful.\n"
            'Return JSON only: {"subqueries": ["...", "..."]}.'
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
