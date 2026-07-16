"""検索は1回のまま、上位候補をLLMに読ませて最終提出論文を選ぶエージェント。

一次検索の recall が高くても、正解論文が固定カットオフより下位にいると
単純な順位カットオフでは拾えない。このエージェントは上位候補をまとめて
LLM に提示し、質問への根拠として本当に必要な paper_id だけを選ばせる。

SimpleAgent との差＝「LLMに選ばせること」の効果。反復はしないので、ReadingAgent
との差＝「反復して検索し直すこと」の効果になる。この引き算が成立するように、
top_k と paper_cutoff は他のエージェントと揃えて使うこと。
"""

from __future__ import annotations

from littraceqa.di_pipeline.agent.json_utils import parse_json_object as _parse_json_object
from littraceqa.di_pipeline.agent.task_family import TaskFamilyClassifier, apply_paper_cutoff
from littraceqa.di_pipeline.contracts import Answer, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever, to_gold_papers


@register("agent", "verifying")
class VerifyingAgent:
    """一次検索の上位候補をLLMに判定させ、内容ベースで最終提出論文を選ぶエージェント。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        top_k: int = 20,
        max_candidates: int = 15,
        paper_cutoff: str = "task_family",
        max_papers: int = 10,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.paper_cutoff = paper_cutoff
        self.max_papers = max_papers
        self.task_family = TaskFamilyClassifier(llm)

    def run(self, query: Query) -> Prediction:
        results = self.retriever.retrieve(query.question, self.top_k)
        candidate_ids = to_gold_papers(results)
        if not candidate_ids:
            return Prediction(
                query_id=query.query_id, gold_papers=[], evidence=[], answer=Answer()
            )

        judged_ids = candidate_ids[: self.max_candidates]
        judged = self._judge(query, judged_ids, self._best_snippet_by_paper(results))

        paper_ids = None
        if judged is not None:
            candidate_set = set(judged_ids)
            paper_ids = [pid for pid in judged if pid in candidate_set]

        if not paper_ids:
            # LLM が使える判定を返さなかった場合は検索の順位のまま出す。
            paper_ids = candidate_ids

        paper_ids = apply_paper_cutoff(
            paper_ids, query, self.task_family, self.paper_cutoff, self.max_papers
        )

        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": pid} for pid in paper_ids],
            evidence=[],
            answer=Answer(),
        )

    def _best_snippet_by_paper(self, results: list[RetrievalResult]) -> dict[str, str]:
        best: dict[str, RetrievalResult] = {}
        for r in results:
            current = best.get(r.paper_id)
            if current is None or r.score > current.score:
                best[r.paper_id] = r
        return {paper_id: r.text[:200] for paper_id, r in best.items()}

    def _judge(
        self, query: Query, candidate_ids: list[str], snippets_by_paper: dict[str, str]
    ) -> list[str] | None:
        listing = "\n".join(
            f"- {pid}: {snippets_by_paper.get(pid, '')}" for pid in candidate_ids
        )
        prompt = (
            "あなたは、検索候補の論文一覧から、質問への回答の根拠として本当に必要な論文だけを"
            "選び出す作業をしています。\n"
            f"質問: {query.question}\n"
            "検索候補（関連度が高い順）:\n"
            f"{listing}\n"
            "この中から、質問に答えるために根拠として必要な論文の paper_id だけを、"
            "過不足なく選んでください。候補一覧に無い paper_id を作り出さないでください。\n"
            '出力は JSON のみとし、{"paper_ids": ["...", "..."]} の形式で答えてください。'
        )
        try:
            response = self.llm(prompt)
            parsed = _parse_json_object(response)
            if parsed is None:
                return None
            paper_ids = parsed.get("paper_ids")
            if isinstance(paper_ids, list):
                return [str(pid) for pid in paper_ids]
        except Exception:
            pass
        return None
