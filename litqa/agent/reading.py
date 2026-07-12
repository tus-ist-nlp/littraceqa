"""検索 → 読解 → 不足分の再検索 を繰り返す、本命の反復エージェント。

既存の2つは片翼ずつしか持っていない:

* IterativeAgent は反復するが中身を検証しない。停止条件が
  「検索が返した論文の本数 >= しきい値」なので、top_k=20 で検索した時点で
  初回から満たされてしまい、2周目に入らない（_refine は一度も呼ばれない）。
* VerifyingAgent は LLM に候補を読ませて選ばせるが、1回で終わる。足りなくても
  検索し直さない。

ReadingAgent はこの2つを合体させる。停止条件を「LLM が根拠として確認できた論文が
質問に答えるのに足りているか」に置き換えることで、初めて反復が意味を持つ。

    1. 質問をサブクエリに分解する
    2. 各サブクエリで検索する
    3. 上位候補のチャンクを**全文**（既定1800文字）LLM に読ませ、
       本当に根拠になる論文と、その根拠チャンクを選ばせる
    4. LLM が「まだ足りない」と言えば、何が欠けているかを聞いて再分解し 2 に戻る
    5. 足りたか、max_steps に達したら終了

選ばれた根拠チャンクから Evidence を組み立てる（litqa/agent/evidence.py 参照）。
Answer（freeform / multiple_choice / table）はまだ埋めない。
"""

from __future__ import annotations

from litqa.agent.evidence import evidence_from_result
from litqa.agent.json_utils import parse_json_object
from litqa.agent.task_family import MULTI, TaskFamilyClassifier, apply_paper_cutoff
from litqa.contracts import Answer, Evidence, Prediction, Query, RetrievalResult
from litqa.llm.base import LLMClient
from litqa.registry import register
from litqa.retrieve.hybrid import HybridRetriever, to_gold_papers


@register("agent", "reading")
class ReadingAgent:
    """候補を読んで根拠を確定し、足りなければ検索し直す反復エージェント。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        max_steps: int = 3,
        top_k: int = 20,
        max_candidates: int = 15,
        chunks_per_paper: int = 2,
        snippet_chars: int = 1800,
        paper_cutoff: str = "task_family",
        max_papers: int = 10,
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.chunks_per_paper = chunks_per_paper
        self.snippet_chars = snippet_chars
        self.paper_cutoff = paper_cutoff
        self.max_papers = max_papers
        self.task_family = TaskFamilyClassifier(llm)

    def run(self, query: Query) -> Prediction:
        subqueries = self._decompose(query)
        tried: list[str] = []
        chunks: dict[str, RetrievalResult] = {}
        verdict: dict | None = None
        trace: list[dict] = []

        for step in range(self.max_steps):
            tried.extend(subqueries)
            for subquery in subqueries:
                for result in self.retriever.retrieve(subquery, self.top_k):
                    chunks[result.chunk_id] = result

            candidates = self._candidate_papers(chunks)
            new_verdict = self._read_and_judge(query, candidates, chunks)
            if new_verdict is not None:
                verdict = new_verdict

            trace.append(
                {
                    "step": step,
                    "subqueries": subqueries,
                    "n_chunks": len(chunks),
                    "n_candidates": len(candidates),
                    "selected": [] if verdict is None else verdict["paper_ids"],
                    "sufficient": None if verdict is None else verdict["sufficient"],
                    "missing": None if verdict is None else verdict["missing"],
                }
            )

            # ここが IterativeAgent との決定的な違い。検索が何本返したかではなく、
            # LLM が根拠として確認できた論文で足りているかどうかで打ち切る。
            if verdict is not None and verdict["sufficient"]:
                break
            if step == self.max_steps - 1:
                break

            missing = "" if verdict is None else verdict["missing"]
            subqueries = self._refine(query, missing, tried)
            if not subqueries:
                break

        return self._build_prediction(query, verdict, chunks, trace)

    # ---- 1. 分解 ----------------------------------------------------------

    def _decompose(self, query: Query) -> list[str]:
        """質問を検索用のサブクエリに分解する。

        IterativeAgent は multi_paper のときしか分解しなかったが、single_paper でも
        「どの論文か」と「その中のどの表か」は別々の検索語になりうるので常に分解する。
        """
        if self.task_family.infer(query) == MULTI:
            count_hint = "3〜6個"
            note = (
                "この質問は複数の論文にまたがる根拠を必要とします。"
                "回答に必要な論文をすべて拾えるように分解してください。"
            )
        else:
            count_hint = "1〜3個"
            note = (
                "この質問の根拠は1本の論文の中に閉じています。"
                "その論文を確実に引き当てられる言い換えを作ってください。"
            )

        prompt = (
            "あなたは、研究に関する質問を、科学論文コーパスに対する検索用のサブクエリに"
            "分解する作業を手伝っています。\n"
            f"質問: {query.question}\n"
            f"{note}\n"
            f"{count_hint}の、短く自己完結した検索サブクエリに分解してください。\n"
            '出力は JSON のみとし、{"subqueries": ["...", "..."]} の形式で答えてください。'
        )
        subqueries = self._ask_for_list(prompt, "subqueries")
        return subqueries or [query.question]

    # ---- 2. 候補の組み立て ------------------------------------------------

    def _candidate_papers(
        self, chunks: dict[str, RetrievalResult]
    ) -> list[tuple[str, list[RetrievalResult]]]:
        """チャンク集合を論文単位にまとめ、スコア上位の論文を候補として返す。"""
        by_paper: dict[str, list[RetrievalResult]] = {}
        for result in chunks.values():
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
            "あなたは、検索でヒットした論文の抜粋を読み、質問への回答の根拠として"
            "本当に必要な論文だけを選び出す作業をしています。\n\n"
            f"質問: {query.question}\n"
            f"{self._format_answer_spec(query)}\n\n"
            "候補（関連度が高い順。各チャンクは論文本文・表・図キャプションの抜粋）:\n"
            f"{listing}\n\n"
            "抜粋を読んだうえで、次を判断してください。\n"
            "1. 質問に答えるための根拠を実際に含む論文はどれか（含まないものは選ばない）。\n"
            "2. 各論文について、根拠になっているチャンクの chunk_id はどれか。\n"
            "3. これで質問に完全に答えられるか。答えられないなら、何がまだ欠けているか"
            "（探すべき手法名・データセット名・論文の特徴など）を具体的に書く。\n\n"
            "候補一覧に無い paper_id / chunk_id を作り出さないでください。\n"
            "出力は JSON のみとし、次の形式で答えてください:\n"
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
        parts = [f"回答形式: {', '.join(query.answer_types) or '(指定なし)'}"]
        if query.table_schema:
            columns = ", ".join(str(c.get("name")) for c in query.table_schema)
            parts.append(f"回答テーブルの列: {columns}")
        return "\n".join(parts)

    # ---- 4. 不足分の再分解 ------------------------------------------------

    def _refine(self, query: Query, missing: str, tried: list[str]) -> list[str]:
        """「何が欠けているか」を踏まえて、次に投げる検索サブクエリを作る。"""
        tried_text = "\n".join(f"- {sq}" for sq in dict.fromkeys(tried))
        missing_text = missing or "（LLMからの指摘なし。別の角度から探してください）"
        prompt = (
            f"元の質問: {query.question}\n\n"
            "ここまでの検索では、根拠がまだ足りていません。欠けているのは次の点です:\n"
            f"{missing_text}\n\n"
            "すでに試した検索サブクエリ（同じもの・似たものを繰り返さないでください）:\n"
            f"{tried_text}\n\n"
            "この不足を埋めるための、新しい検索サブクエリを提案してください。"
            "これ以上検索しても見つかる見込みが無ければ、空のリストを返してください。\n"
            '出力は JSON のみとし、{"subqueries": ["...", "..."]} の形式で答えてください。'
        )
        return self._ask_for_list(prompt, "subqueries")

    # ---- 5. 提出物の組み立て ----------------------------------------------

    def _build_prediction(
        self,
        query: Query,
        verdict: dict | None,
        chunks: dict[str, RetrievalResult],
        trace: list[dict],
    ) -> Prediction:
        if verdict is None:
            # LLM が一度も使える判定を返さなかった場合は検索の順位のまま出す。
            ranked = to_gold_papers(list(chunks.values()))
            evidence: list[Evidence] = []
        else:
            ranked = verdict["paper_ids"]

        paper_ids = apply_paper_cutoff(
            ranked, query, self.task_family, self.paper_cutoff, self.max_papers
        )

        if verdict is not None:
            # 打ち切りで落ちた論文の evidence は出さない。
            kept = set(paper_ids)
            evidence = [
                evidence_from_result(chunks[chunk_id])
                for chunk_id in dict.fromkeys(verdict["evidence_chunk_ids"])
                if chunks[chunk_id].paper_id in kept
            ]

        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=evidence,
            answer=Answer(),
            trace=trace,
        )

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
