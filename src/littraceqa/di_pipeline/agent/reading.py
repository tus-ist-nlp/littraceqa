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

選ばれた根拠チャンクから Evidence を組み立てる（src/littraceqa/di_pipeline/agent/evidence.py 参照）。
Answer（freeform / multiple_choice / table）はまだ埋めない。
"""

from __future__ import annotations

from littraceqa.di_pipeline.agent.evidence import evidence_from_result
from littraceqa.di_pipeline.agent.json_utils import parse_json_object
from littraceqa.di_pipeline.agent.task_family import MULTI, TaskFamilyClassifier, apply_paper_cutoff
from littraceqa.di_pipeline.select import build_paper_selector
from littraceqa.di_pipeline.contracts import Answer, Evidence, Prediction, Query, RetrievalResult
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve.base import Retriever
from littraceqa.di_pipeline.retrieve.hybrid import to_gold_papers


# 予測ファイルに残す候補論文の本数。看板指標は recall@20 だが、再実行が高コストなので
# あとから recall@50 まで測れるよう多めに持たせる。
CANDIDATE_PAPERS_LIMIT = 50


@register("agent", "reading")
class ReadingAgent:
    """候補を読んで根拠を確定し、足りなければ検索し直す反復エージェント。"""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        max_steps: int = 3,
        top_k: int = 20,
        max_candidates: int = 15,
        chunks_per_paper: int = 2,
        snippet_chars: int = 1800,
        paper_cutoff: str = "task_family",
        max_papers: int = 10,
        paper_selector: dict | None = None,
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
        self.paper_selector = build_paper_selector(paper_selector)
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
                    # 同じチャンクが複数のサブクエリで当たったら、スコアが高いほうを残す。
                    # 後勝ちにすると、サブクエリ1で最上位だったチャンクがサブクエリ3の
                    # 低いスコアで上書きされ、_candidate_papers の論文順位が
                    # 「最後に投げたサブクエリ」に引きずられる。
                    previous = chunks.get(result.chunk_id)
                    if previous is None or result.score > previous.score:
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
        # 反復中に溜めたチャンクを論文順位に直したもの。打ち切り前の「検索が拾えた候補」
        # なので、recall@k の分析はこちらを見る必要がある（gold_papers は LLM 選定と
        # cutoff の後なので、検索力と選定力が混ざってしまう）。
        candidate_papers = to_gold_papers(
            list(chunks.values()), max_papers=CANDIDATE_PAPERS_LIMIT
        )

        if verdict is None:
            # LLM が一度も使える判定を返さなかった場合は検索の順位のまま出す。
            # 直後の apply_paper_cutoff が最大 max_papers 本に切るので、上限50でも等価。
            ranked = candidate_papers
            evidence: list[Evidence] = []
        else:
            ranked = verdict["paper_ids"]

        if self.paper_selector is not None:
            # select_style を指定したときは、そちらが提出本数を決める。
            # 根拠が取れた論文を渡すので、require_evidence 付きの構成では
            # 読解で裏が取れなかった候補を落とせる。
            evidence_paper_ids = (
                {chunks[chunk_id].paper_id for chunk_id in verdict["evidence_chunk_ids"]}
                if verdict is not None
                else None
            )
            paper_ids = list(
                self.paper_selector.select(
                    query.question, ranked, evidence_paper_ids
                ).paper_ids
            )
        else:
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

        # 検索で選んだ根拠チャンクを渡して回答（freeform/multiple_choice/table）を生成する。
        # 根拠チャンクが空なら、残った論文の上位チャンクにフォールバックする。
        answer = self._generate_answer(query, evidence_results, chunks, set(paper_ids))

        return Prediction(
            query_id=query.query_id,
            gold_papers=[{"paper_id": paper_id} for paper_id in paper_ids],
            evidence=evidence,
            answer=answer,
            trace=trace,
            candidate_papers=candidate_papers,
        )

    # ---- 6. 回答生成 ------------------------------------------------------

    def _generate_answer(
        self,
        query: Query,
        evidence_results: list[RetrievalResult],
        chunks: dict[str, RetrievalResult],
        kept_paper_ids: set[str],
    ) -> Answer:
        """検索が取ってきた根拠チャンクを渡し、回答を生成する（gold は渡さない）。"""
        if not query.answer_types:
            return Answer()

        # 回答の根拠は「LLM が選んだ evidence チャンク」を最優先。空なら残った論文の
        # スコア上位チャンクで補う（回答に必要な値が evidence 外にあることもあるため）。
        context_results = list(evidence_results)
        if not context_results:
            fallback = [r for r in chunks.values() if r.paper_id in kept_paper_ids]
            fallback.sort(key=lambda r: r.score, reverse=True)
            context_results = fallback[: self.max_candidates]
        if not context_results:
            return Answer()

        context = "\n\n".join(
            f"[{r.chunk_id}] {r.text[: self.snippet_chars]}" for r in context_results
        )
        schema_fields: list[str] = []
        blocks = [
            "以下は検索で見つかった根拠チャンクです。この内容だけを根拠に質問に答えてください。",
            context,
            f"質問: {query.question}",
        ]
        if "freeform" in query.answer_types:
            schema_fields.append(
                '"freeform": {"text": "原文からの短い逐語的な値・語句（数値も引用符付き文字列で）"}'
            )
        if "multiple_choice" in query.answer_types and query.options:
            opt_lines = "\n".join(f"{k}: {v}" for k, v in query.options.items())
            blocks.append(f"選択肢:\n{opt_lines}")
            schema_fields.append('"multiple_choice": {"gold": "選んだ選択肢のアルファベット1文字"}')
        if "table" in query.answer_types and query.table_schema:
            columns = "\n".join(
                f'- "{c.get("name")}" (type: {c.get("type", "string")}'
                f'{", row key" if c.get("is_row_key") else ""})'
                for c in query.table_schema
                if isinstance(c, dict) and c.get("name")
            )
            blocks.append(f"必要な表の列:\n{columns}")
            schema_fields.append('"table": {"rows": [{"列名": "値", ...}, ...]}')

        if not schema_fields:
            return Answer()

        blocks.append(
            "次の JSON 形式だけで答えてください（説明文は不要）:\n"
            "{ " + ", ".join(schema_fields) + " }"
        )
        payload = self._ask_for_json("\n\n".join(blocks)) or {}
        return Answer(
            freeform=payload.get("freeform") if "freeform" in query.answer_types else None,
            multiple_choice=(
                payload.get("multiple_choice")
                if "multiple_choice" in query.answer_types
                else None
            ),
            table=payload.get("table") if "table" in query.answer_types else None,
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
