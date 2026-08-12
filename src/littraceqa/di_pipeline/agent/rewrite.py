"""検索結果を材料にしたクエリ書き換え（仕様: docs/search_agent2_spec.md）。

**サブクエリを、質問文からではなく「コーパスが返した本文」から作る。**

現行の `ReadingAgent._refine()` はコーパスの反応を一度も見ない（材料は読解 LLM の
`missing` だけ）。その結果、3ステップぶんのサブクエリが**ひとつの語彙ファミリーの
言い換え**に収束し、別の語彙で自分を説明している gold に一度も届かないことがある。

実例（q_022「ICML 2025 で reference-free な preference optimization を提案した
論文を列挙せよ」）:

  * gold の LOGO は abstract に "reference-free preference optimization" と書いて
    いるので1位で取れる。
  * gold の AlphaPO は**自分を一度も reference-free と呼ばない**（本文で該当するのは
    参考文献リストの1件だけ）。自称は "Direct Alignment Algorithm" / "reward shape" /
    "likelihood displacement" で、どれも質問文に出てこない語。
  * 実際に投げられた23本のサブクエリは全部 "reference-free …" 系で、AlphaPO は28位。
    BM25 単体でも `Direct Alignment Algorithm, reward shape` と投げれば1位で取れる。

そこで、書き換えの材料を2系統に分けて LLM に渡す。

  * 材料A（検索の上位論文）… **いま何を引き当てているか**。ヒットしたチャンクを
    見せるのが要点で、「質問のどの側面に当たっているか」はそこにしか無い。
  * 材料B（論文→論文展開の上位論文）… **その論文は自分を何と呼ぶか**。展開は
    paper_id しか返さないので ChunkStore から abstract を引く。

材料A は深さの違う2つのビューを渡す。詳細ビューだけだと分布の異常が分布に見えない
——q_022 の候補列では上位5本に長文脈論文が1本しか無く（ノイズに見える）、20本まで
広げて初めて10本が長文脈だと分かる（＝展開の軸がズレている）。

このモジュールは **LLM を呼ばない**。プロンプト文字列を組み立てて返すだけで、
呼び出しと JSON 解釈は `ReadingAgent` 側に残す（テストしやすさのため）。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from littraceqa.chunk_store import ChunkStore
from littraceqa.di_pipeline.contracts import RetrievalResult

# 論文全体を1ドキュメントとして索引する indexer。ここ由来の「チャンク」は
# **論文全文**（`bm25s_paper` の text は _paper_prefix + 全本文の連結）で、
# chunk_id も "{paper_id}#paper" という擬似 ID なので、材料A の
# 「ヒットしたチャンク」としては使えない:
#
#   * snippet で切ると先頭 = タイトル + 本文冒頭 になり、すぐ上に出している
#     abstract とほぼ重複する。
#   * 「質問のどこに当たったか」という情報がゼロ（全文が当たっているので位置が無い）。
#
# その論文が paper 索引でしか当たっていなければヒットチャンクは0本になるが、
# それは正しい表示（「論文全体としては合っているが特定の箇所には当たっていない」）。
PAPER_LEVEL_SOURCES = frozenset({"bm25s_paper"})

ABSTRACT_CHUNK_TYPE = "title_abstract"

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class QueryRewriter:
    """材料の組み立てと書き換えプロンプトの生成。

    `enabled` が false のときは `ReadingAgent` 側で一切呼ばれない
    （このクラスを構築しない）。
    """

    def __init__(
        self,
        *,
        at_step: int = 1,
        chunk_store: str | None = None,
        image_root: str | None = None,
        from_a: dict | None = None,
        from_b: dict | None = None,
    ):
        self.at_step = at_step
        self.chunk_store_path = chunk_store
        self.image_root = image_root

        from_a = dict(from_a or {})
        self.detail_papers = int(from_a.get("detail_papers", 5))
        self.hit_chunks_per_paper = int(from_a.get("hit_chunks_per_paper", 2))
        self.include_abstract_a = bool(from_a.get("include_abstract", True))
        self.listing_papers = int(from_a.get("listing_papers", 20))

        from_b = dict(from_b or {})
        self.b_top_papers = int(from_b.get("top_papers", 20))
        self.include_abstract_b = bool(from_b.get("include_abstract", True))
        self.b_body_chunks = int(from_b.get("body_chunks_per_paper", 0))

        self._store: ChunkStore | None = None

    # ---- ChunkStore（遅延ロード） ----------------------------------------

    @property
    def store(self) -> ChunkStore | None:
        """初回アクセスで索引を作る（実測: 構築23秒・1論文0.7ms・索引1.0MB）。

        パスが未設定・ファイルが無い構成では None を返し、abstract 抜きで動く。
        材料が痩せるだけで落ちはしない。
        """
        if self._store is None and self.chunk_store_path:
            try:
                self._store = ChunkStore(self.chunk_store_path, image_root=self.image_root)
            except (OSError, ValueError):
                self.chunk_store_path = None
        return self._store

    def _chunks_of(self, paper_id: str) -> list[dict[str, Any]]:
        store = self.store
        if store is None or paper_id not in store:
            return []
        return store.load_paper(paper_id)

    def _abstract(self, paper_id: str) -> str:
        for chunk in self._chunks_of(paper_id):
            if chunk.get("chunk_type") == ABSTRACT_CHUNK_TYPE:
                return str(chunk.get("text") or "")
        return ""

    def _body_by_question(self, paper_id: str, question: str, limit: int) -> list[str]:
        """元の質問に近い本文チャンクを選ぶ（`select_by: question`）。

        展開はスコアを持たないうえ、**展開ソースのスコアで選ぶと anchor の軸に
        引きずられる**。q_022 の anchor は長文脈の論文なので、そのまま選ぶと
        長文脈の話ばかりが材料になり、質問の軸（preference optimization）から
        外れる。質問との語の重なりで選べば、同じ論文からでも質問に近い側の
        本文が出る。埋め込みは使わない（クエリ書き換えのたびにモデルを回さない）。
        """
        if limit <= 0:
            return []
        want = _tokens(question)
        scored = [
            (_jaccard(want, _tokens(str(chunk.get("text") or "")[:2000])), str(chunk.get("text") or ""))
            for chunk in self._chunks_of(paper_id)
            if chunk.get("chunk_type") != ABSTRACT_CHUNK_TYPE
        ]
        # 同点は本文の並び順に任せる（安定させるため score のみで sort しない）。
        scored.sort(key=lambda item: -item[0])
        return [text for _, text in scored[:limit] if text]

    # ---- 材料 -------------------------------------------------------------

    @staticmethod
    def _header(paper_id: str, results: Sequence[RetrievalResult]) -> str:
        metadata = (results[0].metadata or {}) if results else {}
        title = metadata.get("title", "")
        venue = metadata.get("venue", "")
        year = metadata.get("year", "")
        return f"[paper_id: {paper_id}] {title} ({venue} {year})"

    def material_a(
        self,
        papers: Sequence[tuple[str, list[RetrievalResult]]],
        snippet_chars: int,
    ) -> str:
        """材料A 詳細 — タイトル / abstract / **実際にヒットしたチャンク**。"""
        blocks = []
        for paper_id, results in list(papers)[: self.detail_papers]:
            lines = [self._header(paper_id, results)]
            if self.include_abstract_a:
                abstract = self._abstract(paper_id)
                if abstract:
                    lines.append(f"  abstract: {abstract[:snippet_chars]}")
            hits = [r for r in results if r.source not in PAPER_LEVEL_SOURCES]
            for result in hits[: self.hit_chunks_per_paper]:
                lines.append(
                    f"  matched chunk (type={result.chunk_type}): "
                    f"{result.text[:snippet_chars]}"
                )
            if not hits:
                # 論文単位索引でしか当たっていない = 特定の箇所には当たっていない。
                lines.append("  matched chunk: (none — matched at whole-paper level only)")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def listing_a(self, merged: Iterable[RetrievalResult]) -> str:
        """材料A 俯瞰 — `[venue year] title` だけを並べる（軸ズレ検出用）。

        タイトルのみなので分量はほぼゼロ。上位が全部よその話題だと見えれば、
        「質問の軸から外れている」という事実がその場で分かる。
        """
        seen: list[str] = []
        head: dict[str, RetrievalResult] = {}
        for result in merged:
            if result.paper_id not in head:
                head[result.paper_id] = result
                seen.append(result.paper_id)
            if len(seen) >= self.listing_papers:
                break
        lines = []
        for rank, paper_id in enumerate(seen, 1):
            metadata = head[paper_id].metadata or {}
            lines.append(
                f"{rank}. [{metadata.get('venue', '')} {metadata.get('year', '')}] "
                f"{metadata.get('title', '')}"
            )
        return "\n".join(lines)

    def material_b(self, paper_ids: Sequence[str], question: str, snippet_chars: int) -> str:
        """材料B — 展開で拾った論文の自己記述。

        **本文チャンクは既定で見せず、本数を稼ぐ**（`body_chunks_per_paper: 0`）。
        欲しいのは「この論文は自分を何と呼ぶか」という語彙だけで、そのために
        本文は要らない。abstract だけなら20本でも詳細5本ぶんと同じ分量に収まる。
        """
        blocks = []
        for paper_id in list(paper_ids)[: self.b_top_papers]:
            lines = [f"[paper_id: {paper_id}]"]
            if self.include_abstract_b:
                abstract = self._abstract(paper_id)
                if abstract:
                    lines.append(f"  {abstract[:snippet_chars]}")
            for text in self._body_by_question(paper_id, question, self.b_body_chunks):
                lines.append(f"  body: {text[:snippet_chars]}")
            if len(lines) > 1:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---- プロンプト -------------------------------------------------------

    def prompt(
        self,
        *,
        question: str,
        material_a: str,
        listing_a: str,
        material_b: str,
        tried: Sequence[str],
        dead: Sequence[str],
        corpus_note: str,
        constraint_note: str = "",
        missing: str = "",
    ) -> str:
        """書き換えプロンプト。

        指示の要は「**語彙だけ変えろ、性質は変えるな**」。素朴に「関連論文を探せ」と
        書かせると anchor の話題に引きずられる（q_022 の LOGO = 長文脈）ので、
        元の質問を必ず入れて軸を固定する。
        """
        parts = [
            "You are rewriting search subqueries against a local corpus of papers.",
            f"Original question: {question}",
        ]
        if missing:
            parts.append(f"What the reader said is still missing:\n{missing}")
        if material_a:
            parts.append(
                "Papers the search is currently retrieving, with the chunks that "
                f"actually matched:\n{material_a}"
            )
        if listing_a:
            parts.append(
                "The current top candidates, titles only. If these are mostly about a "
                "different topic than the question asks for, the search has drifted "
                f"off-axis and you should correct it:\n{listing_a}"
            )
        if material_b:
            parts.append(
                "Papers related to the top candidate (found by citation/embedding "
                "neighbourhood, not by the question). These are the papers whose own "
                f"wording you should borrow:\n{material_b}"
            )
        if tried:
            parts.append(
                "Subqueries already tried:\n"
                + "\n".join(f"- {sq}" for sq in dict.fromkeys(tried))
            )
        if dead:
            parts.append(
                "Of those, these returned nothing that survived into the top candidates:\n"
                + "\n".join(f"- {sq}" for sq in dict.fromkeys(dead))
            )
        if constraint_note:
            parts.append(constraint_note)
        parts.append(corpus_note)
        parts.append(
            "Write new search subqueries using the words these papers use to describe "
            "THEMSELVES. Do not paraphrase the question: the question's own wording has "
            "already been searched. Keep the property the question asks about unchanged "
            "and change only the vocabulary — for example, if the question asks for a "
            "property that a paper would never state about itself, search instead for "
            "the method family, the technique name, or the baseline it extends.\n"
            "Each subquery must go after papers that are not already in the candidate "
            "list above. Return an empty list if there is nothing new to look for.\n"
            'Respond with JSON only, in the form {"subqueries": ["...", "..."]}.'
        )
        return "\n\n".join(part for part in parts if part)


class SubqueryDeduper:
    """**引いてくる論文が重なるか**でサブクエリの重複を判定する。

    本数を先に決めない。LLM に N 本と指定しても、返ってくるのは言い回しを変えた
    だけの同じクエリになりがちで、そのぶん検索と reranker が空回りする。
    **何本作らせるかではなく、何本残すかを中身で決める。**

    文字列の重複では捕まえられない。`reference-free …` と
    `Direct Alignment Algorithm …` は文字列としては全く別物だが**別の論文を引く**
    （残すべき）。逆に言い回し違いのクエリは文字列上は別に見えるのに**同じ論文しか
    引かない**（捨てるべき）。

    **BM25 だけで篩う**のは、コストが桁で違うから。本番の検索1本は reranker が
    pool_k 件を推論するが、BM25 の引き当ては索引を1回叩くだけ。重複したクエリに
    reranker を1回も走らせない。
    """

    def __init__(
        self,
        indexer: Any | None,
        *,
        probe_k: int = 20,
        max_overlap: float = 0.7,
        max_queries: int = 4,
    ):
        self.indexer = indexer
        self.probe_k = probe_k
        self.max_overlap = max_overlap
        self.max_queries = max_queries

    @staticmethod
    def from_retriever(retriever: Any, config: dict | None) -> "SubqueryDeduper | None":
        """`retriever.indexers` から BM25 索引を名前で探す。

        **`subquery_dedup` を書かなければ None を返す**（既存の yaml の挙動を
        1ビットも変えないため）。`method: none` も同じ。

        索引が見つからない構成（埋め込みだけの retriever、テスト用スタブ）では
        `indexer=None` の Deduper を返し、**本数の上限だけ**を効かせる。
        重複除去のために埋め込みや reranker を回すことはしない。
        """
        if not config:
            return None
        config = dict(config)
        if config.get("method", "bm25_overlap") == "none":
            return None
        indexer = None
        for candidate in getattr(retriever, "indexers", []) or []:
            if getattr(candidate, "name", "") == "bm25s":
                indexer = candidate
                break
        return SubqueryDeduper(
            indexer,
            probe_k=int(config.get("probe_k", 20)),
            max_overlap=float(config.get("max_overlap", 0.7)),
            max_queries=int(config.get("max_queries", 4)),
        )

    def _papers(self, subquery: str) -> set[str]:
        if self.indexer is None:
            return set()
        try:
            results = self.indexer.search(subquery, self.probe_k)
        except RuntimeError:
            # 索引が未ロード（build()/load() 前）のときだけ握りつぶす。それ以外は
            # 通す——**広く握りつぶすと重複判定が黙って空集合になり、除去が
            # 効かないまま「全部残す」動作に化ける**（実際にテストで踏んだ）。
            return set()
        return {r.paper_id for r in results}

    def filter(
        self, candidates: Sequence[str], already: Sequence[str] = ()
    ) -> list[str]:
        """重複しないものだけを順に残す。

        `already` は同じクエリに対して**すでに投げたサブクエリ**。新しい候補が
        それらと同じ論文しか引かないなら、投げる意味がない。
        """
        seen_sets = [self._papers(sq) for sq in dict.fromkeys(already)]
        kept: list[str] = []
        kept_texts: set[str] = set()
        for subquery in candidates:
            if len(kept) >= self.max_queries:
                break
            text = subquery.strip()
            if not text or text in kept_texts:
                continue
            papers = self._papers(text)
            if papers and any(_jaccard(papers, seen) > self.max_overlap for seen in seen_sets):
                continue
            kept.append(text)
            kept_texts.add(text)
            seen_sets.append(papers)
        return kept
