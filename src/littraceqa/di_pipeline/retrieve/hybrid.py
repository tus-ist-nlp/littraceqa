"""複数の Indexer と Fuser（任意で Reranker）を束ねる Retriever 本体。

各 Indexer で検索した結果を Fuser で1つのランキングに統合し、
必要に応じて Reranker で再ランクして返す。

質問が会議名で検索範囲を明示している場合（「Which NAACL 2025 papers ...」）は、
各 Indexer から多めに取ってから metadata で落とす（attribute_filter.py 参照）。
索引側は無改修で済み、どの indexer にも同じように効く。
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.attribute_filter import (
    AttributeExtractor,
    AttributeFilter,
    filter_results,
)
from littraceqa.di_pipeline.retrieve.paper_rrf import PaperRRFFuser
from littraceqa.di_pipeline.retrieve.reranker import Qwen3Reranker


@dataclass(frozen=True)
class RerankBlend:
    """reranker に順位を置き換えさせず、融合前の順位と RRF で混ぜる設定。

    既定（`rerank_blend=None`）では reranker が順位を**完全に置き換える**。だが
    reranker は「質問に答えるか」で判定するので、質問文が名指ししないピア gold を
    必ず下げる（`agent/reading.py` の `_combine_rrf` がランキングB を reranker に
    通さないのと同じ理由）。置換だとその危険が検索ランキングの内部で無防備になる。
    """

    original_weight: float = 0.6
    rerank_weight: float = 0.4
    rrf_k: int = 60
    # 融合前の上位N件の「集合」を先頭に残す（21位以下が無条件に上位へ侵入するのを防ぐ）。
    protect_top: int = 0


@dataclass(frozen=True)
class SeedExpansion:
    """1位論文の title+abstract の先頭 `query_chars` 文字を質問に足して引き直す設定。

    **質問文は「その論文が自分をどう呼ぶか」を知らない。** ある gold 論文は自分を
    一度も `reference-free` と呼ばず `Direct Alignment Algorithm` と名乗る——質問文に
    無い語なので、質問だけを投げ続ける限り当たらない。上位論文からコーパス内の語彙を
    借りるのがこの機構の役目。LLM は1回も呼ばない。
    """

    query_chars: int = 512


class HybridRetriever:
    def __init__(
        self,
        indexers: list,
        fuser: PaperRRFFuser,
        # None なら融合した順位をそのまま返す（再ランクしない）。
        reranker: Qwen3Reranker | None = None,
        per_index_k: int = 100,
        pool_k: int | None = None,
        attribute_extractor: AttributeExtractor | None = None,
        fetch_safety: float = 1.5,
        max_fetch_k: int = 5000,
        min_filtered_results: int = 10,
        rerank_blend: RerankBlend | None = None,
        seed_expansion: SeedExpansion | None = None,
        anchor_store: object | None = None,
    ):
        self.indexers = indexers
        self.fuser = fuser
        self.reranker = reranker
        self.per_index_k = per_index_k
        # reranker に渡す候補プールの件数。未指定なら top_k*3 (旧来の既定値)。
        self.pool_k = pool_k
        # None（既定）なら reranker の結果でそのまま置き換える（従来どおり）。
        # dict を渡すと融合前の順位と RRF で混ぜる（_blend_rerank 参照）。
        self.rerank_blend = rerank_blend
        # None（既定）なら Seed Expansion は走らない（従来どおり検索1回）。
        # dict を渡すと1位論文の語彙を質問に足して引き直す（_seed_expand 参照）。
        self.seed_expansion = seed_expansion
        # anchor の title+abstract を引くための ChunkStore（Seed Expansion 専用）。
        self.anchor_store = anchor_store
        # None なら属性フィルタは完全に無効（従来どおりのコードパスを通る）。
        self.attribute_extractor = attribute_extractor
        self.fetch_safety = fetch_safety
        self.max_fetch_k = max_fetch_k
        self.min_filtered_results = min_filtered_results

    def retrieve(
        self,
        query: str,
        top_k: int,
        attribute_filter: AttributeFilter | None = None,
    ) -> list[RetrievalResult]:
        """検索する。

        attribute_filter を渡すとその制約で絞り込む。渡さなかった場合は
        attribute_extractor があれば query 自体から抽出する（生の質問を直接
        投げる呼び出し用のフォールバック）。**本走行ではこの経路は通らない**——
        ReadingAgent はサブクエリではなく元の質問から1回だけ抽出したものを渡してくる
        （サブクエリからは会議名が落ちるため。agent/reading.py の
        `_extract_attribute_filter` 参照）。
        """
        if not self.indexers:
            return []

        if attribute_filter is None and self.attribute_extractor is not None:
            attribute_filter = self.attribute_extractor.extract(query)

        runs = self._run_indexers(query, attribute_filter)
        if self.reranker is not None:
            fuse_k = self.pool_k if self.pool_k is not None else top_k * 3
        else:
            fuse_k = top_k
        fused = self.fuser.fuse(runs, top_k=fuse_k)
        # Seed Expansion は **reranker の前**に置く。reranker を2回走らせると
        # 推論コストが倍になるが、索引を2回引くだけなら安い（reranker は元の質問で
        # 1回だけ走らせる）。
        fused = self._seed_expand(query, fused, attribute_filter, fuse_k)

        if self.reranker is None:
            return fused[:top_k]
        if self.rerank_blend is None:
            return self.reranker.rerank(query, fused, top_k)
        # 融合するには切る前の全順位が要るので len(fused) で呼ぶ。**推論コストは
        # 増えない**——Qwen3Reranker は候補を全件スコアしてから top_k で切っている
        # だけ（retrieve/reranker.py の rerank）。
        reranked = self.reranker.rerank(query, fused, len(fused))
        return self._blend_rerank(fused, reranked)[:top_k]

    def _blend_rerank(
        self, fused: list[RetrievalResult], reranked: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """融合前の順位と reranker の順位を RRF で混ぜる（`rerank_blend`）。

        既定（`rerank_blend` を書かない）では reranker が順位を**完全に置き換える**。
        reranker は「質問に答えるか」で判定するので、質問文が名指ししないピア gold を
        必ず下げる（`agent/reading.py` の `_combine_rrf` が B を reranker に通さない
        のと同じ理由）。置換だとその危険が検索ランキングの内部で無防備になる。

            score(c) = w_orig / (k + rank_fused) + w_rerank / (k + rank_reranked)

        **スコアではなく順位だけを見る。** RRF スコアと reranker の yes 確率は
        スケールが違って足せない（同じ理由で `_combine_rrf` も順位しか使わない）。

        `protect_top` を指定すると、**融合前の上位N件の「集合」**を先頭に残す
        （並び順は融合結果に従う）。21位以下が無条件に上位へ侵入するのを防ぐ。

        **融合後の順位は `score` に書き戻す。** 下流はどこも `score` で並べ直す
        （`agent/reading.py` の貯め込みと `_candidate_papers`、`to_gold_papers`）ので、
        返り値の並び順だけに順位を乗せると捨てられる。既存の `Qwen3Reranker.rerank`
        が score を上書きしているのと同じ事情。
        """
        blend = self.rerank_blend or RerankBlend()
        k = blend.rrf_k
        w_orig = blend.original_weight
        w_rerank = blend.rerank_weight
        protect_top = blend.protect_top

        scores: dict[str, float] = {}
        for rank, result in enumerate(fused):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + w_orig / (k + rank + 1)
        for rank, result in enumerate(reranked):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + w_rerank / (k + rank + 1)

        # **保護は score に載せる。** 並び順だけで先頭に寄せても、下流が score で
        # 並べ直した瞬間に元に戻ってしまう。保護対象に「最大スコア + 1」を足せば、
        # 群の中の RRF 順位を保ったまま必ず上に来る。
        if protect_top and scores:
            boost = max(scores.values()) + 1.0
            for result in fused[:protect_top]:
                scores[result.chunk_id] += boost

        # 本文は reranker が返したもの（score だけ後で上書きする）を使う。
        # 融合前に無くて reranker にだけあることは無いが、両方を見て取りこぼさない。
        by_id = {result.chunk_id: result for result in fused}
        by_id.update({result.chunk_id: result for result in reranked})
        ordered = sorted(by_id.values(), key=lambda r: -scores[r.chunk_id])
        return [replace(r, score=scores[r.chunk_id]) for r in ordered]

    def _seed_expand(
        self,
        query: str,
        fused: list[RetrievalResult],
        attribute_filter: AttributeFilter | None,
        fuse_k: int,
    ) -> list[RetrievalResult]:
        """1位論文の語彙を質問に足して引き直し、初回の順位と融合する。

        **LLM によるクエリ分解ではなく pseudo relevance feedback。**

            expanded = 元の質問 + 1位論文の title+abstract の先頭 query_chars 文字

        質問文は「その論文が自分をどう呼ぶか」を知らない。q_022 の gold（AlphaPO）は
        自分を一度も `reference-free` と呼ばず `Direct Alignment Algorithm` /
        `reward shape` と名乗る——質問文に無い語なので、質問だけを投げ続ける限り
        当たらない。**上位論文からコーパス内の語彙を借りる**のがこの機構の役目。

        `agent/rewrite.py`（LLM に書き換えさせる版）との違いは、**LLM を1回も
        呼ばない**ことと、元の質問を必ず残すこと。rewrite は LLM が候補論文の
        タイトルをそのままクエリにしてしまい、既に持っている論文を引き直すだけに
        なっていた（候補列のタイトル語を8割以上含むサブクエリが 1% -> 16%）。
        機械的に連結すればその失敗モードは起きない。

        **融合は `self.fuser` にそのまま任せる。** `fuser: rrf` ならチャンク単位、
        `fuser: paper_rrf` なら論文単位で混ざる。つまり Seed Expansion と
        論文単位RRF は独立に足せて、両方書けば「初回と拡張の論文単位RRF」になる。

        **`reranker` の前に置く**ので、reranker の推論回数は1回のまま増えない。
        増えるのは索引の検索1回ぶんだけ。
        """
        if not self.seed_expansion or not fused:
            return fused
        anchor_text = self._anchor_text(fused[0])
        if not anchor_text:
            return fused
        query_chars = self.seed_expansion.query_chars
        expanded = f"{query}\n{anchor_text[:query_chars]}"
        runs = self._run_indexers(expanded, attribute_filter)
        expanded_fused = self.fuser.fuse(runs, top_k=fuse_k)
        if not expanded_fused:
            return fused
        # 初回と拡張を2つの run として融合する。順位しか見ないので、
        # 初回の RRF スコアと拡張の RRF スコアのスケール差は問題にならない。
        return self.fuser.fuse([fused, expanded_fused], top_k=fuse_k)

    def _anchor_text(self, anchor: RetrievalResult) -> str:
        """anchor 論文の title+abstract。無ければヒットしたチャンク本文で代用する。

        `title_abstract` チャンク（`{paper_id}#c0000`）は
        `"[venue year] title\\n" + abstract` の形なので、先頭から切るだけで
        「タイトル + 概要」になる（`preprocess/mineru_chunker.py`）。
        """
        store = self.anchor_store
        if store is not None:
            try:
                for chunk in store.load_paper(anchor.paper_id):
                    if chunk.get("chunk_type") == "title_abstract":
                        return str(chunk.get("text") or "")
            except Exception:  # noqa: BLE001 - 本文が引けなくても検索は続ける
                pass
        # ChunkStore が無い構成（テストなど）や論文が見つからない場合。
        # 論文単位索引の擬似チャンクは text が論文全文なので、そのまま先頭を切れば
        # やはり title+abstract になる。
        title = str((anchor.metadata or {}).get("title") or "")
        if title and not anchor.text.startswith("["):
            return f"{title}\n{anchor.text}"
        return anchor.text

    def _run_indexers(
        self, query: str, attribute_filter: AttributeFilter | None
    ) -> list[list[RetrievalResult]]:
        """各 indexer を引く。制約があれば多めに取ってから落とす。"""
        if attribute_filter is None or attribute_filter.is_empty():
            return [indexer.search(query, self.per_index_k) for indexer in self.indexers]

        fetch_k = self._fetch_k(attribute_filter)
        runs = []
        for indexer in self.indexers:
            raw = indexer.search(query, fetch_k)
            kept = filter_results(raw, attribute_filter)
            # 絞り込みで候補が枯れたら、そのランだけ制約なしに戻す。誤抽出や
            # 取得件数不足で recall を落とすくらいなら雑音を許す（fail-open）。
            if len(kept) < self.min_filtered_results:
                kept = raw
            runs.append(kept[: self.per_index_k])
        return runs

    def _fetch_k(self, attribute_filter: AttributeFilter) -> int:
        """絞り込み後に per_index_k 件が残るよう、取得件数を選択率から逆算する。"""
        selectivity = 1.0
        if self.attribute_extractor is not None:
            selectivity = self.attribute_extractor.selectivity(attribute_filter)
        if selectivity <= 0:
            return self.max_fetch_k
        needed = int(self.per_index_k / selectivity * self.fetch_safety)
        return max(self.per_index_k, min(needed, self.max_fetch_k))


def to_gold_papers(
    results: list[RetrievalResult],
    max_papers: int | None = None,
    agg: str = "max",
    skip_chunk_types: Collection[str] = (),
) -> list[str]:
    """チャンクのランキングを論文のランキングに畳む。

    `skip_chunk_types` に入れた種別は**論文の代表スコアに 0 として入る**。
    その種別しか持たない論文はスコア 0 になり A の最下位に沈むが、**候補列からは
    消えない**（ランキングB が押し上げれば戻ってくる）。`chunk_type: "table"` が実測での最良で、
    表チャンクは数値と短いラベルが密なので BM25 も reranker も語の重なりだけで
    高いスコアを出し、**論文が質問の主題でなくても表1枚で代表スコアが跳ね上がる**。

    重みを掛ける形（代表スコア = max(表以外, w × 表)）で `w` を振ると 0.85 以下は
    完全に同値なので、閾値ではなく規則そのものが効いている。だから `w` ではなく
    「使わない」という真偽値で書ける（**自由パラメータが無い**）。

    **チャンクプールは変えない。** 落とすのは代表スコアの計算だけで、表チャンクは
    読解 LLM にそのまま渡り `evidence` にも出せる（gold の `primary_evidence_type` は
    table が17件で最多なので、ここは絶対に落とせない）。むしろ読解には表を見せたほうが
    確認率が高い（見せた 71% / 見せない 51%）。

    `figure` / `equation_algorithm` を一緒に外すと**悪化する**ので `table` だけにする。
    `agg="sum"` では効果がほぼ消える——これは max 集約に固有の歪み。

    **「表しか無い論文には表スコアを使う」というフォールバックを入れてはいけない。**
    親切に見えるが実測で負ける（multi@5 0.758 -> 0.720、4分割で悪化3セル）。
    表しか手掛かりが無い論文が488本あり、**それを沈めること自体が効いている**。
    """
    scores = paper_scores(results, agg=agg, skip_chunk_types=skip_chunk_types)
    # 同点は挿入順（= 融合ランキングでの出現順）で決まる。sorted が安定なため。
    papers = sorted(scores, key=lambda paper_id: scores[paper_id], reverse=True)
    if max_papers is not None:
        papers = papers[:max_papers]
    return papers


def paper_scores(
    results: list[RetrievalResult],
    agg: str = "max",
    skip_chunk_types: Collection[str] = (),
) -> dict[str, float]:
    """論文ごとの代表スコア。`to_gold_papers()` が順位に畳む前の値。

    順位だけでは足りない場面（`anchor_from: "score"` の起点選び）で要るので、
    集約の規則を1か所に持たせる。**`to_gold_papers()` はこの関数を並べ替えるだけ**
    なので、代表スコアの定義が2つに割れることはない。
    """
    skip = set(skip_chunk_types)
    scores: dict[str, float] = {}
    for result in results:
        value = 0.0 if result.chunk_type in skip else result.score
        if agg == "max":
            scores[result.paper_id] = max(scores.get(result.paper_id, value), value)
        elif agg == "sum":
            scores[result.paper_id] = scores.get(result.paper_id, 0.0) + value
        else:
            raise ValueError(f"unknown agg: {agg!r}")
    return scores
