"""論文単位の RRF（1論文1票）で複数 Indexer の検索結果を統合する。

既定の `RRFFuser`（`retrieve/rrf.py`）は**チャンク単位**で融合する:

    s(c) = Σ_i  w_i / (k + chunk_rank_i(c))

このとき、**同じ論文の複数チャンクがそれぞれ独立に票を持つ**。長い論文や表が多い
論文は単純にチャンク数が多いので上位を占有しやすく、「論文としてこの質問に近いか」
とは別の理由で順位が上がる。評価は論文単位（`candidate_recall`）なので、この歪みは
そのまま指標に効く。

ここでは**先に論文へ畳んでから**融合する:

    s(p) = Σ_i  w_i / (k + paper_rank_i(p))
    paper_rank_i(p) = run i の中で p が最初に現れた位置（0起点の密順位）

**1つの run の中では、同じ論文に何チャンク当たっても1票**になる。

出力は従来どおり**チャンクの列**（reranker も読解エージェントも evidence も
chunk_id で動くため）。論文の順位を主キー、論文内のチャンク順位を副キーにして並べ、
`score` には**下流が並べ直しても同じ順序が再現される値**を書く
（`agent/reading.py` の貯め込み・`_candidate_papers`・`to_gold_papers` は
すべて score で並べ直すので、返り値の並び順だけに順序を持たせると捨てられる）。

**1論文が出せるチャンク数は `chunks_per_paper` 本に制限する。** 制限しないと
チャンクを100本持つ論文1本が `pool_k` を食い潰し、reranker が1論文しか見なくなる。
既定の3本は「読解エージェントが1論文あたり2チャンク見せる」よりわずかに多い値。

**`bm25s_paper` 由来の擬似チャンク（`{paper_id}#paper`）は代表に選ばない。**
`chunk_id` が実在しないので evidence に使えず、text は論文全文なので読解にも渡せない。
論文の順位付けには使い（それがこの索引の役目）、代表チャンクには実チャンクを充てる。
"""

from __future__ import annotations

import dataclasses

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.registry import register

# 論文単位索引の擬似チャンク。順位付けには使うが、代表チャンクには選ばない。
PAPER_LEVEL_SOURCES = frozenset({"bm25s_paper"})

# 同じ論文の中でチャンクの順序を保つための微小オフセット。論文スコアの差
# （1/(k+r) - 1/(k+r+1) ≈ 1/(k+r)^2、k=60・r=200 でも 1.5e-5）より十分小さくして、
# 論文をまたぐ順序を絶対に壊さないようにする。
_CHUNK_ORDER_EPS = 1e-9


def is_paper_level(result: RetrievalResult) -> bool:
    """論文単位索引が作った擬似チャンクか。"""
    return result.source in PAPER_LEVEL_SOURCES or result.chunk_id.endswith("#paper")


def paper_rrf_fuse(
    runs: list[list[RetrievalResult]],
    top_k: int,
    k: int = 60,
    weights: dict[str, float] | None = None,
    chunks_per_paper: int = 3,
) -> list[RetrievalResult]:
    """論文単位 RRF。`PaperRRFFuser.fuse()` の実体（他所からも呼べるよう関数に出す）。"""
    weights = weights or {}
    paper_scores: dict[str, float] = {}
    # チャンク側は従来どおりのチャンク単位 RRF スコア。論文内の並び順にだけ使う。
    chunk_scores: dict[str, float] = {}
    chunk_of: dict[str, RetrievalResult] = {}
    chunks_of_paper: dict[str, list[str]] = {}

    for run in runs:
        # paper_id -> (この run での密順位, その論文を最初に出したチャンクの source)
        seen_papers: dict[str, tuple[int, str]] = {}
        for rank, result in enumerate(run):
            weight = weights.get(result.source, 1.0)
            chunk_scores[result.chunk_id] = chunk_scores.get(result.chunk_id, 0.0) + weight / (
                k + rank + 1
            )
            if result.chunk_id not in chunk_of:
                chunk_of[result.chunk_id] = result
                chunks_of_paper.setdefault(result.paper_id, []).append(result.chunk_id)
            # **1 run 内では1論文1票。** 最初に現れた位置だけを順位として使う。
            if result.paper_id not in seen_papers:
                seen_papers[result.paper_id] = (len(seen_papers), result.source)
        for paper_id, (paper_rank, source) in seen_papers.items():
            # 論文の票は「その run を出した索引」の重みで測る（1 run = 1索引）。
            paper_scores[paper_id] = paper_scores.get(paper_id, 0.0) + weights.get(
                source, 1.0
            ) / (k + paper_rank + 1)

    ordered_papers = sorted(paper_scores, key=lambda p: (-paper_scores[p], p))

    fused: list[RetrievalResult] = []
    for paper_id in ordered_papers:
        candidates = sorted(
            chunks_of_paper.get(paper_id, []),
            # 擬似チャンクは最後に回す（実チャンクがあればそちらを代表にする）。
            key=lambda cid: (is_paper_level(chunk_of[cid]), -chunk_scores[cid], cid),
        )
        for offset, chunk_id in enumerate(candidates[:chunks_per_paper]):
            fused.append(
                dataclasses.replace(
                    chunk_of[chunk_id],
                    score=paper_scores[paper_id] - offset * _CHUNK_ORDER_EPS,
                )
            )
            if len(fused) >= top_k:
                return fused
    return fused


@register("fuser", "paper_rrf")
class PaperRRFFuser:
    def __init__(
        self,
        k: int = 60,
        weights: dict[str, float] | None = None,
        chunks_per_paper: int = 3,
    ):
        self.k = k
        self.weights = weights or {}
        self.chunks_per_paper = chunks_per_paper

    def fuse(
        self, runs: list[list[RetrievalResult]], top_k: int
    ) -> list[RetrievalResult]:
        return paper_rrf_fuse(
            runs,
            top_k,
            k=self.k,
            weights=self.weights,
            chunks_per_paper=self.chunks_per_paper,
        )
