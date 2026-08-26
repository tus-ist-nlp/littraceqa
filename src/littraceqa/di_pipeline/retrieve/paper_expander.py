"""候補1位論文の「近い論文」で候補列を拡張する（論文→論文展開）。

近さの測り方は3種類あり、registry の "expander" として差し替え・併用できる:

- ``specter2``: SPECTER2(proximity) の埋め込み近傍。意味的な近さ。
- ``bib_coupling``: 書誌結合。参考文献の arXiv ID 集合の Jaccard で測る。
- ``bm25_mlt``: 論文全文の more-like-this。anchor の title+abstract をクエリにして
  構築済みの ``bm25s_paper`` 索引を引く。レキシカルな近さ。

取り込み方は**関連ランキングと候補列の RRF 統合**の1通りだけ。``rank()`` が
既存候補も含めた関連度順を返す（重なった論文を加点するのが目的なので落としてはいけない）。
統合の式と根拠は agent/reading.py の ``_combine_rrf`` を参照。

かつては「追記すべき論文を候補列の決まった位置に差し込む」位置挿入方式もあったが、
順位融合に全指標で負けたので実装ごと削除した。

**3つとも違う gold を拾うので併用する価値がある**（実測: 候補圏外 gold 37本の回収は
SPECTER2 15本 / 書誌結合 11本 / 全文MLT 16本で、MLT だけが拾えたのが2本、
既存2つだけが拾えたのが6本、重複14本）。``fused`` が各ソースの近傍を RRF で融合する。

書誌結合が効くのは、このコーパスが2024〜2025年の論文しか持たないため。
同時期の論文は互いに引用できない（TCM は sCT / IMM を引用していない）ので
引用グラフそのものは繋がらない——実測で anchor から解決できたコーパス内引用は
1本だけだった。しかし**同じ古い文献を引いている**ので書誌結合なら繋がる
（TCM とピア3本の Jaccard 0.19〜0.24 に対し、無作為30本は中央値 0.000・最大 0.054）。

multi_paper の gold は「トピッククラスタの主要論文」の使い回しで、
質問文が名指ししないピア論文は**質問→論文**の検索では原理的に拾いにくい
（クエリ品質監査の実測: 候補50位に入らない evidence 持ち gold 17本）。
一方でそれらは**正解論文からは近い**。SPECTER2 の proximity アダプタは
引用近接で学習された論文単位の類似埋め込みなので、候補1位（たいてい
supporting 本体）を anchor に近傍を引けばクラスタの残りが拾える。

検証55件・predictions_8b_chunk_b_merged での実測（anchor上位1×近傍20）:

    candidate_recall          0.836 -> 0.914 (+7.8pt)
    evidence_candidate_recall 0.908 -> 0.956 (+4.8pt)
    候補列の伸び              50 -> 平均57本程度（重複除去後）

anchor を3本に増やしても ecr は 0.956 から動かず候補だけ増えた（クラスタの
中心1本で十分）ため、既定は anchors=1 / neighbors=20。

実装上の要点:

- **構築済みの faiss_specter2_abstract 索引をそのまま読む**。anchor のベクトルは
  索引から reconstruct() で取り出すので、クエリ時に SPECTER2 モデルのロードも
  GPU も不要（faiss の CPU 検索1回）。
- 展開結果の挿入位置は呼び出し側（ReadingAgent）が決める。このクラスは
  「追加すべき論文IDのリスト」を返すだけ。実測では **max_candidates 直後への
  挿入**が最良（末尾追記比で cr@20 同一のまま cr@50 が 0.855 -> 0.880。
  10位挿入は LLM 可視域の候補を押し出して cr@20 を壊す）。
- 提出（gold_papers）には影響しない。挿入分は LLM が読む max_candidates の
  範囲外で、apply_paper_cutoff の対象にもならない。
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

import bm25s
import faiss

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.registry import register

_INDEX_FILENAME = "index.faiss"
_CHUNKS_FILENAME = "chunks.jsonl"
# bm25s_paper 索引が並べて書き出す「1論文=1ドキュメント」の本文。
_PAPERS_FILENAME = "papers.jsonl"
_PAPER_ID_RE = re.compile(r'"paper_id":\s*"([^"]+)"')

# 参考文献から arXiv ID を拾う。MinerU の出力は "ar X iv : 2403.06807" のように
# 字間が割れることがあるので、間の空白を許す。
_ARXIV_RE = re.compile(r"ar\s*X\s*iv[:\s]*(\d{4}\.\d{4,5})")

class PaperExpander(Protocol):
    """論文→論文の近さで「起点に似た論文」を順位付けする（ランキングB）。

    **起点（anchor）は呼び出し側が決めて渡す。** どの論文を起点にするかは
    A/B 統合の判断（`agent/reading.py` の `_anchor_papers`）なので、expander は
    「渡された起点の近傍を返す」ことだけを担う。

    返り値から**既存候補を除外しない**。既存候補と重なる論文こそ RRF 統合での
    加点対象なので、ここで落とすと統合の意味が消える（`_combine_rrf` 参照）。
    """

    def rank(self, anchors: list[str]) -> list[str]: ...


def _interleave(
    pools: list[list[str]], limit: int, exclude: set[str] | None = None
) -> list[str]:
    """複数 anchor の近傍リストをランク順に交互配置して1本にする。

    1つの anchor の遠い近傍より、別の anchor の近い近傍を先に置く。
    ``exclude`` に入っている論文は飛ばす（``rank()`` は何も除外しない）。
    """
    seen = set(exclude or ())
    merged: list[str] = []
    for rank in range(limit):
        for pool in pools:
            if rank < len(pool) and pool[rank] not in seen:
                seen.add(pool[rank])
                merged.append(pool[rank])
    return merged[:limit]


@register("expander", "specter2")
class Specter2PaperExpander:
    """SPECTER2(proximity) 埋め込みの近傍。構築済みの faiss 索引をそのまま読む。"""

    def __init__(self, index_dir: str, neighbors: int = 20):
        self.index_dir = Path(index_dir)
        self.neighbors = neighbors
        # 索引のロードは初回 rank() まで遅延する（--build だけの実行や
        # テストで索引が無くても構築できるように）。
        self._index: faiss.Index | None = None
        self._row_of: dict[str, int] = {}
        self._pid_of: dict[int, str] = {}
        # rerank 用に title+abstract の本文も持っておく（この索引の chunks.jsonl に
        # 入っているので、別途チャンクストアを読む必要はない）。
        self._chunk_of: dict[str, dict] = {}

    def _load(self) -> None:
        self._index = faiss.read_index(str(self.index_dir / _INDEX_FILENAME))
        with open(self.index_dir / _CHUNKS_FILENAME, encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                chunk = json.loads(line)
                paper_id = chunk.get("paper_id", "")
                # abstract 索引は1論文1行だが、複数行あっても最初の行を使う。
                if paper_id and paper_id not in self._row_of:
                    self._row_of[paper_id] = row
                    self._pid_of[row] = paper_id
                    self._chunk_of[paper_id] = chunk

    def _pools(self, anchors: list[str]) -> list[list[str]]:
        """anchor ごとの近傍リスト（近い順）。"""
        if not anchors:
            return []
        if self._index is None:
            self._load()
        assert self._index is not None

        pools: list[list[str]] = []
        for anchor in anchors:
            row = self._row_of.get(anchor)
            if row is None:
                continue
            vector = self._index.reconstruct(row).reshape(1, -1)
            _, ids = self._index.search(vector, self.neighbors + 1)
            pools.append(
                [
                    self._pid_of[i]
                    for i in ids[0]
                    if i >= 0 and i in self._pid_of and self._pid_of[i] != anchor
                ]
            )
        return pools

    def rank(self, anchors: list[str]) -> list[str]:
        return _interleave(self._pools(anchors), self.neighbors)

    def text_of(self, paper_id: str) -> str:
        """rerank に渡す代表テキスト（title+abstract）。無ければ空文字。"""
        if self._index is None:
            self._load()
        return (self._chunk_of.get(paper_id) or {}).get("text", "")


@register("expander", "bib_coupling")
class BibCouplingExpander:
    """書誌結合（参考文献の共有）で近い論文を返す。

    各論文の全文から参考文献の arXiv ID を抜き、ID 集合の Jaccard で近さを測る。
    **引用グラフ（A が B を引く）ではない**。このコーパスは2024〜2025年しか無く
    同時期の論文は互いに引用できないので、引用グラフは繋がらない（実測で anchor から
    解決できたコーパス内引用は1本）。一方、同じ古典を引いているかは測れる。

    索引はコーパス全走査で作り、``cache_path`` に pickle で保存する（実測47秒、
    25,012論文 / 68,418 ID）。2回目以降はキャッシュを読むだけ。GPU 不要。
    """

    def __init__(
        self,
        chunks: str,
        cache_path: str,
        neighbors: int = 20,
        min_shared: int = 2,
    ):
        self.chunks_path = Path(chunks)
        self.cache_path = Path(cache_path)
        self.neighbors = neighbors
        # 共有文献がこの本数未満のペアは切る。1本だけの共有は汎用的な引用
        # （Adam, ResNet 等）で繋がってしまい、ノイズにしかならない。
        self.min_shared = min_shared
        self._refs: dict[str, set[str]] | None = None
        self._inv: dict[str, set[str]] = {}
        self._text: dict[str, str] = {}

    def _load(self) -> None:
        if self.cache_path.exists():
            payload = pickle.loads(self.cache_path.read_bytes())
        else:
            payload = self._build()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(payload))
        self._refs = payload["refs"]
        self._inv = payload["inv"]
        self._text = payload["text"]

    def _build(self) -> dict:
        """chunks.jsonl を1回走査して {論文 -> 引用arXiv ID} と転置索引を作る。"""
        refs: dict[str, set[str]] = {}
        text: dict[str, str] = {}
        current: str | None = None
        buffer: list[str] = []
        title_abstract = ""

        def flush() -> None:
            if current is None:
                return
            ids = set(_ARXIV_RE.findall(" ".join(buffer)))
            if ids:
                refs[current] = ids
            if title_abstract:
                text[current] = title_abstract

        with open(self.chunks_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                paper_id = chunk.get("paper_id", "")
                if paper_id != current:
                    flush()
                    current, buffer, title_abstract = paper_id, [], ""
                body = chunk.get("text", "")
                buffer.append(body)
                if chunk.get("chunk_type") == "title_abstract" and not title_abstract:
                    title_abstract = body
        flush()

        inv: dict[str, set[str]] = {}
        for paper_id, ids in refs.items():
            for arxiv_id in ids:
                inv.setdefault(arxiv_id, set()).add(paper_id)
        return {"refs": refs, "inv": inv, "text": text}

    def _neighbors(self, paper_id: str) -> list[str]:
        assert self._refs is not None
        own = self._refs.get(paper_id)
        if not own:
            return []
        shared: Counter[str] = Counter()
        for arxiv_id in own:
            for other in self._inv.get(arxiv_id, ()):
                if other != paper_id:
                    shared[other] += 1
        scores = {
            other: count / len(own | self._refs[other])
            for other, count in shared.items()
            if count >= self.min_shared
        }
        # 同点は paper_id で決める。scores は set を走査して作るので、これが無いと
        # 同点の並びが**プロセスごとに変わる**（文字列ハッシュの乱択）。実測で候補列の
        # 40%強のクエリが実行のたびに入れ替わり、cr@20 が 0.4pt ぶれた。
        return sorted(scores, key=lambda p: (-scores[p], p))[: self.neighbors]

    def _pools(self, anchors: list[str]) -> list[list[str]]:
        if not anchors:
            return []
        if self._refs is None:
            self._load()
        return [self._neighbors(a) for a in anchors]

    def rank(self, anchors: list[str]) -> list[str]:
        """書誌結合の近さ順。"""
        return _interleave(self._pools(anchors), self.neighbors)

    def text_of(self, paper_id: str) -> str:
        if self._refs is None:
            self._load()
        return self._text.get(paper_id, "")


def _json_string_prefix(line: str, start: int, max_chars: int) -> str:
    """JSON1行の ``start`` 位置から始まる文字列値の先頭 ``max_chars`` 文字を復号する。

    papers.jsonl は1行が数百KB〜数MBあるので、行全体を ``json.loads`` すると
    27,489行ぶんで数分かかる。閉じ引用符まで読まずに先頭だけ取りたいので、
    エスケープを自前で解きながら必要な分だけ進める。
    """
    out: list[str] = []
    i = start
    while i < len(line) and len(out) < max_chars:
        char = line[i]
        if char == '"':  # 文字列値の終わり（本文がとても短い論文）
            break
        if char == "\\":
            escape = line[i : i + 6] if line[i + 1 : i + 2] == "u" else line[i : i + 2]
            try:
                out.append(json.loads(f'"{escape}"'))
            except json.JSONDecodeError:
                break
            i += len(escape)
            continue
        out.append(char)
        i += 1
    return "".join(out)


@register("expander", "bm25_mlt")
class BM25MLTExpander:
    """論文全文の more-like-this。anchor の title+abstract で `bm25s_paper` 索引を引く。

    SPECTER2（abstract の意味近傍）とも書誌結合（引用文献の共有）とも違い、
    **本文全体のレキシカル一致**で近さを測る。LLM 呼び出しゼロ・追加の索引構築ゼロで、
    構築済みの `bm25s_paper`（論文1本=1ドキュメントの BM25）をそのまま読む。

    ``papers.jsonl`` は 2.5GB あるが**クエリ時には読まない**:

    - BM25 本体は ``mmap=True`` で開くので npy(490MB) を RAM に載せない。
    - 行番号 -> paper_id と anchor 用の title+abstract は初回1回だけ流し読みして
      pickle にキャッシュする（BibCouplingExpander の refs.pkl と同じ作法）。
      各行の text は "[venue year] title\\n" + abstract + 本文… の順なので、
      先頭 ``query_chars`` 文字を取れば title+abstract になる。
    """

    def __init__(
        self,
        index_dir: str,
        cache_path: str,
        neighbors: int = 20,
        # anchor のクエリに使う先頭文字数。title+abstract を覆う程度に取る
        # （短い abstract では本文の冒頭まで入るが、MLT のクエリとしては害がない）。
        query_chars: int = 1200,
    ):
        self.index_dir = Path(index_dir)
        self.cache_path = Path(cache_path)
        self.neighbors = neighbors
        self.query_chars = query_chars
        self._bm25 = None
        self._pids: list[str] = []
        self._text: dict[str, str] = {}

    def _load(self) -> None:
        if self.cache_path.exists():
            payload = pickle.loads(self.cache_path.read_bytes())
        else:
            payload = self._build()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(payload))
        self._pids = payload["pids"]
        self._text = payload["text"]
        self._bm25 = bm25s.BM25.load(str(self.index_dir), load_corpus=False, mmap=True)

    def _build(self) -> dict:
        """papers.jsonl を1回流し読みして {行番号 -> paper_id} と title+abstract を作る。"""
        pids: list[str] = []
        text: dict[str, str] = {}
        with open(self.index_dir / _PAPERS_FILENAME, encoding="utf-8") as handle:
            for line in handle:
                match = _PAPER_ID_RE.search(line, 0, 200)
                if match is None:
                    continue
                paper_id = match.group(1)
                pids.append(paper_id)
                marker = line.find('"text": "', match.end())
                if marker >= 0:
                    text[paper_id] = _json_string_prefix(
                        line, marker + len('"text": "'), self.query_chars
                    )
        return {"pids": pids, "text": text}

    def _neighbors(self, paper_id: str) -> list[str]:
        query = self._text.get(paper_id)
        if not query:
            return []
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k = min(self.neighbors + 1, len(self._pids))
        if k <= 0:
            return []
        doc_indices, _ = self._bm25.retrieve(tokens, k=k, show_progress=False)
        return [
            self._pids[int(i)]
            for i in doc_indices[0]
            if 0 <= int(i) < len(self._pids) and self._pids[int(i)] != paper_id
        ]

    def _pools(self, anchors: list[str]) -> list[list[str]]:
        if not anchors:
            return []
        if self._bm25 is None:
            self._load()
        return [self._neighbors(a) for a in anchors]

    def rank(self, anchors: list[str]) -> list[str]:
        """全文 BM25 の近さ順。"""
        return _interleave(self._pools(anchors), self.neighbors)

    def text_of(self, paper_id: str) -> str:
        if self._bm25 is None:
            self._load()
        return self._text.get(paper_id, "")


@register("expander", "fused")
class FusedPaperExpander:
    """複数の expander の近傍を RRF で融合する。

    SPECTER2（意味的な近さ）と書誌結合（引用文献の共有）は違う gold を拾うので、
    片方に寄せず順位融合する。重み付けの根拠が無いので RRF（順位のみを使う、
    スコアのスケールに依存しない）を使う——検索側の fuser と同じ考え方。
    """

    def __init__(
        self,
        sources: list[PaperExpander],
        neighbors: int = 20,
        rrf_k: int = 60,
    ):
        self.sources = sources
        self.neighbors = neighbors
        self.rrf_k = rrf_k

    def _fuse(self, per_source: list[list[str]]) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in per_source:
            for rank, paper_id in enumerate(ranking):
                scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ordered = sorted(scores, key=lambda p: -scores[p])
        return ordered[: self.neighbors]

    def rank(self, anchors: list[str]) -> list[str]:
        """各ソースの近傍ランキングを RRF 融合する。"""
        return self._fuse([source.rank(anchors) for source in self.sources])

    def text_of(self, paper_id: str) -> str:
        for source in self.sources:
            text = source.text_of(paper_id)
            if text:
                return text
        return ""


