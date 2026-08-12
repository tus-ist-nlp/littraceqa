"""コーパス内論文どうしの**明示的な関係**で近い論文を返す expander 2種。

既存の3ソース（``specter2`` / ``bib_coupling`` / ``bm25_mlt``）はどれも
「内容が近い」を測っている。ここで足すのは「**名指ししているか**」で、
測っているものが違う。

- ``title_mention``: A の本文に B の名前（正式タイトル or コロン前の見出し）が
  出てくる。A→B の直接リンクと、ハブを避けた2ホップ。
- ``method_comention``: A と B が**同じ論文の名前を挙げている**（共言及）。
  Jaccard で測る。名指しの向きではなく「同じ手法群を論じている仲間か」を見る。

**CLAUDE.md の「引用グラフはほぼ張れない（実測でコーパス内引用1本）」は
arXiv ID 解決ベースの結論**で、ここでは覆る可能性がある。参考文献の arXiv ID が
取れなくても、本文に名前が書かれていれば繋がるため。実際に繋がるかは
replay_expansion.py で測って判断する。

### spec からの設計変更: 手法名の正規表現抽出をやめた

``docs/candidate_ranking_spec.md`` 3.6 は ``We propose X`` を正規表現で抜いて
「手法名 -> 提案元論文」を作る設計だったが、**MinerU の出力では成立しない**。
タイトルの実データが ``T oken S hapley:`` / ``H i A gent:`` / ``AIMSC heck:`` の
ように大文字の前で分かち書きが壊れており、``We propose <名前>`` の <名前> を
トークン列から復元すると同じ崩れに当たる。

代わりに**タイトルのコロン前の見出しを手法名として使う**。65%のタイトルが
コロンを持ち、その前が手法名になっている（``D-FINE:`` / ``SECRET:`` / ``HSCR:``）。
「提案元がコーパス内で一意」という spec の要件は、正規表現ではなく
``paper_titles.TitleIndex`` の一意性フィルタがそのまま満たす。

さらに **owner/relation の2種類ではなく共言及（co-mention）にした**。
owner への辺（「手法 X を挙げている論文」→「X の提案元」）は
``title_mention`` の辺と同じものになるので足しても情報が増えない。
ピア gold（q_036 の TCM に対する IMM / sCT のような、質問文が名指ししない
同トピック論文）に届くのは「**A と B が同じ手法群を挙げている**」ほうで、
これは title_mention では作れない辺になる。
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from littraceqa.di_pipeline.registry import register
from littraceqa.di_pipeline.retrieve import paper_titles
from littraceqa.di_pipeline.retrieve.paper_expander import (
    _interleave,
    _set_combine,
)

_REFERENCE_SECTIONS = ("references", "reference", "bibliography", "referencesandnotes")


def _is_reference_section(chunk: dict) -> bool:
    """参考文献の節か。名前が並んでいるだけの節で辺を張らないために外す。"""
    section = (chunk.get("metadata") or {}).get("section") or ""
    return paper_titles.normalize(section) in _REFERENCE_SECTIONS


def build_relations(chunks_path: Path, max_key_degree: int = 20) -> dict:
    """chunks.jsonl を2回走査して「どの論文がどの論文を名指ししているか」を作る。

    1周目はタイトルだけ（``title_abstract`` の行を文字列判定で拾うので安い）。
    全論文の名前が揃わないと本文の照合ができないので分ける。
    2周目が本体で、本文を正規化・連結してから前置フィルタ付きで照合する
    （詳細は paper_titles のモジュール docstring）。

    **``max_key_degree`` 本を超える論文から名指しされたキーは捨てる。**
    大文字小文字を一致させても、``HTML`` / ``MUST`` / ``FLAME`` / ``FLARE`` のように
    「ALLCAPS の普通の英単語・略語」を手法名にした論文が残る（コーパスの21%で実測
    したとき ``HTML`` が123本から「名指しされて」いた）。こういう名前は
    **どの論文とも繋がるので関係の弁別に使えない**——出現本数で機械的に落とす。
    外部チームの「同じ手法名に繋がる論文が10本を超えたら曖昧な名前として除外」
    と同じ考え方で、判定を固定リストではなくコーパスの分布に任せる。
    """
    index = paper_titles.TitleIndex()
    owners: dict[str, set[str]] = {}
    with open(chunks_path, encoding="utf-8") as handle:
        for line in handle:
            # json.loads を全行に掛けないための粗い前置判定。
            if '"title_abstract"' not in line:
                continue
            chunk = json.loads(line)
            title = (chunk.get("metadata") or {}).get("title") or ""
            paper_id = chunk.get("paper_id") or ""
            if paper_id and title:
                index.add(paper_id, title, owners)
    index.finalize(owners)

    # まずキー単位で拾う（論文IDに落とす前に、ハブになった名前を数えて捨てるため）。
    hits: dict[str, set[str]] = {}
    text: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is None or not buffer:
            return
        found = index.lookup_keys(paper_titles.Mentions(" ".join(buffer)))
        # 自分自身の名前は数えない（自分の論文には必ず出てくる）。
        found -= {k for k in found if index.owner.get(k) == current}
        if found:
            hits[current] = found

    with open(chunks_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            paper_id = chunk.get("paper_id", "")
            if paper_id != current:
                flush()
                current, buffer = paper_id, []
            if chunk.get("chunk_type") == "title_abstract":
                text.setdefault(paper_id, chunk.get("text", ""))
                continue
            if _is_reference_section(chunk):
                continue
            buffer.append(chunk.get("text", ""))
    flush()

    degree: Counter[str] = Counter()
    for keys in hits.values():
        degree.update(keys)
    hubs = {key for key, count in degree.items() if count > max_key_degree}

    mentions: dict[str, set[str]] = {}
    for paper_id, keys in hits.items():
        targets = {index.owner[key] for key in keys - hubs}
        targets.discard(paper_id)
        if targets:
            mentions[paper_id] = targets

    mentioned_by: dict[str, set[str]] = {}
    for paper_id, targets in mentions.items():
        for target in targets:
            mentioned_by.setdefault(target, set()).add(paper_id)

    return {
        "mentions": mentions,
        "mentioned_by": mentioned_by,
        "text": text,
        "n_keys": len(index),
        "max_key_degree": max_key_degree,
        # 質問文の名指し保護（agent/reading.py）も同じ辞書を使う。索引を2度作らない。
        "title_index": index,
        "hub_keys": frozenset(hubs),
        # 捨てたハブ名（出現本数の多い順）。何が落ちたかを目で確認するため。
        "dropped_hubs": sorted(
            ((key, degree[key]) for key in hubs), key=lambda kv: -kv[1]
        )[:50],
    }


def load_relations(chunks_path: Path, cache_path: Path, max_key_degree: int = 20) -> dict:
    """キャッシュがあれば読む。無ければ作って保存する（bib_coupling と同じ作法）。"""
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())
    payload = build_relations(chunks_path, max_key_degree=max_key_degree)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(payload))
    return payload


def load_title_index(
    chunks_path: Path, cache_path: Path
) -> tuple[paper_titles.TitleIndex, frozenset[str]]:
    """名指し保護（`agent/reading.py`）が使う識別子辞書とハブ名の集合。

    関係グラフと**同じキャッシュ**を読む。索引を2度作らないため。
    """
    payload = load_relations(chunks_path, cache_path)
    return payload["title_index"], payload.get("hub_keys", frozenset())


class _RelationExpanderBase:
    """``paper_expander`` の各 expander と同じ形（rank / rank_pools / text_of）。"""

    def __init__(
        self,
        chunks: str,
        cache_path: str,
        neighbors: int = 20,
        anchors: int = 1,
        max_key_degree: int = 20,
        **combine_kwargs,
    ):
        _set_combine(self, combine_kwargs)
        self.chunks_path = Path(chunks)
        self.cache_path = Path(cache_path)
        self.neighbors = neighbors
        self.anchors = anchors
        # キャッシュが既にあるときは使われない（作り直すときだけ効く）。
        self.max_key_degree = max_key_degree
        self._payload: dict | None = None

    def _load(self) -> None:
        self._payload = load_relations(
            self.chunks_path, self.cache_path, self.max_key_degree
        )

    def _ensure(self) -> dict:
        if self._payload is None:
            self._load()
        assert self._payload is not None
        return self._payload

    def _neighbors(self, paper_id: str) -> list[str]:
        raise NotImplementedError

    def _pools(self, ranked_paper_ids: list[str]) -> list[list[str]]:
        if not ranked_paper_ids:
            return []
        self._ensure()
        return [self._neighbors(a) for a in ranked_paper_ids[: self.anchors]]

    def rank(self, ranked_paper_ids: list[str]) -> list[str]:
        return _interleave(self._pools(ranked_paper_ids), self.neighbors)

    def rank_pools(self, ranked_paper_ids: list[str]) -> list[list[str]]:
        return [pool[: self.neighbors] for pool in self._pools(ranked_paper_ids)]

    def text_of(self, paper_id: str) -> str:
        return self._ensure()["text"].get(paper_id, "")


@register("expander", "title_mention")
class TitleMentionExpander(_RelationExpanderBase):
    """本文でのタイトル言及（双方向）で近い論文を返す。

    直接リンクを重み 1.0、2ホップを ``two_hop_weight``(0.05) で足す。
    2ホップは**中継論文の次数が ``max_hub_degree``(4) 以下のときだけ**使う。
    survey のように大量の論文へ繋がる論文を経由すると、無関係な論文が
    まとめて上がってノイズにしかならないため。
    """

    def __init__(self, *args, two_hop_weight: float = 0.05, max_hub_degree: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.two_hop_weight = two_hop_weight
        self.max_hub_degree = max_hub_degree

    def _linked(self, paper_id: str) -> set[str]:
        """向きを問わない直接リンク（A が B を挙げる / B が A を挙げる）。"""
        payload = self._ensure()
        return payload["mentions"].get(paper_id, set()) | payload["mentioned_by"].get(
            paper_id, set()
        )

    def _neighbors(self, paper_id: str) -> list[str]:
        direct = self._linked(paper_id)
        scores: dict[str, float] = {target: 1.0 for target in direct}
        if self.two_hop_weight > 0:
            for hop in direct:
                linked = self._linked(hop)
                if len(linked) > self.max_hub_degree:
                    continue
                for target in linked:
                    if target == paper_id or target in direct:
                        continue
                    scores[target] = scores.get(target, 0.0) + self.two_hop_weight
        scores.pop(paper_id, None)
        # 同点は paper_id で決める（プロセスごとに順序がブレると実験が再現しない。
        # bib_coupling で実際に起きた問題。CLAUDE.md 参照）。
        return sorted(scores, key=lambda p: (-scores[p], p))[: self.neighbors]


@register("expander", "method_comention")
class MethodCoMentionExpander(_RelationExpanderBase):
    """**同じ論文群を名指ししている**論文を返す（共言及の Jaccard）。

    title_mention が「A が B を挙げた」という辺なのに対し、こちらは
    「A と B が同じ C を挙げている」という辺。質問文が名指ししない
    ピア gold（q_036 の TCM に対する IMM / sCT）は、gold どうしが
    互いを挙げているとは限らないが、**同じ手法群を挙げている**ことは多い。

    ``min_shared`` は bib_coupling と同じ理由で 2 が既定。共有1本だけの
    ペアは有名手法1つで繋がってしまう。
    """

    def __init__(self, *args, min_shared: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_shared = min_shared

    def _load(self) -> None:
        super()._load()
        assert self._payload is not None
        # 「その論文を挙げている論文」の転置は mentioned_by がそのまま使える。
        self._inv = self._payload["mentioned_by"]

    def _neighbors(self, paper_id: str) -> list[str]:
        payload = self._ensure()
        own = payload["mentions"].get(paper_id)
        if not own:
            return []
        shared: Counter[str] = Counter()
        for target in own:
            for other in self._inv.get(target, ()):  # 同じ論文を挙げている仲間
                if other != paper_id:
                    shared[other] += 1
        scores: dict[str, float] = {}
        for other, count in shared.items():
            if count < self.min_shared:
                continue
            other_own = payload["mentions"].get(other) or set()
            union = len(own | other_own)
            if union:
                scores[other] = count / union
        return sorted(scores, key=lambda p: (-scores[p], p))[: self.neighbors]
