"""質問文が明示した会議名・年で検索結果を絞り込むための抽出器とフィルタ。

LitTraceQA の質問には「Which NAACL 2025 papers ...」「Among ICML 2025 papers ...」
のように**検索範囲を会議名で明示している**ものがある。検証55件では5件が該当し、
そのとき gold 論文がその制約を満たす率は 18/18 = 100% だった。つまり制約が
述べられていればフィルタは常に正しい。

索引の改修は要らない。``RetrievalResult.metadata`` には既に venue / year が
入っている（``preprocess/mineru_chunker.py`` の metadata_base）ので、
検索結果を後段で落とすだけでどの indexer にも同じように効く。

**発火条件は「会議名が一意に取れたとき」だけ**にしてある。理由:

* 年だけで絞っても意味が薄い。コーパスは 2025 が 91.3%、2024 が 8.7% の2値しか
  なく、2025 で絞っても 8.7% しか削れない。
* 「Across all venues, among 2025 ...」のように**明示的に全会議を対象**とする
  質問がある（gold は iccv / neurips / icml にまたがっていた）。
* 「Which CVPR 2025 papers cite UniAD (..., CVPR2023)」のように、**引用先の
  会議名**が混ざることがある。会議名が2種類以上見つかったら諦める。

取れなかったときは空の AttributeFilter を返し、呼び出し側は現状どおりの
コードパスを通る（＝本番の質問が会議名を書かなければ挙動は一切変わらない）。

**正規表現だけで足りている。** 検証55件で取り逃がしはゼロ（会議名に言及する7件を
発火5件／`all venues` の意図的除外2件ですべて正しく処理）。別名 "NIPS" や語順違いに
備えて LLM を後段に足した版も作ったが、実測での利得がゼロだったので消してある。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from littraceqa.di_pipeline.agent.json_utils import parse_json_object

# コーパスに存在する会議名（data/paper_metadata.jsonl の venue 全9値）。
# 表記ゆれの吸収のため、小文字化したキーで引く。
_VENUES = ("NeurIPS", "ICLR", "EMNLP", "ACL", "ICML", "CVPR", "ICCV", "ECCV", "NAACL")

# 「全会議が対象」と明示している質問。会議名を拾ってはいけない。
_ALL_VENUES_RE = re.compile(r"\ball\s+venues\b", re.I)

# 会議名の直後（空白・カンマ・アポストロフィを挟んで）に付く年だけを採用する。
# "CVPR 2025 papers cite UniAD (..., CVPR2023)" のように離れた年に引きずられないため。
_YEAR_RE = r"(20\d{2})"


@dataclass(frozen=True)
class AttributeFilter:
    """検索結果に掛ける属性の制約。空なら制約なし。"""

    venue: str | None = None
    year: int | None = None

    def is_empty(self) -> bool:
        return self.venue is None and self.year is None

    def matches(self, metadata: dict | None) -> bool:
        """チャンクの metadata がこの制約を満たすか。"""
        metadata = metadata or {}
        if self.venue is not None and metadata.get("venue") != self.venue:
            return False
        if self.year is not None and metadata.get("year") != self.year:
            return False
        return True


class AttributeExtractor:
    """質問文から AttributeFilter を作り、その選択率を答える。

    選択率は paper_metadata.jsonl の論文数から求める（チャンク数ではなく論文数の
    比を使う。1論文あたりのチャンク数は会議によって大きくは変わらないので、
    取得件数の逆算にはこれで足りる）。
    """

    def __init__(self, paper_metadata: str | Path):
        self._venue_by_lower = {v.lower(): v for v in _VENUES}
        self._total = 0
        self._counts: dict[tuple[str | None, int | None], int] = {}
        self._load(Path(paper_metadata))
        # 会議名は語境界で拾う。ACL が NAACL の一部に一致しないよう \b を両側に置く。
        self._venue_re = re.compile(
            r"\b(" + "|".join(re.escape(v) for v in _VENUES) + r")\b", re.I
        )

    def _load(self, path: Path) -> None:
        papers: list[tuple[str, int]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                venue = record.get("venue")
                year = record.get("year")
                papers.append((venue, int(year) if year is not None else None))

        self._total = len(papers)
        counts: dict[tuple[str | None, int | None], int] = {}
        for venue, year in papers:
            # (venue, None) / (venue, year) / (None, year) を数えておき、
            # 会議名・年のどちらを欠いた制約でも選択率を引けるようにする。
            # (None, year) は LLM 抽出が「年だけ」の制約を返しうるため必要
            # （正規表現の抽出器は年だけの制約を作らない）。
            for key in ((venue, None), (venue, year), (None, year)):
                counts[key] = counts.get(key, 0) + 1
        self._counts = counts

    def exists(self, venue: str | None, year: int | None) -> bool:
        """その (会議名, 年) の組がコーパスに1本でもあるか。"""
        if venue is None and year is None:
            return False
        return self._counts.get((venue, year), 0) > 0

    @staticmethod
    def canonical_venue(name: str | None) -> str | None:
        """表記ゆれを吸収してコーパスの会議名表記に直す。無い会議なら None。"""
        if not name:
            return None
        for venue in _VENUES:
            if venue.lower() == name.strip().lower():
                return venue
        return None

    def extract(self, question: str) -> AttributeFilter:
        """質問文から制約を取り出す。取れなければ空の AttributeFilter。"""
        if not question or _ALL_VENUES_RE.search(question):
            return AttributeFilter()

        found = {self._venue_by_lower[m.group(1).lower()] for m in self._venue_re.finditer(question)}
        if len(found) != 1:
            # 0個（会議名なし）でも2個以上（引用先が混ざっている）でも諦める。
            return AttributeFilter()
        venue = next(iter(found))

        return AttributeFilter(venue=venue, year=self._adjacent_year(question, venue))

    def _adjacent_year(self, question: str, venue: str) -> int | None:
        """会議名に隣接する年だけを返す。離れた場所の年は無視する。

        "Which CVPR 2025 papers cite UniAD (Planning-oriented ..., CVPR2023)" では
        CVPR2023 も隣接年として拾えてしまうので、複数見つかったら採用しない。
        """
        pattern = re.compile(r"\b" + re.escape(venue) + r"\b[\s,'’]*" + _YEAR_RE, re.I)
        years = {int(m.group(1)) for m in pattern.finditer(question)}
        if len(years) != 1:
            return None
        year = next(iter(years))
        # コーパスに存在しない年で絞ると必ず空になる。その場合は年を使わない。
        if self._counts.get((venue, year), 0) == 0:
            return None
        return year

    def selectivity(self, attribute_filter: AttributeFilter) -> float:
        """制約を満たす論文の割合。0除算を避けるため下限を置く。"""
        if attribute_filter.is_empty() or self._total == 0:
            return 1.0
        matched = self._counts.get((attribute_filter.venue, attribute_filter.year), 0)
        if matched <= 0:
            return 1.0
        return matched / self._total


def filter_results(results: list, attribute_filter: AttributeFilter) -> list:
    """RetrievalResult の列から制約を満たすものだけを返す。"""
    if attribute_filter.is_empty():
        return list(results)
    return [r for r in results if attribute_filter.matches(r.metadata)]
