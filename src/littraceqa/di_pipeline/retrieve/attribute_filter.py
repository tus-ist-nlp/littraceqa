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

正規表現が構造上拾えない書き方（別名 "NIPS"、正式名称、"a 2025 NeurIPS method" の
ような語順、年だけの指定）に備えて ``LLMAttributeExtractor`` を後ろに足せる。
正規表現が空を返したときだけ LLM に聞き、返答はコーパスに実在する (会議名, 年) の
組でのみ採用する。既定では無効（search_style の ``attribute_filter.llm_extract``）。
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


_LLM_PROMPT = """\
You are deciding whether a research question restricts the SEARCH SCOPE to a particular \
publication venue or year.

The corpus contains papers from exactly these venue/year pairs:
  NeurIPS 2025, ICLR 2025, EMNLP 2025, ACL 2025, ICML 2025, CVPR 2025, ICCV 2025, \
NAACL 2025, ECCV 2024
Note that every venue except ECCV is 2025, so a year is only informative when it \
distinguishes ECCV 2024 from the rest.

Question: {question}

Rules:
* Report a venue ONLY if the question limits which papers to search. Resolve common \
aliases to the names above (e.g. "NIPS" / "Neural Information Processing Systems" -> NeurIPS).
* If the question mentions a venue only to identify a CITED or BASELINE paper (e.g. \
"papers that cite UniAD (CVPR2023)"), that is not a scope restriction - do not report it.
* If the question explicitly covers every venue ("across all venues", "regardless of venue"), \
set all_venues to true and report no venue.
* If two or more different venues are in scope, set all_venues to true.
* Report a year ONLY if the question states which year to search. Do not infer a year \
from a venue.
* When in doubt, report nothing. A wrong restriction removes correct papers from the \
search results; reporting nothing simply keeps the search as it is.

Respond with JSON only, in the form \
{{"venue": "NAACL" or null, "year": 2025 or null, "all_venues": false}}"""


class LLMAttributeExtractor:
    """正規表現で取れなかったときだけ LLM に会議名・年を判定させる抽出器。

    正規表現版（``AttributeExtractor``）を置き換えるのではなく**後ろに足す**。
    検証55件では正規表現の取り逃がしが無く（該当7件を発火5/意図的除外2で正しく処理）、
    LLM に置き換える利得は実測ゼロだったので、確実に効いている経路には触らない。
    LLM が担うのは正規表現が構造上拾えない書き方——別名（NIPS）、正式名称、
    語順違い（"a 2025 NeurIPS method"）、年だけの指定——だけになる。

    ``extract()`` は正規表現のままにしてある。``HybridRetriever`` は制約を渡されな
    かったとき**サブクエリ1本ごとに** ``extract()`` を呼ぶので、ここを LLM にすると
    1クエリあたり十数回 API を叩くことになる。LLM を通すのは ``ReadingAgent`` が
    元の質問に対して1回だけ呼ぶ ``extract_with_llm()`` に限定する。
    """

    def __init__(self, base: AttributeExtractor, llm, prompt: str | None = None):
        self._base = base
        self._llm = llm
        self._prompt = prompt or _LLM_PROMPT
        self._cache: dict[str, AttributeFilter] = {}

    # --- AttributeExtractor と同じインタフェース（HybridRetriever 用）---------

    def extract(self, question: str) -> AttributeFilter:
        """正規表現だけで抽出する。LLM は呼ばない（サブクエリ経路を無料に保つ）。"""
        return self._base.extract(question)

    def selectivity(self, attribute_filter: AttributeFilter) -> float:
        return self._base.selectivity(attribute_filter)

    # --- LLM を通す経路（ReadingAgent がクエリごとに1回だけ呼ぶ）-------------

    def extract_with_llm(self, question: str) -> AttributeFilter:
        """正規表現 → 取れなければ LLM、の順で制約を取る。"""
        regex_filter = self._base.extract(question)
        if not regex_filter.is_empty():
            return regex_filter
        if not question:
            return AttributeFilter()
        if question in self._cache:
            return self._cache[question]

        result = self._validate(self._ask(question))
        self._cache[question] = result
        return result

    def _ask(self, question: str) -> dict | None:
        try:
            return parse_json_object(self._llm(self._prompt.format(question=question)))
        except Exception:
            # 抽出に失敗しても検索は続けたい。制約なし＝従来の挙動に落ちるだけ。
            return None

    def _validate(self, payload: dict | None) -> AttributeFilter:
        """LLM の返答をコーパスの実在値と突き合わせる。

        捏造した会議名や、コーパスに存在しない組み合わせをそのまま信じると
        絞り込みで gold を全部落としうる。実在する組だけを通す。
        """
        if not isinstance(payload, dict) or payload.get("all_venues"):
            return AttributeFilter()

        raw_venue = payload.get("venue")
        venue = AttributeExtractor.canonical_venue(raw_venue)
        if raw_venue and venue is None:
            # コーパスに無い会議名を答えた時点でその返答は信用できない。年だけ拾うと
            # 「AAAI 2025」への返答から year=2025 が残って ECCV を落としてしまうので、
            # 返答ごと捨てる。
            return AttributeFilter()

        year = payload.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None

        # コーパスに無い組み合わせなら年を落とす（会議名だけなら残せることが多い）。
        if year is not None and not self._base.exists(venue, year):
            year = None
        if venue is None and year is None:
            return AttributeFilter()
        return AttributeFilter(venue=venue, year=year)


def filter_results(results: list, attribute_filter: AttributeFilter) -> list:
    """RetrievalResult の列から制約を満たすものだけを返す。"""
    if attribute_filter.is_empty():
        return list(results)
    return [r for r in results if attribute_filter.matches(r.metadata)]
