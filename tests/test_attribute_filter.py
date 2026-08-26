"""会議名・年による属性フィルタのテスト。

一番大事なのは「**抽出してはいけない質問で抽出しないこと**」。本番の質問分布は
分からないので、発火しなければ現状と同じ動作に戻る（＝損失ゼロ）という性質だけが
安全の担保になる。誤って絞り込むと gold を候補から落として recall が壊れる。
"""

from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.retrieve.attribute_filter import (
    AttributeExtractor,
    AttributeFilter,
    LLMAttributeExtractor,
    filter_results,
)
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever
from littraceqa.di_pipeline.retrieve.paper_rrf import PaperRRFFuser

# 本物の paper_metadata.jsonl と同じ venue 構成（件数だけ縮めたもの）。
_PAPERS = (
    [{"paper_id": f"naacl2025_{i:03d}", "venue": "NAACL", "year": 2025} for i in range(5)]
    + [{"paper_id": f"cvpr2025_{i:03d}", "venue": "CVPR", "year": 2025} for i in range(10)]
    + [{"paper_id": f"icml2025_{i:03d}", "venue": "ICML", "year": 2025} for i in range(10)]
    + [{"paper_id": f"acl2025_{i:03d}", "venue": "ACL", "year": 2025} for i in range(10)]
    + [{"paper_id": f"eccv2024_{i:03d}", "venue": "ECCV", "year": 2024} for i in range(5)]
)


@pytest.fixture
def extractor(tmp_path):
    path = tmp_path / "paper_metadata.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in _PAPERS:
            f.write(json.dumps(record) + "\n")
    return AttributeExtractor(path)


def _result(chunk_id: str, venue: str, year: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("#")[0],
        score=score,
        text=f"body of {chunk_id}",
        chunk_type="text_span",
        metadata={"venue": venue, "year": year},
    )


# ---- 抽出する例 -----------------------------------------------------------


def test_extracts_venue_and_year(extractor):
    f = extractor.extract(
        "Which NAACL 2025 papers explicitly mention or reference MCTS in their figure?"
    )
    assert f == AttributeFilter(venue="NAACL", year=2025)


def test_extracts_venue_without_year(extractor):
    f = extractor.extract("By how much does a NeurIPS method improve reasoning?")
    assert f.venue == "NeurIPS"
    assert f.year is None


def test_acl_does_not_match_inside_naacl(extractor):
    """ACL が NAACL の部分文字列として拾われないこと（語境界）。"""
    f = extractor.extract("Among NAACL 2025 papers, which one uses MCTS?")
    assert f.venue == "NAACL"


# ---- 抽出してはいけない例 -------------------------------------------------


def test_all_venues_disables_extraction(extractor):
    """「全会議が対象」と明示されている質問では絞り込まない。

    実例では gold が iccv / neurips / icml にまたがっていた。
    """
    f = extractor.extract(
        "Across all venues, among 2025 inference-time scaling methods for "
        "text-to-image generation evaluated on GenEval, what base model is used?"
    )
    assert f.is_empty()


def test_two_venues_disables_extraction(extractor):
    """引用先の会議名が混ざったら諦める。"""
    f = extractor.extract(
        "Which CVPR 2025 papers cite a NeurIPS baseline in their comparison table?"
    )
    assert f.is_empty()


def test_year_only_is_not_extracted(extractor):
    """年だけの言及では絞り込まない（2025 で絞っても 8.7% しか削れない）。"""
    f = extractor.extract("What is the best 2025 method for long-context evaluation?")
    assert f.is_empty()


def test_spaceless_cited_venue_is_ignored(extractor):
    """引用先の "CVPR2023" に引きずられないこと。

    実例: "Which CVPR 2025 papers cite UniAD (..., CVPR2023)"
    語境界(\\b)により "CVPR2023" は会議名としても隣接年としても拾われないので、
    質問本来の制約 CVPR 2025 だけが残る（gold は全て cvpr2025_* なので正しい）。
    """
    f = extractor.extract(
        "Which CVPR 2025 papers cite UniAD (Planning-oriented Autonomous Driving, CVPR2023)?"
    )
    assert f == AttributeFilter(venue="CVPR", year=2025)


def test_conflicting_adjacent_years_drop_the_year(extractor):
    """同じ会議名に別々の年が空白付きで付いていたら年は使わない（会議名だけ残す）。"""
    f = extractor.extract("Which CVPR 2025 papers cite a CVPR 2023 baseline?")
    assert f.venue == "CVPR"
    assert f.year is None


def test_year_absent_from_corpus_is_dropped(extractor):
    """コーパスに無い年で絞ると必ず空になるので、年は使わない。"""
    f = extractor.extract("Which NAACL 2019 papers use MCTS?")
    assert f == AttributeFilter(venue="NAACL", year=None)


def test_empty_question(extractor):
    assert extractor.extract("").is_empty()


# ---- フィルタと選択率 -----------------------------------------------------


def test_filter_results_keeps_only_matching():
    kept = filter_results(
        [
            _result("naacl2025_000#c1", "NAACL", 2025),
            _result("cvpr2025_000#c1", "CVPR", 2025),
            _result("eccv2024_000#c1", "ECCV", 2024),
        ],
        AttributeFilter(venue="NAACL", year=2025),
    )
    assert [r.paper_id for r in kept] == ["naacl2025_000"]


def test_empty_filter_keeps_everything():
    results = [_result("naacl2025_000#c1", "NAACL", 2025)]
    assert filter_results(results, AttributeFilter()) == results


def test_selectivity(extractor):
    # NAACL 2025 は 5 / 40 本
    assert extractor.selectivity(AttributeFilter(venue="NAACL", year=2025)) == pytest.approx(0.125)
    assert extractor.selectivity(AttributeFilter()) == 1.0


# ---- HybridRetriever との結合 ---------------------------------------------


class _StubIndexer:
    """要求された件数だけ、NAACL と CVPR を交互に返すダミー索引。"""

    name = "stub"

    def __init__(self):
        self.requested_k: list[int] = []

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self.requested_k.append(top_k)
        results = []
        for i in range(top_k):
            venue = "NAACL" if i % 2 == 0 else "CVPR"
            prefix = "naacl2025" if venue == "NAACL" else "cvpr2025"
            results.append(_result(f"{prefix}_{i:03d}#c1", venue, 2025, score=top_k - i))
        return results


def _retriever(indexer, extractor, **kwargs) -> HybridRetriever:
    return HybridRetriever(
        indexers=[indexer],
        fuser=PaperRRFFuser(),
        per_index_k=10,
        attribute_extractor=extractor,
        **kwargs,
    )


def test_no_filter_path_is_unchanged(extractor):
    """制約が取れない質問では per_index_k をそのまま要求すること（従来動作）。"""
    indexer = _StubIndexer()
    _retriever(indexer, extractor).retrieve("what is the best method?", top_k=5)
    assert indexer.requested_k == [10]


def test_filter_over_fetches_and_drops(extractor):
    """制約があるときは多めに取り、条件に合わないものを落とすこと。"""
    indexer = _StubIndexer()
    results = _retriever(indexer, extractor, min_filtered_results=1).retrieve(
        "Which NAACL 2025 papers use MCTS?", top_k=5
    )
    # 10 / 0.125 * 1.5 = 120 件を要求する
    assert indexer.requested_k == [120]
    assert results
    assert all(r.metadata["venue"] == "NAACL" for r in results)


def test_fetch_k_is_capped(extractor):
    indexer = _StubIndexer()
    _retriever(indexer, extractor, max_fetch_k=50, min_filtered_results=1).retrieve(
        "Which NAACL 2025 papers use MCTS?", top_k=5
    )
    assert indexer.requested_k == [50]


def test_fails_open_when_filter_empties_the_run(extractor):
    """絞り込み後が min_results 未満なら制約なしに戻すこと（fail-open）。"""
    indexer = _StubIndexer()
    results = _retriever(indexer, extractor, min_filtered_results=1000).retrieve(
        "Which NAACL 2025 papers use MCTS?", top_k=5
    )
    assert any(r.metadata["venue"] == "CVPR" for r in results)


# ---- LLM 抽出（正規表現が空のときだけ後ろに足す）---------------------------


class _StubLLM:
    """決め打ちの応答を返し、呼ばれた回数とプロンプトを記録する LLM。"""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else "{}"


def _llm_extractor(extractor, *responses) -> tuple[LLMAttributeExtractor, _StubLLM]:
    llm = _StubLLM(*responses)
    return LLMAttributeExtractor(extractor, llm), llm


def test_llm_is_not_called_when_regex_succeeds(extractor):
    """正規表現で取れた質問では LLM を呼ばない（検証55件で正規表現は100%正しい）。"""
    llm_extractor, llm = _llm_extractor(extractor)
    f = llm_extractor.extract_with_llm("Which NAACL 2025 papers use MCTS?")
    assert f == AttributeFilter(venue="NAACL", year=2025)
    assert llm.prompts == []


def test_llm_resolves_venue_alias(extractor):
    """正規表現が拾えない別名（NIPS）を LLM が正規の表記に直せること。"""
    llm_extractor, llm = _llm_extractor(
        extractor, '{"venue": "NeurIPS", "year": null, "all_venues": false}'
    )
    f = llm_extractor.extract_with_llm("Among NIPS papers, which one uses MCTS?")
    assert f == AttributeFilter(venue="NeurIPS", year=None)
    assert len(llm.prompts) == 1


def test_llm_extracts_year_only(extractor):
    """会議名なしで年だけの制約も通すこと（ECCV だけが 2024 なので情報量がある）。"""
    llm_extractor, _ = _llm_extractor(
        extractor, '{"venue": null, "year": 2024, "all_venues": false}'
    )
    assert llm_extractor.extract_with_llm("Which 2024 papers use MCTS?") == AttributeFilter(
        venue=None, year=2024
    )


def test_llm_result_is_cached(extractor):
    """同じ質問で2回目の API 呼び出しをしないこと。"""
    llm_extractor, llm = _llm_extractor(
        extractor, '{"venue": "NeurIPS", "year": null, "all_venues": false}'
    )
    question = "Among NIPS papers, which one uses MCTS?"
    first = llm_extractor.extract_with_llm(question)
    assert llm_extractor.extract_with_llm(question) == first
    assert len(llm.prompts) == 1


def test_llm_all_venues_is_empty(extractor):
    llm_extractor, _ = _llm_extractor(
        extractor, '{"venue": null, "year": null, "all_venues": true}'
    )
    assert llm_extractor.extract_with_llm("Across every venue, which paper wins?").is_empty()


def test_llm_hallucinated_venue_is_rejected(extractor):
    """コーパスに無い会議名を答えたら、年も含めて返答ごと捨てること。

    年だけ残すと「AAAI 2025」への返答から year=2025 が生き残り、
    会議名の誤りが ECCV を落とす絞り込みに化ける。
    """
    llm_extractor, _ = _llm_extractor(
        extractor, '{"venue": "AAAI", "year": 2025, "all_venues": false}'
    )
    assert llm_extractor.extract_with_llm("Which AAAI 2025 papers use MCTS?").is_empty()


def test_llm_impossible_pair_drops_the_year(extractor):
    """コーパスに無い (会議名, 年) の組では年だけ落として会議名を残すこと。"""
    llm_extractor, _ = _llm_extractor(
        extractor, '{"venue": "NAACL", "year": 2024, "all_venues": false}'
    )
    assert llm_extractor.extract_with_llm(
        "Among 2024 papers from the Americas chapter meeting, which uses MCTS?"
    ) == AttributeFilter(venue="NAACL", year=None)


def test_llm_garbage_response_is_empty(extractor):
    llm_extractor, _ = _llm_extractor(extractor, "I could not determine the venue.")
    assert llm_extractor.extract_with_llm("Which papers use MCTS?").is_empty()


def test_llm_failure_does_not_raise(extractor):
    """LLM が例外を投げても制約なしに落ちるだけで、検索は続くこと。"""

    def _boom(prompt: str) -> str:
        raise RuntimeError("API is down")

    llm_extractor = LLMAttributeExtractor(extractor, _boom)
    assert llm_extractor.extract_with_llm("Which papers use MCTS?").is_empty()


def test_llm_extractor_extract_stays_regex_only(extractor):
    """HybridRetriever が呼ぶ extract() は LLM を通さないこと。

    制約を渡されなかったとき HybridRetriever は**サブクエリ1本ごとに**
    extract() を呼ぶので、ここが LLM だと1クエリで十数回 API を叩いてしまう。
    """
    llm_extractor, llm = _llm_extractor(
        extractor, '{"venue": "NeurIPS", "year": null, "all_venues": false}'
    )
    assert llm_extractor.extract("Among NIPS papers, which one uses MCTS?").is_empty()
    assert llm.prompts == []


def test_year_only_selectivity(extractor):
    """年だけの制約でも選択率が引けること（fetch_k の逆算に使う）。"""
    # ECCV 2024 は 5 / 40 本
    assert extractor.selectivity(AttributeFilter(year=2024)) == pytest.approx(0.125)


def test_explicit_filter_overrides_extraction(extractor):
    """呼び出し側が渡した制約が、クエリからの抽出より優先されること。

    ReadingAgent は元の質問から抽出した制約をサブクエリ検索に渡してくる。
    """
    indexer = _StubIndexer()
    results = _retriever(indexer, extractor, min_filtered_results=1).retrieve(
        "number of subfigures",  # サブクエリ側には会議名が無い
        top_k=5,
        attribute_filter=AttributeFilter(venue="NAACL", year=2025),
    )
    assert all(r.metadata["venue"] == "NAACL" for r in results)
