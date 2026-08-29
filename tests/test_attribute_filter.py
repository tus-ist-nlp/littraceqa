"""The venue/year attribute filter.

**What matters most is not extracting where extraction would be wrong.** The
distribution of production questions is unknown, so the only thing that makes this
safe is that not firing falls back to exactly the previous behaviour — zero loss.
Narrowing wrongly drops gold from the candidates and breaks recall outright.
"""

from __future__ import annotations

import json

import pytest

from littraceqa.search.contracts import RetrievalResult
from littraceqa.search.retrieve import (
    AttributeExtractor,
    AttributeFilter,
    HybridRetriever,
    PaperRRFFuser,
    filter_results,
)

# The same venue mix as the real paper_metadata.jsonl, with the counts scaled down.
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


# ---- what should be extracted ---------------------------------------------


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
    """ACL is not picked up inside NAACL — word boundaries."""
    f = extractor.extract("Among NAACL 2025 papers, which one uses MCTS?")
    assert f.venue == "NAACL"


# ---- what must not be extracted -------------------------------------------


def test_all_venues_disables_extraction(extractor):
    """A question that says "all venues" narrows nothing.

    In the real case its gold spanned iccv, neurips and icml.
    """
    f = extractor.extract(
        "Across all venues, among 2025 inference-time scaling methods for "
        "text-to-image generation evaluated on GenEval, what base model is used?"
    )
    assert f.is_empty()


def test_two_venues_disables_extraction(extractor):
    """Two venue names means one of them belongs to a cited paper; give up."""
    f = extractor.extract(
        "Which CVPR 2025 papers cite a NeurIPS baseline in their comparison table?"
    )
    assert f.is_empty()


def test_year_only_is_not_extracted(extractor):
    """A bare year narrows nothing: filtering to 2025 removes only 8.7%."""
    f = extractor.extract("What is the best 2025 method for long-context evaluation?")
    assert f.is_empty()


def test_spaceless_cited_venue_is_ignored(extractor):
    """A cited paper's "CVPR2023" does not drag the extraction along.

    The real case: "Which CVPR 2025 papers cite UniAD (..., CVPR2023)". Word
    boundaries (\\b) keep "CVPR2023" from matching either as a venue or as an
    adjacent year, so only the question's own constraint, CVPR 2025, survives —
    correctly, since all its gold is cvpr2025_*.
    """
    f = extractor.extract(
        "Which CVPR 2025 papers cite UniAD (Planning-oriented Autonomous Driving, CVPR2023)?"
    )
    assert f == AttributeFilter(venue="CVPR", year=2025)


def test_conflicting_adjacent_years_drop_the_year(extractor):
    """One venue with two spaced years drops the year and keeps the venue."""
    f = extractor.extract("Which CVPR 2025 papers cite a CVPR 2023 baseline?")
    assert f.venue == "CVPR"
    assert f.year is None


def test_year_absent_from_corpus_is_dropped(extractor):
    """A year the corpus does not have would always filter to nothing, so it is dropped."""
    f = extractor.extract("Which NAACL 2019 papers use MCTS?")
    assert f == AttributeFilter(venue="NAACL", year=None)


def test_empty_question(extractor):
    assert extractor.extract("").is_empty()


# ---- filtering and selectivity --------------------------------------------


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
    # NAACL 2025 is 5 of the 40
    assert extractor.selectivity(AttributeFilter(venue="NAACL", year=2025)) == pytest.approx(0.125)
    assert extractor.selectivity(AttributeFilter()) == 1.0


# ---- wired into HybridRetriever -------------------------------------------


class _StubIndexer:
    """A dummy index returning as many results as asked, alternating NAACL and CVPR."""

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
    """With no constraint, per_index_k is requested unchanged — the previous behaviour."""
    indexer = _StubIndexer()
    _retriever(indexer, extractor).retrieve("what is the best method?", top_k=5)
    assert indexer.requested_k == [10]


def test_filter_over_fetches_and_drops(extractor):
    """With a constraint, over-fetch and drop what does not match."""
    indexer = _StubIndexer()
    results = _retriever(indexer, extractor, min_filtered_results=1).retrieve(
        "Which NAACL 2025 papers use MCTS?", top_k=5
    )
    # asks for 10 / 0.125 * 1.5 = 120
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
    """Fewer than min_results after filtering falls back to no constraint (fail-open)."""
    indexer = _StubIndexer()
    results = _retriever(indexer, extractor, min_filtered_results=1000).retrieve(
        "Which NAACL 2025 papers use MCTS?", top_k=5
    )
    assert any(r.metadata["venue"] == "CVPR" for r in results)


# ---- selectivity and a caller-supplied constraint -------------------------


def test_year_only_selectivity(extractor):
    """A year-only constraint still has a selectivity, which fetch_k works back from."""
    # ECCV 2024 is 5 of the 40
    assert extractor.selectivity(AttributeFilter(year=2024)) == pytest.approx(0.125)


def test_explicit_filter_overrides_extraction(extractor):
    """A constraint passed in wins over one extracted from the query.

    ReadingAgent extracts once from the original question and passes that down to
    each subquery search.
    """
    indexer = _StubIndexer()
    results = _retriever(indexer, extractor, min_filtered_results=1).retrieve(
        "number of subfigures",  # the subquery itself names no venue
        top_k=5,
        attribute_filter=AttributeFilter(venue="NAACL", year=2025),
    )
    assert all(r.metadata["venue"] == "NAACL" for r in results)
