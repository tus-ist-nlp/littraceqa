"""Tests for deterministic paper-level BM25 aggregation."""

from __future__ import annotations

import json

import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.index.paper_bm25 import PaperBM25Index, aggregate_papers


def _chunk(
    chunk_id: str,
    paper_id: str,
    text: str,
    chunk_type: str = "text_span",
    **metadata,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        chunk_type=chunk_type,
        metadata={"title": f"Title {paper_id}", **metadata},
    )


def test_aggregate_papers_removes_repeated_prefix_and_references():
    chunks = [
        _chunk("p1#c0", "p1", "[ACL 2025] Title p1\nAbstract text", "title_abstract"),
        _chunk("p1#c1", "p1", "[ACL 2025] Title p1\n1 Method", text_level=2),
        _chunk("p1#c2", "p1", "[ACL 2025] Title p1\nUseful method details"),
        _chunk("p1#c3", "p1", "[ACL 2025] Title p1\nReferences", text_level=2),
        _chunk("p1#c4", "p1", "[ACL 2025] Title p1\nUnrelated citation title"),
    ]

    papers = list(aggregate_papers(chunks))

    assert len(papers) == 1
    assert papers[0].chunk_id == "p1#paper"
    assert papers[0].text.count("Title p1") == 1
    assert "1 Method\nUseful method details" in papers[0].text
    assert "Unrelated citation title" not in papers[0].text
    assert papers[0].metadata["source_chunk_count"] == 5


def test_aggregate_papers_rejects_noncontiguous_papers():
    chunks = [
        _chunk("p1#c1", "p1", "prefix\none"),
        _chunk("p2#c1", "p2", "prefix\ntwo"),
        _chunk("p1#c2", "p1", "prefix\nthree"),
    ]

    with pytest.raises(ValueError, match="contiguous"):
        list(aggregate_papers(chunks))


def test_paper_bm25_build_load_and_search(tmp_path):
    chunks = [
        _chunk("p1#c1", "p1", "prefix\nlayer parallel speculative decoding"),
        _chunk("p2#c1", "p2", "prefix\nvisual object detection"),
    ]
    index_dir = tmp_path / "paper-bm25"
    index = PaperBM25Index(str(index_dir), result_text_chars=20)
    index.build(chunks)

    loaded = PaperBM25Index(str(index_dir), result_text_chars=20)
    loaded.load()
    results = loaded.search("speculative decoding", top_k=2)

    assert results[0].paper_id == "p1"
    assert results[0].chunk_type == "paper"
    assert results[0].source == "paper_bm25"
    assert len(results[0].text) <= 20


def test_get_document_lazily_reuses_full_built_chunk(tmp_path):
    index = PaperBM25Index(str(tmp_path / "index"), result_text_chars=5)
    index.build(
        [
            _chunk("p1#c1", "p1", "prefix\ncomplete full-paper text"),
            _chunk("p2#c1", "p2", "prefix\nanother document"),
        ]
    )

    assert index._document_by_paper_id is None

    document = index.get_document("p1")

    assert document is index._delegate._chunks[0]
    assert document.text.endswith("complete full-paper text")
    assert index.get_document("missing") is None
    assert index.get_document("p1") is document


def test_get_document_cache_is_invalidated_after_build_and_load(tmp_path):
    index_dir = tmp_path / "index"
    index = PaperBM25Index(str(index_dir))
    index.build([_chunk("p1#c1", "p1", "prefix\nfirst document")])
    first_document = index.get_document("p1")

    index.build([_chunk("p2#c1", "p2", "prefix\nsecond document")])

    assert index.get_document("p1") is None
    assert index.get_document("p2") is not first_document

    index.get_document("p2")
    replacement = PaperBM25Index(str(index_dir))
    replacement.build([_chunk("p3#c1", "p3", "prefix\nthird document")])
    index.load()

    assert index.get_document("p2") is None
    assert index.get_document("p3") is index._delegate._chunks[0]


def test_live_owner_lookup_uses_current_extractor_on_bounded_papers(tmp_path):
    index = PaperBM25Index(str(tmp_path / "index"))
    index.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nThe result table labels D2PO (ours).",
            ),
            _chunk(
                "other#c1",
                "other",
                "prefix\nThis paper compares unrelated methods.",
            ),
        ]
    )

    records = index.find_method_owners_in_papers(
        ("D²PO",),
        ("owner", "other"),
    )

    assert records == (
        {
            "paper_id": "owner",
            "aliases": ["D2PO"],
            "strength": 2,
        },
    )


def test_live_owner_lookup_rejects_ambiguous_candidate_owners(tmp_path):
    index = PaperBM25Index(str(tmp_path / "index"))
    index.build(
        [
            _chunk("p1#c1", "p1", "prefix\nD2PO (ours)."),
            _chunk("p2#c1", "p2", "prefix\nD2PO (ours)."),
        ]
    )

    assert (
        index.find_method_owners_in_papers(
            ("D2PO",),
            ("p1", "p2"),
        )
        == ()
    )


def test_paper_index_uses_shared_papers_filename(tmp_path):
    index = PaperBM25Index(str(tmp_path / "index"))

    assert index._delegate.records_filename == "papers.jsonl"


def test_paper_index_load_falls_back_to_legacy_chunks_filename(monkeypatch, tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.jsonl").write_text("", encoding="utf-8")
    index = PaperBM25Index(str(index_dir))
    loaded_filenames: list[str] = []
    monkeypatch.setattr(
        index._delegate,
        "load",
        lambda: loaded_filenames.append(index._delegate.records_filename),
    )

    index.load()

    assert loaded_filenames == ["chunks.jsonl"]


def test_method_extraction_is_opt_in_and_persists_evidence(tmp_path):
    chunks = [
        _chunk(
            "p1#c1",
            "p1",
            "prefix\nWe propose Alpha Learning (ALPHA).",
        )
    ]
    disabled = PaperBM25Index(str(tmp_path / "disabled"))
    disabled.build(chunks)

    enabled = PaperBM25Index(
        str(tmp_path / "enabled"),
        extract_method_names=True,
    )
    enabled.build(chunks)

    assert "method_names" not in disabled.get_document("p1").metadata
    assert enabled.get_document("p1").metadata["method_names"] == ["ALPHA"]
    assert enabled.get_document("p1").metadata["method_alias_evidence"] == [
            {
                "alias": "ALPHA",
                "source": "parenthetical_definition",
                "start": 36,
                "end": 41,
                "long_name": "Alpha Learning",
            }
    ]
    persisted = json.loads(
        (tmp_path / "enabled" / "papers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert persisted["metadata"]["method_names"] == ["ALPHA"]


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_method_extraction_flag_requires_boolean(tmp_path, value):
    with pytest.raises(TypeError, match="boolean"):
        PaperBM25Index(
            str(tmp_path / "index"),
            extract_method_names=value,
        )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("10", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_method_max_degree_requires_positive_integer(tmp_path, value, error):
    with pytest.raises(error, match="method_max_degree"):
        PaperBM25Index(
            str(tmp_path / "index"),
            method_max_degree=value,
        )


def test_method_graph_is_exact_case_undirected_and_deterministic(tmp_path):
    chunks = [
        _chunk(
            "owner#c1",
            "owner",
            "prefix\nWe propose a method called ALPHA. "
            "We propose a method called BETA.",
        ),
        _chunk(
            "related#c1",
            "related",
            "prefix\nWe compare ALPHA with BETA in all settings.",
        ),
        _chunk(
            "lowercase#c1",
            "lowercase",
            "prefix\nThe alpha and beta settings are unrelated.",
        ),
    ]
    index = PaperBM25Index(
        str(tmp_path / "index"),
        extract_method_names=True,
    )
    index.build(chunks)

    assert index.get_method_neighbors("owner") == (
        {
            "paper_id": "related",
            "aliases": ["ALPHA", "BETA"],
            "strength": 2,
        },
    )
    assert index.get_method_neighbors("related") == (
        {
            "paper_id": "owner",
            "aliases": ["ALPHA", "BETA"],
            "strength": 2,
        },
    )
    assert index.get_method_neighbors("lowercase") == ()
    assert index.get_method_neighbors("owner") == index.get_method_neighbors(
        "owner"
    )


def test_ambiguous_method_owner_is_discarded(tmp_path):
    chunks = [
        _chunk(
            "p1#c1",
            "p1",
            "prefix\nWe propose a method called ALPHA.",
        ),
        _chunk(
            "p2#c1",
            "p2",
            "prefix\nWe introduce an approach called ALPHA.",
        ),
        _chunk("p3#c1", "p3", "prefix\nWe evaluate ALPHA."),
    ]
    index = PaperBM25Index(
        str(tmp_path / "index"),
        extract_method_names=True,
    )
    index.build(chunks)

    assert index.find_method_owners("ALPHA") == ()
    assert index.get_method_neighbors("p1") == ()
    assert index.get_method_neighbors("p2") == ()
    assert index.get_method_neighbors("p3") == ()


def test_high_degree_method_alias_is_discarded(tmp_path):
    chunks = [
        _chunk(
            "owner#c1",
            "owner",
            "prefix\nWe propose a model called ALPHA.",
        ),
        *[
            _chunk(
                f"mention-{number}#c1",
                f"mention-{number}",
                "prefix\nOur evaluation includes ALPHA.",
            )
            for number in range(3)
        ],
    ]
    index = PaperBM25Index(
        str(tmp_path / "index"),
        extract_method_names=True,
        method_max_degree=2,
    )
    index.build(chunks)

    assert index.find_method_owners("ALPHA") == ()
    assert index.get_method_neighbors("owner") == ()


def test_method_graph_ignores_references_and_wrong_case(tmp_path):
    chunks = [
        _chunk(
            "owner#c1",
            "owner",
            "prefix\nWe propose a model called ALPHA.",
        ),
        _chunk(
            "references#c1",
            "references",
            "prefix\nReferences",
            text_level=2,
        ),
        _chunk(
            "references#c2",
            "references",
            "prefix\nThe ALPHA paper.",
        ),
        _chunk(
            "wrong-case#c1",
            "wrong-case",
            "prefix\nWe evaluate Alpha.",
        ),
    ]
    index = PaperBM25Index(
        str(tmp_path / "index"),
        exclude_references=False,
        extract_method_names=True,
    )
    index.build(chunks)

    assert "ALPHA paper" in index.get_document("references").text
    assert index.get_method_neighbors("owner") == ()


def test_method_graph_validates_long_name_context(tmp_path):
    index = PaperBM25Index(
        str(tmp_path / "index"),
        extract_method_names=True,
    )
    index.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nWe propose Truncated Consistency Models (TCM).",
            ),
            _chunk(
                "thermal#c1",
                "thermal",
                "prefix\nThe thermal control module (TCM) regulates heat.",
            ),
            _chunk(
                "consistency-only#c1",
                "consistency-only",
                "prefix\nOur consistency experiments compare TCM.",
            ),
            _chunk(
                "validated#c1",
                "validated",
                "prefix\nTruncated consistency experiments compare TCM.",
            ),
        ]
    )

    assert index.get_document("owner").metadata[
        "method_alias_evidence"
    ][0]["long_name"] == "Truncated Consistency Models"
    assert index.get_method_neighbors("owner") == (
        {
            "paper_id": "validated",
            "aliases": ["TCM"],
            "strength": 1,
        },
    )
    assert index.get_method_neighbors("thermal") == ()
    assert index.get_method_neighbors("consistency-only") == ()
    assert index.get_method_neighbors("validated") == (
        {
            "paper_id": "owner",
            "aliases": ["TCM"],
            "strength": 1,
        },
    )


def test_short_method_alias_requires_two_long_name_context_words(tmp_path):
    index = PaperBM25Index(
        str(tmp_path / "index"),
        extract_method_names=True,
    )
    index.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nWe propose Mixture of Decoding (MoD).",
            ),
            _chunk(
                "mixture-only#c1",
                "mixture-only",
                "prefix\nThe mixture baseline is compared with MoD.",
            ),
            _chunk(
                "validated#c1",
                "validated",
                "prefix\nMixture decoding experiments compare MoD.",
            ),
        ]
    )

    assert index.get_method_neighbors("owner") == (
        {
            "paper_id": "validated",
            "aliases": ["MoD"],
            "strength": 1,
        },
    )
    assert index.get_method_neighbors("mixture-only") == ()


def test_find_method_owners_uses_normalized_exact_method_names(tmp_path):
    index = PaperBM25Index(
        str(tmp_path / "index"),
        extract_method_names=True,
    )
    index.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nWe propose a model called D-FINE.",
            ),
            _chunk(
                "fine-owner#c1",
                "fine-owner",
                "prefix\nWe propose a model called FiNE.",
            ),
            _chunk(
                "svd-owner#c1",
                "svd-owner",
                "prefix\nWe propose a method called SVD.",
            ),
            _chunk(
                "scm-owner#c1",
                "scm-owner",
                "prefix\nWe propose a model called sCM.",
            ),
            _chunk(
                "sct-owner#c1",
                "sct-owner",
                "prefix\nWe propose a model called sCT.",
            ),
            _chunk(
                "uppercase-sct-owner#c1",
                "uppercase-sct-owner",
                "prefix\nWe propose a model called SCT.",
            ),
            _chunk("other#c1", "other", "prefix\nOther text."),
        ]
    )

    assert index.find_method_owners("Does D-FINE improve recall?") == (
        {
            "paper_id": "owner",
            "aliases": ["D-FINE"],
            "strength": 2,
        },
    )
    assert index.find_method_owners("d fine") == (
        {
            "paper_id": "owner",
            "aliases": ["D-FINE"],
            "strength": 1,
        },
    )
    assert index.find_method_owners("SVD") == (
        {
            "paper_id": "svd-owner",
            "aliases": ["SVD"],
            "strength": 2,
        },
    )
    assert index.find_method_owners("Dobi-SVD") == ()
    assert index.find_method_owners("sCM") == (
        {
            "paper_id": "scm-owner",
            "aliases": ["sCM"],
            "strength": 2,
        },
    )
    assert index.find_method_owners("SCM") == ()
    assert index.find_method_owners("sCT") == (
        {
            "paper_id": "sct-owner",
            "aliases": ["sCT"],
            "strength": 2,
        },
    )
    assert index.find_method_owners("SCT") == (
        {
            "paper_id": "uppercase-sct-owner",
            "aliases": ["SCT"],
            "strength": 2,
        },
    )


def test_method_graph_sidecar_is_atomic_and_reused_after_load(
    monkeypatch,
    tmp_path,
):
    index_dir = tmp_path / "index"
    built = PaperBM25Index(
        str(index_dir),
        extract_method_names=True,
    )
    built.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nWe propose a model called ALPHA.",
            ),
            _chunk("related#c1", "related", "prefix\nWe use ALPHA."),
        ]
    )
    sidecar = index_dir / "method_alias_graph.json"
    assert sidecar.is_file()
    assert not list(index_dir.glob(".method_alias_graph.json.*.tmp"))

    loaded = PaperBM25Index(
        str(index_dir),
        extract_method_names=True,
    )
    loaded.load()
    monkeypatch.setattr(
        loaded,
        "_build_method_index_from_documents",
        lambda **kwargs: pytest.fail("valid sidecar should be reused"),
    )

    assert loaded.get_method_neighbors("owner")[0]["paper_id"] == "related"


def test_legacy_index_is_lazily_enriched_without_writing_sidecar(
    monkeypatch,
    tmp_path,
):
    index_dir = tmp_path / "index"
    legacy = PaperBM25Index(str(index_dir))
    legacy.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nWe propose a model called ALPHA.",
            ),
            _chunk("related#c1", "related", "prefix\nWe use ALPHA."),
        ]
    )
    loaded = PaperBM25Index(
        str(index_dir),
        extract_method_names=True,
    )
    loaded.load()
    original = loaded._build_method_index_from_documents
    calls: list[bool] = []

    def observe_build(*, enrich=True):
        calls.append(enrich)
        return original(enrich=enrich)

    monkeypatch.setattr(
        loaded,
        "_build_method_index_from_documents",
        observe_build,
    )

    assert loaded._method_owner_by_alias is None
    assert loaded.get_document("owner").metadata["method_names"] == ["ALPHA"]
    assert calls == [False]
    assert loaded.get_method_neighbors("owner")[0]["paper_id"] == "related"
    assert not (index_dir / "method_alias_graph.json").exists()


def test_corrupt_method_sidecar_rebuilds_in_memory_without_overwrite(tmp_path):
    index_dir = tmp_path / "index"
    built = PaperBM25Index(
        str(index_dir),
        extract_method_names=True,
    )
    built.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nWe propose a model called ALPHA.",
            ),
            _chunk("related#c1", "related", "prefix\nWe use ALPHA."),
        ]
    )
    sidecar = index_dir / "method_alias_graph.json"
    sidecar.write_text("{broken", encoding="utf-8")

    loaded = PaperBM25Index(
        str(index_dir),
        extract_method_names=True,
    )
    loaded.load()

    assert loaded.get_method_neighbors("owner")[0]["paper_id"] == "related"
    assert sidecar.read_text(encoding="utf-8") == "{broken"


def test_old_method_sidecar_schema_rebuilds_without_overwrite(tmp_path):
    index_dir = tmp_path / "index"
    built = PaperBM25Index(
        str(index_dir),
        extract_method_names=True,
    )
    built.build(
        [
            _chunk(
                "owner#c1",
                "owner",
                "prefix\nWe propose a model called ALPHA.",
            ),
            _chunk("related#c1", "related", "prefix\nWe use ALPHA."),
        ]
    )
    sidecar = index_dir / "method_alias_graph.json"
    old_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    old_payload["schema_version"] = 2
    sidecar.write_text(json.dumps(old_payload), encoding="utf-8")

    loaded = PaperBM25Index(
        str(index_dir),
        extract_method_names=True,
    )
    loaded.load()

    assert loaded.get_method_neighbors("owner")[0]["paper_id"] == "related"
    assert json.loads(sidecar.read_text(encoding="utf-8"))[
        "schema_version"
    ] == 2
