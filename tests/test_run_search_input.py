"""Tests for production-compatible query loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from littraceqa.di_pipeline.contracts import Chunk
from littraceqa.di_pipeline.preprocess.checkpoint import MergeResult, PreprocessCache
from scripts.run_search import (
    build_indexers_with_resume,
    iter_chunks,
    load_paper_ids_file,
    load_queries,
    normalize_paper_ids,
    override_max_chars_per_chunk,
    preprocess_selected_papers,
    resolve_preprocess_cache_root,
    select_papers_for_bounded_build,
    validate_build_ceiling,
    validate_build_mode,
    validate_large_build_selection,
    validate_preprocess_cache_root,
    validate_write_paths,
)


def _write_query(path) -> None:
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "question": "What is reported?",
                "answer_types": ["freeform"],
                "table_schema": None,
                "task_family": "hidden_source_single_paper",
                "primary_evidence_type": "table",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_options(path) -> None:
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "answer": {
                    "multiple_choice": {
                        "options": {"A": "first", "B": "second"},
                        "gold": "A",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_query_loading_discards_validation_labels_by_default(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    _write_query(path)

    query = load_queries(path)[0]

    assert query.task_family is None
    assert query.primary_evidence_type is None


def test_query_loading_requires_explicit_oracle_mode_for_labels(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    _write_query(path)

    query = load_queries(path, production_input=False)[0]

    assert query.task_family == "hidden_source_single_paper"
    assert query.primary_evidence_type == "table"


def test_production_loading_never_joins_oracle_options(tmp_path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    options_path = tmp_path / "validation.jsonl"
    _write_query(queries_path)
    _write_options(options_path)

    production = load_queries(queries_path, options_path=options_path)[0]
    oracle = load_queries(
        queries_path,
        production_input=False,
        options_path=options_path,
    )[0]

    assert production.options is None
    assert oracle.options == {"A": "first", "B": "second"}


def test_bounded_selection_preserves_metadata_order(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    path.write_text(
        "".join(
            json.dumps({"paper_id": paper_id}) + "\n"
            for paper_id in ("p1", "p2", "p3")
        ),
        encoding="utf-8",
    )

    papers = select_papers_for_bounded_build(path, ["p3", "p1"], None)

    assert [paper["paper_id"] for paper in papers] == ["p1", "p3"]


def test_unbounded_build_selection_is_rejected(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    path.write_text(json.dumps({"paper_id": "p1"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --paper-id and/or --limit"):
        select_papers_for_bounded_build(path, [], None)


def test_limit_cannot_silently_drop_requested_paper_ids(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    path.write_text(
        json.dumps({"paper_id": "p1"}) + "\n" + json.dumps({"paper_id": "p2"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be smaller"):
        select_papers_for_bounded_build(path, ["p1", "p2"], 1)


def test_paper_ids_file_normalizes_blank_and_duplicate_lines(tmp_path) -> None:
    path = tmp_path / "paper_ids.txt"
    path.write_text(" p2 \n\np1\np2\n  \np3\n", encoding="utf-8")

    assert load_paper_ids_file(path) == ["p2", "p1", "p3"]


def test_paper_ids_file_stops_at_the_explicit_ceiling(tmp_path) -> None:
    path = tmp_path / "paper_ids.txt"
    path.write_text(
        "".join(f"p{index:05d}\n" for index in range(5_001)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="more than 5000 distinct"):
        load_paper_ids_file(path)
    assert len(load_paper_ids_file(path, max_papers=10_000)) == 5_001


def test_cli_and_file_paper_ids_are_merged_deterministically() -> None:
    assert normalize_paper_ids(
        [" p1", "p2", "p1"],
        ["p2", "p3", ""],
    ) == ["p1", "p2", "p3"]


def test_large_build_requires_nonempty_id_file_and_exact_confirmation(
    tmp_path,
) -> None:
    id_file = tmp_path / "paper_ids.txt"
    id_file.write_text("p1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --paper-ids-file"):
        validate_large_build_selection(
            201,
            paper_ids_file=None,
            confirm_paper_count=201,
        )
    with pytest.raises(ValueError, match="must equal"):
        validate_large_build_selection(
            201,
            paper_ids_file=id_file,
            confirm_paper_count=None,
        )
    with pytest.raises(ValueError, match="must equal"):
        validate_large_build_selection(
            2_000,
            paper_ids_file=id_file,
            confirm_paper_count=1_999,
        )

    validate_large_build_selection(
        2_000,
        paper_ids_file=id_file,
        confirm_paper_count=2_000,
    )


def test_existing_200_paper_build_needs_no_large_build_confirmation() -> None:
    validate_large_build_selection(
        200,
        paper_ids_file=None,
        confirm_paper_count=None,
    )


def test_bounded_selection_never_adds_unspecified_papers(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    path.write_text(
        "".join(
            json.dumps({"paper_id": paper_id}) + "\n"
            for paper_id in ("outside-1", "wanted-2", "outside-2", "wanted-1")
        ),
        encoding="utf-8",
    )

    papers = select_papers_for_bounded_build(
        path,
        normalize_paper_ids(["wanted-1"], ["wanted-2"]),
        None,
    )

    assert [paper["paper_id"] for paper in papers] == ["wanted-2", "wanted-1"]


def test_bounded_selection_accepts_5000_but_rejects_5001(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    path.write_text(
        "".join(
            json.dumps({"paper_id": f"p{index:04d}"}) + "\n"
            for index in range(5_000)
        ),
        encoding="utf-8",
    )

    assert len(select_papers_for_bounded_build(path, [], 5_000)) == 5_000
    with pytest.raises(ValueError, match="between 1 and 5000"):
        select_papers_for_bounded_build(path, [], 5_001)
    with pytest.raises(ValueError, match="at most 5000 distinct"):
        select_papers_for_bounded_build(
            path,
            [f"p{index:04d}" for index in range(5_001)],
            None,
        )


def test_explicit_build_cap_accepts_exactly_10000(tmp_path) -> None:
    path = tmp_path / "papers.jsonl"
    path.write_text(
        "".join(
            json.dumps({"paper_id": f"p{index:05d}"}) + "\n"
            for index in range(10_000)
        ),
        encoding="utf-8",
    )
    paper_ids = [f"p{index:05d}" for index in range(10_000)]

    papers = select_papers_for_bounded_build(
        path,
        paper_ids,
        10_000,
        max_build_papers=10_000,
    )

    assert len(papers) == 10_000


def test_extended_build_requires_all_exact_count_confirmations(tmp_path) -> None:
    id_file = tmp_path / "paper_ids.txt"
    id_file.write_text("p1\n", encoding="utf-8")
    common = {
        "selected_count": 10_000,
        "paper_ids_file": id_file,
        "confirm_paper_count": 10_000,
        "limit": 10_000,
        "max_build_papers": 10_000,
    }

    validate_large_build_selection(**common)
    for key, value in (
        ("paper_ids_file", None),
        ("confirm_paper_count", None),
        ("confirm_paper_count", 9_999),
        ("limit", None),
        ("limit", 9_999),
        ("max_build_papers", 5_000),
        ("max_build_papers", 9_999),
    ):
        invalid = dict(common)
        invalid[key] = value
        with pytest.raises(ValueError):
            validate_large_build_selection(**invalid)

    incomplete = dict(common)
    incomplete["selected_count"] = 200
    with pytest.raises(ValueError, match="must equal --max-build-papers"):
        validate_large_build_selection(**incomplete)


def test_absolute_build_cap_is_rejected_before_metadata_access() -> None:
    missing = Path("/path/that/must/not/be/read.jsonl")

    with pytest.raises(ValueError, match="between 1 and 10000"):
        select_papers_for_bounded_build(
            missing,
            [],
            1,
            max_build_papers=10_001,
        )
    with pytest.raises(ValueError, match="between 1 and 10000"):
        select_papers_for_bounded_build(
            missing,
            [],
            10_001,
            max_build_papers=10_000,
        )
    with pytest.raises(ValueError, match="at most 10000 distinct"):
        select_papers_for_bounded_build(
            missing,
            [f"p{index:05d}" for index in range(10_001)],
            None,
            max_build_papers=10_000,
        )
    with pytest.raises(ValueError, match="between 1 and 10000"):
        validate_build_ceiling(10_001)


def test_mineru_chunk_size_override_does_not_mutate_source_config() -> None:
    source = {"name": "mineru", "params": {"max_chars_per_chunk": 2000}}

    updated = override_max_chars_per_chunk(source, 4000)

    assert source["params"]["max_chars_per_chunk"] == 2000
    assert updated["params"]["max_chars_per_chunk"] == 4000


@pytest.mark.parametrize("value", [0, 100_001])
def test_mineru_chunk_size_override_rejects_unsafe_values(value: int) -> None:
    with pytest.raises(ValueError, match="must be between"):
        override_max_chars_per_chunk({"name": "mineru", "params": {}}, value)


def test_chunk_size_override_rejects_other_preprocessors() -> None:
    with pytest.raises(ValueError, match="supports MinerU only"):
        override_max_chars_per_chunk({"name": "marker", "params": {}}, 4000)


def test_resume_requires_build_mode() -> None:
    with pytest.raises(ValueError, match="--resume requires --build"):
        validate_build_mode(
            build=False,
            build_only=False,
            resume=True,
            max_chars_per_chunk=None,
        )

    validate_build_mode(
        build=True,
        build_only=True,
        resume=True,
        max_chars_per_chunk=4000,
    )
    with pytest.raises(
        ValueError,
        match="--preprocess-cache-root requires --build",
    ):
        validate_build_mode(
            build=False,
            build_only=False,
            resume=False,
            max_chars_per_chunk=None,
            preprocess_cache_root=Path("/tmp/cache"),
        )


def test_preprocess_cache_root_keeps_prior_artifact_default(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"

    assert resolve_preprocess_cache_root(None, artifact_root) == (
        artifact_root / "preprocess"
    ).resolve()


def test_preprocess_cache_root_rejects_input_overlap(tmp_path) -> None:
    read_only_root = tmp_path / "read-only"
    source_root = tmp_path / "source"
    read_only_root.mkdir()
    source_root.mkdir()

    for dangerous_root in (
        read_only_root,
        read_only_root / "cache",
        source_root,
        source_root / "cache",
        tmp_path,
    ):
        with pytest.raises(ValueError, match="must not overlap"):
            validate_preprocess_cache_root(
                dangerous_root,
                read_only_root=read_only_root,
                source_roots=[source_root],
            )

    validate_preprocess_cache_root(
        tmp_path / "safe-cache",
        read_only_root=read_only_root,
        source_roots=[source_root],
    )


@pytest.mark.parametrize("internal_name", ["papers", "manifest.jsonl"])
def test_preprocess_cache_root_rejects_internal_symlinks(
    tmp_path,
    internal_name,
) -> None:
    read_only_root = tmp_path / "read-only"
    source_root = tmp_path / "source"
    cache_root = tmp_path / "safe-cache"
    read_only_root.mkdir()
    source_root.mkdir()
    cache_root.mkdir()
    target = read_only_root / internal_name
    if internal_name == "papers":
        target.mkdir()
    else:
        target.write_text("", encoding="utf-8")
    (cache_root / internal_name).symlink_to(
        target,
        target_is_directory=internal_name == "papers",
    )

    with pytest.raises(ValueError, match="must not be symlinks"):
        validate_preprocess_cache_root(
            cache_root,
            read_only_root=read_only_root,
            source_roots=[source_root],
        )


class _FakeCheckpointPreprocessor:
    def __init__(self, source_dir, fail_ids=()) -> None:
        self.source_dir = source_dir
        self.fail_ids = set(fail_ids)
        self.calls: list[str] = []

    def content_list_path(self, paper_id: str):
        return self.source_dir / f"{paper_id}.json"

    def process(self, paper: dict) -> list[Chunk]:
        paper_id = paper["paper_id"]
        self.calls.append(paper_id)
        if paper_id in self.fail_ids:
            raise ValueError(f"broken source for {paper_id}")
        return [
            Chunk(
                chunk_id=f"{paper_id}#c0000",
                paper_id=paper_id,
                text=f"text for {paper_id}",
                chunk_type="text_span",
                metadata={},
            )
        ]


def test_paper_checkpoints_resume_only_failed_paper_and_preserve_order(
    tmp_path,
) -> None:
    papers = [{"paper_id": paper_id} for paper_id in ("p1", "p2", "p3")]
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for paper in papers:
        (source_dir / f"{paper['paper_id']}.json").write_text(
            "[]",
            encoding="utf-8",
        )

    cache_root = tmp_path / "preprocess"
    chunks_path = tmp_path / "chunks.jsonl"
    failures_path = tmp_path / "failures.jsonl"
    process_config = {"name": "fake", "params": {"version": 1}}
    first_preprocessor = _FakeCheckpointPreprocessor(source_dir, fail_ids={"p2"})
    first_cache = PreprocessCache(
        cache_root,
        process_config=process_config,
        source_module_path=Path(__file__),
    )

    first = preprocess_selected_papers(
        preprocessor=first_preprocessor,
        selected_papers=papers,
        cache=first_cache,
        chunks_path=chunks_path,
        failures_path=failures_path,
        resume=False,
    )

    assert first_preprocessor.calls == ["p1", "p2", "p3"]
    assert first.processed_count == 2
    assert first.reused_count == 0
    assert [failure["paper_id"] for failure in first.failures] == ["p2"]
    assert first.merge_result is None
    assert not chunks_path.exists()
    assert json.loads(failures_path.read_text(encoding="utf-8"))["paper_id"] == "p2"

    second_preprocessor = _FakeCheckpointPreprocessor(source_dir)
    second_cache = PreprocessCache(
        cache_root,
        process_config=process_config,
        source_module_path=Path(__file__),
    )
    second = preprocess_selected_papers(
        preprocessor=second_preprocessor,
        selected_papers=papers,
        cache=second_cache,
        chunks_path=chunks_path,
        failures_path=failures_path,
        resume=True,
    )

    assert second_preprocessor.calls == ["p2"]
    assert second.processed_count == 1
    assert second.reused_count == 2
    assert second.failures == ()
    assert second.merge_result is not None
    assert second.merge_result.paper_count == 3
    assert [chunk.paper_id for chunk in iter_chunks(chunks_path)] == [
        "p1",
        "p2",
        "p3",
    ]
    assert failures_path.read_text(encoding="utf-8") == ""


def test_shared_preprocess_cache_reuses_papers_across_artifact_roots(
    tmp_path,
) -> None:
    papers = [{"paper_id": paper_id} for paper_id in ("p1", "p2", "p3")]
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for paper in papers:
        (source_dir / f"{paper['paper_id']}.json").write_text(
            "[]",
            encoding="utf-8",
        )

    shared_cache_root = tmp_path / "shared-cache"
    process_config = {"name": "fake", "params": {"version": 1}}
    first_artifact = tmp_path / "artifacts-c3000"
    first_preprocessor = _FakeCheckpointPreprocessor(source_dir)
    first = preprocess_selected_papers(
        preprocessor=first_preprocessor,
        selected_papers=papers[:2],
        cache=PreprocessCache(
            shared_cache_root,
            process_config=process_config,
            source_module_path=Path(__file__),
        ),
        chunks_path=first_artifact / "chunks.jsonl",
        failures_path=first_artifact / "failures.jsonl",
        resume=False,
    )

    second_artifact = tmp_path / "artifacts-c5000"
    second_preprocessor = _FakeCheckpointPreprocessor(source_dir)
    second = preprocess_selected_papers(
        preprocessor=second_preprocessor,
        selected_papers=papers,
        cache=PreprocessCache(
            shared_cache_root,
            process_config=process_config,
            source_module_path=Path(__file__),
        ),
        chunks_path=second_artifact / "chunks.jsonl",
        failures_path=second_artifact / "failures.jsonl",
        resume=True,
    )

    assert first.processed_count == 2
    assert second_preprocessor.calls == ["p3"]
    assert second.reused_count == 2
    assert second.processed_count == 1
    assert [chunk.paper_id for chunk in iter_chunks(second_artifact / "chunks.jsonl")] == [
        "p1",
        "p2",
        "p3",
    ]
    assert [chunk.paper_id for chunk in iter_chunks(first_artifact / "chunks.jsonl")] == [
        "p1",
        "p2",
    ]


class _FakeCheckpointIndexer:
    def __init__(self, name: str, index_dir: Path, *, fail_build: bool = False):
        self.name = name
        self.index_dir = index_dir
        self.fail_build = fail_build
        self.build_calls = 0
        self.load_calls = 0

    def build(self, chunks) -> None:
        self.build_calls += 1
        list(chunks)
        if self.fail_build:
            raise RuntimeError(f"{self.name} failed")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        (self.index_dir / "complete").write_text("ok", encoding="utf-8")

    def load(self) -> None:
        self.load_calls += 1
        if not (self.index_dir / "complete").is_file():
            raise FileNotFoundError(self.index_dir / "complete")


def _fake_indexer_config(indexer: _FakeCheckpointIndexer) -> dict:
    return {
        "name": indexer.name,
        "params": {"index_dir": str(indexer.index_dir), "option": "stable"},
    }


def test_index_build_passes_exact_checkpoint_signature_when_supported(
    tmp_path,
) -> None:
    class SignatureAwareIndexer(_FakeCheckpointIndexer):
        def __init__(self, name: str, index_dir: Path):
            super().__init__(name, index_dir)
            self.received_signatures: list[str] = []

        def build_with_signature(self, chunks, build_signature: str) -> None:
            self.received_signatures.append(build_signature)
            self.build(chunks)

    chunk = Chunk(
        chunk_id="p1#c0000",
        paper_id="p1",
        text="text",
        chunk_type="text_span",
        metadata={},
    )
    payload = json.dumps(chunk.to_dict(), sort_keys=True).encode() + b"\n"
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(payload)
    chunks = MergeResult(
        paper_count=1,
        chunk_count=1,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    state_path = tmp_path / "index_build_state.json"
    indexer = SignatureAwareIndexer("signed", tmp_path / "signed")

    run = build_indexers_with_resume(
        indexers=[indexer],
        indexer_configs=[_fake_indexer_config(indexer)],
        chunks_path=chunks_path,
        chunks=chunks,
        state_path=state_path,
        resume=False,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    recorded_signature = state["indexers"]["000:signed"]["signature"]
    assert run.built_count == 1
    assert indexer.build_calls == 1
    assert indexer.received_signatures == [recorded_signature]
    assert len(recorded_signature) == 64
    assert set(recorded_signature) <= set("0123456789abcdef")


def test_index_build_keeps_ordinary_build_interface(tmp_path) -> None:
    chunk = Chunk(
        chunk_id="p1#c0000",
        paper_id="p1",
        text="text",
        chunk_type="text_span",
        metadata={},
    )
    payload = json.dumps(chunk.to_dict(), sort_keys=True).encode() + b"\n"
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(payload)
    chunks = MergeResult(
        paper_count=1,
        chunk_count=1,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    indexer = _FakeCheckpointIndexer("ordinary", tmp_path / "ordinary")

    run = build_indexers_with_resume(
        indexers=[indexer],
        indexer_configs=[_fake_indexer_config(indexer)],
        chunks_path=chunks_path,
        chunks=chunks,
        state_path=tmp_path / "index_build_state.json",
        resume=False,
    )

    assert run.built_count == 1
    assert indexer.build_calls == 1


def test_index_checkpoints_resume_after_failure_and_invalidate_on_chunk_change(
    tmp_path,
) -> None:
    chunk = Chunk(
        chunk_id="p1#c0000",
        paper_id="p1",
        text="text",
        chunk_type="text_span",
        metadata={},
    )
    payload = json.dumps(chunk.to_dict(), sort_keys=True).encode() + b"\n"
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(payload)
    chunks = MergeResult(
        paper_count=1,
        chunk_count=1,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    state_path = tmp_path / "index_build_state.json"

    first = [
        _FakeCheckpointIndexer("first", tmp_path / "first"),
        _FakeCheckpointIndexer("second", tmp_path / "second", fail_build=True),
        _FakeCheckpointIndexer("third", tmp_path / "third"),
    ]
    configs = [_fake_indexer_config(indexer) for indexer in first]
    with pytest.raises(RuntimeError, match="second failed"):
        build_indexers_with_resume(
            indexers=first,
            indexer_configs=configs,
            chunks_path=chunks_path,
            chunks=chunks,
            state_path=state_path,
            resume=False,
        )

    assert [indexer.build_calls for indexer in first] == [1, 1, 0]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["indexers"]["000:first"]["status"] == "complete"
    assert state["indexers"]["001:second"]["status"] == "failed"

    resumed = [
        _FakeCheckpointIndexer("first", tmp_path / "first"),
        _FakeCheckpointIndexer("second", tmp_path / "second"),
        _FakeCheckpointIndexer("third", tmp_path / "third"),
    ]
    resumed_run = build_indexers_with_resume(
        indexers=resumed,
        indexer_configs=configs,
        chunks_path=chunks_path,
        chunks=chunks,
        state_path=state_path,
        resume=True,
    )

    assert resumed_run.loaded_count == 1
    assert resumed_run.built_count == 2
    assert [indexer.load_calls for indexer in resumed] == [1, 0, 0]
    assert [indexer.build_calls for indexer in resumed] == [0, 1, 1]

    (tmp_path / "second" / "complete").unlink()
    load_failed = [
        _FakeCheckpointIndexer("first", tmp_path / "first"),
        _FakeCheckpointIndexer("second", tmp_path / "second"),
        _FakeCheckpointIndexer("third", tmp_path / "third"),
    ]
    load_failed_run = build_indexers_with_resume(
        indexers=load_failed,
        indexer_configs=configs,
        chunks_path=chunks_path,
        chunks=chunks,
        state_path=state_path,
        resume=True,
    )

    assert load_failed_run.loaded_count == 1
    assert load_failed_run.built_count == 2
    assert [indexer.load_calls for indexer in load_failed] == [1, 1, 0]
    assert [indexer.build_calls for indexer in load_failed] == [0, 1, 1]

    changed_chunks = MergeResult(
        paper_count=1,
        chunk_count=1,
        byte_count=len(payload),
        sha256="changed-input-sha256",
    )
    invalidated = [
        _FakeCheckpointIndexer("first", tmp_path / "first"),
        _FakeCheckpointIndexer("second", tmp_path / "second"),
        _FakeCheckpointIndexer("third", tmp_path / "third"),
    ]
    invalidated_run = build_indexers_with_resume(
        indexers=invalidated,
        indexer_configs=configs,
        chunks_path=chunks_path,
        chunks=changed_chunks,
        state_path=state_path,
        resume=True,
    )

    assert invalidated_run.loaded_count == 0
    assert invalidated_run.built_count == 3
    assert [indexer.load_calls for indexer in invalidated] == [0, 0, 0]
    assert [indexer.build_calls for indexer in invalidated] == [1, 1, 1]


def test_index_checkpoint_invalidates_on_declared_dependency_change(
    tmp_path,
) -> None:
    chunk = Chunk(
        chunk_id="p1#c0000",
        paper_id="p1",
        text="text",
        chunk_type="text_span",
        metadata={},
    )
    payload = json.dumps(chunk.to_dict(), sort_keys=True).encode() + b"\n"
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(payload)
    chunks = MergeResult(
        paper_count=1,
        chunk_count=1,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    state_path = tmp_path / "index_build_state.json"
    dependency = tmp_path / "method_aliases.py"
    dependency.write_text("VERSION = 1\n", encoding="utf-8")

    first = _FakeCheckpointIndexer("paper", tmp_path / "paper")
    first.checkpoint_dependencies = (dependency,)
    config = _fake_indexer_config(first)
    build_indexers_with_resume(
        indexers=[first],
        indexer_configs=[config],
        chunks_path=chunks_path,
        chunks=chunks,
        state_path=state_path,
        resume=False,
    )

    same = _FakeCheckpointIndexer("paper", tmp_path / "paper")
    same.checkpoint_dependencies = (dependency,)
    same_run = build_indexers_with_resume(
        indexers=[same],
        indexer_configs=[config],
        chunks_path=chunks_path,
        chunks=chunks,
        state_path=state_path,
        resume=True,
    )
    assert same_run.loaded_count == 1
    assert same.build_calls == 0

    dependency.write_text("VERSION = 2\n", encoding="utf-8")
    changed = _FakeCheckpointIndexer("paper", tmp_path / "paper")
    changed.checkpoint_dependencies = (dependency,)
    changed_run = build_indexers_with_resume(
        indexers=[changed],
        indexer_configs=[config],
        chunks_path=chunks_path,
        chunks=chunks,
        state_path=state_path,
        resume=True,
    )
    assert changed_run.loaded_count == 0
    assert changed.build_calls == 1


def test_build_outputs_must_stay_outside_shared_input(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    shared_root = tmp_path / "shared"

    validate_write_paths(
        [artifact_root / "chunks" / "chunks.jsonl"],
        artifact_root,
        shared_root,
    )
    with pytest.raises(ValueError, match="read-only input"):
        validate_write_paths(
            [shared_root / "chunks.jsonl"],
            artifact_root,
            shared_root,
        )
