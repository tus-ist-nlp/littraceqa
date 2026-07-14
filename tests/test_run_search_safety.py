"""Tests for bounded preprocessing and production-compatible query loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_search
from litqa.contracts import Chunk, Query
from litqa.config import compose_config
from run_search import (
    _SAFE_PAPER_LIMIT,
    _build_argument_parser,
    _default_read_only_roots,
    build_paper_input_fingerprint,
    build_process_fingerprint,
    combine_read_only_roots,
    load_chunks,
    load_queries,
    merge_selected_shards,
    preprocess_selected_papers,
    process_worker_limit,
    read_requested_paper_ids,
    require_bounded_artifact_root,
    select_papers,
    validate_artifact_root,
    validate_artifact_path,
    validate_bounded_index_dirs,
    write_predictions_atomic,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class _FakePreprocessor:
    def __init__(self, failing_ids: set[str] | None = None):
        self.failing_ids = failing_ids or set()
        self.calls: list[str] = []

    def process(self, paper: dict) -> list[Chunk]:
        paper_id = paper["paper_id"]
        self.calls.append(paper_id)
        if paper_id in self.failing_ids:
            raise RuntimeError("synthetic failure")
        return [
            Chunk(
                chunk_id=f"{paper_id}#c0000",
                paper_id=paper_id,
                text=f"text for {paper_id}",
                chunk_type="text_span",
                metadata={"page": 1},
            )
        ]


class _FileBackedFakePreprocessor(_FakePreprocessor):
    def __init__(self, source: Path):
        super().__init__()
        self.source = source

    def input_paths(self, paper: dict) -> list[Path]:
        return [self.source]


def test_select_papers_preserves_requested_order_and_reports_missing(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        metadata,
        [{"paper_id": "p1"}, {"paper_id": "p2"}, {"paper_id": "p3"}],
    )

    selected, missing, rejected = select_papers(
        metadata, ["p3", "missing", "p1"], limit=3
    )

    assert [paper["paper_id"] for paper in selected] == ["p3", "p1"]
    assert missing == ["missing"]
    assert rejected == []


def test_select_papers_rejects_more_explicit_ids_than_limit(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(metadata, [{"paper_id": "p1"}])
    with pytest.raises(ValueError, match="--limit"):
        select_papers(metadata, ["p1", "p2"], limit=1)


def test_limit_selection_records_invalid_and_duplicate_paper_ids(tmp_path: Path):
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(
        metadata,
        [
            {"paper_id": "p1"},
            {"paper_id": "p1"},
            {"paper_id": "../unsafe"},
            {"paper_id": 7},
            {"paper_id": "outside-limit"},
        ],
    )

    selected, missing, rejected = select_papers(metadata, [], limit=4)

    assert [paper["paper_id"] for paper in selected] == ["p1"]
    assert missing == []
    assert [record["error_type"] for record in rejected] == [
        "DuplicatePaperMetadata",
        "InvalidPaperMetadata",
        "InvalidPaperMetadata",
    ]
    assert rejected[0]["paper_id"] == "p1"
    assert rejected[-1]["paper_id"] == "7"
    assert all(record["status"] == "failed" for record in rejected)


def test_requested_paper_ids_reject_duplicates_before_selection(tmp_path: Path):
    ids_file = tmp_path / "paper_ids.txt"
    ids_file.write_text("p2\np1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate requested paper_id"):
        read_requested_paper_ids(["p1"], ids_file)


def test_artifact_root_must_not_overlap_read_only_input(tmp_path: Path):
    source = tmp_path / "shared" / "pdfs" / "mineru"
    source.mkdir(parents=True)

    with pytest.raises(ValueError, match="overlaps"):
        validate_artifact_root(source / "output", source, [tmp_path / "shared"])

    safe = validate_artifact_root(tmp_path / "user-output", source, [tmp_path / "shared"])
    assert safe == (tmp_path / "user-output").resolve()


def test_artifact_path_rejects_symlinked_subtree(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifact_root.mkdir()
    outside.mkdir()
    (artifact_root / "images").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_artifact_path(
            artifact_root / "images" / "mineru",
            artifact_root,
        )

    assert validate_artifact_path(
        artifact_root / "chunks" / "mineru",
        artifact_root,
    ) == artifact_root / "chunks" / "mineru"


def test_per_paper_output_rejects_process_subdirectory_symlink(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    (artifact_root / "chunks").mkdir(parents=True)
    outside.mkdir()
    (artifact_root / "chunks" / "mineru").symlink_to(
        outside, target_is_directory=True
    )
    preprocessor = _FakePreprocessor()

    result = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=artifact_root,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=artifact_root / "failures.jsonl",
        process_fingerprint="fingerprint-v1",
    )

    assert result[0]["status"] == "failed"
    assert result[0]["error_type"] == "ValueError"
    assert "symlink" in result[0]["error"]
    assert preprocessor.calls == []
    assert list(outside.iterdir()) == []


def test_bounded_process_requires_artifact_root_before_backend_construction():
    with pytest.raises(ValueError, match="both build and search"):
        require_bounded_artifact_root("mineru", None)

    root = Path("artifacts")
    assert require_bounded_artifact_root("mineru", root) == root


@pytest.mark.parametrize("build", [False, True], ids=["search", "build"])
def test_bounded_main_requires_artifact_root_for_build_and_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, build: bool
):
    source = tmp_path / "shared" / "pdfs" / "mineru"
    source.mkdir(parents=True)
    config_values = {
        "paths.yaml": {
            "mineru_root": str(source),
            "chunks_dir": str(tmp_path / "shared-index" / "chunks"),
            "index_dir": str(tmp_path / "shared-index" / "index"),
        },
        "process.yaml": {
            "name": "mineru",
            "source": "mineru",
            "bounded_build": True,
            "max_workers": 1,
            "path_key": "mineru_root",
            "params": {},
        },
        "search.yaml": {},
        "agent.yaml": {},
    }
    for filename, value in config_values.items():
        (tmp_path / filename).write_text(yaml.safe_dump(value), encoding="utf-8")

    argv = [
        "run_search.py",
        "--paths",
        str(tmp_path / "paths.yaml"),
        "--process",
        str(tmp_path / "process.yaml"),
        "--search",
        str(tmp_path / "search.yaml"),
        "--agent",
        str(tmp_path / "agent.yaml"),
        "--queries",
        str(tmp_path / "queries.jsonl"),
        "--output",
        str(tmp_path / "predictions.jsonl"),
    ]
    if build:
        argv.extend(["--build", "--limit", "1"])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit, match="2"):
        run_search.main()

    assert (
        "requires --artifact-root for both build and search"
        in capsys.readouterr().err
    )


def test_bounded_indexes_must_stay_under_artifact_root(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    safe_cfg = {
        "retriever": {
            "indexers": [
                {
                    "name": "bm25s",
                    "params": {
                        "index_dir": str(artifact_root / "index" / "bm25s")
                    },
                }
            ]
        }
    }
    validate_bounded_index_dirs(safe_cfg, artifact_root)

    unsafe_cfg = {
        "retriever": {
            "indexers": [
                {
                    "name": "bm25s",
                    "params": {"index_dir": str(tmp_path / "outside")},
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="outside artifact root"):
        validate_bounded_index_dirs(unsafe_cfg, artifact_root)


def test_shared_pdf_source_protects_the_owner_root_by_default(tmp_path: Path):
    source = tmp_path / "shared" / "pdfs"
    source.mkdir(parents=True)

    protected = _default_read_only_roots(source)

    assert protected == [tmp_path / "shared"]
    with pytest.raises(ValueError, match="overlaps"):
        validate_artifact_root(tmp_path / "shared" / "output", source, protected)


def test_nested_pdf_source_protects_the_owner_root_by_default(tmp_path: Path):
    source = tmp_path / "shared" / "pdfs" / "pdfs"
    source.mkdir(parents=True)

    assert _default_read_only_roots(source) == [tmp_path / "shared"]


def test_shared_mineru_source_protects_the_owner_root_by_default(tmp_path: Path):
    source = tmp_path / "shared" / "pdfs" / "mineru"
    source.mkdir(parents=True)

    assert _default_read_only_roots(source) == [tmp_path / "shared"]


def test_explicit_read_only_root_adds_to_automatic_owner_protection(
    tmp_path: Path,
):
    source = tmp_path / "shared" / "pdfs" / "pdfs"
    explicit = tmp_path / "another-shared-root"
    source.mkdir(parents=True)
    explicit.mkdir()

    protected = combine_read_only_roots(source, [explicit])

    assert protected == [tmp_path / "shared", explicit.resolve()]
    with pytest.raises(ValueError, match="overlaps"):
        validate_artifact_root(
            tmp_path / "shared" / "output", source, protected
        )


def test_bounded_cli_defaults_to_one_worker_and_one_item_batch():
    args = _build_argument_parser().parse_args(
        [
            "--paths",
            "paths.yaml",
            "--process",
            "process.yaml",
            "--search",
            "search.yaml",
            "--agent",
            "agent.yaml",
            "--queries",
            "queries.jsonl",
            "--output",
            "predictions.jsonl",
        ]
    )

    assert args.workers == 1
    assert args.batch_size == 1
    assert args.resume is True
    assert _SAFE_PAPER_LIMIT == 200


@pytest.mark.parametrize(
    "filename",
    [
        "mineru.yaml",
        "mineru_v2.yaml",
    ],
)
def test_process_styles_use_bounded_single_worker_builds(filename: str):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "process_style"
        / filename
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["bounded_build"] is True
    assert config["max_workers"] == 1


def test_process_worker_limit_can_only_narrow_the_global_cap():
    assert process_worker_limit({}) == 8
    assert process_worker_limit({"max_workers": 2}) == 2
    assert process_worker_limit({"max_workers": 99}) == 8
    with pytest.raises(ValueError, match="positive integer"):
        process_worker_limit({"max_workers": 0})


def test_process_fingerprint_is_deterministic_and_config_sensitive():
    first = build_process_fingerprint(
        {"name": "mineru", "params": {"content_version": "v1"}}
    )
    repeated = build_process_fingerprint(
        {"params": {"content_version": "v1"}, "name": "mineru"}
    )
    changed = build_process_fingerprint(
        {"name": "mineru", "params": {"content_version": "v2"}}
    )

    assert first == repeated
    assert first != changed


def test_process_fingerprint_changes_with_resolved_input_root():
    first = build_process_fingerprint(
        {"name": "pypdf", "params": {"pdf_dir": "/shared/pdfs-v1"}}
    )
    changed = build_process_fingerprint(
        {"name": "pypdf", "params": {"pdf_dir": "/shared/pdfs-v2"}}
    )

    assert first != changed


def test_composed_preprocessor_fingerprint_changes_with_paths_root():
    process = {"name": "pypdf", "params": {"max_chars_per_chunk": 2000}}
    search = {
        "per_index_k": 1,
        "indexers": [{"name": "bm25s", "params": {}}],
        "fuser": {"name": "rrf", "params": {}},
        "reranker": {"name": "none", "params": {}},
    }
    agent = {"name": "simple", "params": {}}
    first = compose_config(
        paths={
            "pdf_dir": "/shared/pdfs-v1",
            "chunks_dir": "/tmp/chunks",
            "index_dir": "/tmp/index",
        },
        process=process,
        search=search,
        agent=agent,
    )
    changed = compose_config(
        paths={
            "pdf_dir": "/shared/pdfs-v2",
            "chunks_dir": "/tmp/chunks",
            "index_dir": "/tmp/index",
        },
        process=process,
        search=search,
        agent=agent,
    )

    assert build_process_fingerprint(first["preprocessor"]) != (
        build_process_fingerprint(changed["preprocessor"])
    )


def test_composed_mineru_fingerprint_changes_with_paths_root():
    process = {
        "name": "mineru",
        "path_key": "mineru_root",
        "params": {"content_version": "v1"},
    }
    search = {
        "per_index_k": 1,
        "indexers": [{"name": "bm25s", "params": {}}],
        "fuser": {"name": "rrf", "params": {}},
        "reranker": {"name": "none", "params": {}},
    }
    agent = {"name": "simple", "params": {}}
    common_paths = {"chunks_dir": "/tmp/chunks", "index_dir": "/tmp/index"}
    first = compose_config(
        paths={**common_paths, "mineru_root": "/shared/mineru-v1"},
        process=process,
        search=search,
        agent=agent,
    )
    changed = compose_config(
        paths={**common_paths, "mineru_root": "/shared/mineru-v2"},
        process=process,
        search=search,
        agent=agent,
    )

    assert build_process_fingerprint(first["preprocessor"]) != (
        build_process_fingerprint(changed["preprocessor"])
    )


def test_paper_input_fingerprint_tracks_metadata_and_source_bytes(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text('{"version": 1}', encoding="utf-8")
    preprocessor = _FileBackedFakePreprocessor(source)
    paper = {"paper_id": "p1", "title": "First title"}

    initial = build_paper_input_fingerprint(paper, preprocessor)
    repeated = build_paper_input_fingerprint(dict(paper), preprocessor)
    metadata_changed = build_paper_input_fingerprint(
        {**paper, "title": "Changed title"}, preprocessor
    )
    source.write_text('{"version": 2}', encoding="utf-8")
    source_changed = build_paper_input_fingerprint(paper, preprocessor)

    assert initial == repeated
    assert initial != metadata_changed
    assert initial != source_changed


def test_resume_reprocesses_when_source_artifact_changes(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text('{"version": 1}', encoding="utf-8")
    failures = tmp_path / "failures.jsonl"
    preprocessor = _FileBackedFakePreprocessor(source)
    kwargs = {
        "papers": [{"paper_id": "p1", "title": "Title"}],
        "preprocessor": preprocessor,
        "artifact_root": tmp_path,
        "process_name": "mineru",
        "workers": 1,
        "batch_size": 1,
        "resume": True,
        "failures_path": failures,
        "process_fingerprint": "fingerprint-v1",
    }

    first = preprocess_selected_papers(**kwargs)
    repeated = preprocess_selected_papers(**kwargs)
    source.write_text('{"version": 2}', encoding="utf-8")
    changed = preprocess_selected_papers(**kwargs)

    assert first[0]["status"] == "success"
    assert repeated[0]["status"] == "skipped"
    assert changed[0]["status"] == "success"
    assert first[0]["input_fingerprint"] != changed[0]["input_fingerprint"]
    assert preprocessor.calls == ["p1", "p1"]


def test_failure_is_recorded_without_stopping_other_papers_and_resume_skips_success(
    tmp_path: Path,
):
    papers = [{"paper_id": "ok"}, {"paper_id": "bad"}, {"paper_id": "later"}]
    failures = tmp_path / "failures.jsonl"
    preprocessor = _FakePreprocessor({"bad"})

    first = preprocess_selected_papers(
        papers=papers,
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v1",
    )

    assert [result["status"] for result in first] == ["success", "failed", "success"]
    assert [json.loads(line)["paper_id"] for line in failures.read_text().splitlines()] == [
        "bad"
    ]

    second = preprocess_selected_papers(
        papers=papers,
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v1",
    )

    assert [result["status"] for result in second] == ["skipped", "failed", "skipped"]
    assert preprocessor.calls == ["ok", "bad", "later", "bad"]

    changed = preprocess_selected_papers(
        papers=papers,
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v2",
    )

    assert [result["status"] for result in changed] == [
        "success",
        "failed",
        "success",
    ]
    assert preprocessor.calls == [
        "ok",
        "bad",
        "later",
        "bad",
        "ok",
        "bad",
        "later",
    ]


def test_resume_reprocesses_structurally_invalid_shard(tmp_path: Path):
    failures = tmp_path / "failures.jsonl"
    preprocessor = _FakePreprocessor()
    first = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v1",
    )
    shard = Path(first[0]["shard"])
    record = json.loads(shard.read_text(encoding="utf-8"))
    record["metadata"] = None
    _write_jsonl(shard, [record])

    repeated = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v1",
    )

    assert repeated[0]["status"] == "success"
    assert preprocessor.calls == ["p1", "p1"]


def test_resume_reprocesses_valid_shard_with_checksum_mismatch(tmp_path: Path):
    failures = tmp_path / "failures.jsonl"
    preprocessor = _FakePreprocessor()
    first = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v1",
    )
    shard = Path(first[0]["shard"])
    record = json.loads(shard.read_text(encoding="utf-8"))
    record["text"] = "tampered but structurally valid"
    _write_jsonl(shard, [record])

    repeated = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v1",
    )

    assert repeated[0]["status"] == "success"
    assert preprocessor.calls == ["p1", "p1"]


def test_failed_new_fingerprint_never_reuses_old_shard(tmp_path: Path):
    failures = tmp_path / "failures.jsonl"
    preprocessor = _FakePreprocessor()
    first = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v1",
    )
    assert first[0]["status"] == "success"

    preprocessor.failing_ids.add("p1")
    failed = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v2",
    )
    assert failed[0]["status"] == "failed"

    preprocessor.failing_ids.clear()
    retried = preprocess_selected_papers(
        papers=[{"paper_id": "p1"}],
        preprocessor=preprocessor,
        artifact_root=tmp_path,
        process_name="mineru",
        workers=1,
        batch_size=1,
        resume=True,
        failures_path=failures,
        process_fingerprint="fingerprint-v2",
    )

    assert retried[0]["status"] == "success"
    assert preprocessor.calls == ["p1", "p1", "p1"]


def test_selected_shards_connect_to_common_chunk_input(tmp_path: Path):
    papers = [{"paper_id": "p2"}, {"paper_id": "p1"}]
    results = preprocess_selected_papers(
        papers=papers,
        preprocessor=_FakePreprocessor(),
        artifact_root=tmp_path,
        process_name="mineru",
        workers=2,
        batch_size=2,
        resume=True,
        failures_path=tmp_path / "failures.jsonl",
    )
    merged = tmp_path / "chunks" / "mineru_chunks.jsonl"
    merge_selected_shards(merged, results)

    chunks = load_chunks(merged)

    class FakeIndexer:
        def __init__(self) -> None:
            self.received: list[Chunk] = []

        def build(self, received: list[Chunk]) -> None:
            self.received = received

    indexer = FakeIndexer()
    indexer.build(chunks)

    assert [chunk.paper_id for chunk in chunks] == ["p2", "p1"]
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert indexer.received == chunks


def test_load_queries_accepts_only_production_fields(tmp_path: Path):
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "question": "What is reported?",
                "answer_types": ["table"],
                "table_schema": [{"name": "Paper", "type": "string"}],
                "multiple_choice_options": {"A": "first", "B": "second"},
            }
        ],
    )

    loaded = load_queries(queries)

    assert loaded[0].query_id == "q1"
    assert loaded[0].task_family is None
    assert loaded[0].primary_evidence_type is None
    assert loaded[0].table_schema == [{"name": "Paper", "type": "string"}]
    assert loaded[0].multiple_choice_options == {"A": "first", "B": "second"}


def test_load_queries_ignores_development_only_labels(tmp_path: Path):
    queries = tmp_path / "queries.jsonl"
    _write_jsonl(
        queries,
        [
            {
                "query_id": "q1",
                "question": "What is reported?",
                "answer_types": ["freeform"],
                "table_schema": [],
                "task_family": "multi_paper",
                "primary_evidence_type": "text_span",
            }
        ],
    )

    loaded = load_queries(queries)

    assert loaded[0].task_family is None
    assert loaded[0].primary_evidence_type is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query_id", "", "non-empty query_id"),
        ("query_id", " q1", "surrounding whitespace"),
        ("question", "", "non-empty question"),
        ("question", "What is reported? ", "surrounding whitespace"),
        ("answer_types", "freeform", "list answer_types"),
        ("answer_types", [], "answer_types must not be empty"),
        ("answer_types", [7], r"answer_types\[0\].*non-empty string"),
        ("answer_types", [" freeform"], r"answer_types\[0\].*whitespace"),
        ("table_schema", {}, "table_schema must be a list"),
        ("table_schema", ["Paper"], r"table_schema\[0\].*object"),
        (
            "multiple_choice_options",
            "A or B",
            "multiple_choice_options must be an object or list",
        ),
    ],
)
def test_load_queries_rejects_invalid_production_fields(
    tmp_path: Path, field: str, value: object, message: str
):
    queries = tmp_path / "queries.jsonl"
    record = {
        "query_id": "q1",
        "question": "What is reported?",
        "answer_types": ["freeform"],
        "table_schema": [],
    }
    record[field] = value
    _write_jsonl(queries, [record])

    with pytest.raises(ValueError, match=message):
        load_queries(queries)


def test_load_queries_rejects_duplicate_query_ids(tmp_path: Path):
    queries = tmp_path / "queries.jsonl"
    record = {
        "query_id": "q1",
        "question": "What is reported?",
        "answer_types": ["freeform"],
        "table_schema": [],
    }
    _write_jsonl(queries, [record, {**record, "question": "And next?"}])

    with pytest.raises(ValueError, match="duplicate query_id"):
        load_queries(queries)


class _FakePrediction:
    def __init__(self, query_id: str) -> None:
        self.query_id = query_id

    def to_dict(self) -> dict[str, str]:
        return {"query_id": self.query_id}


class _FakeAgent:
    def __init__(self, failing_query_id: str | None = None) -> None:
        self.failing_query_id = failing_query_id

    def run(self, query: Query) -> _FakePrediction:
        if query.query_id == self.failing_query_id:
            raise RuntimeError("synthetic query failure")
        return _FakePrediction(query.query_id)


def _queries_for_atomic_output() -> list[Query]:
    return [
        Query(query_id="q1", question="First?", answer_types=["freeform"]),
        Query(query_id="q2", question="Second?", answer_types=["freeform"]),
    ]


def test_prediction_output_is_atomically_published_after_success(tmp_path: Path):
    output = tmp_path / "predictions.jsonl"
    output.write_text("old output\n", encoding="utf-8")

    write_predictions_atomic(output, _queries_for_atomic_output(), _FakeAgent())

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records == [{"query_id": "q1"}, {"query_id": "q2"}]
    assert not output.with_suffix(".jsonl.tmp").exists()


def test_prediction_failure_keeps_previous_output_and_removes_tmp(tmp_path: Path):
    output = tmp_path / "predictions.jsonl"
    output.write_text("old output\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="synthetic query failure"):
        write_predictions_atomic(
            output,
            _queries_for_atomic_output(),
            _FakeAgent(failing_query_id="q2"),
        )

    assert output.read_text(encoding="utf-8") == "old output\n"
    assert not output.with_suffix(".jsonl.tmp").exists()
