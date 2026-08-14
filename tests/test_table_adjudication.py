from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import littraceqa.table_adjudication as adjudication
from littraceqa.table_adjudication import (
    AdjudicationError,
    compose_submission,
    create_review,
)


def _write_jsonl(path: Path, records: list[dict], *, compact: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            if compact:
                encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            else:
                encoded = json.dumps(record, ensure_ascii=False)
            handle.write(encoded + "\n")


def _inputs() -> list[dict]:
    return [
        {
            "query_id": "q_mc",
            "benchmark": "LitTraceQA",
            "question": "Choose one.",
            "answer_types": ["multiple_choice"],
            "multiple_choice_options": [
                {"label": "A", "text": "Alpha"},
                {"label": "B", "text": "Beta"},
            ],
        },
        {
            "query_id": "q_table",
            "benchmark": "LitTraceQA",
            "question": "Report the scores.",
            "answer_types": ["table"],
            "table_schema": [
                {"name": "method", "type": "string", "is_row_key": True},
                {"name": "score", "type": "number", "is_row_key": False},
            ],
        },
        {
            "query_id": "q_both",
            "benchmark": "LitTraceQA",
            "question": "Choose and tabulate.",
            "answer_types": ["multiple_choice", "table"],
            "multiple_choice_options": [
                {"label": "A", "text": "Alpha"},
                {"label": "B", "text": "Beta"},
            ],
            "table_schema": [
                {"name": "setting", "type": "string", "is_row_key": True},
                {"name": "value", "type": "string", "is_row_key": False},
            ],
        },
    ]


def _paper(paper_id: str) -> list[dict]:
    return [{"paper_id": paper_id}]


def _evidence(paper_id: str) -> list[dict]:
    return [
        {
            "paper_id": paper_id,
            "source_type": "table",
            "locator": {"page": 2, "table_id": "Table 1"},
        }
    ]


def _base() -> list[dict]:
    return [
        {
            "query_id": "q_mc",
            "gold_papers": _paper("p_mc"),
            "evidence": _evidence("p_mc"),
            "answer": {"multiple_choice": {"gold": "A"}},
        },
        {
            "query_id": "q_table",
            "gold_papers": _paper("p_table"),
            "evidence": _evidence("p_table"),
            "answer": {
                "table": {"rows": [{"method": "Base", "score": 1.0}]}
            },
        },
        {
            "query_id": "q_both",
            "gold_papers": _paper("p_both"),
            "evidence": _evidence("p_both"),
            "answer": {
                "multiple_choice": {"gold": "B"},
                "table": {"rows": [{"setting": "Base", "value": "old"}]},
            },
        },
    ]


def _full_candidate() -> list[dict]:
    candidate = copy.deepcopy(_base())
    candidate[0]["answer"]["multiple_choice"]["gold"] = "B"
    candidate[1]["gold_papers"] = _paper("candidate_paper")
    candidate[1]["evidence"] = _evidence("candidate_paper")
    candidate[1]["answer"]["table"] = {
        "rows": [{"method": "Candidate", "score": 2.0}]
    }
    candidate[2]["answer"]["multiple_choice"]["gold"] = "A"
    candidate[2]["answer"]["table"] = {
        "rows": [{"setting": "Candidate", "value": "new"}]
    }
    return candidate


def _table_only_candidate() -> list[dict]:
    return [
        {
            "query_id": "q_table",
            "answer": {
                "table": {"rows": [{"method": "Alt", "score": 3.0}]}
            },
        },
        {
            "query_id": "q_both",
            "answer": {
                "table": {"rows": [{"setting": "Alt", "value": "third"}]}
            },
        },
    ]


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    inputs = tmp_path / "inputs.jsonl"
    base = tmp_path / "base.jsonl"
    full = tmp_path / "full.jsonl"
    table_only = tmp_path / "table_only.jsonl"
    _write_jsonl(inputs, _inputs(), compact=True)
    _write_jsonl(base, _base(), compact=False)
    _write_jsonl(full, _full_candidate(), compact=True)
    _write_jsonl(table_only, _table_only_candidate(), compact=True)
    return inputs, base, full, table_only


def _complete_decisions(template_path: Path) -> Path:
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    selections = {"q_table": "full", "q_both": "table_only"}
    papers = {"q_table": "p_table", "q_both": "p_both"}
    for decision in payload["decisions"]:
        query_id = decision["query_id"]
        decision["selected_candidate"] = selections[query_id]
        decision["source_checked"] = True
        decision["notes"] = "Checked against the cited source table."
        decision["locator"] = [
            {
                "paper_id": papers[query_id],
                "source_type": "table",
                "locator": {"page": 2, "table_id": "Table 1"},
            }
        ]
    path = template_path.with_name("decisions.json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_review_supports_full_and_table_only_candidates(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)

    result = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )

    review = json.loads(Path(result["review_json"]).read_text(encoding="utf-8"))
    template = json.loads(
        Path(result["decision_template"]).read_text(encoding="utf-8")
    )
    assert review["table_query_count"] == 2
    assert [item["coverage"] for item in review["candidates"]] == [
        "full",
        "table_only",
    ]
    assert [item["query_id"] for item in review["queries"]] == [
        "q_table",
        "q_both",
    ]
    assert review["queries"][0]["frozen_gold_papers"] == _paper("p_table")
    assert review["queries"][0]["frozen_evidence"] == _evidence("p_table")
    assert all(not item["source_checked"] for item in template["decisions"])
    markdown = Path(result["review_markdown"]).read_text(encoding="utf-8")
    assert "q_table" in markdown
    assert "Frozen papers and evidence" in markdown
    assert '"table_id": "Table 1"' in markdown


def test_review_refuses_to_overwrite_generated_artifacts(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    kwargs = {
        "inputs_path": inputs,
        "base_path": base,
        "candidate_specs": [("full", full), ("table_only", table_only)],
        "output_dir": tmp_path / "review",
    }
    create_review(**kwargs)

    with pytest.raises(AdjudicationError, match="already exists"):
        create_review(**kwargs)


def test_review_interrupt_after_link_rolls_back_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    output_dir = tmp_path / "interrupted_review"
    original_publish = adjudication._publish_staged_new
    interrupted = False

    def publish_then_interrupt(
        temporary: Path, path: Path
    ) -> tuple[int, int]:
        nonlocal interrupted
        identity = original_publish(temporary, path)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected review interrupt")
        return identity

    monkeypatch.setattr(
        adjudication, "_publish_staged_new", publish_then_interrupt
    )
    with pytest.raises(KeyboardInterrupt, match="injected review interrupt"):
        create_review(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            output_dir=output_dir,
        )

    assert list(output_dir.glob("table_*")) == []


def test_review_markdown_contains_untrusted_cells_without_breaking_markup(
    tmp_path: Path,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    records = _table_only_candidate()
    records[0]["answer"]["table"]["rows"][0]["method"] = "Alt ``` | </code>"
    _write_jsonl(table_only, records)

    result = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "safe_markdown",
    )

    markdown = Path(result["review_markdown"]).read_text(encoding="utf-8")
    assert "````json" in markdown
    assert "&#124;" in markdown
    assert "&lt;/code&gt;" in markdown


def test_review_markdown_escapes_untrusted_query_metadata(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    input_records = _inputs()
    injected_id = "q_table\n\n### forged candidate"
    injected_question = "Question text\n\n### forged candidate\n\n```json"
    input_records[1]["query_id"] = injected_id
    input_records[1]["question"] = injected_question
    base_records = _base()
    base_records[1]["query_id"] = injected_id
    full_records = _full_candidate()
    full_records[1]["query_id"] = injected_id
    table_records = _table_only_candidate()
    table_records[0]["query_id"] = injected_id
    _write_jsonl(inputs, input_records, compact=True)
    _write_jsonl(base, base_records, compact=True)
    _write_jsonl(full, full_records, compact=True)
    _write_jsonl(table_only, table_records, compact=True)

    result = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "escaped_metadata",
    )

    markdown = Path(result["review_markdown"]).read_text(encoding="utf-8")
    assert "## Query <code>&quot;q_table\\n\\n### forged candidate&quot;</code>" in markdown
    assert "\n### forged candidate\n" not in markdown


def test_compose_changes_only_selected_tables_and_preserves_non_table_bytes(
    tmp_path: Path,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "composed.jsonl"

    result = compose_submission(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        decisions_path=decisions,
        output_path=output,
    )

    base_lines = base.read_bytes().splitlines(keepends=True)
    output_lines = output.read_bytes().splitlines(keepends=True)
    assert output_lines[0] == base_lines[0]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    base_records = _base()
    assert records[1]["answer"]["table"]["rows"][0]["method"] == "Candidate"
    assert records[2]["answer"]["table"]["rows"][0]["setting"] == "Alt"
    assert records[2]["answer"]["multiple_choice"] == {"gold": "B"}
    for before, after in zip(base_records, records, strict=True):
        assert after["gold_papers"] == before["gold_papers"]
        assert after["evidence"] == before["evidence"]
    audit = json.loads(Path(result["audit"]).read_text(encoding="utf-8"))
    assert audit["changed_query_ids"] == ["q_table", "q_both"]
    assert all(audit["freeze_checks"].values())


def test_compose_all_base_is_byte_identical(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    for decision in payload["decisions"]:
        decision["selected_candidate"] = "base"
    decisions.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "unchanged.jsonl"

    result = compose_submission(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        decisions_path=decisions,
        output_path=output,
    )

    assert output.read_bytes() == base.read_bytes()
    assert result["changed_query_ids"] == []


def test_compose_rejects_casefold_alias_between_output_and_audit(
    tmp_path: Path,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))

    with pytest.raises(AdjudicationError, match="distinct new artifact"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "Result.JSONL",
            audit_path=tmp_path / "result.jsonl",
        )


def test_compose_rejects_dangling_output_symlink(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "linked.jsonl"
    target = tmp_path / "unexpected.jsonl"
    output.symlink_to(target)

    with pytest.raises(AdjudicationError, match="output already exists"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=output,
        )
    assert not target.exists()


def test_compose_preserves_json_numeric_type_and_audit_hash(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    candidate_records = _full_candidate()
    candidate_records[1]["answer"]["table"] = {
        "rows": [{"method": "Base", "score": 1}]
    }
    _write_jsonl(full, candidate_records, compact=True)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "typed.jsonl"

    result = compose_submission(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        decisions_path=decisions,
        output_path=output,
    )

    written = [json.loads(line) for line in output.read_text().splitlines()]
    score = written[1]["answer"]["table"]["rows"][0]["score"]
    assert score == 1
    assert isinstance(score, int)
    table = written[1]["answer"]["table"]
    expected_hash = hashlib.sha256(
        json.dumps(
            table,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    audit = json.loads(Path(result["audit"]).read_text(encoding="utf-8"))
    q_table_audit = next(
        item for item in audit["decisions"] if item["query_id"] == "q_table"
    )
    assert q_table_audit["after_table_sha256"] == expected_hash


def test_compose_hashes_the_exact_decision_bytes_it_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    original_bytes = decisions.read_bytes()
    original_selected_table = adjudication._selected_table
    mutated = False

    def mutate_after_load(*args: object, **kwargs: object) -> dict:
        nonlocal mutated
        if not mutated:
            decisions.write_text('{"mutated":true}\n', encoding="utf-8")
            mutated = True
        return original_selected_table(*args, **kwargs)

    monkeypatch.setattr(adjudication, "_selected_table", mutate_after_load)
    result = compose_submission(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        decisions_path=decisions,
        output_path=tmp_path / "toctou.jsonl",
    )

    audit = json.loads(Path(result["audit"]).read_text(encoding="utf-8"))
    assert audit["decisions_sha256"] == hashlib.sha256(original_bytes).hexdigest()


def test_compose_publishes_audit_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "ordered.jsonl"
    audit = tmp_path / "ordered.audit.json"
    original_publish = adjudication._publish_staged_new
    published: list[Path] = []

    def observe_publish(temporary: Path, path: Path) -> tuple[int, int]:
        if path == audit:
            assert not output.exists()
        if path == output:
            assert audit.exists()
        identity = original_publish(temporary, path)
        published.append(path)
        return identity

    monkeypatch.setattr(adjudication, "_publish_staged_new", observe_publish)
    result = compose_submission(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        decisions_path=decisions,
        output_path=output,
    )

    assert output.exists()
    assert Path(result["audit"]).exists()
    assert published == [audit, output]


def test_staging_failure_removes_private_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(adjudication.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        adjudication._stage_bytes(tmp_path / "artifact.json", b"sensitive")

    assert list(tmp_path.glob(".artifact.json.*.tmp")) == []


def test_temporary_cleanup_failure_cannot_remove_published_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "cleanup.jsonl"
    original_unlink = Path.unlink
    injected = False

    def fail_output_temp_once(
        path: Path, missing_ok: bool = False
    ) -> None:
        nonlocal injected
        if path.name.startswith(".cleanup.jsonl.") and not injected:
            injected = True
            raise OSError("injected temporary unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_output_temp_once)
    result = compose_submission(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        decisions_path=decisions,
        output_path=output,
    )

    assert injected
    assert output.exists()
    assert Path(result["audit"]).exists()
    for temporary in tmp_path.glob(".cleanup.jsonl.*.tmp"):
        original_unlink(temporary)


def test_output_directory_fsync_failure_cleans_both_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "fsync.jsonl"
    audit = tmp_path / "fsync.audit.json"
    original_fsync_parent = adjudication._fsync_parent_directory
    failed_once = False

    def fail_output_parent_fsync(path: Path) -> None:
        nonlocal failed_once
        if path == output and not failed_once:
            failed_once = True
            raise OSError("injected directory fsync failure")
        original_fsync_parent(path)

    monkeypatch.setattr(
        adjudication, "_fsync_parent_directory", fail_output_parent_fsync
    )
    with pytest.raises(OSError, match="injected directory fsync failure"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=output,
            audit_path=audit,
        )

    assert not output.exists()
    assert not audit.exists()


def test_undurable_output_cleanup_retains_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "undurable.jsonl"
    audit = tmp_path / "undurable.audit.json"
    original_fsync_parent = adjudication._fsync_parent_directory

    def fail_every_output_parent_fsync(path: Path) -> None:
        if path == output:
            raise OSError("injected directory fsync failure")
        original_fsync_parent(path)

    monkeypatch.setattr(
        adjudication, "_fsync_parent_directory", fail_every_output_parent_fsync
    )
    with pytest.raises(AdjudicationError, match="retaining audit"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=output,
            audit_path=audit,
        )

    assert not output.exists()
    assert audit.exists()


def test_failed_output_cleanup_retains_its_published_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "retained.jsonl"
    audit = tmp_path / "retained.audit.json"
    original_fsync_parent = adjudication._fsync_parent_directory
    original_unlink = Path.unlink

    def fail_output_parent_fsync(path: Path) -> None:
        if path == output:
            raise OSError("injected directory fsync failure")
        original_fsync_parent(path)

    def fail_output_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path == output:
            raise OSError("injected output cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        adjudication, "_fsync_parent_directory", fail_output_parent_fsync
    )
    monkeypatch.setattr(Path, "unlink", fail_output_cleanup)
    with pytest.raises(OSError, match="injected output cleanup failure"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=output,
            audit_path=audit,
        )

    assert output.exists()
    assert audit.exists()
    original_unlink(output)
    original_unlink(audit)


def test_interrupt_after_output_link_cleans_output_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    output = tmp_path / "interrupted.jsonl"
    audit = tmp_path / "interrupted.audit.json"
    original_link = adjudication.os.link

    def link_then_interrupt(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(
            source,
            destination,
            follow_symlinks=follow_symlinks,
        )
        if Path(destination) == output:
            raise KeyboardInterrupt("injected post-link interrupt")

    monkeypatch.setattr(adjudication.os, "link", link_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected post-link interrupt"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=output,
            audit_path=audit,
        )

    assert not output.exists()
    assert not audit.exists()


@pytest.mark.parametrize("missing", ["source_checked", "notes", "locator"])
def test_compose_requires_source_check_notes_and_locator(
    tmp_path: Path, missing: str
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    if missing == "source_checked":
        payload["decisions"][0][missing] = False
    elif missing == "notes":
        payload["decisions"][0][missing] = ""
    else:
        payload["decisions"][0][missing] = []
    decisions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AdjudicationError, match=missing):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("query_id", ["q_table"], "query_id must be a non-empty string"),
        ("selected_candidate", {"source": "full"}, "unknown selected_candidate"),
    ],
)
def test_compose_rejects_non_string_decision_identifiers(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    payload["decisions"][0][field] = invalid_value
    decisions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AdjudicationError, match=message):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_compose_rejects_candidate_changed_after_review(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    changed = _table_only_candidate()
    changed[0]["answer"]["table"]["rows"][0]["score"] = 99.0
    _write_jsonl(table_only, changed, compact=True)

    with pytest.raises(AdjudicationError, match="sealed field mismatch: candidates"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_review_rejects_invalid_base_evidence_locator(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    records = _base()
    del records[1]["evidence"][0]["locator"]["table_id"]
    _write_jsonl(base, records)

    with pytest.raises(AdjudicationError, match="incomplete for table"):
        create_review(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            output_dir=tmp_path / "invalid_base_review",
        )


def test_compose_requires_review_locator_from_frozen_evidence(
    tmp_path: Path,
) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    payload["decisions"][0]["locator"][0]["locator"] = {
        "page": 3,
        "table_id": "Table 2",
    }
    decisions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AdjudicationError, match="present in frozen evidence"):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_compose_rejects_tampered_schema_hash_in_decision(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    review = create_review(
        inputs_path=inputs,
        base_path=base,
        candidate_specs=[("full", full), ("table_only", table_only)],
        output_dir=tmp_path / "review",
    )
    decisions = _complete_decisions(Path(review["decision_template"]))
    payload = json.loads(decisions.read_text(encoding="utf-8"))
    payload["table_schema_sha256"] = "0" * 64
    decisions.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        AdjudicationError, match="sealed field mismatch: table_schema_sha256"
    ):
        compose_submission(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            decisions_path=decisions,
            output_path=tmp_path / "no.jsonl",
        )


def test_review_rejects_candidate_order_and_schema_errors(tmp_path: Path) -> None:
    inputs, base, full, table_only = _artifacts(tmp_path)
    reversed_records = list(reversed(_table_only_candidate()))
    _write_jsonl(table_only, reversed_records, compact=True)

    with pytest.raises(AdjudicationError, match="query order"):
        create_review(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            output_dir=tmp_path / "review_order",
        )

    invalid = _table_only_candidate()
    del invalid[0]["answer"]["table"]["rows"][0]["score"]
    _write_jsonl(table_only, invalid, compact=True)
    with pytest.raises(AdjudicationError, match="must contain exactly"):
        create_review(
            inputs_path=inputs,
            base_path=base,
            candidate_specs=[("full", full), ("table_only", table_only)],
            output_dir=tmp_path / "review_schema",
        )
