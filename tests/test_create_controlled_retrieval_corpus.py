"""Tests for deterministic nested controlled retrieval corpora."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.create_controlled_retrieval_corpus import (
    DEFAULT_ORDER_SEED,
    DEFAULT_SELECTION_SEED,
    Paper,
    create_nested_corpora,
    largest_remainder_quotas,
    load_gold_paper_ids,
    mineru_content_list_path,
    preflight_mineru_files,
    select_nested_corpora,
    validate_corpus_size_limits,
    validate_output_root,
)


def _papers() -> list[Paper]:
    return [
        Paper(f"a{index}", "A", 2025)
        for index in range(1, 6)
    ] + [
        Paper(f"b{index}", "B", 2024)
        for index in range(1, 4)
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _create_mineru_files(root: Path, paper_ids: list[str]) -> None:
    for paper_id in paper_ids:
        path = mineru_content_list_path(root, paper_id)
        path.parent.mkdir(parents=True)
        path.write_text("[]\n", encoding="utf-8")


def test_largest_remainder_quotas_have_exact_proportional_size() -> None:
    quotas = largest_remainder_quotas(
        {("A", 2025): 5, ("B", 2024): 3},
        6,
    )

    assert quotas == {("A", 2025): 4, ("B", 2024): 2}
    assert sum(quotas.values()) == 6


def test_nested_selection_is_deterministic_and_contains_all_gold() -> None:
    gold = {"a1", "b1"}

    selected, quotas = select_nested_corpora(
        list(reversed(_papers())),
        gold,
        [4, 6, 8],
    )
    repeated, _ = select_nested_corpora(_papers(), gold, [8, 4, 6])

    assert selected == repeated
    assert set(selected[4]) < set(selected[6]) < set(selected[8])
    assert all(gold <= set(selected[size]) for size in selected)
    assert [sum(quotas[size].values()) for size in (4, 6, 8)] == [4, 6, 8]
    assert selected[4] == ["a5", "a2", "b1", "a1"]
    assert DEFAULT_SELECTION_SEED == "littraceqa-controlled-stratified-v1"
    assert DEFAULT_ORDER_SEED == "littraceqa-controlled-order-v1"


def test_gold_loader_uses_only_gold_paper_ids(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [
            {
                "query_id": "q1",
                "task_family": "irrelevant",
                "primary_evidence_type": "irrelevant",
                "gold_papers": [{"paper_id": "gold-1"}],
                "evidence": [{"paper_id": "must-not-be-selected"}],
            }
        ],
    )

    assert load_gold_paper_ids(validation_path) == {"gold-1"}


def test_mineru_preflight_checks_exact_paths_and_rejects_empty_files(
    tmp_path: Path,
) -> None:
    mineru_root = tmp_path / "mineru"
    _create_mineru_files(mineru_root, ["p1"])
    empty_path = mineru_content_list_path(mineru_root, "p2")
    empty_path.parent.mkdir(parents=True)
    empty_path.touch()

    assert preflight_mineru_files(mineru_root, ["p1"])[
        "checked_exact_selected_paths"
    ] == 1
    with pytest.raises(ValueError, match="missing=1, empty=1"):
        preflight_mineru_files(mineru_root, ["p1", "p2", "p3"])


def test_output_root_cannot_overlap_shared_input(tmp_path: Path) -> None:
    shared_root = tmp_path / "shared"

    with pytest.raises(ValueError, match="must not overlap"):
        validate_output_root(shared_root / "output", [shared_root])
    with pytest.raises(ValueError, match="must not overlap"):
        validate_output_root(tmp_path, [shared_root])


def test_default_and_explicit_corpus_size_limits() -> None:
    validate_corpus_size_limits([5_000])
    validate_corpus_size_limits(
        [5_000, 10_000, 20_000, 27_487],
        max_papers=27_487,
        confirm_paper_count=27_487,
    )

    with pytest.raises(ValueError, match="exceeds --max-papers"):
        validate_corpus_size_limits([5_001])
    with pytest.raises(ValueError, match="between 1 and 27487"):
        validate_corpus_size_limits(
            [27_487],
            max_papers=27_488,
            confirm_paper_count=27_487,
        )
    with pytest.raises(ValueError, match="exceeds --max-papers"):
        validate_corpus_size_limits(
            [27_488],
            max_papers=27_487,
            confirm_paper_count=27_488,
        )


@pytest.mark.parametrize("confirmation", [None, 27_486, 27_488])
def test_large_corpus_requires_exact_confirmation(
    confirmation: int | None,
) -> None:
    with pytest.raises(ValueError, match="--confirm-paper-count"):
        validate_corpus_size_limits(
            [5_000, 27_487],
            max_papers=27_487,
            confirm_paper_count=confirmation,
        )


def test_large_corpus_rejection_precedes_input_access(tmp_path: Path) -> None:
    output_root = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="exceeds --max-papers"):
        create_nested_corpora(
            metadata_path=tmp_path / "missing-metadata.jsonl",
            validation_path=tmp_path / "missing-validation.jsonl",
            mineru_root=tmp_path / "missing-mineru",
            output_root=output_root,
            sizes=[5_001],
            read_only_roots=[],
        )

    assert not output_root.exists()


def test_create_nested_corpora_writes_lists_and_refuses_implicit_overwrite(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    mineru_root = tmp_path / "mineru"
    output_root = tmp_path / "output"
    papers = _papers()
    _write_jsonl(
        metadata_path,
        [
            {"paper_id": paper.paper_id, "venue": paper.venue, "year": paper.year}
            for paper in papers
        ],
    )
    _write_jsonl(
        validation_path,
        [{"query_id": "q1", "gold_papers": [{"paper_id": "a1"}, {"paper_id": "b1"}]}],
    )
    _create_mineru_files(mineru_root, [paper.paper_id for paper in papers])

    selected = create_nested_corpora(
        metadata_path=metadata_path,
        validation_path=validation_path,
        mineru_root=mineru_root,
        output_root=output_root,
        sizes=[4, 6, 8],
        read_only_roots=[],
    )

    assert (
        output_root / "accuracy_4" / "paper_ids_4.txt"
    ).read_text(encoding="utf-8").splitlines() == selected[4]
    root_manifest = json.loads(
        (output_root / "nested_corpus_manifest.json").read_text(encoding="utf-8")
    )
    assert root_manifest["nested"] is True
    assert root_manifest["sizes"] == [4, 6, 8]
    assert root_manifest["generation_safety"] == {
        "default_max_papers": 5_000,
        "absolute_max_papers": 27_487,
        "max_papers": 5_000,
        "confirmed_paper_count": None,
    }
    with pytest.raises(ValueError, match="already exist"):
        create_nested_corpora(
            metadata_path=metadata_path,
            validation_path=validation_path,
            mineru_root=mineru_root,
            output_root=output_root,
            sizes=[4, 6, 8],
            read_only_roots=[],
        )
