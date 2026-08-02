"""Keep a build from writing into shared, read-only or input trees.

Preprocessing inputs and prebuilt indexes are shared between users, so every
destination is checked against them before anything is created.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve()
    right = right.expanduser().resolve()
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def validate_write_paths(
    output_paths: list[Path],
    artifact_root: Path,
    read_only_root: Path,
) -> None:
    """Reject writes outside the requested artifact root or into shared input."""
    artifact_root = artifact_root.expanduser().resolve()
    read_only_root = read_only_root.expanduser().resolve()
    if paths_overlap(artifact_root, read_only_root):
        raise ValueError("--artifact-root must not overlap --read-only-root")
    for output_path in output_paths:
        resolved = output_path.expanduser().resolve()
        if paths_overlap(resolved, read_only_root):
            raise ValueError(f"refusing to write inside read-only input: {resolved}")
        try:
            resolved.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(
                f"build output must stay under --artifact-root: {resolved}"
            ) from exc


def resolve_preprocess_cache_root(
    explicit_root: Path | None,
    artifact_root: Path,
) -> Path:
    """Resolve an optional shared cache while preserving the prior default."""
    root = explicit_root if explicit_root is not None else artifact_root / "preprocess"
    return root.expanduser().resolve()


def preprocessing_source_roots(preprocessor: Any) -> list[Path]:
    """Return roots containing the artifacts read by a preprocessor."""
    mineru_dir = getattr(preprocessor, "mineru_dir", None)
    if mineru_dir is not None:
        return [Path(mineru_dir).expanduser().resolve()]
    pdf_dir = getattr(preprocessor, "pdf_dir", None)
    if pdf_dir is not None:
        return [Path(pdf_dir).expanduser().resolve()]
    return []


def validate_preprocess_cache_root(
    cache_root: Path,
    *,
    read_only_root: Path,
    source_roots: Sequence[Path],
) -> None:
    """Reject cache locations that could overwrite or contain input data."""
    cache_root = cache_root.expanduser().resolve()
    protected_roots = [
        read_only_root.expanduser().resolve(),
        *(root.expanduser().resolve() for root in source_roots),
    ]
    for protected_root in protected_roots:
        if paths_overlap(cache_root, protected_root):
            raise ValueError(
                "preprocess cache must not overlap read-only or source input: "
                f"{protected_root}"
            )

    for internal_path in (
        cache_root / "papers",
        cache_root / "manifest.jsonl",
    ):
        if internal_path.is_symlink():
            raise ValueError(
                "preprocess cache internal paths must not be symlinks: "
                f"{internal_path}"
            )
        resolved_internal = internal_path.resolve()
        try:
            resolved_internal.relative_to(cache_root)
        except ValueError as exc:
            raise ValueError(
                "preprocess cache internal path escapes its root: "
                f"{resolved_internal}"
            ) from exc
        for protected_root in protected_roots:
            if paths_overlap(resolved_internal, protected_root):
                raise ValueError(
                    "preprocess cache internal path overlaps protected input: "
                    f"{resolved_internal}"
                )

    if cache_root.exists():
        if not cache_root.is_dir():
            raise ValueError(
                f"preprocess cache root is not a directory: {cache_root}"
            )
        if cache_root.stat().st_uid != os.getuid():
            raise ValueError(
                f"preprocess cache root is not owned by the current user: {cache_root}"
            )
        if not os.access(cache_root, os.W_OK | os.X_OK):
            raise ValueError(
                f"preprocess cache root is not writable: {cache_root}"
            )
        return

    writable_parent = cache_root
    while not writable_parent.exists() and writable_parent != writable_parent.parent:
        writable_parent = writable_parent.parent
    if not writable_parent.is_dir() or not os.access(
        writable_parent,
        os.W_OK | os.X_OK,
    ):
        raise ValueError(
            "preprocess cache root has no writable parent owned or accessible "
            f"to the current user: {cache_root}"
        )
