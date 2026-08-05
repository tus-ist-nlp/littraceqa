#!/usr/bin/env python3
"""Download a pinned, hash-verified LitTraceQA organizer release.

The pristine organizer files are kept under a revision directory instead of
overwriting repository code that adds local diagnostics.  This makes schema and
evaluator parity inspectable while keeping custom candidate-recall metrics
separate from the official reference script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/official_release_manifest.json"
DEFAULT_DESTINATION = ROOT / "artifacts/official_release"
BASE_URL = "https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned official LitTraceQA release"
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--destination", default=str(DEFAULT_DESTINATION))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download missing files; only verify the existing snapshot",
    )
    return parser


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("official release manifest must be an object")
    repository = manifest.get("repository")
    revision = manifest.get("revision")
    files = manifest.get("files")
    if not isinstance(repository, str) or repository != "LitTraceQA/LitTraceQA":
        raise ValueError("manifest repository must be LitTraceQA/LitTraceQA")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("manifest revision must be a 40-character commit hash")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest files must be a non-empty mapping")
    for relative, digest in files.items():
        parts = PurePosixPath(str(relative)).parts
        if not parts or PurePosixPath(str(relative)).is_absolute() or ".." in parts:
            raise ValueError(f"unsafe release path: {relative!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid SHA-256 for {relative}")
    return manifest


def sync_release(
    manifest: dict[str, Any],
    destination: str | Path,
    *,
    verify_only: bool = False,
) -> Path:
    snapshot_root = Path(destination).expanduser().resolve() / manifest["revision"]
    snapshot_root.mkdir(parents=True, exist_ok=True)
    for relative, expected_sha256 in manifest["files"].items():
        target = snapshot_root.joinpath(*PurePosixPath(relative).parts)
        if target.is_file() and _sha256(target) == expected_sha256:
            print(f"verified {relative}")
            continue
        if verify_only:
            raise FileNotFoundError(
                f"missing or hash-mismatched official file: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        url = BASE_URL.format(
            repository=manifest["repository"],
            revision=manifest["revision"],
            path=relative,
        )
        _download_verified(url, target, expected_sha256)
        print(f"downloaded {relative}")
    (snapshot_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot_root


def _download_verified(url: str, target: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    temporary.write(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {target.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    snapshot = sync_release(
        manifest,
        args.destination,
        verify_only=args.verify_only,
    )
    print(f"official release ready: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
