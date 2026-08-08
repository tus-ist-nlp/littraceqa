#!/usr/bin/env python3
"""Download the exact Qwen3 reranker revision selected by a search config."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_SEARCH_CONFIG = Path(
    "configs/search_style/seed_expansion_structured_filter.yaml"
)


def reranker_spec(search_config: Path) -> tuple[str, str]:
    """Return a pinned Qwen3 model and revision from a search YAML file."""

    import yaml

    with search_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    reranker = config.get("reranker") if isinstance(config, dict) else None
    if not isinstance(reranker, dict) or reranker.get("name") != "qwen3":
        raise ValueError(f"search config does not select qwen3: {search_config}")
    params = reranker.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"qwen3 params are missing: {search_config}")
    model = params.get("model")
    revision = params.get("revision")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"qwen3 model is missing: {search_config}")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"qwen3 revision must be pinned: {search_config}")
    return model, revision


def download(
    search_config: Path,
    cache_dir: Path | None,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> Path:
    """Download one pinned model snapshot and return its local directory."""

    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        snapshot_download = hf_snapshot_download
    model, revision = reranker_spec(search_config)
    kwargs: dict[str, Any] = {"repo_id": model, "revision": revision}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir.expanduser())
    return Path(snapshot_download(**kwargs))


def build_parser() -> argparse.ArgumentParser:
    """Define the model-download command line."""

    parser = argparse.ArgumentParser(
        description="Download the pinned Qwen3 reranker for offline evaluation."
    )
    parser.add_argument(
        "--search",
        type=Path,
        default=DEFAULT_SEARCH_CONFIG,
        help="Search YAML whose qwen3 model and revision should be downloaded.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Hugging Face cache root; use the same path for evaluation.",
    )
    return parser


def main() -> None:
    """Download the configured snapshot and print the resolved local path."""

    args = build_parser().parse_args()
    snapshot = download(args.search, args.cache_dir)
    print(f"Downloaded model snapshot: {snapshot}")
    if args.cache_dir is not None:
        print(f"Use HF_HUB_CACHE={args.cache_dir.expanduser()} during evaluation.")


if __name__ == "__main__":
    main()
