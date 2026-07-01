#!/usr/bin/env python3
"""Validate the local Azure .env file without printing secret values."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from .common import ROOT

REQUIRED = [
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_ADMIN_KEY",
    "AZURE_SEARCH_INDEX_NAME",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DIMENSIONS",
]

KNOWN_OPTIONAL = {
    "AZURE_DOCUMENT_INTELLIGENCE_MODEL",
    "AZURE_DOCUMENT_INTELLIGENCE_API_VERSION",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_USE_V1",
    "AZURE_OPENAI_EMBEDDING_REQUEST_DIMENSIONS",
}


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, raw_value = stripped.split("=", 1)
            values[name.strip()] = raw_value.strip().strip("\"'")
    return values


def looks_like_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def looks_like_endpoint_origin(value: str) -> bool:
    parsed = urlparse(value)
    path = parsed.path.strip("/")
    return looks_like_https_url(value) and not path and not parsed.query


def looks_like_deployment_name(value: str) -> bool:
    if not value:
        return False
    if looks_like_https_url(value) or "/openai/deployments/" in value:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate .env without printing secrets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()

    if not args.env_file.exists():
        print(f"missing env file: {args.env_file}", file=sys.stderr)
        return 1

    values = parse_env(args.env_file)
    issues: list[str] = []

    for name in REQUIRED:
        value = values.get(name, "")
        if not value:
            issues.append(f"missing: {name}")
            continue
        if name.endswith("_ENDPOINT") and not looks_like_endpoint_origin(value):
            issues.append(f"{name}: endpoint should be an https origin without path/query")
        if name == "AZURE_OPENAI_EMBEDDING_DIMENSIONS" and not value.isdigit():
            issues.append(f"{name}: must be an integer")
        if name.endswith("_DEPLOYMENT") and not looks_like_deployment_name(value):
            issues.append(f"{name}: should be a deployment name, not a URL")

    unknown = sorted(set(values) - set(REQUIRED) - KNOWN_OPTIONAL)
    for name in unknown:
        issues.append(f"unknown variable: {name}")

    print("checked:", args.env_file)
    for name in REQUIRED:
        value = values.get(name, "")
        status = "OK" if value else "MISSING"
        print(f"{status:7s} {name:45s} length={len(value)}")

    if issues:
        print("\nissues:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("\n.env looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
