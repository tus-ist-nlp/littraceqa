#!/usr/bin/env python3
"""Run the production-safe corpus reader over a candidate-paper sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from littraceqa.candidate_handoff import load_candidate_handoffs, read_jsonl
from littraceqa.chunk_store import ChunkStore
from littraceqa.corpus_preflight import inspect_corpus
from littraceqa.di_pipeline.agent.corpus_qa import CorpusQAAgent
from littraceqa.di_pipeline.llm.fake import FakeLLM
from littraceqa.di_pipeline.llm.openai_compatible import OpenAICompatibleLLM
from littraceqa.submission import TOP_LEVEL_KEYS, prediction_to_submission


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

_RUNTIME_FILES = tuple(
    sorted((ROOT / "src/littraceqa").rglob("*.py"))
) + (
    Path(__file__),
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"agent config is not an object: {path}")
    return payload


def build_llm(config: dict[str, Any]):
    llm_config = config.get("llm")
    if not isinstance(llm_config, dict):
        raise ValueError("agent config must contain llm")
    name = str(llm_config.get("name") or "")
    params = llm_config.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("llm.params must be an object")
    if name == "openai_compatible":
        return OpenAICompatibleLLM(**params)
    if name == "fake":
        return FakeLLM(**params)
    raise ValueError(
        f"unsupported corpus QA llm {name!r}; use openai_compatible or fake"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(resolved),
    }


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    # Generated submissions/checkpoints are often placed inside the repository.
    # Including `git status` would therefore make an otherwise identical resume
    # reject its own files.  Relevant dirty/untracked runtime files are already
    # content-fingerprinted explicitly in `runtime` and `inputs`.
    return {"sha": run("rev-parse", "HEAD")}


def build_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    preflight_report: dict[str, Any],
) -> dict[str, Any]:
    llm_config = config.get("llm") or {}
    llm_params = llm_config.get("params") or {}
    print("fingerprinting inputs and corpus for reproducible resume...")
    return {
        "version": 1,
        "inputs": {
            "queries": _fingerprint(args.queries),
            "candidates": _fingerprint(args.candidates),
            "paper_metadata": _fingerprint(args.paper_metadata),
            "chunks": _fingerprint(args.chunks),
            "agent_config": _fingerprint(args.agent),
        },
        "runtime": {
            str(path.relative_to(ROOT)): _fingerprint(path)
            for path in _RUNTIME_FILES
        },
        "llm": {
            "adapter": llm_config.get("name"),
            "model": llm_params.get("model") or os.environ.get("OPENAI_CHAT_MODEL"),
            "base_url": llm_params.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
        },
        "run": {
            "limit": args.limit,
            "image_root": str(Path(args.image_root).resolve()) if args.image_root else None,
            "chunk_index": (
                str(Path(args.chunk_index).resolve()) if args.chunk_index else None
            ),
        },
        # Includes full-corpus ID coverage, locator/modality coverage and a
        # content digest over every readable candidate table/figure image.
        "preflight": preflight_report,
        "git": _git_state(),
    }


def ensure_manifest(
    manifest_path: Path, manifest: dict[str, Any], resume: bool
) -> None:
    if resume:
        if not manifest_path.exists():
            raise ValueError(f"resume manifest does not exist: {manifest_path}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                "resume manifest does not match current model/config/data/code; "
                "start a new output path"
            )
        return
    if manifest_path.exists():
        raise ValueError(f"refusing to overwrite existing manifest: {manifest_path}")
    tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, manifest_path)
    finally:
        tmp.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    records: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if set(record) != {"query_id", "submission", "analysis"}:
                raise ValueError(f"invalid checkpoint shape on line {line_number}")
            query_id = str(record.get("query_id") or "")
            submission = record.get("submission")
            analysis = record.get("analysis")
            if (
                not query_id
                or query_id in query_ids
                or not isinstance(submission, dict)
                or set(submission) != TOP_LEVEL_KEYS
                or submission.get("query_id") != query_id
                or not isinstance(analysis, dict)
                or analysis.get("query_id") != query_id
            ):
                raise ValueError(f"invalid checkpoint record on line {line_number}")
            query_ids.add(query_id)
            records.append(record)
    return records, query_ids


def write_checkpoint(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace the durable per-query checkpoint."""

    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def materialize_outputs(
    ordered_records: list[dict[str, Any]],
    output_path: Path,
    analysis_path: Path,
) -> None:
    if output_path.resolve() == analysis_path.resolve():
        raise ValueError("submission and analysis paths must be different")
    output_tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    analysis_tmp = analysis_path.with_name(f".{analysis_path.name}.{os.getpid()}.tmp")
    try:
        with output_tmp.open("x", encoding="utf-8") as submission_file, analysis_tmp.open(
            "x", encoding="utf-8"
        ) as analysis_file:
            for record in ordered_records:
                submission_file.write(
                    json.dumps(record["submission"], ensure_ascii=False) + "\n"
                )
                analysis_file.write(
                    json.dumps(record["analysis"], ensure_ascii=False) + "\n"
                )
        os.replace(output_tmp, output_path)
        os.replace(analysis_tmp, analysis_path)
    finally:
        output_tmp.unlink(missing_ok=True)
        analysis_tmp.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer LitTraceQA from candidate papers and MinerU chunks."
    )
    parser.add_argument("--queries", required=True, help="Official four-field JSONL")
    parser.add_argument(
        "--candidates",
        required=True,
        help="Sanitized query_id + candidate_papers sidecar",
    )
    parser.add_argument("--chunks", required=True, help="MinerU chunks JSONL")
    parser.add_argument(
        "--chunk-index",
        default=None,
        help="Writable byte-offset index path (recommended for read-only corpus)",
    )
    parser.add_argument(
        "--paper-metadata",
        default="data/paper_metadata.jsonl",
        help="Canonical paper metadata used to validate candidate IDs",
    )
    parser.add_argument("--image-root", default=None)
    parser.add_argument(
        "--agent", default="configs/agent_style/corpus_qa.yaml", help="Agent YAML"
    )
    parser.add_argument("--output", required=True, help="Official submission JSONL")
    parser.add_argument(
        "--analysis-output",
        default=None,
        help="Separate trace/candidate JSONL (default: <output>.analysis.jsonl)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue an identical fingerprinted run"
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test first N queries")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    analysis_path = (
        Path(args.analysis_output)
        if args.analysis_output
        else output_path.with_suffix(output_path.suffix + ".analysis.jsonl")
    )
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint.jsonl")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    all_paths = {path.resolve() for path in (output_path, analysis_path, checkpoint_path, manifest_path)}
    if len(all_paths) != 4:
        raise SystemExit("output, analysis, checkpoint and manifest paths must be different")
    if not args.resume:
        existing = [
            path
            for path in (output_path, analysis_path, checkpoint_path, manifest_path)
            if path.exists()
        ]
        if existing:
            raise SystemExit(f"refusing to overwrite existing run files: {existing}")

    handoffs = load_candidate_handoffs(
        args.queries, args.candidates, paper_metadata_path=args.paper_metadata
    )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        handoffs = handoffs[: args.limit]

    config = load_yaml(args.agent)
    if config.get("name") != "corpus_qa":
        raise ValueError("agent config name must be corpus_qa")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_paper_ids = {
        str(record.get("paper_id") or "")
        for record in read_jsonl(args.paper_metadata)
        if record.get("paper_id")
    }
    if not canonical_paper_ids:
        raise ValueError("paper metadata has no canonical paper IDs")
    store = ChunkStore(
        args.chunks,
        index_path=args.chunk_index,
        image_root=args.image_root,
    )
    preflight_report, preflight_errors = inspect_corpus(
        handoffs, store, canonical_paper_ids
    )
    print(
        "preflight: "
        f"{preflight_report['corpus_papers']} corpus papers, "
        f"{preflight_report['unique_candidate_papers']} candidate papers, "
        f"{preflight_report['image_paths']['existing']} readable images"
    )
    for warning in preflight_report["warnings"]:
        print(f"preflight warning: {warning}")
    if preflight_errors:
        raise RuntimeError(
            "corpus QA preflight failed:\n"
            + json.dumps(preflight_report, ensure_ascii=False, indent=2)
        )
    manifest = build_manifest(args, config, preflight_report)
    ensure_manifest(manifest_path, manifest, args.resume)

    checkpoint_records, completed = load_checkpoint(checkpoint_path)
    expected_ids = {handoff.query.query_id for handoff in handoffs}
    if not completed.issubset(expected_ids):
        raise ValueError(
            f"checkpoint has query_ids outside this input: {sorted(completed - expected_ids)}"
        )

    llm = build_llm(config)
    agent = CorpusQAAgent(store, llm, **(config.get("params") or {}))

    total = len(handoffs)
    for index, handoff in enumerate(handoffs, start=1):
        query = handoff.query
        if query.query_id in completed:
            continue
        prediction = agent.run(query, handoff.candidate_papers)
        record = {
            "query_id": query.query_id,
            "submission": prediction_to_submission(query, prediction),
            "analysis": prediction.to_dict(),
        }
        checkpoint_records.append(record)
        write_checkpoint(checkpoint_path, checkpoint_records)
        completed.add(query.query_id)
        print(f"[{index}/{total}] {query.query_id}")

    if completed != expected_ids:
        missing = sorted(expected_ids - completed)
        raise RuntimeError(f"checkpoint/query mismatch: missing={missing}")
    by_id = {record["query_id"]: record for record in checkpoint_records}
    ordered = [by_id[handoff.query.query_id] for handoff in handoffs]
    materialize_outputs(ordered, output_path, analysis_path)
    print(f"wrote {len(ordered)} submission records to {output_path}")
    print(f"wrote analysis traces separately to {analysis_path}")
    print(f"kept atomic resume checkpoint at {checkpoint_path}")


if __name__ == "__main__":
    main()
