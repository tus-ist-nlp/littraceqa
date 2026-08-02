#!/usr/bin/env python3
"""Run preprocessing, indexing, retrieval, and evaluation end to end.

The four independent YAML files under ``configs`` select paths,
preprocessing, retrieval, and agent behavior.

Usage:
    # Build at most three papers without constructing or calling an LLM agent.
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/bm25.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output ~/littraceqa_data/mineru_eval/unused.jsonl \\
      --artifact-root ~/littraceqa_data/mineru_eval/smoke \\
      --limit 3 --build --build-only

    # Load a prebuilt index and run retrieval plus the configured agent.
    uv run python scripts/run_search.py \\
      --paths configs/paths/default.yaml \\
      --process configs/process_style/mineru.yaml \\
      --search configs/search_style/abstract_specter2_body_qwen3.yaml \\
      --agent configs/agent_style/reading.yaml \\
      --queries data/validation_inputs.jsonl \\
      --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from littraceqa.di_pipeline.build.index_state import (
    build_indexers_with_resume,
    fingerprint_chunk_file,
    implementation_source_paths,
)
from littraceqa.di_pipeline.build.paper_selection import (
    ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS,
    DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
    LARGE_BUILD_THRESHOLD,
    load_paper_ids_file,
    normalize_paper_ids,
    select_papers_for_bounded_build,
    validate_build_ceiling,
    validate_build_mode,
    validate_large_build_selection,
)
from littraceqa.di_pipeline.build.write_guard import (
    paths_overlap,
    preprocessing_source_roots,
    resolve_preprocess_cache_root,
    validate_preprocess_cache_root,
    validate_write_paths,
)
from littraceqa.di_pipeline.experiment_log import (
    generate_comment,
    log_experiment,
    write_report,
)
from littraceqa.di_pipeline.preprocess.orchestration import (
    MAX_CHARS_PER_CHUNK,
    override_max_chars_per_chunk,
    preprocess_selected_papers,
)
from littraceqa.di_pipeline.preprocess.checkpoint import (
    MergeResult,
    PreprocessCache,
)
from littraceqa.di_pipeline.query_input import load_queries


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line surface."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="configs/paths/*.yaml")
    parser.add_argument("--process", required=True, help="configs/process_style/*.yaml")
    parser.add_argument("--search", required=True, help="configs/search_style/*.yaml")
    parser.add_argument("--agent", required=True, help="configs/agent_style/*.yaml")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--build", action="store_true", help="前処理 + 索引構築をする（初回のみ）"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build a bounded index without constructing or calling an agent.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse valid per-paper preprocessing and completed indexer "
            "checkpoints during --build."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="User-owned output root required for bounded index builds.",
    )
    parser.add_argument(
        "--preprocess-cache-root",
        type=Path,
        help=(
            "Optional user-owned per-paper cache shared by multiple artifact "
            "roots; defaults to <artifact-root>/preprocess."
        ),
    )
    parser.add_argument(
        "--read-only-root",
        type=Path,
        default=Path("/data2/iseakira"),
        help="Shared input root that must never receive writes.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Paper ID to include in a bounded build; repeat as needed.",
    )
    parser.add_argument(
        "--paper-ids-file",
        type=Path,
        help="File containing one paper ID per line for a bounded build.",
    )
    parser.add_argument(
        "--confirm-paper-count",
        type=int,
        help=(
            "Exact selected paper count required when building more than "
            f"{LARGE_BUILD_THRESHOLD} papers."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Exact maximum papers for a bounded build. The default safety "
            f"ceiling is {DEFAULT_MAX_BOUNDED_BUILD_PAPERS}."
        ),
    )
    parser.add_argument(
        "--max-build-papers",
        type=int,
        default=DEFAULT_MAX_BOUNDED_BUILD_PAPERS,
        help=(
            "Safety ceiling for this build. Values above "
            f"{DEFAULT_MAX_BOUNDED_BUILD_PAPERS} require redundant exact-count "
            f"confirmation and cannot exceed {ABSOLUTE_MAX_BOUNDED_BUILD_PAPERS}."
        ),
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        help=(
            "Override MinerU text chunk size for a bounded build; "
            f"maximum {MAX_CHARS_PER_CHUNK}."
        ),
    )
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        help="Override the enabled reranker's candidate pool (1-1000).",
    )
    input_mode = parser.add_mutually_exclusive_group()
    input_mode.add_argument(
        "--production-input",
        dest="production_input",
        action="store_true",
        default=True,
        help="Use only query_id, question, answer_types, and table_schema (default).",
    )
    input_mode.add_argument(
        "--allow-validation-labels",
        dest="production_input",
        action="store_false",
        help="[oracle] Allow validation-only labels such as task_family.",
    )
    parser.add_argument(
        "--options-file",
        default=None,
        help="[oracle] multiple_choice の options を結合する jsonl（gold は読まない）。"
        "本番入力に options は無いので、これを付けた実行は「選択肢を教えてもらえたら"
        "何点取れるか」を見る ablation であり本番の点数ではない。"
        "--production-input との併用時は無視される。",
    )
    return parser


def validate_cli_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[str]:
    """Reject unusable flag combinations and resolve the requested paper IDs."""

    try:
        validate_build_mode(
            build=args.build,
            build_only=args.build_only,
            resume=args.resume,
            max_chars_per_chunk=args.max_chars_per_chunk,
            preprocess_cache_root=args.preprocess_cache_root,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        validate_build_ceiling(args.max_build_papers)
        if (
            args.build
            and args.max_build_papers
            > DEFAULT_MAX_BOUNDED_BUILD_PAPERS
        ):
            validate_large_build_selection(
                args.max_build_papers,
                paper_ids_file=args.paper_ids_file,
                confirm_paper_count=args.confirm_paper_count,
                limit=args.limit,
                max_build_papers=args.max_build_papers,
            )
    except ValueError as exc:
        parser.error(str(exc))

    if paths_overlap(Path(args.output), args.read_only_root):
        parser.error("--output must not overlap --read-only-root")

    requested_paper_ids = normalize_paper_ids(args.paper_id)
    if args.build and args.paper_ids_file is not None:
        try:
            file_paper_ids = load_paper_ids_file(
                args.paper_ids_file,
                max_papers=args.max_build_papers,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        requested_paper_ids = normalize_paper_ids(
            requested_paper_ids,
            file_paper_ids,
        )
    return requested_paper_ids


def resolve_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[dict, Path | None, Path | None]:
    """Compose the four config files and derive the build artifact paths."""

    # Import optional retrieval dependencies only when the CLI is executed.
    # Query-loading and path-safety helpers remain testable with the base extra.
    from littraceqa.di_pipeline.config import (
        compose_config,
        load_config,
        override_rerank_pool,
    )

    artifact_root: Path | None = None
    preprocess_cache_root: Path | None = None
    paths_cfg = load_config(args.paths)
    if args.build:
        if args.artifact_root is None:
            parser.error("--build requires --artifact-root")
        artifact_root = args.artifact_root.expanduser().resolve()
        preprocess_cache_root = resolve_preprocess_cache_root(
            args.preprocess_cache_root,
            artifact_root,
        )
        paths_cfg = dict(paths_cfg)
        paths_cfg["chunks_dir"] = str(artifact_root / "chunks")
        paths_cfg["index_dir"] = str(artifact_root / "index")

    process_cfg = load_config(args.process)
    try:
        process_cfg = override_max_chars_per_chunk(
            process_cfg,
            args.max_chars_per_chunk,
        )
    except ValueError as exc:
        parser.error(str(exc))

    search_cfg = load_config(args.search)
    try:
        search_cfg = override_rerank_pool(search_cfg, args.rerank_pool_k)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = compose_config(
        paths=paths_cfg,
        process=process_cfg,
        search=search_cfg,
        agent=load_config(args.agent),
    )
    if args.build:
        index_paths = [
            Path(indexer["params"]["index_dir"])
            for indexer in cfg["retriever"]["indexers"]
        ]
        validate_write_paths(
            [Path(cfg["paths"]["chunks"]), *index_paths],
            artifact_root,
            args.read_only_root,
        )
    return cfg, artifact_root, preprocess_cache_root


def select_build_papers(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    cfg: dict,
    requested_paper_ids: list[str],
) -> list[dict]:
    """Choose the bounded paper set that a build will preprocess."""

    metadata_path = Path(
        cfg.get("paths", {}).get("paper_metadata", "data/paper_metadata.jsonl")
    )
    try:
        selected_papers = select_papers_for_bounded_build(
            metadata_path,
            requested_paper_ids,
            args.limit,
            max_build_papers=args.max_build_papers,
        )
        validate_large_build_selection(
            len(selected_papers),
            paper_ids_file=args.paper_ids_file,
            confirm_paper_count=args.confirm_paper_count,
            limit=args.limit,
            max_build_papers=args.max_build_papers,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return selected_papers


def run_build(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    cfg: dict,
    *,
    preprocessor,
    retriever,
    selected_papers: list[dict] | None,
    artifact_root: Path | None,
    preprocess_cache_root: Path | None,
) -> None:
    """Preprocess the selected papers, then build every index checkpoint."""

    if preprocessor is None or preprocess_cache_root is None:
        raise RuntimeError("build preprocessing cache was not initialized")
    try:
        validate_preprocess_cache_root(
            preprocess_cache_root,
            read_only_root=args.read_only_root,
            source_roots=[
                *preprocessing_source_roots(preprocessor),
                Path(cfg["paths"]["paper_metadata"]),
            ],
        )
    except ValueError as exc:
        parser.error(str(exc))

    chunks_path = Path(cfg["paths"]["chunks"])
    merged_chunks = _preprocess_for_build(
        args,
        cfg,
        preprocessor=preprocessor,
        selected_papers=selected_papers,
        chunks_path=chunks_path,
        artifact_root=artifact_root,
        preprocess_cache_root=preprocess_cache_root,
    )
    index_build = build_indexers_with_resume(
        indexers=retriever.indexers,
        indexer_configs=cfg["retriever"]["indexers"],
        chunks_path=chunks_path,
        chunks=merged_chunks,
        state_path=artifact_root / "index_build_state.json",
        resume=args.resume,
    )
    print(
        f"Index checkpoints: {index_build.built_count} built, "
        f"{index_build.loaded_count} loaded"
    )
    print("索引構築完了")


def _preprocess_for_build(
    args: argparse.Namespace,
    cfg: dict,
    *,
    preprocessor,
    selected_papers: list[dict] | None,
    chunks_path: Path,
    artifact_root: Path,
    preprocess_cache_root: Path,
) -> MergeResult:
    """Publish the merged chunk file the index build reads from."""

    if preprocessor is None:
        if not chunks_path.exists():
            print(f"エラー: {chunks_path} が存在しません", file=sys.stderr)
            sys.exit(1)
        return fingerprint_chunk_file(chunks_path)

    if selected_papers is None:
        raise RuntimeError("bounded paper selection was not initialized")

    failures_path = artifact_root / "failures.jsonl"
    implementation_paths = implementation_source_paths(preprocessor)
    cache = PreprocessCache(
        preprocess_cache_root,
        process_config=cfg["preprocessor"],
        source_module_path=implementation_paths[0],
        source_dependency_paths=implementation_paths[1:],
    )
    preprocessing = preprocess_selected_papers(
        preprocessor=preprocessor,
        selected_papers=selected_papers,
        cache=cache,
        chunks_path=chunks_path,
        failures_path=failures_path,
        resume=args.resume,
    )
    print(
        "Bounded preprocessing: "
        f"{preprocessing.processed_count} processed, "
        f"{preprocessing.reused_count} reused, "
        f"{len(preprocessing.failures)} failed; "
        f"failures: {failures_path}"
    )
    if preprocessing.failures:
        print(
            "Preprocessing stopped before global index construction. "
            "Completed papers remain checkpointed; rerun the same build "
            "with --resume to retry only failed or stale papers.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if preprocessing.merge_result is None:
        raise RuntimeError("preprocessing did not publish merged chunks")
    print(
        f"{preprocessing.merge_result.chunk_count} chunks from "
        f"{preprocessing.merge_result.paper_count} papers were "
        f"atomically saved to {chunks_path}"
    )
    return preprocessing.merge_result


def load_existing_indexes(retriever) -> None:
    """Load every prebuilt index, exiting with guidance if one is missing."""

    print("既存の索引を読み込み中...")
    for indexer in retriever.indexers:
        try:
            indexer.load()
        except Exception as exc:
            print(
                f"エラー: {indexer.name} の索引読み込みに失敗しました: {exc}\n"
                f"先に --build を付けて索引を構築してください。",
                file=sys.stderr,
            )
            sys.exit(1)
    print("読み込み完了")


def resolve_options_path(args: argparse.Namespace) -> Path | None:
    """Decide where multiple_choice options come from.

    Production input has no options, so joining them is always an explicit
    oracle setting. Combining ``--options-file`` with ``--production-input``
    is contradictory and therefore leaves options disabled.
    """

    options_path = Path(args.options_file) if args.options_file else None
    if options_path is not None and args.production_input:
        print(
            "警告: --production-input と --options-file は併用できません"
            "（本番入力に options は無い）。options の結合をスキップします。",
            file=sys.stderr,
        )
        return None
    return options_path


def load_and_announce_queries(
    args: argparse.Namespace,
    options_path: Path | None,
) -> list:
    """Load the query file and report which input mode is in effect."""

    queries = load_queries(
        Path(args.queries),
        production_input=args.production_input,
        options_path=options_path,
    )
    if args.production_input:
        print("本番と同じ4フィールド（query_id/question/answer_types/table_schema）で走らせます")
    if options_path is not None:
        n_opt = sum(1 for q in queries if q.options)
        print(
            f"[oracle] multiple_choice options を {options_path} から結合しました"
            f"（{n_opt}件）。本番では与えられないので、この点数は本番の点数ではありません。"
        )
    return queries


def predict_all(agent, queries: list) -> list[dict]:
    """Run the agent over every query, reporting progress every ten queries."""

    predictions = []
    for i, query in enumerate(queries):
        pred = agent.run(query)
        predictions.append(pred.to_dict())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} 完了")
    return predictions


def write_predictions(output_path: Path, predictions: list[dict]) -> None:
    """Write one prediction per line and report where they landed."""

    with output_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"予測結果を {output_path} に書き出しました")


def score_predictions(output_path: Path) -> dict | None:
    """Score the predictions, returning ``None`` when the output is unusable."""

    print("\n採点中...")
    result = subprocess.run(
        [
            "uv", "run", "python", "scripts/evaluate.py",
            "--gold", "data/validation.jsonl",
            "--pred", str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    try:
        return json.loads(result.stdout)["metrics"]
    except (json.JSONDecodeError, KeyError):
        print("採点結果を解釈できなかったので実験ログには残しません", file=sys.stderr)
        return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    requested_paper_ids = validate_cli_args(parser, args)
    cfg, artifact_root, preprocess_cache_root = resolve_config(parser, args)
    selected_papers = (
        select_build_papers(parser, args, cfg, requested_paper_ids)
        if args.build
        else None
    )

    from littraceqa.di_pipeline.config import build_pipeline

    preprocessor, retriever, agent = build_pipeline(
        cfg,
        build_agent=not args.build_only,
        build_preprocessor=args.build,
    )

    if args.build:
        run_build(
            parser,
            args,
            cfg,
            preprocessor=preprocessor,
            retriever=retriever,
            selected_papers=selected_papers,
            artifact_root=artifact_root,
            preprocess_cache_root=preprocess_cache_root,
        )
        if args.build_only:
            print("Build-only mode completed without constructing or calling an agent.")
            return
    else:
        load_existing_indexes(retriever)

    options_path = resolve_options_path(args)
    queries = load_and_announce_queries(args, options_path)
    print(f"{len(queries)} 件の質問に対して検索中...")

    if agent is None:
        raise RuntimeError("agent was not built")

    predictions = predict_all(agent, queries)
    output_path = Path(args.output)
    write_predictions(output_path, predictions)

    metrics = score_predictions(output_path)
    if metrics is None:
        return
    options_joined = options_path is not None
    log_experiment(args, metrics, len(queries), cfg, retriever, options_joined)
    comment = generate_comment(getattr(agent, "llm", None), args, metrics, len(queries))
    write_report(args, metrics, len(queries), comment, cfg, retriever, options_joined)


if __name__ == "__main__":
    main()
