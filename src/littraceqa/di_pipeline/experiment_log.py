"""Record what an evaluation run used and what it scored.

Writes the append-only ``results/experiments.jsonl`` row and the per-run
Markdown report, both of which are local records rather than shared artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from littraceqa.di_pipeline.agent.json_utils import parse_json_object


# Reranker attributes recorded as effective settings (instance name -> log name).
# Persist defaults such as instruction and compile even when YAML omits them so
# that each result remains reproducible.
_RERANKER_EFFECTIVE_ATTRS = {
    "model_name": "model",
    "revision": "revision",
    "device": "device",
    "fp16": "fp16",
    "batch_size": "batch_size",
    "max_tokens": "max_tokens",
    "instruction": "instruction",
    "compile": "compile",
    "base_rank_weight": "base_rank_weight",
    "rank_fusion_k": "rank_fusion_k",
}


def git_sha() -> str | None:
    """Return the current commit hash, or ``None`` outside a Git worktree."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:  # noqa: BLE001 - Provenance lookup must not stop an experiment.
        pass
    return None


def _flatten(prefix: str, params: dict | None) -> dict:
    """Flatten parameters such as ``{"k": 60}`` to ``{"fuser_k": 60}``."""
    if not isinstance(params, dict):
        return {}
    return {f"{prefix}_{key}": value for key, value in params.items()}


def tuned_params(cfg: dict, retriever_obj: Any = None) -> dict:
    """Collect tunable parameters in a flat dictionary.

    The experiment record also stores the fully resolved configuration, but it
    is long and contains environment-specific values such as ``index_dir``.
    This view keeps the parameters most useful for comparing runs.

    Nested parameter groups are flattened so that one changed value appears as
    one changed field. Effective reranker values come from the initialized
    instance because defaults omitted from YAML can still affect results.
    """
    retriever = cfg.get("retriever", {})
    preprocessor = cfg.get("preprocessor", {})
    preprocessor_params = {
        key: value
        for key, value in (preprocessor.get("params") or {}).items()
        if key not in {"pdf_dir", "mineru_dir"}
    }
    agent = cfg.get("agent", {})
    agent_params = agent.get("params", {})
    fuser = retriever.get("fuser", {})
    reranker = retriever.get("reranker", {})

    # Prefer values from the live reranker and fall back to its YAML settings.
    reranker_effective = _flatten("reranker", reranker.get("params"))
    obj = getattr(retriever_obj, "reranker", None)
    if obj is not None:
        for attr, label in _RERANKER_EFFECTIVE_ATTRS.items():
            if hasattr(obj, attr):
                reranker_effective[f"reranker_{label}"] = getattr(obj, attr)

    return {
        # Preprocessing
        "preprocessor": preprocessor.get("name"),
        **_flatten("preprocessor", preprocessor_params),
        # Retrieval
        "per_index_k": retriever.get("per_index_k"),
        "pool_k": retriever.get("pool_k"),
        "indexers": [ix.get("index_name", ix["name"]) for ix in retriever.get("indexers", [])],
        "fuser": fuser.get("name"),
        **_flatten("fuser", fuser.get("params")),
        "reranker": reranker.get("name"),
        **reranker_effective,
        # Agent
        "agent": agent.get("name"),
        "agent_llm": (agent.get("llm") or {}).get("name"),
        **{f"agent_{k}": v for k, v in agent_params.items()},
    }


def log_experiment(
    args: argparse.Namespace,
    metrics: dict,
    n_queries: int,
    cfg: dict,
    retriever_obj: Any = None,
    options_joined: bool = False,
) -> None:
    """Append one resolved experiment record to ``results/experiments.jsonl``.

    Recording only configuration paths is insufficient when the same YAML file
    is edited between runs. Store the values resolved by ``compose_config()`` so
    that each result remains traceable to its parameters.

    ``options_joined`` marks an oracle run supplied with multiple-choice
    options. Production input omits them, so accuracy from such a run must not
    be interpreted as production performance.
    """
    path = Path("results/experiments.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "paths": args.paths,
        "process": args.process,
        "search": args.search,
        "agent": args.agent,
        "queries": args.queries,
        "production_input": args.production_input,
        "options_joined": options_joined,
        "n_queries": n_queries,
        "output": args.output,
        "git_sha": git_sha(),
        "tuned_params": tuned_params(cfg, retriever_obj),
        "config": cfg,
        "metrics": metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"実験結果を {path} に追記しました")


def _load_matching_experiments(
    process: str, search: str, agent: str, limit: int = 3
) -> list[dict]:
    """Return the latest matching records from ``results/experiments.jsonl``."""
    path = Path("results/experiments.jsonl")
    if not path.exists():
        return []
    matches = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (record.get("process"), record.get("search"), record.get("agent")) == (
                process,
                search,
                agent,
            ):
                matches.append(record)
    return matches[-limit:]


def generate_comment(llm, args: argparse.Namespace, metrics: dict, n_queries: int) -> str:
    """Ask the LLM for a concise metric review, or return a fallback message.

    This runs after ``log_experiment()`` appends the current record, so the last
    matching entry is excluded before prior runs are added to the prompt.
    """
    if llm is None:
        return "(LLMコメントなし: このagent_styleはLLMを使用しない設定です)"

    history = _load_matching_experiments(args.process, args.search, args.agent, limit=4)[:-1]
    history_text = "\n".join(
        f"- {record['timestamp']}: {json.dumps(record['metrics'], ensure_ascii=False)}"
        for record in history
    ) or "(No prior runs with the same configuration)"

    prompt = (
        "You are a researcher reviewing retrieval-system experiments. Assess "
        "the metrics, note concerns, and suggest a useful next experiment. "
        "Write the comment concisely in Japanese.\n\n"
        f"Configuration: process={args.process}, search={args.search}, agent={args.agent}\n"
        f"Query count: {n_queries} (production_input={args.production_input})\n"
        f"Current metrics: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        f"Prior runs with the same configuration, oldest first:\n{history_text}\n\n"
        'Return only JSON in the form {"comment": "..."}, using 3 to 5 sentences.'
    )
    try:
        parsed = parse_json_object(llm(prompt))
    except Exception as exc:
        return f"(LLMコメントの生成に失敗しました: {exc})"
    if not parsed or not isinstance(parsed.get("comment"), str):
        return "(LLMコメントの生成に失敗しました: 応答をパースできませんでした)"
    return parsed["comment"]


def write_report(
    args: argparse.Namespace,
    metrics: dict,
    n_queries: int,
    comment: str,
    cfg: dict,
    retriever_obj: Any = None,
    options_joined: bool = False,
) -> None:
    """Write one Markdown report with settings, metrics, and LLM commentary."""
    process_name = Path(args.process).stem
    search_name = Path(args.search).stem
    agent_name = Path(args.agent).stem
    now = datetime.now()

    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{process_name}_{search_name}_{agent_name}.md"
    path = report_dir / filename

    lines = [
        f"# {process_name} + {search_name} + {agent_name}",
        "",
        f"- 実行日時: {now.isoformat(timespec='seconds')}",
        f"- paths: `{args.paths}`",
        f"- process: `{args.process}`",
        f"- search: `{args.search}`",
        f"- agent: `{args.agent}`",
        f"- queries: `{args.queries}` ({n_queries}件, production_input={args.production_input})",
        f"- output: `{args.output}`",
    ]
    sha = git_sha()
    if sha:
        lines.append(f"- git: `{sha[:12]}`")
    if options_joined:
        lines.append(
            "- **[oracle] multiple_choice の選択肢を与えて実行**（本番入力に options は"
            "無いため、multiple_choice_accuracy は本番の点数ではない）"
        )
    # Embed resolved parameters so the report remains useful after YAML changes.
    lines.extend(
        [
            "",
            "## 設定（この実行時の実際の値）",
            "",
            "| パラメータ | 値 |",
            "|---|---|",
        ]
    )
    for key, value in tuned_params(cfg, retriever_obj).items():
        if value is None:
            continue
        lines.append(f"| {key} | `{json.dumps(value, ensure_ascii=False)}` |")
    lines.extend(
        [
            "",
            "## 指標",
            "",
            "| 指標 | 値 |",
            "|---|---|",
        ]
    )
    for key, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {formatted} |")
    lines.extend(["", "## コメント", "", comment, ""])

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"レポートを {path} に書き出しました")
