#!/usr/bin/env python3
"""Create deterministic nested corpora for controlled retrieval evaluation.

The generated corpora contain every paper referenced by ``gold_papers[].paper_id``
and venue/year-stratified distractors. Selection and output ordering use separate
SHA-256 seeds so that metadata file order cannot affect the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_SELECTION_SEED = "littraceqa-controlled-stratified-v1"
DEFAULT_ORDER_SEED = "littraceqa-controlled-order-v1"
DEFAULT_SIZES = (500, 1_000, 2_000)
DEFAULT_READ_ONLY_ROOT = Path("/data2/iseakira")
DEFAULT_MAX_PAPERS = 5_000
ABSOLUTE_MAX_PAPERS = 10_000

Year = int | str
Stratum = tuple[str, Year]


@dataclass(frozen=True)
class Paper:
    """Metadata required to select a controlled retrieval corpus."""

    paper_id: str
    venue: str
    year: Year

    @property
    def stratum(self) -> Stratum:
        """Return the venue/year sampling stratum."""
        return (self.venue, self.year)


def validate_corpus_size_limits(
    sizes: Sequence[int],
    *,
    max_papers: int = DEFAULT_MAX_PAPERS,
    confirm_paper_count: int | None = None,
) -> None:
    """Reject unbounded corpus generation before metadata or MinerU access."""
    if (
        isinstance(max_papers, bool)
        or not isinstance(max_papers, int)
        or not 1 <= max_papers <= ABSOLUTE_MAX_PAPERS
    ):
        raise ValueError(
            f"--max-papers must be between 1 and {ABSOLUTE_MAX_PAPERS}"
        )
    normalized_sizes = sorted(set(sizes))
    if not normalized_sizes:
        raise ValueError("at least one corpus size is required")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in normalized_sizes
    ):
        raise ValueError("corpus sizes must be positive integers")
    largest_size = normalized_sizes[-1]
    if largest_size > max_papers:
        raise ValueError(
            f"largest corpus size ({largest_size}) exceeds --max-papers "
            f"({max_papers})"
        )
    if largest_size > DEFAULT_MAX_PAPERS:
        if max_papers != largest_size:
            raise ValueError(
                "--max-papers must equal the largest requested corpus size "
                f"({largest_size}) above {DEFAULT_MAX_PAPERS}"
            )
        if confirm_paper_count != largest_size:
            raise ValueError(
                "--confirm-paper-count must equal the largest requested corpus "
                f"size ({largest_size}) above {DEFAULT_MAX_PAPERS}"
            )
    elif (
        confirm_paper_count is not None
        and confirm_paper_count != largest_size
    ):
        raise ValueError(
            "--confirm-paper-count must equal the largest requested corpus "
            f"size ({largest_size})"
        )


def _normalize_paper_id(value: object, *, source: str) -> str:
    paper_id = str(value or "").strip()
    if not paper_id:
        raise ValueError(f"{source} has an empty paper_id")
    if Path(paper_id).name != paper_id or paper_id in {".", ".."}:
        raise ValueError(f"{source} has an unsafe paper_id: {paper_id!r}")
    return paper_id


def _normalize_year(value: object) -> Year:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value or "").strip()
    return text or "UNKNOWN"


def _stratum_sort_key(stratum: Stratum) -> tuple[str, str]:
    return (stratum[0], str(stratum[1]))


def _seeded_sort_key(seed: str, paper_id: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}\0{paper_id}".encode("utf-8")).digest()
    return (digest, paper_id)


def load_metadata(path: Path) -> list[Paper]:
    """Load paper IDs and strata from a JSONL metadata file."""
    papers: list[Paper] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            paper_id = _normalize_paper_id(
                record.get("paper_id"),
                source=f"{path}:{line_number}",
            )
            if paper_id in seen:
                raise ValueError(f"{path}:{line_number} duplicates paper_id {paper_id!r}")
            seen.add(paper_id)
            papers.append(
                Paper(
                    paper_id=paper_id,
                    venue=str(record.get("venue") or "").strip() or "UNKNOWN",
                    year=_normalize_year(record.get("year")),
                )
            )
    if not papers:
        raise ValueError(f"metadata contains no papers: {path}")
    return papers


def load_gold_paper_ids(path: Path) -> set[str]:
    """Load only ``gold_papers[].paper_id`` values from validation JSONL."""
    gold_paper_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            gold_papers = record.get("gold_papers")
            if not isinstance(gold_papers, list):
                raise ValueError(f"{path}:{line_number} has no gold_papers list")
            for index, gold_paper in enumerate(gold_papers):
                if not isinstance(gold_paper, dict):
                    raise ValueError(
                        f"{path}:{line_number} gold_papers[{index}] is not an object"
                    )
                gold_paper_ids.add(
                    _normalize_paper_id(
                        gold_paper.get("paper_id"),
                        source=f"{path}:{line_number} gold_papers[{index}]",
                    )
                )
    if not gold_paper_ids:
        raise ValueError(f"validation data contains no gold paper IDs: {path}")
    return gold_paper_ids


def largest_remainder_quotas(
    population_counts: Mapping[Stratum, int],
    target_size: int,
) -> dict[Stratum, int]:
    """Allocate an exact sample size proportionally with largest remainders."""
    total_population = sum(population_counts.values())
    if total_population <= 0:
        raise ValueError("population must contain at least one paper")
    if not 1 <= target_size <= total_population:
        raise ValueError(
            f"target size must be between 1 and {total_population}: {target_size}"
        )

    quotas: dict[Stratum, int] = {}
    remainders: dict[Stratum, int] = {}
    for stratum, count in population_counts.items():
        if count <= 0:
            raise ValueError(f"stratum {_stratum_sort_key(stratum)!r} is empty")
        quota, remainder = divmod(count * target_size, total_population)
        quotas[stratum] = quota
        remainders[stratum] = remainder

    remaining = target_size - sum(quotas.values())
    ranked_strata = sorted(
        population_counts,
        key=lambda stratum: (
            -remainders[stratum],
            _stratum_sort_key(stratum),
        ),
    )
    for stratum in ranked_strata[:remaining]:
        quotas[stratum] += 1
    return quotas


def select_nested_corpora(
    papers: Sequence[Paper],
    gold_paper_ids: set[str],
    sizes: Sequence[int] = DEFAULT_SIZES,
    *,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    order_seed: str = DEFAULT_ORDER_SEED,
) -> tuple[dict[int, list[str]], dict[int, dict[Stratum, int]]]:
    """Select exact, deterministic, nested venue/year-stratified corpora."""
    normalized_sizes = sorted(set(sizes))
    if not normalized_sizes:
        raise ValueError("at least one corpus size is required")
    if any(size <= 0 for size in normalized_sizes):
        raise ValueError("corpus sizes must be positive")
    if normalized_sizes[-1] > len(papers):
        raise ValueError("largest corpus size exceeds metadata population")
    if normalized_sizes[0] < len(gold_paper_ids):
        raise ValueError("every corpus size must be at least the gold paper count")

    paper_by_id = {paper.paper_id: paper for paper in papers}
    if len(paper_by_id) != len(papers):
        raise ValueError("paper IDs must be unique")
    missing_gold = gold_paper_ids - paper_by_id.keys()
    if missing_gold:
        raise ValueError(f"gold paper IDs missing from metadata: {sorted(missing_gold)}")

    papers_by_stratum: dict[Stratum, list[str]] = defaultdict(list)
    for paper in papers:
        papers_by_stratum[paper.stratum].append(paper.paper_id)
    population_counts = {
        stratum: len(paper_ids)
        for stratum, paper_ids in papers_by_stratum.items()
    }
    gold_counts = Counter(paper_by_id[paper_id].stratum for paper_id in gold_paper_ids)
    ranked_distractors = {
        stratum: sorted(
            (
                paper_id
                for paper_id in paper_ids
                if paper_id not in gold_paper_ids
            ),
            key=lambda paper_id: _seeded_sort_key(selection_seed, paper_id),
        )
        for stratum, paper_ids in papers_by_stratum.items()
    }

    selected_by_size: dict[int, list[str]] = {}
    quotas_by_size: dict[int, dict[Stratum, int]] = {}
    previous_selected: set[str] = set()
    previous_quotas: dict[Stratum, int] | None = None
    for size in normalized_sizes:
        quotas = largest_remainder_quotas(population_counts, size)
        if previous_quotas is not None:
            shrinking = [
                stratum
                for stratum, quota in quotas.items()
                if quota < previous_quotas[stratum]
            ]
            if shrinking:
                labels = [_stratum_sort_key(stratum) for stratum in shrinking]
                raise ValueError(
                    "requested sizes do not have nested largest-remainder quotas: "
                    f"{labels}"
                )

        selected = set(gold_paper_ids)
        for stratum, quota in quotas.items():
            required_gold = gold_counts[stratum]
            if required_gold > quota:
                raise ValueError(
                    "gold papers exceed the proportional quota for "
                    f"{_stratum_sort_key(stratum)!r} at size {size}: "
                    f"{required_gold} > {quota}"
                )
            distractor_count = quota - required_gold
            selected.update(ranked_distractors[stratum][:distractor_count])

        if len(selected) != size:
            raise RuntimeError(f"selection produced {len(selected)} papers, expected {size}")
        if not previous_selected.issubset(selected):
            raise RuntimeError(f"corpus size {size} is not a superset of the prior corpus")

        selected_by_size[size] = sorted(
            selected,
            key=lambda paper_id: _seeded_sort_key(order_seed, paper_id),
        )
        quotas_by_size[size] = quotas
        previous_selected = selected
        previous_quotas = quotas

    return selected_by_size, quotas_by_size


def mineru_content_list_path(mineru_root: Path, paper_id: str) -> Path:
    """Return the one expected MinerU content-list path for a paper."""
    return mineru_root / paper_id / "auto" / f"{paper_id}_content_list.json"


def preflight_mineru_files(
    mineru_root: Path,
    paper_ids: Iterable[str],
) -> dict[str, int | bool]:
    """Check only exact selected MinerU paths without traversing the root."""
    checked = 0
    missing: list[str] = []
    empty: list[str] = []
    for paper_id in paper_ids:
        checked += 1
        path = mineru_content_list_path(mineru_root, paper_id)
        if not path.is_file():
            missing.append(paper_id)
        elif path.stat().st_size == 0:
            empty.append(paper_id)
    if missing or empty:
        examples = ", ".join((missing + empty)[:10])
        raise ValueError(
            "MinerU preflight failed: "
            f"missing={len(missing)}, empty={len(empty)}; examples={examples}"
        )
    return {
        "checked_exact_selected_paths": checked,
        "missing_files": 0,
        "empty_files": 0,
        "json_contents_loaded": False,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
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


def validate_output_root(output_root: Path, read_only_roots: Iterable[Path]) -> None:
    """Reject an output root that overlaps any read-only input tree."""
    for read_only_root in read_only_roots:
        if _paths_overlap(output_root, read_only_root):
            raise ValueError(
                "output root must not overlap read-only input: "
                f"{read_only_root.expanduser().resolve()}"
            )


def _paper_ids_sha256(paper_ids: Sequence[str]) -> str:
    content = "".join(f"{paper_id}\n" for paper_id in paper_ids)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _manifest_distribution(
    papers: Sequence[Paper],
    gold_paper_ids: set[str],
    quotas: Mapping[Stratum, int],
) -> list[dict[str, object]]:
    population_counts = Counter(paper.stratum for paper in papers)
    paper_by_id = {paper.paper_id: paper for paper in papers}
    gold_counts = Counter(paper_by_id[paper_id].stratum for paper_id in gold_paper_ids)
    total_population = len(papers)
    sample_size = sum(quotas.values())
    return [
        {
            "venue": stratum[0],
            "year": stratum[1],
            "population_count": population_counts[stratum],
            "population_fraction": population_counts[stratum] / total_population,
            "selected_count": quotas[stratum],
            "selected_fraction": quotas[stratum] / sample_size,
            "gold_count": gold_counts[stratum],
        }
        for stratum in sorted(quotas, key=_stratum_sort_key)
    ]


def write_nested_corpora(
    output_root: Path,
    papers: Sequence[Paper],
    gold_paper_ids: set[str],
    selected_by_size: Mapping[int, Sequence[str]],
    quotas_by_size: Mapping[int, Mapping[Stratum, int]],
    *,
    metadata_path: Path,
    validation_path: Path,
    mineru_root: Path,
    preflight: Mapping[str, int | bool],
    selection_seed: str,
    order_seed: str,
    max_papers: int,
    confirm_paper_count: int | None,
    overwrite: bool = False,
) -> list[Path]:
    """Write paper-ID lists and manifests after checking all destinations."""
    output_root = output_root.expanduser().resolve()
    output_files = [output_root / "nested_corpus_manifest.json"]
    for size in sorted(selected_by_size):
        corpus_dir = output_root / f"accuracy_{size}"
        output_files.extend(
            [
                corpus_dir / f"paper_ids_{size}.txt",
                corpus_dir / "corpus_manifest.json",
            ]
        )
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    common = {
        "controlled_diagnostic": True,
        "selection": (
            "All unique gold_papers[].paper_id values plus deterministic "
            "venue/year-stratified distractors. Integer quotas use the "
            "largest-remainder method; distractors and final input order use "
            "separate fixed SHA-256 seeds."
        ),
        "selection_seed": selection_seed,
        "order_seed": order_seed,
        "metadata_file": str(metadata_path.expanduser().resolve()),
        "validation_file": str(validation_path.expanduser().resolve()),
        "mineru_root": str(mineru_root.expanduser().resolve()),
        "mineru_preflight": dict(preflight),
        "generation_safety": {
            "default_max_papers": DEFAULT_MAX_PAPERS,
            "absolute_max_papers": ABSOLUTE_MAX_PAPERS,
            "max_papers": max_papers,
            "confirmed_paper_count": confirm_paper_count,
        },
    }
    root_manifest = {
        **common,
        "sizes": sorted(selected_by_size),
        "gold_paper_count": len(gold_paper_ids),
        "nested": all(
            set(selected_by_size[smaller]).issubset(selected_by_size[larger])
            for smaller, larger in zip(
                sorted(selected_by_size),
                sorted(selected_by_size)[1:],
            )
        ),
        "corpora": {
            str(size): {
                "paper_count": len(selected_by_size[size]),
                "paper_ids_sha256": _paper_ids_sha256(selected_by_size[size]),
                "paper_ids_file": str(
                    output_root
                    / f"accuracy_{size}"
                    / f"paper_ids_{size}.txt"
                ),
            }
            for size in sorted(selected_by_size)
        },
    }

    for path in output_files:
        path.parent.mkdir(parents=True, exist_ok=True)
    (output_root / "nested_corpus_manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for size in sorted(selected_by_size):
        paper_ids = selected_by_size[size]
        corpus_dir = output_root / f"accuracy_{size}"
        ids_path = corpus_dir / f"paper_ids_{size}.txt"
        ids_path.write_text(
            "".join(f"{paper_id}\n" for paper_id in paper_ids),
            encoding="utf-8",
        )
        manifest = {
            "paper_count": size,
            "gold_paper_count": len(gold_paper_ids),
            "distractor_count": size - len(gold_paper_ids),
            **common,
            "paper_ids_sha256": _paper_ids_sha256(paper_ids),
            "paper_ids_file": str(ids_path),
            "distribution": _manifest_distribution(
                papers,
                gold_paper_ids,
                quotas_by_size[size],
            ),
        }
        (corpus_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return output_files


def create_nested_corpora(
    *,
    metadata_path: Path,
    validation_path: Path,
    mineru_root: Path,
    output_root: Path,
    sizes: Sequence[int] = DEFAULT_SIZES,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    order_seed: str = DEFAULT_ORDER_SEED,
    read_only_roots: Sequence[Path] = (DEFAULT_READ_ONLY_ROOT,),
    max_papers: int = DEFAULT_MAX_PAPERS,
    confirm_paper_count: int | None = None,
    overwrite: bool = False,
) -> dict[int, list[str]]:
    """Build, preflight, and write nested controlled corpus definitions."""
    validate_corpus_size_limits(
        sizes,
        max_papers=max_papers,
        confirm_paper_count=confirm_paper_count,
    )
    validate_output_root(output_root, [mineru_root, *read_only_roots])
    papers = load_metadata(metadata_path)
    gold_paper_ids = load_gold_paper_ids(validation_path)
    selected_by_size, quotas_by_size = select_nested_corpora(
        papers,
        gold_paper_ids,
        sizes,
        selection_seed=selection_seed,
        order_seed=order_seed,
    )
    largest_size = max(selected_by_size)
    preflight = preflight_mineru_files(
        mineru_root,
        selected_by_size[largest_size],
    )
    write_nested_corpora(
        output_root,
        papers,
        gold_paper_ids,
        selected_by_size,
        quotas_by_size,
        metadata_path=metadata_path,
        validation_path=validation_path,
        mineru_root=mineru_root,
        preflight=preflight,
        selection_seed=selection_seed,
        order_seed=order_seed,
        max_papers=max_papers,
        confirm_paper_count=confirm_paper_count,
        overwrite=overwrite,
    )
    return selected_by_size


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--mineru-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--max-papers",
        type=int,
        default=DEFAULT_MAX_PAPERS,
        help=(
            "Safety ceiling for the largest corpus. Values above "
            f"{DEFAULT_MAX_PAPERS} require exact confirmation and cannot exceed "
            f"{ABSOLUTE_MAX_PAPERS}."
        ),
    )
    parser.add_argument(
        "--confirm-paper-count",
        type=int,
        help=(
            "Exact largest corpus size required when generating more than "
            f"{DEFAULT_MAX_PAPERS} papers."
        ),
    )
    parser.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--order-seed", default=DEFAULT_ORDER_SEED)
    parser.add_argument(
        "--read-only-root",
        type=Path,
        action="append",
        default=[DEFAULT_READ_ONLY_ROOT],
        help="Additional tree that must never overlap the output root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing corpus lists and manifests without deleting directories.",
    )
    return parser


def main() -> None:
    """Run the controlled-corpus generator."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        selected_by_size = create_nested_corpora(
            metadata_path=args.metadata,
            validation_path=args.validation,
            mineru_root=args.mineru_root,
            output_root=args.output_root,
            sizes=args.sizes,
            selection_seed=args.selection_seed,
            order_seed=args.order_seed,
            read_only_roots=args.read_only_root,
            max_papers=args.max_papers,
            confirm_paper_count=args.confirm_paper_count,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for size, paper_ids in sorted(selected_by_size.items()):
        print(f"created size={size} papers={len(paper_ids)}")


if __name__ == "__main__":
    main()
