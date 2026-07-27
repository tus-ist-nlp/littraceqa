"""Read-only search over precomputed paper-level embeddings."""

from __future__ import annotations

import hashlib
import json
import operator
import stat
from pathlib import Path
from typing import Any

import numpy as np

from littraceqa.di_pipeline.contracts import Chunk, RetrievalResult

_SCHEMA_VERSION = 1
_EMBEDDINGS_FILENAME = "embeddings.npy"
_PAPERS_FILENAME = "papers.jsonl"
_CONFIG_FILENAME = "index_config.json"
_HEX_DIGITS = frozenset("0123456789abcdef")
_VALIDATION_BLOCK_ROWS = 4096
_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "paper_count",
        "dimension",
        "files",
        "source",
    }
)
_FILE_KEYS = frozenset({_EMBEDDINGS_FILENAME, _PAPERS_FILENAME})
_FILE_RECORD_KEYS = frozenset({"sha256", "size"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _regular_file_size(path: Path, label: str) -> int:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return file_stat.st_size


class PaperEmbeddingStore:
    """Load and search a model-free paper embedding sidecar.

    The row order in ``embeddings.npy`` must match the common ``Chunk`` records
    in ``papers.jsonl``. Vectors are memory-mapped, while the bounded paper
    metadata is loaded once to support deterministic ID lookup and tie breaks.
    """

    name = "paper_embedding"

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self._embeddings: np.ndarray | None = None
        self._papers: list[Chunk] = []
        self._row_by_paper_id: dict[str, int] = {}
        self._source: dict[str, Any] = {}

    @property
    def paper_count(self) -> int:
        """Return the number of loaded paper records."""

        return len(self._papers)

    @property
    def dimension(self) -> int:
        """Return the loaded embedding dimension, or zero before loading."""

        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    @property
    def source(self) -> dict[str, Any]:
        """Return a shallow copy of the recorded sidecar provenance."""

        return dict(self._source)

    def load(self) -> None:
        """Load a verified sidecar without loading its matrix into RAM."""

        config_path = self.index_dir / _CONFIG_FILENAME
        embeddings_path = self.index_dir / _EMBEDDINGS_FILENAME
        papers_path = self.index_dir / _PAPERS_FILENAME
        observed_sizes = {}
        for filename, path, label in (
            (_CONFIG_FILENAME, config_path, "paper embedding config"),
            (
                _EMBEDDINGS_FILENAME,
                embeddings_path,
                "paper embedding matrix",
            ),
            (_PAPERS_FILENAME, papers_path, "paper embedding records"),
        ):
            observed_sizes[filename] = _regular_file_size(path, label)

        config = self._load_config(config_path)
        for filename, path in (
            (_EMBEDDINGS_FILENAME, embeddings_path),
            (_PAPERS_FILENAME, papers_path),
        ):
            file_record = config["files"][filename]
            observed_size = observed_sizes[filename]
            if observed_size != file_record["size"]:
                raise ValueError(
                    f"{filename} size does not match index_config.json "
                    f"({observed_size} != {file_record['size']})"
                )
            observed_checksum = _sha256_file(path)
            if observed_checksum != file_record["sha256"]:
                raise ValueError(
                    f"{filename} checksum does not match index_config.json"
                )

        papers = self._load_papers(papers_path)
        expected_count = config["paper_count"]
        if len(papers) != expected_count:
            raise ValueError(
                "paper record count does not match index_config.json "
                f"({len(papers)} != {expected_count})"
            )

        try:
            embeddings = np.load(
                embeddings_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot load paper embedding matrix: {embeddings_path}"
            ) from exc
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("paper embedding matrix must be a NumPy array")
        self._validate_embeddings(
            embeddings,
            expected_count=expected_count,
            expected_dimension=config["dimension"],
        )

        self._papers = papers
        self._row_by_paper_id = {
            paper.paper_id: row for row, paper in enumerate(papers)
        }
        self._embeddings = embeddings
        self._source = dict(config["source"])

    def search_by_paper_id(
        self,
        paper_id: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return nearest papers for one stored paper, excluding the query."""

        if self._embeddings is None:
            raise RuntimeError("paper embedding store is not loaded; call load() first")
        if not isinstance(paper_id, str):
            raise TypeError("paper_id must be a string")
        if not paper_id:
            raise ValueError("paper_id must not be empty")
        try:
            limit = operator.index(top_k)
        except TypeError:
            raise TypeError("top_k must be an integer") from None
        if isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if limit <= 0:
            return []

        query_row = self._row_by_paper_id.get(paper_id)
        if query_row is None:
            return []

        query_embedding = np.asarray(self._embeddings[query_row])
        scores = np.asarray(self._embeddings @ query_embedding, dtype=np.float32)
        candidate_rows = [
            row for row in range(len(self._papers)) if row != query_row
        ]
        candidate_rows.sort(
            key=lambda row: (
                -float(scores[row]),
                self._papers[row].paper_id,
                self._papers[row].chunk_id,
            )
        )

        results: list[RetrievalResult] = []
        for row in candidate_rows[:limit]:
            paper = self._papers[row]
            results.append(
                RetrievalResult(
                    chunk_id=paper.chunk_id,
                    paper_id=paper.paper_id,
                    score=float(scores[row]),
                    text=paper.text,
                    chunk_type=paper.chunk_type,
                    metadata=dict(paper.metadata),
                    source=self.name,
                )
            )
        return results

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid paper embedding config: {path}") from exc
        if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
            raise ValueError(
                "paper embedding config must contain exactly "
                f"{sorted(_CONFIG_KEYS)}"
            )
        if (
            isinstance(value["schema_version"], bool)
            or value["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported paper embedding schema version: "
                f"{value['schema_version']!r}"
            )
        for name in ("paper_count", "dimension"):
            item = value[name]
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"{name} must be a positive integer")

        files = value["files"]
        if not isinstance(files, dict) or set(files) != _FILE_KEYS:
            raise ValueError(
                "paper embedding files must contain exactly "
                f"{sorted(_FILE_KEYS)}"
            )
        for filename, file_record in files.items():
            if (
                not isinstance(file_record, dict)
                or set(file_record) != _FILE_RECORD_KEYS
            ):
                raise ValueError(
                    f"paper embedding file record for {filename} must contain "
                    f"exactly {sorted(_FILE_RECORD_KEYS)}"
                )
            if not _is_sha256(file_record["sha256"]):
                raise ValueError(
                    f"paper embedding checksum for {filename} is not SHA-256"
                )
            size = file_record["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(
                    f"paper embedding size for {filename} must be a "
                    "non-negative integer"
                )
        if not isinstance(value["source"], dict):
            raise ValueError("paper embedding source must be a JSON object")
        return value

    @staticmethod
    def _load_papers(path: Path) -> list[Chunk]:
        papers: list[Chunk] = []
        seen_paper_ids: set[str] = set()
        seen_chunk_ids: set[str] = set()
        try:
            handle = path.open(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"cannot open paper embedding records: {path}") from exc
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"{path}:{line_number} must not be an empty JSONL record"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number} is not valid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number} must be a JSON object"
                    )
                try:
                    paper = Chunk(**value)
                except TypeError as exc:
                    raise ValueError(
                        f"{path}:{line_number} does not satisfy the Chunk contract"
                    ) from exc
                if (
                    not isinstance(paper.paper_id, str)
                    or not paper.paper_id
                    or not isinstance(paper.chunk_id, str)
                    or not paper.chunk_id
                    or not isinstance(paper.text, str)
                    or not paper.text.strip()
                    or not isinstance(paper.chunk_type, str)
                    or not paper.chunk_type
                    or not isinstance(paper.metadata, dict)
                ):
                    raise ValueError(
                        f"{path}:{line_number} does not satisfy the Chunk contract"
                    )
                if paper.paper_id in seen_paper_ids:
                    raise ValueError(
                        f"duplicate paper_id in paper embedding records: "
                        f"{paper.paper_id}"
                    )
                if paper.chunk_id in seen_chunk_ids:
                    raise ValueError(
                        f"duplicate chunk_id in paper embedding records: "
                        f"{paper.chunk_id}"
                    )
                seen_paper_ids.add(paper.paper_id)
                seen_chunk_ids.add(paper.chunk_id)
                papers.append(paper)
        return papers

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
        *,
        expected_count: int,
        expected_dimension: int,
    ) -> None:
        if embeddings.dtype != np.dtype(np.float32):
            raise ValueError(
                "paper embedding matrix must have dtype float32, "
                f"not {embeddings.dtype}"
            )
        if embeddings.ndim != 2:
            raise ValueError(
                "paper embedding matrix must be two-dimensional, "
                f"not {embeddings.ndim}D"
            )
        expected_shape = (expected_count, expected_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                "paper embedding matrix shape does not match index_config.json "
                f"({embeddings.shape} != {expected_shape})"
            )

        for start in range(0, expected_count, _VALIDATION_BLOCK_ROWS):
            block = np.asarray(
                embeddings[start : start + _VALIDATION_BLOCK_ROWS]
            )
            if not np.isfinite(block).all():
                raise ValueError(
                    "paper embedding matrix must contain only finite values"
                )
            norms = np.linalg.norm(block, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError(
                    "paper embedding matrix rows must be L2-normalized"
                )
