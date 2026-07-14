"""Exact NumPy index for normalized BGE-M3 dense embeddings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from litqa.contracts import Chunk, RetrievalResult
from litqa.registry import register


_EMBEDDINGS_FILENAME = "embeddings.npy"
_CHUNKS_FILENAME = "chunks.jsonl"
_CONFIG_FILENAME = "index_config.json"
_DEFAULT_MODEL = "BAAI/bge-m3"
_DEFAULT_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_VALIDATION_BATCH_ROWS = 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@register("indexer", "bge_m3_numpy")
class BGEM3NumpyIndex:
    """Embed selected common Chunks and search them by exact cosine similarity."""

    name = "bge_m3_numpy"

    def __init__(
        self,
        index_dir: str,
        model: str = _DEFAULT_MODEL,
        revision: str = _DEFAULT_REVISION,
        model_path: str | None = None,
        batch_size: int = 1,
        device: str = "cpu",
        max_length: int = 512,
        local_files_only: bool = True,
        include_chunk_types: list[str] | tuple[str, ...] | None = None,
    ):
        if not isinstance(model, str) or not isinstance(revision, str):
            raise TypeError("model and revision must be strings")
        if not model.strip() or not revision.strip():
            raise ValueError("model and revision must not be empty")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or not 1 <= max_length <= 8192
        ):
            raise ValueError("max_length must be an integer from 1 to 8192")
        if not isinstance(device, str):
            raise TypeError("device must be a string")
        if not device.strip():
            raise ValueError("device must not be empty")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")

        normalized_types: tuple[str, ...] | None = None
        if include_chunk_types is not None:
            if not isinstance(include_chunk_types, (list, tuple)):
                raise TypeError("include_chunk_types must be a list, tuple, or None")
            if any(not isinstance(value, str) for value in include_chunk_types):
                raise TypeError("include_chunk_types values must be strings")
            normalized_types = tuple(
                dict.fromkeys(value.strip() for value in include_chunk_types)
            )
            if not normalized_types or any(not value for value in normalized_types):
                raise ValueError("include_chunk_types must contain non-empty values")

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self.revision = revision
        self.model_path = Path(model_path).expanduser().resolve() if model_path else None
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.include_chunk_types = normalized_types

        self._model: Any | None = None
        self._embeddings: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: Iterable[Chunk]) -> None:
        """Build and atomically persist a normalized exact-search matrix."""
        selected = [
            chunk
            for chunk in chunks
            if self.include_chunk_types is None
            or chunk.chunk_type in self.include_chunk_types
        ]
        if not selected:
            raise ValueError("BGE-M3 index received no selected Chunks")
        chunk_ids = [chunk.chunk_id for chunk in selected]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("BGE-M3 index requires unique chunk_id values")
        if any(not chunk.text.strip() for chunk in selected):
            raise ValueError("BGE-M3 index cannot embed empty Chunk text")

        embeddings = self._embed([chunk.text for chunk in selected])
        if embeddings.ndim != 2 or embeddings.shape[0] != len(selected):
            raise ValueError("BGE-M3 encoder returned an unexpected matrix shape")
        if embeddings.shape[1] <= 0:
            raise ValueError("BGE-M3 encoder returned invalid embeddings")
        if embeddings.dtype != np.float32:
            raise ValueError("BGE-M3 encoder must return float32 embeddings")
        self._validate_embedding_values(embeddings)

        temporary = self.index_dir / (_EMBEDDINGS_FILENAME + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        temporary.replace(self.index_dir / _EMBEDDINGS_FILENAME)
        self._save_chunks(selected)
        embeddings_path = self.index_dir / _EMBEDDINGS_FILENAME
        chunks_path = self.index_dir / _CHUNKS_FILENAME
        _atomic_json(
            self.index_dir / _CONFIG_FILENAME,
            self._index_config(
                dimension=int(embeddings.shape[1]),
                row_count=len(selected),
                embeddings_sha256=_sha256_file(embeddings_path),
                chunks_sha256=_sha256_file(chunks_path),
            ),
        )
        self._chunks = selected
        self._embeddings = embeddings

    def load(self) -> None:
        """Load a previously built matrix after validating its configuration."""
        config_path = self.index_dir / _CONFIG_FILENAME
        try:
            saved = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid BGE-M3 index config: {config_path}") from exc
        if not isinstance(saved, dict):
            raise ValueError(f"invalid BGE-M3 index config: {config_path}")
        schema_version = saved.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ValueError("unsupported BGE-M3 index config schema version")
        expected = self._index_config(
            dimension=saved.get("dimension"),
            row_count=saved.get("row_count"),
            embeddings_sha256=saved.get("embeddings_sha256"),
            chunks_sha256=saved.get("chunks_sha256"),
        )
        if saved != expected:
            raise ValueError("saved BGE-M3 index config does not match requested settings")

        chunks_path = self.index_dir / _CHUNKS_FILENAME
        embeddings_path = self.index_dir / _EMBEDDINGS_FILENAME
        if _sha256_file(chunks_path) != saved["chunks_sha256"]:
            raise ValueError("saved BGE-M3 Chunk checksum does not match its config")
        if _sha256_file(embeddings_path) != saved["embeddings_sha256"]:
            raise ValueError("saved BGE-M3 embedding checksum does not match its config")
        chunks = self._load_chunks()
        try:
            embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid BGE-M3 embedding matrix: {embeddings_path}") from exc
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("BGE-M3 embedding and Chunk counts do not match")
        if embeddings.shape[0] != saved["row_count"]:
            raise ValueError("BGE-M3 embedding row count does not match its config")
        if embeddings.shape[1] != saved["dimension"]:
            raise ValueError("BGE-M3 embedding dimension does not match its config")
        if embeddings.dtype != np.float32:
            raise ValueError("BGE-M3 embedding matrix is not float32 data")
        self._validate_embedding_values(embeddings)
        self._chunks = chunks
        self._embeddings = embeddings

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return exact inner-product results with deterministic tie breaking."""
        if self._embeddings is None:
            raise RuntimeError("index is not built or loaded; call build() or load() first")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            return []
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")

        query_embedding = self._embed([query])[0]
        if query_embedding.shape != (self._embeddings.shape[1],):
            raise ValueError("BGE-M3 query embedding has an unexpected shape")
        self._validate_embedding_values(query_embedding.reshape(1, -1))
        scores = np.asarray(self._embeddings @ query_embedding, dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("BGE-M3 search produced invalid similarity scores")
        indices = np.lexsort((np.arange(scores.size), -scores))[
            : min(top_k, len(self._chunks))
        ]
        results: list[RetrievalResult] = []
        for index in indices:
            chunk = self._chunks[int(index)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    score=float(scores[int(index)]),
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    metadata=chunk.metadata,
                    source=self.name,
                )
            )
        return results

    def _embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            prompt_name=None,
            prompt="",
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is not None and not self.model_path.is_dir():
            raise FileNotFoundError(f"local BGE-M3 model does not exist: {self.model_path}")
        from sentence_transformers import SentenceTransformer

        target = str(self.model_path) if self.model_path is not None else self.model_name
        kwargs: dict[str, Any] = {
            "device": self.device,
            "local_files_only": self.local_files_only,
        }
        if self.model_path is None:
            kwargs["revision"] = self.revision
        model = SentenceTransformer(target, **kwargs)
        model.max_seq_length = self.max_length
        self._model = model

    def _index_config(
        self,
        dimension: Any,
        row_count: Any,
        embeddings_sha256: Any,
        chunks_sha256: Any,
    ) -> dict[str, Any]:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError("saved BGE-M3 dimension must be a positive integer")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count <= 0
        ):
            raise ValueError("saved BGE-M3 row count must be a positive integer")
        for name, checksum in (
            ("embeddings_sha256", embeddings_sha256),
            ("chunks_sha256", chunks_sha256),
        ):
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise ValueError(f"saved BGE-M3 {name} must be a SHA-256 digest")
        config = {
            "schema_version": 1,
            "indexer": self.name,
            "model": self.model_name,
            "revision": self.revision,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "batch_size": self.batch_size,
            "device": self.device,
            "max_length": self.max_length,
            "local_files_only": self.local_files_only,
            "include_chunk_types": (
                list(self.include_chunk_types)
                if self.include_chunk_types is not None
                else None
            ),
            "pooling": "sentence_transformers_model_config",
            "normalize_embeddings": True,
            "similarity": "exact_inner_product",
            "dtype": "float32",
            "dimension": dimension,
            "row_count": row_count,
            "embeddings_sha256": embeddings_sha256,
            "chunks_sha256": chunks_sha256,
        }
        return config

    @staticmethod
    def _validate_embedding_values(embeddings: np.ndarray) -> None:
        for start in range(0, embeddings.shape[0], _VALIDATION_BATCH_ROWS):
            block = np.asarray(embeddings[start : start + _VALIDATION_BATCH_ROWS])
            if not np.isfinite(block).all():
                raise ValueError("BGE-M3 embeddings must contain finite values")
            norms = np.linalg.norm(block, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError("BGE-M3 embeddings must be L2-normalized")

    def _save_chunks(self, chunks: list[Chunk]) -> None:
        path = self.index_dir / _CHUNKS_FILENAME
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        temporary.replace(path)

    def _load_chunks(self) -> list[Chunk]:
        path = self.index_dir / _CHUNKS_FILENAME
        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                chunks.append(Chunk(**record))
        if not chunks:
            raise ValueError(f"BGE-M3 index has no saved Chunks: {path}")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("saved BGE-M3 index contains duplicate chunk_id values")
        return chunks
