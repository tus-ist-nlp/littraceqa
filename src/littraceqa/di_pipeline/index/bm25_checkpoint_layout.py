"""Where a resumable build keeps each kind of checkpoint, and under what settings.

Path names are part of the on-disk contract: an existing generation directory
must keep resolving to the same files, so the formats here are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from littraceqa.di_pipeline.index.bm25_checkpoint_format import (
    MANIFEST_NAME,
    PARTS_DIR_NAME,
    SCORES_DIR_NAME,
)


@dataclass(frozen=True)
class CheckpointLayout:
    """Resolve every checkpoint path from one generation directory."""

    generation_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.generation_dir / MANIFEST_NAME

    @property
    def parts_dir(self) -> Path:
        return self.generation_dir / PARTS_DIR_NAME

    @property
    def scores_dir(self) -> Path:
        return self.generation_dir / SCORES_DIR_NAME

    def part_path(self, index: int) -> Path:
        return self.parts_dir / f"{index:08d}.json"

    def part_meta_path(self, index: int) -> Path:
        return self.parts_dir / f"{index:08d}.meta.json"

    def score_path(self, index: int) -> Path:
        return self.scores_dir / f"{index:08d}.npz"

    def score_meta_path(self, index: int) -> Path:
        return self.scores_dir / f"{index:08d}.meta.json"


@dataclass(frozen=True)
class BM25Parameters:
    """The scoring parameters that decide what a persisted score means."""

    method: str
    idf_method: str
    k1: float
    b: float
    delta: float
    dtype: np.dtype
    int_dtype: np.dtype


@dataclass(frozen=True)
class GlobalStatistics:
    """Corpus-wide counts every score shard must be computed against.

    Shards are scored with these values rather than shard-local ones, which is
    what makes the merged result identical to a single global BM25 index.
    """

    vocab: dict[str, int]
    document_frequencies: np.ndarray
    num_documents: int
    total_document_length: int
    average_document_length: np.float64
    nonzero_scores: int
    signature: str
    record: dict[str, Any]
