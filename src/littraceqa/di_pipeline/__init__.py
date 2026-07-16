"""LitTraceQA パイプラインのパッケージ。"""

from littraceqa.di_pipeline.contracts import (
    Answer,
    Chunk,
    Evidence,
    EvidenceLocator,
    PaperMeta,
    Prediction,
    Query,
    RetrievalResult,
)
from littraceqa.di_pipeline.registry import build, list_registered, register

__all__ = [
    "Answer",
    "Chunk",
    "Evidence",
    "EvidenceLocator",
    "PaperMeta",
    "Prediction",
    "Query",
    "RetrievalResult",
    "build",
    "list_registered",
    "register",
]
