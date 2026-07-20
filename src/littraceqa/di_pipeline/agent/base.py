"""検索エージェントが実装すべき Protocol 定義。"""

from __future__ import annotations

from typing import Protocol

from littraceqa.di_pipeline.contracts import Prediction, Query


class SearchAgent(Protocol):
    def run(self, query: Query) -> Prediction: ...
