"""検索エージェントが実装すべき Protocol 定義。"""

from __future__ import annotations

from typing import Protocol

from litqa.contracts import Prediction, Query


class SearchAgent(Protocol):
    def run(self, query: Query) -> Prediction: ...
