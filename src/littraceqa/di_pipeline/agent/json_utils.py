"""LLM 応答から JSON を安全に取り出すヘルパ。

複数のエージェントと task_family 推定器が共有するため、独立したモジュールに置く
（agent 同士が互いを import して循環参照になるのを避ける）。
"""

from __future__ import annotations

import json
import re

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_object(text: str) -> dict | None:
    """LLM の出力から JSON オブジェクトを安全に取り出す。失敗したら None。"""
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        candidate = text
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
