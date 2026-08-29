"""Pull a JSON object out of an LLM response without trusting its shape.

Several places in the reading loop need this, so it lives in a module of its own
rather than on one of them (importing each other would make the agent modules
circular).
"""

from __future__ import annotations

import json
import re

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_object(text: str) -> dict | None:
    """Extract a JSON object from an LLM response; None if there is none."""
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
