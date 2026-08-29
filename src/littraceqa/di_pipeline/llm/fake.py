"""A canned-response LLM client, for tests and dry runs."""

from __future__ import annotations



class FakeLLM:
    """Returns ``responses`` one per call, in order.

    Once ``responses`` runs out it keeps returning the last one, so a test only has
    to spell out the calls it actually cares about.
    """

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [""]
        self.calls: list[str] = []
        self._i = 0

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        response = self.responses[min(self._i, len(self.responses) - 1)]
        self._i += 1
        return response
