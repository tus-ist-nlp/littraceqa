"""テスト・ドライラン用の固定応答 LLM クライアント。"""

from __future__ import annotations

from littraceqa.di_pipeline.registry import register


@register("llm", "fake")
class FakeLLM:
    """呼び出すたびに ``responses`` を順番に返すフェイク LLM クライアント。

    ``responses`` を使い切った後は最後の応答を返し続ける。
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
