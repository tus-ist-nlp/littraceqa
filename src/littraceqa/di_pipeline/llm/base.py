"""LLM クライアントの契約。

本番は `llm/azure_openai.py` の `AzureOpenAILLM`、テストは `llm/fake.py` の `FakeLLM`。
`ReadingAgent` はこの1メソッドしか使わないので、テストで差し替えられる。
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def __call__(self, prompt: str) -> str: ...
