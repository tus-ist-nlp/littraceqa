"""The LLM client contract.

Production uses `AzureOpenAILLM` (llm/azure_openai.py); tests use `FakeLLM`
(llm/fake.py). **`ReadingAgent` calls nothing but this one method**, which is what
makes the substitution possible.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def __call__(self, prompt: str) -> str: ...
