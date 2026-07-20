"""LLM クライアントが実装すべき Protocol 定義。

この Protocol を満たすクラスを src/littraceqa/di_pipeline/llm/ 配下に追加すれば、任意の LLM を
DI で差し替えられる（例: src/littraceqa/di_pipeline/llm/fake.py の FakeLLM）。
ただし Claude や GPT など実際の API を呼び出す場合は、この Protocol の定義だけでは
不十分で、以下の実装が別途必要になる:
  - API キー（例: ANTHROPIC_API_KEY）を読み込む処理
  - 対応する SDK（例: anthropic パッケージ）を pyproject.toml に追加
  - この Protocol を満たす __call__ を持つクライアントクラスを実装し、
    @register("llm", "...") で登録する
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def __call__(self, prompt: str) -> str: ...
