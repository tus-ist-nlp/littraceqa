"""The LLM client that calls Azure OpenAI.

Satisfies the `LLMClient` Protocol (base.py). It is constructed in exactly one
place, `pipeline.build_agent()`, and handed to ReadingAgent from there.

Configuration comes from .env — **never from code or config**:

    AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_API_VERSION=2025-04-01-preview
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4     # Azure is called by *deployment* name

Unlike OpenAI proper, Azure takes **the deployment name you chose**, not a model
name, in `model`.

What the deployment 'gpt-5.4' (really gpt-5.4-2026-03-05) actually accepts,
measured:

    max_tokens              -> 400 Unsupported parameter (unusable)
    max_completion_tokens   -> OK
    temperature             -> OK
    response_format         -> OK (json_object forces JSON)
    reasoning_effort        -> OK

Every use in this pipeline (splitting the question into subqueries, reading the
candidates) returns nothing but a short JSON object, so `response_format=
json_object` is on by default. That is what keeps the agent's JSON parsing from
failing — and a parse failure there is silent, it just falls back.

**Missing credentials raise here, at construction.** The agent wraps every LLM call
in try/except and falls back, so an exception raised mid-run would show up as
quietly degraded retrieval rather than an error. Surfacing it while the pipeline is
being assembled is what makes it impossible to miss.
"""

from __future__ import annotations

import os

import openai
from openai import AzureOpenAI


# Kept in Japanese deliberately: this is the prompt the measured system ran with,
# and rewording it would change the model's output.
_SYSTEM = (
    "あなたは科学論文の検索システムの一部として動作しています。"
    "指示された出力フォーマットに厳密に従ってください。"
    "JSON を求められたら、前置きや説明を付けずに JSON だけを出力してください。"
)

_REQUIRED = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
)


class AzureOpenAILLM:
    """One round-trip to Azure OpenAI's Chat Completions, and nothing else."""

    def __init__(
        self,
        deployment: str | None = None,
        max_completion_tokens: int = 16000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = True,
        system: str = _SYSTEM,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        deployment = deployment or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")

        missing = [
            name
            for name, value in zip(
                _REQUIRED, (endpoint, api_key, api_version, deployment)
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Azure OpenAI is not configured; missing: " + ", ".join(missing) + "\n"
                "Put these in .env (never in code):\n"
                "    AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com\n"
                "    AZURE_OPENAI_API_KEY=...\n"
                "    AZURE_OPENAI_API_VERSION=2025-04-01-preview\n"
                "    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4\n"
                "Only tests can run without an LLM (hand ReadingAgent the FakeLLM "
                "from llm/fake.py directly); a real search requires one."
            )

        self.deployment = deployment
        # gpt-5.4 rejects max_tokens (400 Unsupported parameter); measured.
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.system = system

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    def __call__(self, prompt: str) -> str:
        """Send the prompt, return the response text."""
        kwargs: dict = {
            "model": self.deployment,  # Azure takes the deployment name here
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                "Azure OpenAI rejected the credentials. Check AZURE_OPENAI_API_KEY."
            ) from exc

        return response.choices[0].message.content or ""
