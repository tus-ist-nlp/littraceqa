"""OpenAI-compatible text/VLM client for OpenAI, vLLM and local gateways."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from littraceqa.di_pipeline.registry import register


_SYSTEM = (
    "You are the grounded reading component of a scientific-paper QA system. "
    "Use only the supplied corpus excerpts. Return JSON only when requested."
)


@register("llm", "openai_compatible")
class OpenAICompatibleLLM:
    """Small adapter around an OpenAI-compatible Chat Completions endpoint.

    The client is intentionally independent of ``src/littraceqa/azure``.  A
    local vLLM server can be selected with ``OPENAI_BASE_URL`` and
    ``OPENAI_CHAT_MODEL``; an arbitrary non-empty key is accepted by most local
    servers.  ``complete`` also accepts local image paths for multimodal models.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_completion_tokens: int = 12000,
        token_parameter: str = "max_completion_tokens",
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = True,
        system: str = _SYSTEM,
        timeout: float = 180.0,
        max_retries: int = 3,
        image_detail: str = "high",
    ) -> None:
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        model = model or os.environ.get("OPENAI_CHAT_MODEL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not model:
            raise RuntimeError(
                "OpenAI-compatible model is missing; set OPENAI_CHAT_MODEL or model="
            )
        if not api_key:
            if base_url:
                api_key = "not-needed"
            else:
                raise RuntimeError(
                    "OPENAI_API_KEY is required when OPENAI_BASE_URL is not set"
                )

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        if token_parameter not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError(
                "token_parameter must be max_completion_tokens or max_tokens"
            )
        self.token_parameter = token_parameter
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.system = system
        self.image_detail = image_detail

    def __call__(self, prompt: str) -> str:
        return self.complete(prompt)

    def complete(self, prompt: str, image_paths: list[str] | None = None) -> str:
        content: str | list[dict[str, Any]]
        if image_paths:
            content = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(image_path),
                            "detail": self.image_detail,
                        },
                    }
                )
        else:
            content = prompt

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": content},
            ],
        }
        kwargs[self.token_parameter] = self.max_completion_tokens
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
