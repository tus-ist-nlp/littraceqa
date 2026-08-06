"""Azure OpenAI を呼び出す LLM クライアント。

`LLMClient` Protocol（base.py）を満たすので、agent_style の yaml で

    llm: { name: azure_openai, params: {} }

と書けば IterativeAgent / VerifyingAgent / ReadingAgent / TaskFamilyClassifier から使える。

設定は .env から読む（値はコードに書かない）:

    AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_API_VERSION=2025-04-01-preview
    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4     # Azure は「デプロイ名」で呼ぶ

Azure は本家 OpenAI と違い、モデル名ではなく**自分で付けたデプロイ名**を model に渡す。

デプロイ 'gpt-5.4'（実体 gpt-5.4-2026-03-05）で実測した制約:

    max_tokens              -> 400 Unsupported parameter（使えない）
    max_completion_tokens   -> OK
    temperature             -> OK
    response_format         -> OK（json_object で JSON を強制できる）
    reasoning_effort        -> OK

このパイプラインでの用途（task_family 判定・サブクエリ分解・候補論文の選定）は
どれも短い JSON を返すだけなので、response_format=json_object を既定で有効にする。
これでエージェント側の JSON パース失敗（= 静かにフォールバック）を大幅に減らせる。

資格情報が無い状態でこのクラスを構築すると、その場で例外を投げる。エージェント側は
LLM 呼び出しを try/except で囲んでフォールバックする作りなので、実行中に例外を投げると
「LLMが動いていないのに静かに劣化する」状態になる。それを避けるため、設定の不足は
必ずパイプライン組み立て時（build_pipeline）に表面化させる。
"""

from __future__ import annotations

import base64
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import openai
from openai import AzureOpenAI

from littraceqa.di_pipeline.registry import register
from littraceqa.mineru_record import (
    MAX_AOAI_IMAGES_PER_REQUEST,
    MAX_IMAGE_BYTES,
    validate_image_bytes,
    validate_image_file,
)

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

_RATE_LIMIT_HEADER_NAMES = (
    "retry-after-ms",
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

# Empirically calibrated from all 766 v18 calls: these values reserve about
# 44.6k tokens for the observed average 40.9k prompt-plus-output request. They
# intentionally optimize steady-state throughput rather than upper-bound every
# individual tokenizer/image outlier. The deployment margin and outer 429 AIMD
# are the safety net for rare underestimation. Image allowance is independent
# of JPEG/PNG byte size because provider-side tiling can differ from compression.
_ESTIMATED_TEXT_CHARS_PER_TOKEN = 3.2
_ESTIMATED_HIGH_DETAIL_IMAGE_TOKENS = 512


@register("llm", "azure_openai")
class AzureOpenAILLM:
    """Azure OpenAI の Chat Completions を1往復だけ呼ぶクライアント。"""

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
        min_request_interval_seconds: float = 0.0,
        target_tpm: float | None = None,
    ):
        try:
            request_interval = float(min_request_interval_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "min_request_interval_seconds must be a finite number >= 0"
            ) from exc
        if not math.isfinite(request_interval) or request_interval < 0:
            raise ValueError(
                "min_request_interval_seconds must be a finite number >= 0"
            )

        if isinstance(target_tpm, bool):
            raise ValueError("target_tpm must be a finite number > 0")
        if target_tpm is None:
            parsed_target_tpm = None
        else:
            try:
                parsed_target_tpm = float(target_tpm)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "target_tpm must be a finite number > 0"
                ) from exc
            if not math.isfinite(parsed_target_tpm) or parsed_target_tpm <= 0:
                raise ValueError("target_tpm must be a finite number > 0")

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
                "Azure OpenAI の設定が足りません: " + ", ".join(missing) + "\n"
                ".env に次を書いてください（値はコードに書かない）:\n"
                "    AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com\n"
                "    AZURE_OPENAI_API_KEY=...\n"
                "    AZURE_OPENAI_API_VERSION=2025-04-01-preview\n"
                "    AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4\n"
                "LLM を使わずに動かすなら agent_style の yaml から llm を外してください。"
            )

        self.deployment = deployment
        # gpt-5.4 は max_tokens を受け付けない（400 Unsupported parameter）。実測済み。
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.system = system
        self.min_request_interval_seconds = request_interval
        self.target_tpm = parsed_target_tpm
        # One reader instance is shared by every worker thread. Calls acquire
        # this lock only while paying for their launch slot; it is always
        # released before the provider request, so slow AOAI calls still overlap.
        # Keeping the wait inside the lock also preserves arrival order for
        # differently-sized prompts: a long request cannot have its reservation
        # overtaken by a later short request.
        self._request_slot_lock = threading.Lock()
        self._last_request_launch_at: float | None = None

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    def _estimate_reserved_tokens(
        self,
        *,
        prompt: str,
        image_count: int,
        max_completion_tokens: int,
    ) -> int:
        """Estimate AOAI's TPM reservation using measured workload ratios."""

        text_characters = len(self.system) + len(prompt)
        text_tokens = math.ceil(
            text_characters / _ESTIMATED_TEXT_CHARS_PER_TOKEN
        )
        return (
            text_tokens
            + max_completion_tokens
            + image_count * _ESTIMATED_HIGH_DETAIL_IMAGE_TOKENS
        )

    def _pace_provider_invocation(
        self,
        *,
        prompt: str,
        image_count: int,
        max_completion_tokens: int,
    ) -> tuple[int | None, float]:
        """Reserve and wait for one launch slot shared by all client threads.

        With ``target_tpm`` enabled, each call pays its own estimated token cost
        *before* launch.  This pre-paid schedule is important for mixed request
        sizes: charging after launch lets a long prompt follow a short prompt's
        tiny interval and creates a token burst.

        Without ``target_tpm`` the historical behavior is retained: the first
        call launches immediately and later calls use only the fixed floor.
        """

        reserved_tokens: int | None = None
        token_interval = 0.0
        if self.target_tpm is not None:
            reserved_tokens = self._estimate_reserved_tokens(
                prompt=prompt,
                image_count=image_count,
                max_completion_tokens=max_completion_tokens,
            )
            token_interval = reserved_tokens * 60.0 / self.target_tpm
        interval = max(self.min_request_interval_seconds, token_interval)
        if interval == 0:
            return reserved_tokens, 0.0

        with self._request_slot_lock:
            now = time.monotonic()
            if self._last_request_launch_at is None:
                # Token-aware calls pre-pay their first reservation. Fixed-only
                # configurations keep the old immediate-first-call behavior.
                scheduled_at = now + (
                    interval if reserved_tokens is not None else 0.0
                )
            else:
                scheduled_at = max(
                    now,
                    self._last_request_launch_at + interval,
                )
            while True:
                delay = scheduled_at - time.monotonic()
                if delay <= 0:
                    break
                time.sleep(delay)
            self._last_request_launch_at = time.monotonic()

        return reserved_tokens, interval

    def __call__(self, prompt: str) -> str:
        """プロンプトを投げて、応答のテキストを返す。"""
        return self.complete(prompt)

    def complete(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        *,
        max_completion_tokens: int | None = None,
    ) -> str:
        """テキストと任意のローカル画像を1往復で処理する。"""
        return str(
            self.complete_with_metadata(
                prompt,
                image_paths=image_paths,
                max_completion_tokens=max_completion_tokens,
            )["text"]
        )

    def complete_with_metadata(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        *,
        max_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        """応答本文に加え、再現・エラー解析用の利用量等を返す。

        APIキー、endpoint、送信画像のbase64は返さない。呼び出し側はこの辞書を
        checkpointへ安全に保存できる。
        """
        if image_paths and len(image_paths) > MAX_AOAI_IMAGES_PER_REQUEST:
            raise ValueError(
                "Azure OpenAI accepts at most "
                f"{MAX_AOAI_IMAGES_PER_REQUEST} images per request"
            )
        content: str | list[dict[str, Any]]
        if image_paths:
            content = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(image_path),
                            "detail": "high",
                        },
                    }
                )
        else:
            content = prompt

        completion_limit = (
            self.max_completion_tokens
            if max_completion_tokens is None
            else max_completion_tokens
        )
        if isinstance(completion_limit, bool) or not isinstance(
            completion_limit, int
        ) or completion_limit < 1:
            raise ValueError("max_completion_tokens must be a positive integer")

        kwargs: dict = {
            "model": self.deployment,  # Azure ではデプロイ名を渡す
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": completion_limit,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        reserved_tokens, launch_interval = self._pace_provider_invocation(
            prompt=prompt,
            image_count=len(image_paths or ()),
            max_completion_tokens=completion_limit,
        )
        started = time.monotonic()
        try:
            raw_creator = getattr(
                self.client.chat.completions, "with_raw_response", None
            )
            if raw_creator is None:
                # Lightweight test doubles and older compatible adapters may
                # expose only the parsed-response method.
                response = self.client.chat.completions.create(**kwargs)
                response_headers = None
            else:
                raw_response = raw_creator.create(**kwargs)
                response_headers = getattr(raw_response, "headers", None)
                response = raw_response.parse()
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                "Azure OpenAI の認証に失敗しました。AZURE_OPENAI_API_KEY を確認してください。"
            ) from exc

        usage = getattr(response, "usage", None)
        if usage is None:
            usage_payload = None
        elif hasattr(usage, "model_dump"):
            usage_payload = usage.model_dump(mode="json")
        else:
            usage_payload = {
                key: getattr(usage, key, None)
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
            }
        choice = response.choices[0]
        result = {
            "text": choice.message.content or "",
            "request_id": (
                getattr(response, "_request_id", None)
                or getattr(response, "id", None)
            ),
            "model": getattr(response, "model", None),
            "deployment": self.deployment,
            "usage": usage_payload,
            "latency_seconds": time.monotonic() - started,
            "max_completion_tokens": completion_limit,
        }
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            result["finish_reason"] = str(finish_reason)
        rate_limit = _safe_rate_limit_headers(response_headers)
        if rate_limit:
            result["rate_limit"] = rate_limit
        if reserved_tokens is not None:
            result["estimated_reserved_tokens"] = reserved_tokens
            result["launch_interval_seconds"] = launch_interval
            result["target_tpm"] = self.target_tpm
        return result


def _safe_rate_limit_headers(headers: Any) -> dict[str, str]:
    """Return only non-secret provider quota telemetry with bounded values."""

    if headers is None:
        return {}
    safe: dict[str, str] = {}
    for name in _RATE_LIMIT_HEADER_NAMES:
        try:
            value = headers.get(name)
        except (AttributeError, TypeError):
            return {}
        if value is None:
            continue
        rendered = str(value).strip()
        if rendered:
            safe[name] = rendered[:64]
    return safe


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    # Check size and structure before allocating the payload. Read at most one
    # byte over the cap so a file swapped/grown after stat cannot cause an
    # unbounded allocation. Validate the actual bytes again to close the normal
    # stat/read race and derive MIME from content rather than an extension.
    validate_image_file(path)
    with path.open("rb") as handle:
        payload = handle.read(MAX_IMAGE_BYTES + 1)
    mime_type = validate_image_bytes(payload, source=str(path))
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
