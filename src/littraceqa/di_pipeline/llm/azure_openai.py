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
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import openai
from openai import AzureOpenAI

from littraceqa.di_pipeline.registry import register

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

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )

    def __call__(self, prompt: str) -> str:
        """プロンプトを投げて、応答のテキストを返す。"""
        return self.complete(prompt)

    def complete(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> str:
        """テキストと任意のローカル画像を1往復で処理する。"""
        return str(
            self.complete_with_metadata(prompt, image_paths=image_paths)["text"]
        )

    def complete_with_metadata(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> dict[str, Any]:
        """応答本文に加え、再現・エラー解析用の利用量等を返す。

        APIキー、endpoint、送信画像のbase64は返さない。呼び出し側はこの辞書を
        checkpointへ安全に保存できる。
        """
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

        kwargs: dict = {
            "model": self.deployment,  # Azure ではデプロイ名を渡す
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort

        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
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
        return {
            "text": response.choices[0].message.content or "",
            "request_id": (
                getattr(response, "_request_id", None)
                or getattr(response, "id", None)
            ),
            "model": getattr(response, "model", None),
            "deployment": self.deployment,
            "usage": usage_payload,
            "latency_seconds": time.monotonic() - started,
        }


def _image_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
