"""Anthropic Claude API を呼び出す実 LLM クライアント。

`LLMClient` Protocol（base.py）を満たすので、agent_style の yaml で

    llm: { name: claude, params: {} }

と書けば IterativeAgent / VerifyingAgent / TaskFamilyClassifier から使える。

APIキーは環境変数 ANTHROPIC_API_KEY から読む（`ant auth login` のプロファイルや
明示的な api_key パラメータでも可。Anthropic SDK の通常の解決順に従う）。

    export ANTHROPIC_API_KEY=sk-ant-...

キーが無い状態でこのクラスを構築すると、その場で例外を投げる。エージェント側は
LLM 呼び出しを try/except で囲んでフォールバックする作りなので、実行中に例外を
投げると「LLMが動いていないのに静かに劣化する」状態になってしまう。それを避けるため、
認証の失敗は必ずパイプライン組み立て時（build_pipeline）に表面化させる。
"""

from __future__ import annotations

import anthropic

from litqa.registry import register

_SYSTEM = (
    "あなたは科学論文の検索システムの一部として動作しています。"
    "指示された出力フォーマットに厳密に従ってください。"
    "JSON を求められたら、前置きや説明を付けずに JSON だけを出力してください。"
)


@register("llm", "claude")
class ClaudeLLM:
    """Claude の Messages API を1往復だけ呼ぶクライアント。

    このパイプラインでの LLM の用途（task_family の判定、サブクエリ分解、候補論文の
    取捨選択）はどれも短い JSON を返すだけなので、ストリーミングも会話履歴も要らない。

    thinking は adaptive を既定にする。Opus 4.8 は thinking を省略すると思考なしで
    走るため、明示的に指定する必要がある。effort は思考の深さと総トークン量を決める。
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 16000,
        effort: str = "medium",
        thinking: bool = True,
        system: str = _SYSTEM,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self.system = system

        # api_key=None なら SDK が環境変数やプロファイルから解決する。
        self.client = anthropic.Anthropic(
            api_key=api_key, max_retries=max_retries, timeout=timeout
        )

        # SDK は資格情報が無くても構築だけは成功し、最初の API 呼び出しで初めて落ちる。
        # だがエージェントは LLM 呼び出しを try/except で握りつぶしてフォールバックする
        # ので、そのままでは「LLMが動いていないのに静かに劣化する」状態になる。
        # ここで検出してパイプライン組み立て時に止める。
        if not (
            self.client.api_key or self.client.auth_token or self.client.credentials
        ):
            raise RuntimeError(
                "Claude API の資格情報が見つかりません。"
                "環境変数 ANTHROPIC_API_KEY にAPIキーを設定してください:\n"
                "    export ANTHROPIC_API_KEY=sk-ant-...\n"
                "LLM を使わずに動かしたい場合は、agent_style の yaml から llm を外すか、"
                "llm: { name: fake } にしてください。"
            )

    def __call__(self, prompt: str) -> str:
        """プロンプトを投げて、応答のテキストを返す。"""
        if self.thinking:
            thinking_config: dict = {"type": "adaptive"}
        else:
            thinking_config = {"type": "disabled"}

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                thinking=thinking_config,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "Claude API の認証に失敗しました。ANTHROPIC_API_KEY を確認してください。"
            ) from exc

        # thinking ブロックが混ざるので text ブロックだけを拾う。
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
