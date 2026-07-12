# litqa/llm/

`IterativeAgent` / `VerifyingAgent` / `TaskFamilyClassifier` が使うLLMクライアント。
`__call__(prompt: str) -> str` の形で統一する。

- `base.py` — `LLMClient` Protocol
- `claude.py` — `ClaudeLLM`（"claude"）: Anthropic Claude API の実クライアント
- `fake.py` — `FakeLLM`（"fake"）: テスト・ドライラン用。渡した応答を順番に返すだけ

## ClaudeLLM を使う

APIキーを環境変数に設定するだけで動く。

```
export ANTHROPIC_API_KEY=sk-ant-...
```

agent_style の yaml で指定する:

```yaml
name: verifying
llm: { name: claude, params: { effort: medium } }
params: { top_k: 20, max_candidates: 15 }
```

`params` に渡せるもの: `model`（既定 `claude-opus-4-8`）, `max_tokens`, `effort`
(`low`/`medium`/`high`/`xhigh`/`max`), `thinking`, `system`, `api_key`,
`max_retries`, `timeout`。

キーが無い状態で `ClaudeLLM` を構築すると、その場で `RuntimeError` になる。
エージェントは LLM 呼び出しを try/except で囲んでフォールバックする作りなので、
実行中に認証で失敗すると「LLMが動いていないのに静かに劣化する」状態になってしまう。
それを避けるため、認証の失敗はパイプライン組み立て時（`build_pipeline`）に必ず表面化させる。
