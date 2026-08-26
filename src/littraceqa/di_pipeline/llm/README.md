# src/littraceqa/di_pipeline/llm/

`ReadingAgent` / `TaskFamilyClassifier` が使う LLM クライアント。
`__call__(prompt: str) -> str` の形に統一してある。

- `base.py` — `LLMClient` Protocol（このメソッド1つだけ）
- `azure_openai.py` — `AzureOpenAILLM`: 本番。Azure OpenAI の実クライアント
- `fake.py` — `FakeLLM`: テスト用。渡した応答を順番に返すだけ

`pipeline.build_agent()` が既定で `AzureOpenAILLM(reasoning_effort="medium")` を渡す。
テストは第2引数に `FakeLLM` を渡して差し替える。

## 認証

APIキー等はリポジトリ直下の `.env` から読む（コードにも yaml にも書かない）。
必要な環境変数は `.env.example` を参照。

キーが無い状態でクライアントを構築すると、その場で例外になる。**エージェントは
LLM 呼び出しを try/except で囲んでフォールバックする作り**なので、実行中に認証で
失敗すると「LLM が動いていないのに静かに劣化する」状態になってしまう。それを避けるため、
認証の失敗はパイプライン組み立て時に必ず表面化させる。
