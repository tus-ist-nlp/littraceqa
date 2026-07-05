# litqa/llm/

`IterativeAgent`等がクエリ分解・記述生成に使うLLMクライアント。`__call__(prompt: str) -> str` の形で統一する。

- `base.py` — `LLMClient` Protocol。実際にClaude等を使うには、この形を満たすクライアントを別途実装しAPIキー等を設定する必要がある（詳細はdocstring参照）
- `fake.py` — `FakeLLM`（"fake"）: テスト・ドライラン用に、あらかじめ渡した応答を順番に返すだけの固定応答クライアント
