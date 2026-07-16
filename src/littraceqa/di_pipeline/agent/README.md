# src/littraceqa/di_pipeline/agent/

`Query` を受け取り `Prediction` を返す検索エージェント。`retriever`（と必要なら`llm`）を注入して使う。

- `base.py` — `SearchAgent` Protocol（`run(query) -> Prediction`）
- `simple.py` — `SimpleAgent`: 1回検索して終わり。LLM不使用（最終提出は順位カットオフ）
- `iterative.py` — `IterativeAgent`: 検索結果を見てから足りない分をLLMで再分解し、複数回検索する
- `verifying.py` — `VerifyingAgent`: 1回検索した上位候補をLLMに提示し、順位カットオフではなく
  内容判定で最終提出論文を選ぶ（正解論文が固定カットオフより下位でも拾える）。LLM出力が
  壊れている・候補外のIDしか返らない場合は順位カットオフにフォールバックする
