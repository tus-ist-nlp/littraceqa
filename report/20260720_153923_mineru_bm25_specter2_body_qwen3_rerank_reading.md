# mineru + bm25_specter2_body_qwen3_rerank + reading

- 実行日時: 2026-07-20T15:39:23
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_specter2_body_qwen3_rerank.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_specter2_body_qwen3_rerank.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7091 |
| paper_recall_macro | 0.6222 |
| paper_f1_macro | 0.4920 |
| evidence_precision_macro | 0.1152 |
| evidence_recall_macro | 0.0838 |
| evidence_f1_macro | 0.0903 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.0000 |
| table_row_f1_macro | 0.0000 |
| table_cell_accuracy_macro | 0.0000 |
| table_cell_accuracy_micro | None |

## コメント

論文単位では precision 0.71、recall 0.62 と候補論文の当たり自体はそこそこ取れていますが、paper F1 が 0.49 に留まっておりバランス改善の余地があります。特に evidence 系は precision 0.12、recall 0.084、F1 0.090 とかなり低く、関連論文は拾えても根拠箇所の特定にほぼ失敗しているのが最大の課題です。下流タスクも multiple choice / freeform / table 系がすべて 0.0 なので、reading 段での抽出品質か、evidence から回答生成への接続に重大なボトルネックがある可能性が高いです。次は rerank 後の上位文書に対する evidence 抽出範囲の拡張、chunking や body フィールド依存の見直し、また reading agent のプロンプト・出力形式を点検して、まず evidence recall を優先的に改善するとよさそうです。
