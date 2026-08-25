# mineru + bm25_qwen3 + reading

- 実行日時: 2026-07-20T23:30:08
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_qwen3.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.6143 |
| paper_recall_macro | 0.5884 |
| paper_f1_macro | 0.4298 |
| evidence_precision_macro | 0.0818 |
| evidence_recall_macro | 0.0475 |
| evidence_f1_macro | 0.0542 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.0000 |
| table_row_f1_macro | 0.0000 |
| table_cell_accuracy_macro | 0.0000 |
| table_cell_accuracy_micro | None |

## コメント

論文レベルでは precision 0.614、recall 0.588 と一定の取りこぼしはあるものの、候補論文の絞り込み自体は大きく崩れていません。 一方で paper_f1 が 0.430 に留まり、特に evidence 系は precision 0.082、recall 0.047、F1 0.054 と非常に低く、論文は拾えても根拠箇所の特定にほぼ失敗しているのが最大の課題です。 その結果として multiple choice、freeform、table 系の下流指標がすべて 0.0 で、reader/grounding 側か、検索粒度と回答生成の接続に深刻なボトルネックがある可能性が高いです。 次は evidence 抽出用の chunking・再ランキング・reader 設定を優先的に見直し、あわせて retrieval 上位件数やクエリ展開を調整して、論文ヒットから根拠ヒットへの変換率を確認するとよさそうです。
