# mineru + bm25_qwen3 + reading

- 実行日時: 2026-07-21T05:41:03
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_qwen3_answergen.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.5645 |
| paper_recall_macro | 0.6197 |
| paper_f1_macro | 0.4009 |
| evidence_precision_macro | 0.1036 |
| evidence_recall_macro | 0.0520 |
| evidence_f1_macro | 0.0591 |
| multiple_choice_accuracy | 0.4390 |
| freeform_exact_match | 0.0769 |
| table_row_f1_macro | 0.4758 |
| table_cell_accuracy_macro | 0.2409 |
| table_cell_accuracy_micro | 0.2963 |

## コメント

今回の結果は、論文レベルでは paper_recall がやや改善した一方で、paper_precision と paper_f1 は前回より少し低下しており、候補を広めに拾えている反面ノイズが増えていそうです。evidence 系は前回より微増していますが、precision/recall/F1 ともに依然かなり低く、根拠抽出が全体のボトルネックに見えます。multiple_choice_accuracy が 0.439、表関連指標も 0 から有意に改善しており、前回は壊れていた下流処理や評価対象のカバレッジが今回は機能している点は良いです。ただし freeform_exact_match は 0.077 と低いため、検索後の読解・生成段での回答整形や根拠紐付けが弱い可能性があります。次は BM25 の上位件数やチャンク粒度を見直して evidence recall を上げつつ、reading 設定側で根拠抽出の厳格化・回答フォーマット制約を試すとよさそうです。
