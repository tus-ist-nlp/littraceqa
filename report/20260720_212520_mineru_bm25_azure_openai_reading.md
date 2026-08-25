# mineru + bm25_azure_openai + reading

- 実行日時: 2026-07-20T21:25:20
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_azure_openai.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_azure_openai.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7709 |
| paper_recall_macro | 0.6086 |
| paper_f1_macro | 0.5639 |
| evidence_precision_macro | 0.1833 |
| evidence_recall_macro | 0.1520 |
| evidence_f1_macro | 0.1579 |
| multiple_choice_accuracy | 0.5854 |
| freeform_exact_match | 0.1538 |
| table_row_f1_macro | 0.4332 |
| table_cell_accuracy_macro | 0.1302 |
| table_cell_accuracy_micro | 0.1852 |

## コメント

前回と比べると全指標で大きく改善しており、特に multiple_choice_accuracy が 0.0→0.585、table 系も 0 から立ち上がっているため、パイプライン全体としては明確に前進しています。paper_precision 0.771 は良好ですが、paper_recall 0.609 に対して paper_f1 0.564 なので、関連論文の取りこぼしはまだやや気になります。特に evidence 系は precision/recall/f1 が 0.15〜0.18 と低く、論文は当てられても根拠抽出が弱い点がボトルネックに見えます。次は上位論文からの evidence 抽出条件や chunking・rerank の見直し、表形式出力ではセル単位整形と表構造復元の改善を試すとよさそうです。
