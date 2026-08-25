# mineru + bm25_specter2_body_qwen3 + reading

- 実行日時: 2026-07-20T19:35:40
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_specter2_body_qwen3.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_specter2_body_qwen3_rerun.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.6000 |
| paper_recall_macro | 0.5924 |
| paper_f1_macro | 0.3931 |
| evidence_precision_macro | 0.0773 |
| evidence_recall_macro | 0.0429 |
| evidence_f1_macro | 0.0507 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.0000 |
| table_row_f1_macro | 0.0000 |
| table_cell_accuracy_macro | 0.0000 |
| table_cell_accuracy_micro | None |

## コメント

論文レベルの検索は precision 0.60 / recall 0.59 で最低限は取れていますが、過去実行（0.68 / 0.62）よりやや悪化しており、paper_f1 も 0.46→0.39 に下がっています。特に evidence 系は precision 0.077、recall 0.043、F1 0.051 とかなり低く、過去の 0.13 付近 থেকে大きく落ちているため、根拠抽出または本文位置合わせの不安定さが強く疑われます。その結果として multiple choice / freeform / table 系が全て 0.0 のままで、検索された論文を最終回答に結びつけられていません。次は、上位論文は取れている前提で evidence 抽出のチャンク分割・本文対象範囲・rerank 条件を見直し、クエリごとの失敗例を見て root cause を切り分けるのがよさそうです。
