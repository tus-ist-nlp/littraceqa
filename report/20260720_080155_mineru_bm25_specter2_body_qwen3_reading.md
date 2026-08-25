# mineru + bm25_specter2_body_qwen3 + reading

- 実行日時: 2026-07-20T08:01:55
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_specter2_body_qwen3.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_specter2_body_qwen3.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.6845 |
| paper_recall_macro | 0.6157 |
| paper_f1_macro | 0.4552 |
| evidence_precision_macro | 0.1636 |
| evidence_recall_macro | 0.1227 |
| evidence_f1_macro | 0.1309 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.0000 |
| table_row_f1_macro | 0.0000 |
| table_cell_accuracy_macro | 0.0000 |
| table_cell_accuracy_micro | None |

## コメント

論文レベルでは precision 0.685、recall 0.616 と候補論文の拾い上げ自体はそこそこできていますが、paper_f1 が 0.455 にとどまっておりバランスはまだ改善余地があります。特に evidence 系が precision 0.164、recall 0.123、f1 0.131 とかなり低く、論文は見つけても根拠抽出・回答生成に十分つながっていない点が気になります。multiple_choice、freeform、table 系がすべて 0.0 なので、reader/agent 側の抽出・整形処理か、評価対象形式へのマッピングに不具合やミスマッチがないかをまず確認したいです。次は retrieval 上位件数や body 重み付けの見直しに加え、evidence 抽出プロンプトや chunking 設定、表形式出力のパーサ周りを優先的に点検するとよさそうです。
