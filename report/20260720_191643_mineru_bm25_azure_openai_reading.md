# mineru + bm25_azure_openai + reading

- 実行日時: 2026-07-20T19:16:43
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_azure_openai.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_azure_openai.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7073 |
| paper_recall_macro | 0.6101 |
| paper_f1_macro | 0.5075 |
| evidence_precision_macro | 0.1333 |
| evidence_recall_macro | 0.1338 |
| evidence_f1_macro | 0.1255 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.0000 |
| table_row_f1_macro | 0.0000 |
| table_cell_accuracy_macro | 0.0000 |
| table_cell_accuracy_micro | None |

## コメント

論文レベルでは precision 0.71、recall 0.61 と一定の候補は拾えており、検索の入口は極端には悪くない印象です。一方で evidence 系は precision/recall ともに 0.13 前後とかなり低く、必要箇所の特定や抽出が主なボトルネックに見えます。下流の multiple choice、freeform、table 指標がすべて 0.0 なので、読解・回答生成または表処理のパイプラインが実質的に機能していない可能性が高いです。次は上位論文からの chunk 分割・再ランキング・evidence 抽出条件を見直し、あわせて回答生成プロンプトと表抽出の入出力不整合を重点的に確認するとよさそうです。
