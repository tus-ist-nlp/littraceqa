# mineru + bm25_specter2_body_qwen3_rerank + reading

- 実行日時: 2026-07-21T06:16:42
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_specter2_body_qwen3_rerank.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_specter2_body_qwen3_rerank_answergen.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.6329 |
| paper_recall_macro | 0.6268 |
| paper_f1_macro | 0.4640 |
| evidence_precision_macro | 0.1167 |
| evidence_recall_macro | 0.1000 |
| evidence_f1_macro | 0.0970 |
| multiple_choice_accuracy | 0.4878 |
| freeform_exact_match | 0.0385 |
| table_row_f1_macro | 0.4197 |
| table_cell_accuracy_macro | 0.2011 |
| table_cell_accuracy_micro | 0.2963 |

## コメント

今回の結果は、paper系は precision が 0.709→0.633 に下がり、recall は 0.622→0.627 と微増したものの、paper_f1 は 0.492→0.464 でやや悪化しており、候補の広がりと引き換えにノイズが増えた印象です。 一方で evidence 系は低水準ながら前回より少し改善し、multiple choice accuracy も 0.488、table 系指標も 0 から大きく改善しているため、下流の読解・構造化は前回よりかなり安定しています。 気になるのは evidence_f1 が 0.097、freeform_exact_match が 0.038 と依然かなり低く、正しい論文に当たっても根拠抽出や厳密な記述生成が弱い点です。 次は rerank 上位件数や evidence 抽出条件を見直して paper precision を戻しつつ、freeform 生成のプロンプト/出力制約や表抽出の後処理を調整して、根拠整合性と記述精度を重点的に改善するとよさそうです。
