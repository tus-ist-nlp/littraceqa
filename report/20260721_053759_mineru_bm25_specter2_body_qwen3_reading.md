# mineru + bm25_specter2_body_qwen3 + reading

- 実行日時: 2026-07-21T05:37:59
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_specter2_body_qwen3.yaml`
- agent: `configs/agent_style/reading.yaml`
- queries: `data/validation_inputs.jsonl` (55件, production_input=True)
- output: `predictions_bm25_specter2_body_qwen3_answergen.jsonl`

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.5741 |
| paper_recall_macro | 0.6131 |
| paper_f1_macro | 0.3867 |
| evidence_precision_macro | 0.0818 |
| evidence_recall_macro | 0.0409 |
| evidence_f1_macro | 0.0491 |
| multiple_choice_accuracy | 0.5366 |
| freeform_exact_match | 0.1154 |
| table_row_f1_macro | 0.3909 |
| table_cell_accuracy_macro | 0.1272 |
| table_cell_accuracy_micro | 0.1111 |

## コメント

今回の結果は、multiple_choice_accuracy=0.537、table_row_f1=0.391 まで出ており、過去2回でゼロだった回答・表系指標が大きく改善している点は良いです。一方で、paper_f1=0.387 は過去最高には届かず、evidence_f1=0.049 も 7/20 08:01 の 0.131 から大きく低下しており、根拠抽出の弱さが依然としてボトルネックに見えます。特に paper の recall は 0.613 と悪くないのに evidence recall が 0.041 と低いため、検索された論文内の該当箇所特定や引用生成の段で取りこぼしている可能性が高いです。次は、上位論文数や evidence 抽出窓の拡張、body 重みの見直し、BM25 と SPECTER2 のスコア配合調整を試し、paper_f1 を維持しつつ evidence 系を戻せるか確認するとよさそうです。
