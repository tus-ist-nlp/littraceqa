# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_fat

- 実行日時: 2026-08-03T22:22:40
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_fat.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_fat.jsonl`
- git: `a0206042354c`

## 設定（この実行時の実際の値）

| パラメータ | 値 |
|---|---|
| per_index_k | `100` |
| pool_k | `200` |
| indexers | `["bm25s", "faiss_qwen3"]` |
| fuser | `"rrf"` |
| fuser_k | `60` |
| fuser_weights | `{"bm25s": 1.0, "faiss_qwen3": 1.0}` |
| reranker | `"qwen3"` |
| reranker_model | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_devices | `"cuda:1,cuda:2"` |
| reranker_fp16 | `true` |
| reranker_max_batch_tokens | `2048` |
| reranker_batch_size | `4` |
| reranker_max_tokens | `2048` |
| reranker_device | `"cuda"` |
| reranker_instruction | `"Given a scientific question, retrieve passages from research papers that help identify or support the answer"` |
| reranker_compile | `true` |
| agent | `"reading"` |
| agent_llm | `"azure_openai"` |
| agent_max_steps | `3` |
| agent_retrieve_top_k | `100` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.6951 |
| paper_recall_macro | 0.6212 |
| paper_f1_macro | 0.5223 |
| candidate_recall_at1_single_macro | 0.9231 |
| candidate_recall_at5_single_macro | 0.9615 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.1935 |
| candidate_recall_at5_multi_macro | 0.3898 |
| candidate_recall_at10_multi_macro | 0.4770 |
| candidate_recall_at20_multi_macro | 0.5661 |
| candidate_recall_at50_multi_macro | 0.6638 |
| candidate_recall_at70_multi_macro | 0.6638 |
| candidate_recall_at1_total_macro | 0.5384 |
| candidate_recall_at5_total_macro | 0.6601 |
| candidate_recall_at10_total_macro | 0.7242 |
| candidate_recall_at20_total_macro | 0.7712 |
| candidate_recall_at50_total_macro | 0.8227 |
| candidate_recall_at70_total_macro | 0.8227 |
| evidence_candidate_recall_at1_single_macro | 0.9231 |
| evidence_candidate_recall_at5_single_macro | 0.9615 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3716 |
| evidence_candidate_recall_at5_multi_macro | 0.5508 |
| evidence_candidate_recall_at10_multi_macro | 0.6322 |
| evidence_candidate_recall_at20_multi_macro | 0.7155 |
| evidence_candidate_recall_at50_multi_macro | 0.8017 |
| evidence_candidate_recall_at70_multi_macro | 0.8017 |
| evidence_candidate_recall_at1_total_macro | 0.6323 |
| evidence_candidate_recall_at5_total_macro | 0.7449 |
| evidence_candidate_recall_at10_total_macro | 0.8061 |
| evidence_candidate_recall_at20_total_macro | 0.8500 |
| evidence_candidate_recall_at50_total_macro | 0.8955 |
| evidence_candidate_recall_at70_total_macro | 0.8955 |
| evidence_candidate_recall_by_backed_at1_single_macro | 0.9375 |
| evidence_candidate_recall_by_backed_at5_single_macro | 0.9688 |
| evidence_candidate_recall_by_backed_at10_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at20_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at50_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at70_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at1_multi_macro | 0.2077 |
| evidence_candidate_recall_by_backed_at5_multi_macro | 0.4336 |
| evidence_candidate_recall_by_backed_at10_multi_macro | 0.5362 |
| evidence_candidate_recall_by_backed_at20_multi_macro | 0.6413 |
| evidence_candidate_recall_by_backed_at50_multi_macro | 0.7500 |
| evidence_candidate_recall_by_backed_at70_multi_macro | 0.7500 |
| evidence_candidate_recall_by_backed_at1_total_macro | 0.6323 |
| evidence_candidate_recall_by_backed_at5_total_macro | 0.7449 |
| evidence_candidate_recall_by_backed_at10_total_macro | 0.8061 |
| evidence_candidate_recall_by_backed_at20_total_macro | 0.8500 |
| evidence_candidate_recall_by_backed_at50_total_macro | 0.8955 |
| evidence_candidate_recall_by_backed_at70_total_macro | 0.8955 |
| evidence_precision_macro | 0.1870 |
| evidence_recall_macro | 0.1626 |
| evidence_f1_macro | 0.1627 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.2308 |
| table_row_f1_macro | 0.4859 |
| table_cell_accuracy_macro | 0.3420 |
| table_cell_accuracy_micro | 0.4444 |

## コメント

候補回収はかなり良く、single では candidate/evidence ともに@10でほぼ 1.0、total でも evidence_candidate_recall@10=0.81 と上流検索は強そうです。一方で最終成果物は弱く、paper_f1=0.52 に対して evidence_f1=0.16、multiple_choice_accuracy=0.0 で、取得した候補を最終的な根拠・回答に落とし込む段で大きく性能を落としている印象です。特に multi 系の再現率が低め（candidate_recall@10=0.48、evidence_candidate_recall@10=0.63）なので、複数文献・複数根拠を要するクエリへの対応が課題に見えます。次は reranker/reader の出力件数や抽出プロンプトを見直して evidence precision を改善すること、あわせて multi-hop・multi-evidence クエリだけを切り出して attrfilter の閾値や chunk 粒度を調整するとよさそうです。
