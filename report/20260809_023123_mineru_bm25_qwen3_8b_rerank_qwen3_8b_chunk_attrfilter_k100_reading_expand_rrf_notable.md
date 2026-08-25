# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_expand_rrf_notable

- 実行日時: 2026-08-09T02:31:23
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_expand_rrf/notable.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_k100_notable.jsonl`
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
| agent_paper_score_skip_chunk_types | `["table"]` |
| agent_retrieve_top_k | `20` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

## 指標

| 指標 | 値 |
|---|---|
| candidate_recall_at1_single_macro | 0.8846 |
| candidate_recall_at5_single_macro | 1.0000 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2107 |
| candidate_recall_at5_multi_macro | 0.5910 |
| candidate_recall_at10_multi_macro | 0.7529 |
| candidate_recall_at20_multi_macro | 0.8592 |
| candidate_recall_at50_multi_macro | 0.9224 |
| candidate_recall_at70_multi_macro | 0.9224 |
| candidate_recall_at1_total_macro | 0.5293 |
| candidate_recall_at5_total_macro | 0.7843 |
| candidate_recall_at10_total_macro | 0.8697 |
| candidate_recall_at20_total_macro | 0.9258 |
| candidate_recall_at50_total_macro | 0.9591 |
| candidate_recall_at70_total_macro | 0.9591 |
| evidence_candidate_recall_at1_single_macro | 0.8846 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3544 |
| evidence_candidate_recall_at5_multi_macro | 0.7002 |
| evidence_candidate_recall_at10_multi_macro | 0.8391 |
| evidence_candidate_recall_at20_multi_macro | 0.9253 |
| evidence_candidate_recall_at50_multi_macro | 0.9799 |
| evidence_candidate_recall_at70_multi_macro | 0.9799 |
| evidence_candidate_recall_at1_total_macro | 0.6051 |
| evidence_candidate_recall_at5_total_macro | 0.8419 |
| evidence_candidate_recall_at10_total_macro | 0.9152 |
| evidence_candidate_recall_at20_total_macro | 0.9606 |
| evidence_candidate_recall_at50_total_macro | 0.9894 |
| evidence_candidate_recall_at70_total_macro | 0.9894 |
| evidence_candidate_recall_by_backed_at1_single_macro | 0.8750 |
| evidence_candidate_recall_by_backed_at5_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at10_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at20_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at50_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at70_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at1_multi_macro | 0.2295 |
| evidence_candidate_recall_by_backed_at5_multi_macro | 0.6220 |
| evidence_candidate_recall_by_backed_at10_multi_macro | 0.7971 |
| evidence_candidate_recall_by_backed_at20_multi_macro | 0.9058 |
| evidence_candidate_recall_by_backed_at50_multi_macro | 0.9746 |
| evidence_candidate_recall_by_backed_at70_multi_macro | 0.9746 |
| evidence_candidate_recall_by_backed_at1_total_macro | 0.6051 |
| evidence_candidate_recall_by_backed_at5_total_macro | 0.8419 |
| evidence_candidate_recall_by_backed_at10_total_macro | 0.9152 |
| evidence_candidate_recall_by_backed_at20_total_macro | 0.9606 |
| evidence_candidate_recall_by_backed_at50_total_macro | 0.9894 |
| evidence_candidate_recall_by_backed_at70_total_macro | 0.9894 |

## コメント

全体として単一正解クエリは非常に強く、candidate/evidence ともに@5でほぼ100%に到達しているため、上位候補生成は良好です。一方で multi クエリの初段性能は弱く、candidate_recall@1=0.211、@5=0.591 と取りこぼしが目立くため、複数文書・複数根拠が必要な質問で順位付けがまだ不十分です。evidence 系は multi でも @20 以降かなり高く、必要根拠自体は広めに取れているので、課題は候補不足というより top-k 前半への押し上げにありそうです。次は multi クエリに絞って、chunk 粒度や attrfilter 条件、reranker の入力特徴量・多様化設定を見直し、@1〜@10 の改善余地を確認するとよさそうです。
