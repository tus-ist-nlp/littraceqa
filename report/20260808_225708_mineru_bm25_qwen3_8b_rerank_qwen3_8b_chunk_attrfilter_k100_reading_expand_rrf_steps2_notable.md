# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_expand_rrf_steps2_notable

- 実行日時: 2026-08-08T22:57:08
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_expand_rrf/steps2_notable.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_k100_steps2_notable.jsonl`
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
| agent_max_steps | `2` |
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
| candidate_recall_at1_multi_macro | 0.2021 |
| candidate_recall_at5_multi_macro | 0.5565 |
| candidate_recall_at10_multi_macro | 0.7146 |
| candidate_recall_at20_multi_macro | 0.8333 |
| candidate_recall_at50_multi_macro | 0.8966 |

| candidate_recall_at1_total_macro | 0.5247 |
| candidate_recall_at5_total_macro | 0.7662 |
| candidate_recall_at10_total_macro | 0.8495 |
| candidate_recall_at20_total_macro | 0.9121 |
| candidate_recall_at50_total_macro | 0.9455 |
| candidate_recall_at70_total_macro | 0.9455 |
| evidence_candidate_recall_at1_single_macro | 0.8846 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3458 |
| evidence_candidate_recall_at5_multi_macro | 0.6628 |
| evidence_candidate_recall_at10_multi_macro | 0.8008 |
| evidence_candidate_recall_at20_multi_macro | 0.8994 |
| evidence_candidate_recall_at50_multi_macro | 0.9511 |
| evidence_candidate_recall_at70_multi_macro | 0.9511 |
| evidence_candidate_recall_at1_total_macro | 0.6005 |
| evidence_candidate_recall_at5_total_macro | 0.8222 |
| evidence_candidate_recall_at10_total_macro | 0.8949 |
| evidence_candidate_recall_at20_total_macro | 0.9470 |
| evidence_candidate_recall_at50_total_macro | 0.9742 |
| evidence_candidate_recall_at70_total_macro | 0.9742 |
| evidence_candidate_recall_by_backed_at1_single_macro | 0.8750 |
| evidence_candidate_recall_by_backed_at5_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at10_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at20_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at50_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at70_single_macro | 1.0000 |
| evidence_candidate_recall_by_backed_at1_multi_macro | 0.2186 |
| evidence_candidate_recall_by_backed_at5_multi_macro | 0.5749 |
| evidence_candidate_recall_by_backed_at10_multi_macro | 0.7488 |
| evidence_candidate_recall_by_backed_at20_multi_macro | 0.8732 |
| evidence_candidate_recall_by_backed_at50_multi_macro | 0.9384 |
| evidence_candidate_recall_by_backed_at70_multi_macro | 0.9384 |
| evidence_candidate_recall_by_backed_at1_total_macro | 0.6005 |
| evidence_candidate_recall_by_backed_at5_total_macro | 0.8222 |
| evidence_candidate_recall_by_backed_at10_total_macro | 0.8949 |
| evidence_candidate_recall_by_backed_at20_total_macro | 0.9470 |
| evidence_candidate_recall_by_backed_at50_total_macro | 0.9742 |
| evidence_candidate_recall_by_backed_at70_total_macro | 0.9742 |

## コメント

全体として single は非常に強く、candidate/evidence ともに recall@5 で 1.0 に到達しているため、単一正解系の取りこぼしはほぼありません。一方で multi は弱く、candidate_recall@1=0.20、@5=0.56 にとどまっており、複数根拠を含む質問では上位順位への集約が課題です。@20 以降では total/evidence ともに 0.95 前後まで伸びるので、候補生成量は十分で、主なボトルネックは rerank や multi-hop 的な拡張・統合の精度に見えます。次は multi クエリに絞った失敗分析を行い、chunk 粒度や attrfilter 条件、reading_expand の展開幅・step 数、RRF/再ランキングの重みを調整して上位 5〜10 件の押し上げを試すとよさそうです。
