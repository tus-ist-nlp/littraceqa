# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100 + reading_cand50

- 実行日時: 2026-08-02T12:43:05
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter_k100.yaml`
- agent: `configs/agent_style/reading_cand50.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_k100_cand50.jsonl`
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
| agent_retrieve_top_k | `50` |
| agent_max_candidates | `50` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7664 |
| paper_recall_macro | 0.6540 |
| paper_f1_macro | 0.5675 |
| candidate_recall_at1_single_macro | 0.9231 |
| candidate_recall_at5_single_macro | 0.9231 |
| candidate_recall_at10_single_macro | 0.9615 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at100_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2050 |
| candidate_recall_at5_multi_macro | 0.4358 |
| candidate_recall_at10_multi_macro | 0.5038 |
| candidate_recall_at20_multi_macro | 0.6264 |
| candidate_recall_at50_multi_macro | 0.6954 |
| candidate_recall_at100_multi_macro | 0.6954 |
| candidate_recall_at1_total_macro | 0.5444 |
| candidate_recall_at5_total_macro | 0.6662 |
| candidate_recall_at10_total_macro | 0.7202 |
| candidate_recall_at20_total_macro | 0.8030 |
| candidate_recall_at50_total_macro | 0.8394 |
| candidate_recall_at100_total_macro | 0.8394 |
| evidence_candidate_recall_at1_single_macro | 0.9231 |
| evidence_candidate_recall_at5_single_macro | 0.9231 |
| evidence_candidate_recall_at10_single_macro | 0.9615 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at100_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3803 |
| evidence_candidate_recall_at5_multi_macro | 0.6169 |
| evidence_candidate_recall_at10_multi_macro | 0.6705 |
| evidence_candidate_recall_at20_multi_macro | 0.7644 |
| evidence_candidate_recall_at50_multi_macro | 0.8247 |
| evidence_candidate_recall_at100_multi_macro | 0.8247 |
| evidence_candidate_recall_at1_total_macro | 0.6369 |
| evidence_candidate_recall_at5_total_macro | 0.7616 |
| evidence_candidate_recall_at10_total_macro | 0.8081 |
| evidence_candidate_recall_at20_total_macro | 0.8758 |
| evidence_candidate_recall_at50_total_macro | 0.9076 |
| evidence_candidate_recall_at100_total_macro | 0.9076 |
| evidence_precision_macro | 0.2377 |
| evidence_recall_macro | 0.1919 |
| evidence_f1_macro | 0.1927 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1923 |
| table_row_f1_macro | 0.5333 |
| table_cell_accuracy_macro | 0.3530 |
| table_cell_accuracy_micro | 0.4444 |

## クエリ診断

- 候補上位10本に gold が1本も入らなかった: **1件** (q_055)
- 一部しか入らなかった(multi の取りこぼし): **27件** (q_022, q_023, q_025, q_028, q_029, q_031, q_032, q_033, q_034, q_035, q_036, q_037, q_038, q_039, q_040, q_041, q_042, q_043, q_044, q_045, q_046, q_047, q_048, q_049, q_050, q_051, q_056)

`gold順位` は候補列(`candidate_papers`)で gold 論文が何位だったか。`-` は候補に入っていない。`cr@k` はその上位k本での recall。`gold` の括弧内は evidence が紐づいている本数で、`ecr@k` はそこだけを分母にした recall（取りに行ける gold だけの検索力）。

| query_id | family | gold | 提出 | 候補 | gold順位 | cr@10 | ecr@10 | cr@50 | ecr@50 | paper_f1 | ev_f1 | 回答 | attrfilter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q_001 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_002 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | ff:○ |  |
| q_003 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | ff:○ |  |
| q_004 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_005 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | ff:× |  |
| q_006 | single | 1 | 1 | 16 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_007 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_008 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_009 | single | 1 | 1 | 22 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_010 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_011 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:○ |  |
| q_012 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:○ |  |
| q_013 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_014 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_015 | single | 1 | 1 | 9 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | mc:× / ff:× |  |
| q_017 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_018 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_019 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_020 | multi | 4 | 4 | 50 | 1, 4, 7, 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.22 | ff:× / table row_f1:0.25 cell:0.25 | NAACL 2025 |
| q_021 | single | 1 | 2 | 50 | 10 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.33 | mc:× / ff:× |  |
| q_022 | multi | 3 | 10 | 50 | 14, -, 1 | 0.33 | 0.33 | 0.67 | 0.67 | 0.15 | 0.00 | table row_f1:0.40 cell:0.00 | ICML 2025 |
| q_023 | multi | 9 | 8 | 50 | 3, 1, 5, 2, 7, 4, 12, 13, 6 | 0.78 | 0.78 | 1.00 | 1.00 | 0.94 | 0.59 | table row_f1:0.82 cell:0.82 | CVPR 2025 |
| q_024 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× | NeurIPS |
| q_025 | multi | 4 | 10 | 50 | 1, 2, -, 16 | 0.50 | 0.50 | 0.75 | 0.75 | 0.43 | 0.00 | ff:× / table row_f1:0.14 cell:0.14 |  |
| q_026 | single | 1 | 10 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | 0.00 | mc:× / ff:× |  |
| q_027 | single | 1 | 1 | 50 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | ff:× / table row_f1:0.00 cell:0.00 | CVPR 2025 |
| q_028 | multi | 4 | 2 | 50 | 2, 14, 11, 9 | 0.50 | 0.50 | 1.00 | 1.00 | 0.33 | 0.33 | table row_f1:0.75 cell:0.75 |  |
| q_029 | multi | 4 | 2 | 50 | 4, 8, 5, 33 | 0.75 | 0.75 | 1.00 | 1.00 | 0.67 | 0.57 | table row_f1:0.75 cell:0.75 |  |
| q_030 | multi | 4 | 3 | 50 | 4, 2, 3, 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 | 0.00 | table row_f1:1.00 cell:0.50 |  |
| q_031 | multi | 4 (3) | 1 | 41 | 4, 3, 1, - | 0.75 | 1.00 | 0.75 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_032 | multi | 4 (1) | 1 | 50 | 3, -, 1, - | 0.50 | 1.00 | 0.50 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_033 | multi | 4 (1) | 1 | 29 | 1, 2, 5, 20 | 0.75 | 1.00 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_034 | multi | 4 (1) | 1 | 50 | 20, 50, 39, 1 | 0.25 | 1.00 | 1.00 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_035 | multi | 4 | 1 | 50 | -, 1, 17, - | 0.25 | 0.25 | 0.50 | 0.50 | 0.40 | 0.00 | mc:× |  |
| q_036 | multi | 4 | 1 | 50 | -, -, 1, - | 0.25 | 0.25 | 0.25 | 0.25 | 0.40 | 0.40 | mc:× |  |
| q_037 | multi | 4 (1) | 1 | 50 | -, -, 1, - | 0.25 | 1.00 | 0.25 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_038 | multi | 4 (1) | 1 | 11 | 1, -, -, - | 0.25 | 1.00 | 0.25 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_039 | multi | 4 (3) | 1 | 28 | -, -, -, 1 | 0.25 | 0.33 | 0.25 | 0.33 | 0.40 | 0.40 | mc:× |  |
| q_040 | multi | 4 (1) | 1 | 50 | 7, -, -, 1 | 0.50 | 1.00 | 0.50 | 1.00 | 0.40 | 0.00 | mc:× |  |
| q_041 | multi | 4 (3) | 1 | 50 | 7, 3, 6, - | 0.75 | 0.67 | 0.75 | 0.67 | 0.40 | 0.00 | mc:× |  |
| q_042 | multi | 4 (2) | 2 | 50 | -, 2, -, 3 | 0.50 | 1.00 | 0.50 | 1.00 | 0.33 | 0.00 | mc:× |  |
| q_043 | multi | 4 | 2 | 50 | 1, 3, 5, - | 0.75 | 0.75 | 0.75 | 0.75 | 0.67 | 0.33 | mc:× |  |
| q_044 | multi | 4 | 2 | 50 | 7, 2, 1, 34 | 0.75 | 0.75 | 1.00 | 1.00 | 0.67 | 0.29 | mc:× |  |
| q_045 | multi | 4 | 2 | 50 | 1, 11, 3, 41 | 0.50 | 0.50 | 1.00 | 1.00 | 0.67 | 0.00 | mc:× |  |
| q_046 | multi | 4 (3) | 1 | 50 | 1, 12, 5, 20 | 0.50 | 0.67 | 1.00 | 1.00 | 0.40 | 0.40 | mc:× |  |
| q_047 | multi | 4 (3) | 1 | 50 | 39, 3, -, - | 0.25 | 0.33 | 0.50 | 0.67 | 0.40 | 0.00 | mc:× |  |
| q_048 | multi | 4 (3) | 1 | 50 | 1, -, -, - | 0.25 | 0.33 | 0.25 | 0.33 | 0.40 | 0.00 | mc:× |  |
| q_049 | multi | 4 (3) | 10 | 50 | -, 15, 43, 1 | 0.25 | 0.33 | 0.75 | 0.67 | 0.14 | 0.00 | mc:× |  |
| q_050 | multi | 4 (3) | 1 | 34 | 12, 32, 1, - | 0.25 | 0.33 | 0.75 | 1.00 | 0.40 | 0.40 | mc:× |  |
| q_051 | multi | 4 (3) | 1 | 50 | -, -, -, 1 | 0.25 | 0.33 | 0.25 | 0.33 | 0.40 | 0.00 | mc:× |  |
| q_052 | single | 1 | 1 | 45 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:1.00 cell:0.67 |  |
| q_053 | single | 1 | 1 | 25 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | mc:× / ff:× |  |
| q_054 | single | 1 | 1 | 20 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | table row_f1:0.00 cell:0.00 |  |
| q_055 | single | 1 | 1 | 50 | 19 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | mc:× / ff:× |  |
| q_056 | multi | 4 | 1 | 50 | 11, 4, 1, 2 | 0.75 | 0.75 | 1.00 | 1.00 | 0.40 | 0.00 | table row_f1:0.75 cell:0.00 |  |

## コメント

候補検索は single では非常に強く、candidate/evidence recall@20 が 1.0 近くまで到達しているので、単一文献の取りこぼしは少なそうです。一方で multi 系は candidate recall@50 でも 0.695、evidence recall@50 でも 0.825 に留まっており、複数根拠・複数文献が必要な問題での検索多様性が主なボトルネックに見えます。paper precision 0.766 に対して recall 0.654、evidence precision/recall も 0.238/0.192 と低めで、さらに paper F1 が相対的にかなり低い点は、クエリごとのばらつきや一部クエリの大崩れを疑いたいです。次は attr filter の緩和やクエリ拡張で multi-hop 候補の多様性を増やしつつ、reranker/reader の evidence 抽出を見直して、特に失敗クエリの内訳を確認するのがよさそうです。

## 訂正（2026-08-02 追記）: 本レポートの @100 の値は比較に使えない

`candidate_recall@100` / `evidence_candidate_recall@100` として載せた値は、
**実際には「@min(100, 候補列の長さ)」**である。予測に残す候補は `reading.py` の
`CANDIDATE_PAPERS_LIMIT`（既定50）で切られるため:

- 展開なしの実験は候補列が **50本** -> その「@100」は @50 と同値
- 展開ありの実験は候補列が **最大70本** -> その「@100」は実質 @70

つまり「@100 が 0.836 -> 0.914 に伸びた」という記述は、**@50 と @70 を比べたもの**で
あり、展開なし側は構造的に51位以降が空欄だった。候補列に gold が増えたこと自体は
事実だが、`@100` という指標名で比較したのは誤り。

**公平に比較できるのは cr@50 / ecr@50 まで**（どちらの実験も50本以上ある）。
本レポート内の @50 以下の数値と結論は影響を受けない。

## candidate_recall（後から追記）

予測ファイル `predictions_8b_chunk_k100_cand50.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 50 本なので、それを超える k は @50 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.6954 |
| candidate_recall_at70_total_macro | 0.8394 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at70_multi_macro | 0.8247 |
| evidence_candidate_recall_at70_total_macro | 0.9076 |

<!-- candidate_recall backfill: pred=predictions_8b_chunk_k100_cand50.jsonl, max_candidates=50 -->
