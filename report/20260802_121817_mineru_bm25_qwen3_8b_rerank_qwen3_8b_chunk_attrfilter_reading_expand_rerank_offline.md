# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter + reading_expand_rerank（オフライン適用）

- 実行日時: 2026-08-02T12:18:17
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter.yaml`
- agent: `configs/agent_style/reading_expand_rerank.yaml`
- queries: `data/validation_inputs.jsonl` (採点 55件, production_input=True)
- output: `predictions_8b_chunk_expand_rerank_offline.jsonl`
- git: `a0206042354c`
- **オフライン適用**: 土台 `predictions_8b_chunk_b_merged.jsonl` の candidate_papers に、
  `compose_config()` で組んだ expander と `reading.py._expand_candidates()` と同じ手順を適用。
- **注意: reranker は 0.6B で代用**（本番構成は 8B だが GPU に空きが無かった）。
  8B なら同等以上が期待できるが未確認。

## 設定（この実行時の実際の値）

| パラメータ | 値 |
|---|---|
| per_index_k | `1000` |
| pool_k | `1000` |
| indexers | `["bm25s", "faiss_qwen3"]` |
| fuser | `"rrf"` |
| fuser_k | `60` |
| reranker | `"qwen3"` |
| reranker_model_本番 | `"Qwen/Qwen3-Reranker-8B"` |
| reranker_model_本実験 | `"Qwen/Qwen3-Reranker-0.6B"` |
| reranker_device | `"cuda:0"` |
| reranker_max_tokens | `1024` |
| attribute_filter_enabled | `true` |
| agent | `"reading"` |
| agent_llm | `"azure_openai"` |
| agent_max_steps | `3` |
| agent_retrieve_top_k | `20` |
| agent_max_candidates | `20` |
| agent_chunks_per_paper | `2` |
| agent_snippet_chars | `1800` |
| agent_paper_cutoff | `"llm"` |
| agent_max_papers | `10` |
| expansion_index | `"faiss_specter2_abstract"` |
| expansion_neighbors | `20` |
| expansion_anchors | `1` |
| expansion_rerank | `true` |
| expansion_rerank_top_k | `5` |
| expansion_insert_at | `15` |
| expansion_rerank_cost | `"55クエリで23秒（0.6B・GPU1枚）"` |

## 指標

| 指標 | 値 |
|---|---|
| paper_precision_macro | 0.7824 |
| paper_recall_macro | 0.6475 |
| paper_f1_macro | 0.5841 |
| candidate_recall_at1_single_macro | 0.9615 |
| candidate_recall_at5_single_macro | 1.0000 |
| candidate_recall_at10_single_macro | 1.0000 |
| candidate_recall_at20_single_macro | 1.0000 |
| candidate_recall_at50_single_macro | 1.0000 |
| candidate_recall_at100_single_macro | 1.0000 |
| candidate_recall_at1_multi_macro | 0.2021 |
| candidate_recall_at5_multi_macro | 0.4071 |
| candidate_recall_at10_multi_macro | 0.4751 |
| candidate_recall_at20_multi_macro | 0.6456 |
| candidate_recall_at50_multi_macro | 0.8017 |
| candidate_recall_at100_multi_macro | 0.8362 |
| candidate_recall_at1_total_macro | 0.5611 |
| candidate_recall_at5_total_macro | 0.6874 |
| candidate_recall_at10_total_macro | 0.7232 |
| candidate_recall_at20_total_macro | 0.8131 |
| candidate_recall_at50_total_macro | 0.8955 |
| candidate_recall_at100_total_macro | 0.9136 |
| evidence_candidate_recall_at1_single_macro | 0.9615 |
| evidence_candidate_recall_at5_single_macro | 1.0000 |
| evidence_candidate_recall_at10_single_macro | 1.0000 |
| evidence_candidate_recall_at20_single_macro | 1.0000 |
| evidence_candidate_recall_at50_single_macro | 1.0000 |
| evidence_candidate_recall_at100_single_macro | 1.0000 |
| evidence_candidate_recall_at1_multi_macro | 0.3831 |
| evidence_candidate_recall_at5_multi_macro | 0.5910 |
| evidence_candidate_recall_at10_multi_macro | 0.6504 |
| evidence_candidate_recall_at20_multi_macro | 0.7634 |
| evidence_candidate_recall_at50_multi_macro | 0.8822 |
| evidence_candidate_recall_at100_multi_macro | 0.9167 |
| evidence_candidate_recall_at1_total_macro | 0.6566 |
| evidence_candidate_recall_at5_total_macro | 0.7843 |
| evidence_candidate_recall_at10_total_macro | 0.8157 |
| evidence_candidate_recall_at20_total_macro | 0.8753 |
| evidence_candidate_recall_at50_total_macro | 0.9379 |
| evidence_candidate_recall_at100_total_macro | 0.9561 |
| evidence_precision_macro | 0.2404 |
| evidence_recall_macro | 0.2328 |
| evidence_f1_macro | 0.2205 |
| multiple_choice_accuracy | 0.0000 |
| freeform_exact_match | 0.1923 |
| table_row_f1_macro | 0.4945 |
| table_cell_accuracy_macro | 0.2900 |
| table_cell_accuracy_micro | 0.2963 |

## 比較

| 指標 | 展開なし | 展開のみ | **展開+rerank（本実行）** |
|---|---|---|---|
| paper_precision_macro | 0.7824 | 0.7824 | **0.7824** |
| paper_recall_macro | 0.6475 | 0.6475 | **0.6475** |
| paper_f1_macro | 0.5841 | 0.5841 | **0.5841** |
| candidate_recall_at10_total_macro | 0.7232 | 0.7232 | **0.7232** |
| candidate_recall_at20_total_macro | 0.7833 | 0.7879 | **0.8131** |
| candidate_recall_at50_total_macro | 0.8364 | 0.8803 | **0.8955** |
| candidate_recall_at20_multi_macro | 0.5891 | 0.5977 | **0.6456** |
| candidate_recall_at50_multi_macro | 0.6897 | 0.7730 | **0.8017** |
| evidence_candidate_recall_at20_total_macro | 0.8697 | 0.8758 | **0.8753** |
| evidence_candidate_recall_at50_total_macro | 0.9076 | 0.9242 | **0.9379** |

## 所見

**cr@20 が初めて動いた。** 0.783 -> 0.813（multi は 0.589 -> 0.646）。従来の展開は
LLM 可視域を汚さないため cr@20 が不変だったが、rerank で絞れば上位に入れられる。
cr@50 も 0.880 -> 0.895 と展開のみを上回る。

**必須条件: 上位K本に絞ってから位置挿入する。** 最初に実装した「reranker のスコアで
既存候補と混ぜる」方式は **cr@20 が 0.773 に悪化**した。診断すると top20 に押し込まれた
91本の gold は **0本**で、代わりに既存 gold を2本追い出していた。原因は、reranker が
展開20本の中での序列は正しく付ける（gold の順位が中央値 9位 -> 5位、上位5位以内が
3本 -> 12本）一方で、**絶対スコアが既存候補と比較可能でない**こと。gold を含まない
38クエリからも高スコアの非 gold が流れ込む。

**パラメータ掃引（insert_at=15）:**

| rerank_top_k | cr@20 | multi@20 | cr@50 | ecr@50 |
|---|---|---|---|---|
| 0（展開なし） | 0.783 | 0.589 | 0.836 | 0.908 |
| **5（採用）** | **0.813** | **0.646** | **0.895** | **0.938** |
| 7 | 0.813 | 0.646 | 0.885 | 0.932 |
| 10 | 0.813 | 0.646 | 0.876 | 0.921 |
| 20 | 0.813 | 0.646 | 0.880 | 0.924 |

K を増やしても cr@20 は頭打ちで、溢れたノイズが50位圏内を圧迫して cr@50 が下がる。
insert_at は 10位と15位が同値、20位だと LLM 可視域に入らず cr@20 が動かない。

**提出指標は不変**（paper_f1 0.584）。展開は candidate_papers にしか触らないため。

## 次に検証すべきこと

- **8B reranker での再検証**（GPU 空き待ち）。本実験は 0.6B 代用なので上振れの余地がある。
- 拾った論文を提出に載せるには、展開をループ内に移して LLM に読ませる改修が要る。
  cr@20 が上がった今なら、可視域に入った gold を LLM が選べる可能性がある。
- プールを完璧に並べ替えた場合の cr@20 天井は 0.914。現状 0.813 との差10ポイントが残る。

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

予測ファイル `predictions_8b_chunk_expand_rerank_offline.jsonl` を現在の `scripts/evaluate.py` で採点し直した値。既存の指標がすべて一致することを確認済み（照合 11指標）。候補列は最大 70 本なので、それを超える k は @70 と同値。

| 指標 | 値 |
|---|---|
| candidate_recall_at70_single_macro | 1.0000 |
| candidate_recall_at70_multi_macro | 0.8362 |
| candidate_recall_at70_total_macro | 0.9136 |
| evidence_candidate_recall_at70_single_macro | 1.0000 |
| evidence_candidate_recall_at70_multi_macro | 0.9167 |
| evidence_candidate_recall_at70_total_macro | 0.9561 |

<!-- candidate_recall backfill: pred=predictions_8b_chunk_expand_rerank_offline.jsonl, max_candidates=70 -->
