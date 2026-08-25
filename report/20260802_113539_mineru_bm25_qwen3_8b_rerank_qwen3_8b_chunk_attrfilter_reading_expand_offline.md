# mineru + bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter + reading_expand（オフライン適用）

- 実行日時: 2026-08-02T11:35:39
- paths: `configs/paths/default.yaml`
- process: `configs/process_style/mineru.yaml`
- search: `configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b_chunk_attrfilter.yaml`
- agent: `configs/agent_style/reading_expand.yaml` の expansion ブロック（neighbors=20, anchors=1）を既存予測にオフライン適用
- 土台: `predictions_8b_chunk_b_merged.jsonl`（7/25-26 の val_a+val_b 結合、reading_topk50）
- output: `predictions_8b_chunk_expand_offline.jsonl`（採点 55件, production_input=True）
- git: `a0206042354c`

## この実行について（GPU なしで厳密な理由）

論文→論文展開（`retrieve/paper_expander.py`）は candidate_papers 組み立て後の**末尾追記のみ**で、
LLM の読解ループ・提出・evidence に一切フィードバックしない。したがって保存済み予測の
candidate_papers に同じ追記を適用した本ファイルは、`reading_expand.yaml` でフル実行した場合の
出力と意味的に同一。伸びは cr@100 / ecr@100 にだけ現れる設計（既存50本の順位は不変。
追記は51位以降に入るため cr@50 はほぼ動かない。base の候補が50本未満のクエリでのみ
50位以内に食い込み、cr@50 が僅かに上がる）。

**paper_recall / precision / f1 / evidence_f1 / 回答系は定義上まったく動かない**（提出に触れないため）。

## 設定（展開部分）

| パラメータ | 値 |
|---|---|
| expansion_index | `faiss_specter2_abstract`（構築済みを再利用、追加構築なし） |
| expansion_neighbors | `20` |
| expansion_anchors | `1`（候補1位論文。3本にしても ecr 不変で候補だけ増える実測により） |
| 追記数 | 平均 15.6 本/クエリ（重複除去後） |
| 土台の検索設定 | 土台実行のレポート（20260726_072645_..._reading_topk50.md）を参照 |

## 指標（土台 → 展開後）

| 指標 | 展開なし | 展開あり |
|---|---|---|
| paper_precision_macro | 0.7824 | 0.7824 |
| paper_recall_macro | 0.6475 | 0.6475 |
| paper_f1_macro | 0.5841 | 0.5841 |
| candidate_recall_at50_total_macro | 0.8364 | 0.8545 **(+0.018)** |
| candidate_recall_at100_total_macro | 0.8364 | 0.9136 **(+0.077)** |
| candidate_recall_at50_multi_macro | 0.6897 | 0.7241 **(+0.034)** |
| candidate_recall_at100_multi_macro | 0.6897 | 0.8362 **(+0.147)** |
| evidence_candidate_recall_at50_total_macro | 0.9076 | 0.9136 **(+0.006)** |
| evidence_candidate_recall_at100_total_macro | 0.9076 | 0.9561 **(+0.048)** |
| evidence_candidate_recall_at50_multi_macro | 0.8247 | 0.8362 **(+0.011)** |
| evidence_candidate_recall_at100_multi_macro | 0.8247 | 0.9167 **(+0.092)** |
| evidence_f1_macro | 0.2205 | 0.2205 |

## 別土台での再現（predictions_8b_chunk_cand50 に同じ適用）

| 指標 | 展開なし | 展開あり |
|---|---|---|
| candidate_recall_at100_total_macro | 0.8318 | 0.9000 **(+0.068)** |
| candidate_recall_at100_multi_macro | 0.6810 | 0.8103 **(+0.129)** |
| evidence_candidate_recall_at100_total_macro | 0.8894 | 0.9455 **(+0.056)** |
| evidence_candidate_recall_at100_multi_macro | 0.7902 | 0.8966 **(+0.106)** |

## 所見

- multi の cr@100 が 0.690 -> 0.836（+14.6pt）、ecr@100 が 0.825 -> 0.917（+9.2pt）。
  クエリ品質監査で判明した「トピッククラスタのピア gold は質問→論文検索では拾えないが、
  正解論文からは近い」という構造に SPECTER2 proximity（引用近接学習の論文類似）が刺さった。
- 独立2土台（topk50 / cand50）で +6.8〜7.8pt が再現。anchor の質への感度は低い。
- 未検出 supporting は展開後も q_025 の ScaleKV 1本のみ（展開でも拾えない、残る唯一の検索失敗）。
- 次の伸びしろは展開で拾った51位以降のピアを**提出に乗せる**選定変更。そこで初めて
  paper_recall_macro が動く（候補内 evidence 持ちを全部取れた場合の oracle 上限 0.647 -> 0.791）。
- nlp01 の GPU が空いたら `reading_expand.yaml` でライブ実行し、experiments.jsonl に
  config 付きの正式な行を残すこと（結果は本レポートと同値になる想定）。

## 追記（同日）: 挿入位置を末尾から max_candidates 直後に変更

「拾ったピアが51位以降に沈む」という指摘を受けて挿入位置を比較した。

| 挿入位置 | cr@20 | cr@50 | cr@100 |
|---|---|---|---|
| （展開なし） | 0.783 | 0.836 | 0.836 |
| 末尾追記（当初実装） | 0.788 | 0.855 | 0.914 |
| **20位直後（採用）** | 0.788 | **0.880** | 0.914 |
| 10位挿入 | 0.773 | 0.880 | 0.914 |

- 20位（= LLM が読む max_candidates）直後への挿入は、LLM 可視域を汚さず
  （cr@20 が末尾追記と同一）、cr@50 が +2.5pt。押し出しで50位圏外に落ちる gold は
  実質ゼロ（cr@100 が同値）。
- 10位挿入は本物の上位候補を押し出して cr@20 が悪化（0.788 -> 0.773）するので不採用。
- 実装は `reading.py` が `inserted_at = min(max_candidates, len(候補))` で挿入する形に変更済み。
  本レポートの予測ファイルと下の指標もこの配置で再生成した（上の「指標」表の
  cr@50/ecr@50 は当初の末尾追記時の値なので、比較はこの追記の表を正とする）。

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
