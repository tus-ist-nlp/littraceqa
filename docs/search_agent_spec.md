# 検索エージェントの反復ループ拡張 仕様

> **注記（2026-08-06）**: この文書が作った `configs/agent_style/reading_loop/` は削除した。
> 拡張キー（`subquery_merge` / `grounded_refine` / `pool_rescore` / `adaptive_depth`）は
> `agent/reading.py` にそのまま残っているので、任意の agent yaml の `params` に直接書けば動く。
> 以下の `--agent configs/agent_style/reading_loop/*.yaml` はその読み替えが必要。


作成日: 2026-08-03 / 状態: **未実装**（実験中のためコードは一切変更していない）

`ReadingAgent`（`src/littraceqa/di_pipeline/agent/reading.py`）の反復ループを4方向に
拡張する仕様。実装するときはこのファイルを読めば着手できるように書いてある。

---

## 1. なぜやるか

### 1.1 現状の構造

反復ステップ間で変えられるのは**サブクエリの集合だけ**で、それ以外
（`retrieve_top_k`、属性フィルタ、reranker、貯めたチャンク）は全ステップ固定。

`predictions_8b_chunk_k100_expand_fused.jsonl` の trace 実測（55件）:

| | クエリ数 | サブクエリ本数（中央値） | 累積チャンク（中央値） |
|---|---|---|---|
| step0 `_decompose()` | 55 | 3（平均3.4, max5） | 45 |
| step1 `_refine()` | 35 | 8（平均8.4, max17） | 117 |
| step2 `_refine()` | 32 | 9（平均9.6, max20） | 227 |

20/55 が step0 で `sufficient` になって終了、32/55 が3ステップ完走。

### 1.2 確認済みの欠陥

- **サブクエリ間のマージが max**（`reading.py:121-123`）。異なるサブクエリに対する
  reranker の yes 確率という**比較可能でない値**を突き合わせている。
- **`_refine()` がコーパスの反応を見ていない。** 材料は読解 LLM の `missing` だけ。
  実測の症状:
  - q_003: `EasySpec` を分光解析ソフトと誤解したまま2ステップ暴走（`Spectragryph` /
    `Fityk` / `Balmer series` / `Savitzky-Golay`）。step0 では gold を1位で引けていた。
  - q_002: 空振りした `NVIDIA A100` の次に `Ubuntu 20.04` / `64GB RAM` を投げ続けた
    （正解は RTX 4090）。
  - q_001: `missing` が step1/step2 でほぼ同文のまま、`Ours500→1` / `Ours 500→1` /
    `500:1` / `500 to 1` と**語彙の言い換え**に流れた。「similar を繰り返すな」という
    指示が意味の転換ではなく表記ゆれの総当たりを生んでいる。
- **プールが積む一方**（45 → 117 → 227）。剪定も再スコアもしない。
- **深さが固定。** 全クエリが同じ `retrieve_top_k: 20`。

### 1.3 伸びしろの所在（設計の前提）

同ファイル、cr@20 0.824 / multi 0.667:

| gold の現在順位 | 1-20位 | 21-50位 | 圏外 |
|---|---|---|---|
| hidden_source_single_paper (26) | **26** | 0 | 0 |
| multi_paper (120) | 82 | **11** | 27（うち evidence なし14） |

**single は完全飽和。打ち手は multi_paper を動かせるかだけで評価する。**

step0 で `sufficient` と言って早期終了した multi 11件の cr@20 は **0.614** で、
3ステップ完走組（16件, 0.661）より低い——停止判定が検索側の状況を見ていない。

狙いは cr@20 の multi（0.667）と、21-50位に沈んでいる11本。

### 1.4 出典

ADORE（retrieval-grounded relevance feedback, [arXiv 2606.13905](https://arxiv.org/html/2606.13905)）/
RAG-Fusion / CRAG / listwise-ranker-guided adaptive retrieval
（[arXiv 2501.09186](https://arxiv.org/pdf/2501.09186)）。
RL によるクエリ生成学習（DeepRetrieval / s3）は**この仕様の範囲外**。

---

## 2. 方針

**すべて `agent_style` の任意キーにし、キーが無ければ現行と完全に同一の経路を通す。**
既存 yaml は触らず、比較用の yaml を新規に足す。

`retrieve/`（`hybrid.py` / `rrf.py` / `reranker.py` / `base.py`）は**無改修**。
`Retriever` Protocol（`retrieve/base.py:20-24`）を変えないので、
`tests/test_reading_agent.py` の `_StubRetriever`（2引数 `retrieve`）も壊れない。

---

## 3. 共通の土台: サブクエリ単位の run を保持する

`ReadingAgent.run()` の `chunks: dict[str, RetrievalResult]` 一本を、

```python
@dataclass
class SubqueryRun:
    step: int
    subquery: str
    results: list[RetrievalResult]   # 検索が返した順
```

の列 `runs: list[SubqueryRun]` に置き換える。

`chunks` は chunk_id → RetrievalResult の**ルックアップ用に残す**
（`_read_and_judge()` の捏造チェックと `_build_prediction()` の evidence 引きが
chunk_id で引いている）。**ランキングの出所だけを `runs` に移す。**

新メソッド `_merged_results(runs) -> list[RetrievalResult]` を置き、
`_candidate_papers()` と `_build_prediction()` の両方がこれを通る形にする。
4〜7 はすべてこの構造の上に乗る。

### ダンプ

`scripts/run_search.py` に `--dump-runs <path>`（省略時は何もしない）を足す。

```json
{"query_id": "q_001", "step": 0, "subquery": "...",
 "results": [{"chunk_id": "...", "paper_id": "...", "rank": 1, "score": 0.93}, ...]}
```

`Prediction.trace` は汚さない（提出ファイルが膨らむため）。

---

## 4. サブクエリ間マージを RRF にする

```yaml
# agent_style params
subquery_merge: rrf      # "max"(既定 = 現行) | "rrf"
subquery_rrf_k: 60
```

`_merged_results()` の `rrf` 分岐は既存の `RRFFuser`（`retrieve/rrf.py:15`）を
**そのまま使う**（`weights` は `result.source` = indexer 名で引くが既定 1.0 なので
素通しになる）。新しい融合ロジックは書かない。

**multi に効く理屈**: multi の gold は「1本のサブクエリだけが見つける論文」が多い。
max マージではその論文が単一サブクエリの絶対スコアで他と競うが、RRF なら
「そのサブクエリの中での順位」で評価される。21-50位の11本が動く見込み。

---

## 5. `_refine()` を検索結果に接地させる（ADORE 系）

```yaml
grounded_refine: true        # 既定 false
grounded_refine_top_n: 10
```

`_refine()`（`reading.py:367`）のプロンプトに2つ足す。
**追加の LLM 呼び出しはゼロ**（プロンプトが太るだけ）。

1. **いま候補上位 N 本が何なのか** — `[venue year] title` の一覧。コーパスが実際に
   返したもの。q_003 なら「上位10本が全部 speculative decoding 系」と見えて、
   分光解析ソフトという前提が崩れる。
2. **各サブクエリの寄与** — そのサブクエリの結果が現在の上位 N に何本残っているか。
   0本のものを「効かなかった」として名指しする（`runs` から計算できる）。

指示文も ADORE の三分法に寄せる: **効いた語を伸ばす / まだ埋まっていない側面を突く /
もう投げない語を挙げる**。

`CORPUS_NOTE`（`reading.py:48`）と `_constraint_note()` はそのまま残す。

---

## 6. プールの再スコアと剪定（CRAG 系）

```yaml
pool_rescore: false    # 既定オフ
pool_prune_to: null    # 既定オフ
```

- `pool_rescore: true`: `_build_prediction()` の直前に **元の質問**で
  `self.retriever.reranker` を1回かけ、プール全体を単一スケールに揃える。
  サブクエリ間のスコア非可換性の根治（4 が順位で回避するのに対し、こちらは直接解消）。
  `_expansion_reranker()`（`reading.py:593`）と同じ流儀で、reranker が無い構成・
  `NoneReranker` の構成では黙って skip する。
- `pool_prune_to: N`: 再スコア後に上位 N 件へ切る。

**既定オフのまま出す。** CLAUDE.md の実測どおり reranker は「質問に答える論文」を
選べているほど no_evidence gold を落とす（ecr↑ / cr↓）ので、剪定は multi の圏外27本
（うち evidence なし14本）をさらに落としうる。**評価は cr と ecr を必ず並べて読む。**

コストは `qwen3_paper`・3GPU で1クエリ 9〜60秒。

---

## 7. 検索の深さをリランカのスコア分布で決める

```yaml
adaptive_depth:
  enabled: true
  probe_rank: 4          # 何位との差を見るか
  gap_threshold: 0.15    # これ以上離れていれば「勝者が明確」
  shallow_k: 10
  deep_k: 40
```

各 `retrieve()` の結果に対し `gap = score[0] - score[probe_rank]` を見て、
大きければ `shallow_k` 件、平坦なら `deep_k` 件だけ `runs` に採る。
reranker 有効時のスコアは yes 確率なので 0〜1 の解釈可能なスケールになる。

**retriever には常に `deep_k` を渡して取り、切るのはエージェント側。**
`HybridRetriever.retrieve()` の呼び出しは1回のままで、`pool_k`（reranker の推論件数）は
search_style 側の値なので**推論コストは増えない**（CLAUDE.md の
「`retrieve_top_k` を増やしても reranker の推論は増えない」がそのまま当てはまる）。

single は1位が飛び抜けるので自動的に浅く、multi は平坦なので深くなる。
**正解率0.67の task_family 推定に依存しない**点が要点。

---

## 8. オフライン再生 `scripts/replay_merge.py`（新規）

`scripts/replay_expansion.py` と同じ流儀で、**ReadingAgent 本体のメソッドを呼ぶ**
（`_merged_results` / 深さ判定 / `to_gold_papers`）。ロジックを書き写さない。

```
uv run python scripts/replay_merge.py \
  --runs runs_fat.jsonl --pred predictions_fat.jsonl \
  --agent configs/agent_style/reading_loop/rrf.yaml \
  --set subquery_merge=rrf --ks 5,10,20,50
```

土台は `retrieve_top_k: 100` の「太い」走行1回。オフラインでは 100 以下の任意の
`retrieve_top_k` と、4・6・7 の全変種を数十秒で振れる。

**限界を docstring に明記する**:
- **5 は再生不可**（サブクエリ自体が変わるのでフル走行が要る）。
- 4・6・7 も「サブクエリを固定した条件下」の数字で、本走行では読解 LLM の選定が
  変わりうる。`replay_expansion.py` と同じく**下限として読む**。

---

## 9. 変更するファイル

| ファイル | 変更 |
|---|---|
| `src/littraceqa/di_pipeline/agent/reading.py` | `SubqueryRun` / `_merged_results()` / 4〜7 の分岐。`__init__` に新 params |
| `scripts/run_search.py` | `--dump-runs` |
| `scripts/replay_merge.py` | 新規 |
| `configs/agent_style/reading_normal/fat.yaml` | 新規（`retrieve_top_k: 100`、ダンプ土台用） |
| `configs/agent_style/reading_loop/rrf.yaml` | 新規（4） |
| `configs/agent_style/reading_loop/grounded.yaml` | 新規（5。4 も入れる） |
| `configs/agent_style/reading_loop/rescore.yaml` | 新規（6） |
| `configs/agent_style/reading_loop/depth.yaml` | 新規（7） |
| `tests/test_reading_agent.py` | 下記のテストを追加 |
| `CLAUDE.md` | 3節の agent_style 一覧に4枚を追記 |

既存 yaml と `retrieve/` は触らない。

## 10. 実装順

3（土台+ダンプ）→ 4 → 7 → 6 → 5。
4・7・6 はダンプができた時点でオフラインで振れる。5 だけフル走行が要るので最後。

---

## 11. 検証

### テスト（`uv run pytest tests/test_reading_agent.py -q`）

- 新 params を一切書かない構成で、既存テストが全部通る（後方互換）
- 4: 2本のサブクエリが別々の論文を返し片方の絶対スコアが高いとき、`rrf` で順位が
  入れ替わり `max` では変わらない
- 5: `grounded_refine: true` のとき `_refine` のプロンプトに候補タイトルと
  「効かなかったサブクエリ」が含まれる（`FakeLLM` に渡ったプロンプトを見る）
- 6: `pool_rescore: true` でスタブ reranker が**元の質問**で1回だけ呼ばれる。
  `NoneReranker` 構成では呼ばれない
- 7: 平坦なスコア分布で `deep_k`、急峻で `shallow_k` 件が採られる
- `_StubRetriever`（2引数 `retrieve`）が全構成で動く

### オフライン（土台のフル走行1回のあと）

```
mkdir -p logs
tmux new-session -d -s littrace-fat "PYTHONUNBUFFERED=1 uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_qwen3_8b_rerank_qwen3_8b/chunk_attrfilter.yaml \
  --agent configs/agent_style/reading_normal/fat.yaml \
  --queries data/validation_inputs.jsonl --output predictions_fat.jsonl \
  --dump-runs runs_fat.jsonl --production-input 2>&1 | tee logs/fat.log"

uv run python scripts/replay_merge.py --runs runs_fat.jsonl --pred predictions_fat.jsonl \
  --agent configs/agent_style/reading_loop/rrf.yaml --ks 5,10,20,50
```

### 合格条件

`candidate_recall_at20_multi_macro`（現状 0.667）と
`evidence_candidate_recall_at20_total_macro`（現状 0.877）を**両方**並べて読む。
`candidate_recall_at20_single_macro` が 1.000 から下がっていないことを毎回確認する
（飽和側を壊さない）。6 は cr が下がって ecr が上がる形になりやすいので、
その向きが出たら既定オフのまま残す。

### フル走行（5 の評価）

同じ tmux 手順で `--agent configs/agent_style/reading_loop/grounded.yaml`。
結果は `results/experiments.jsonl` と `report/` に自動で残る。走行後は
`scripts/audit_report.py` で HTML を作り直し、同じ Artifact URL に再公開する。
LLM は非決定的なので、結論を出す前に複数回まわす。
