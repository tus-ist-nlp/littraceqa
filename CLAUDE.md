# 開発ルール

## 言語
- 返答は必ず日本語で行うこと

## プロジェクト概要
- LitTraceQA コンペ（EMNLP 2026）の検索システム
- **構成は `di_pipeline/pipeline.py` 1ファイルに書き下してある。**
  質問1件が候補50本になるまでの全段と、その全パラメータがそこで読める
- `contracts.py` が各段の入出力契約（dataclass）
- **手法を差し替える仕組みは持たない。** 以前は registry + 4分割 yaml で
  差し替えられるようにしていたが（ablation を回すのに役立った）、最終構成1つを
  出す段階では「yaml のキー → registry のキー → デコレータ → クラス」という
  4ホップの読みにくさしか生まないので畳んだ。旧構成は `iseakira/paper-ablation` にある
- **retrieval（indexer）の目的は gold paper（正解論文ID）の特定であり、
  根拠チャンクの特定（evidence）は ReadingAgent が別途担当する。** indexer を設計・
  追加するときは chunk 単位の粒度を保つことにこだわらなくてよく、論文単位の識別精度を
  優先してよい（例: `bm25s_paper` は論文全体を1ドキュメントとして扱い、
  `chunk_id` は `"{paper_id}#paper"` という擬似IDで evidence 用途には使わない）。

## コーディング規約
- Python 3.11+
- uv でパッケージ管理
- 型アノテーション必須

---

## このブランチの範囲

**論文（EMNLP 2026 投稿）で最終構成として報告した構成だけを載せた最小セット。**
比較対象として測った他の構成と、それを走らせる道具は入っていない。

| ブランチ | 中身 |
|---|---|
| **`iseakira/paper-final`（これ）** | 最終構成 (d) のみ |
| `iseakira/paper-repro` | + 論文 Table 2/3 の残り3構成 (a)(b)(c) と、それを測る道具 |
| `iseakira/paper-ablation` | + 選定の過程で試した ablation すべて |

最終構成の全体は `src/littraceqa/di_pipeline/pipeline.py` にある。

MinerU で PDF をチャンク化し、chunk BM25 + paper BM25 + Qwen3-Embedding-8B を
**論文単位RRF**で融合、Seed Expansion を通してから Qwen3-Reranker-8B で**順位融合**、
ReadingAgent が読解 LLM の判定を使って反復検索し、最後に質問起点のランキングA と
論文間展開のランキングB を順位融合して候補50本を出す。

validation 55件の `candidate_recall@k`（macro, total）:

| @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|
| 0.5172 | 0.7813 | 0.8662 | 0.9136 | **0.9682** |

**選定基準は @50。** 読解チームは候補50本を順位で閾値化せず集合として使うので、
最適化すべきは上位順位の精度ではなく50本の被覆。

## コードの歩き方

**`src/littraceqa/di_pipeline/pipeline.py` を読めば全体が分かる。** 各段の構成と、
振ったつまみの実効値がすべてそこに書いてある。個々の機構の根拠は下の「各機構の設計根拠」と
各モジュールの docstring にある。

```
pipeline.py          構成そのもの（Paths / build_indexers / build_retriever /
                     build_expander / build_agent）
contracts.py         各段の入出力契約（Query / Chunk / RetrievalResult / Prediction …）
index/               索引3本（bm25s / bm25s_paper / faiss_qwen3_8b）と SPECTER2
retrieve/hybrid.py   検索本体（索引 → 融合 → Seed Expansion → reranker → 順位融合）
retrieve/paper_rrf.py    論文単位RRF（1論文1票）
retrieve/reranker.py     Qwen3-Reranker-8B
retrieve/paper_expander.py  論文→論文展開（ランキングB）
agent/reading.py     反復エージェント（分解 → 読解 → 再検索 → A/B 統合）
```

### 設定を変えるとき

**手法のつまみは `pipeline.py` を直接編集する。** 実行環境の場所だけが
`configs/paths/*.yaml` にある（マシンによって置き場所が違うため）。

- エージェントのつまみ … `ReadingConfig`（`agent/reading.py`）。**存在する param の
  定義はこの dataclass だけ**で、`from_params()` が未知のキーを名前を挙げて弾く
- A/B 統合のつまみ … `CombineConfig`（同上）
- 検索のつまみ … `build_retriever()` の引数

**索引パスは `Paths.index(名前)` で導出する。** 名前が重なると**先に作った索引を
上書きする**（数時間かけたビルドが消える）。`test_pipeline.py` が重複を検知する。

**埋め込みのモデル設定は `index/faiss_qwen3.py` の `PRODUCTION_PARAMS` に置く。**
分散ビルド（`scripts/build_faiss_qwen3_shard.py`）が同じ定数を読むので、
**構築時と検索時でモデルや前置詞がズレる事故が起きない。**

実行例:
```
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --production-input

uv run python scripts/evaluate.py --gold data/validation.jsonl --pred predictions.jsonl
```

索引（`bm25s` / `bm25s_paper` / `faiss_qwen3_8b`）と 27,489件分の chunks は構築済みで、
`--build` なしですぐ検索できる。


---

## 各機構の設計根拠

### 論文単位 RRF（`fuser: paper_rrf`）— 1論文1票

実装は `retrieve/paper_rrf.py`。

素朴に**チャンク単位**で融合すると、**同じ論文の複数チャンクがそれぞれ
独立に票を持つ**。長い論文・表が多い論文はチャンク数が多いだけで上位を占有しやすく、
「論文としてこの質問に近いか」とは別の理由で順位が上がる。評価は論文単位
（`candidate_recall`）なので、この歪みはそのまま指標に効く。

    s(p) = Σ_i  w_i / (k + paper_rank_i(p))
    paper_rank_i(p) = run i の中で p が最初に現れた位置（0起点の密順位）

**1つの run の中では、同じ論文に何チャンク当たっても1票。**

実装で外せない3点:

- **出力はチャンクの列のまま。** reranker も読解も evidence も chunk_id で動く。
  論文の順位を主キー、論文内のチャンク順位を副キーにして並べる。
- **論文内の順序も `score` に載せる。** 下流（`agent/reading.py` の貯め込み・
  `_candidate_papers`・`to_gold_papers`）はすべて score で並べ直すので、返り値の
  並び順だけに順序を持たせると捨てられる。論文内オフセットは `1e-9`。
- **`chunks_per_paper` で1論文の占有を止める。** 制限しないとチャンクを100本持つ論文1本が
  `pool_k` を食い潰し、reranker が1論文しか見なくなる。
  ⚠ **`pool_k` の意味が変わる**（200チャンク → 最低67論文）。推論回数は不変。
- **`bm25s_paper` の擬似チャンク（`{paper_id}#paper`）は代表に選ばない**
  （`PAPER_LEVEL_SOURCES`）。順位付けには使うが chunk_id が実在しないので evidence にも
  読解にも渡せない。ただし**実チャンクが無い論文では擬似チャンクを使う**（候補から消さない）。

### Seed Expansion（`seed_expansion`）— 上位論文から語彙を借りる

実装は `retrieve/hybrid.py` の `_seed_expand()`。**LLM を1回も呼ばない。**

    expanded = 元の質問 + 1位論文の title+abstract の先頭 query_chars 文字

その結果を初回の順位と融合してから reranker に渡す。

**質問文は「その論文が自分をどう呼ぶか」を知らない。** 実例として、ある gold 論文は
自分を一度も `reference-free` と呼ばず `Direct Alignment Algorithm` / `reward shape` と
名乗る——質問文に無い語なので、質問だけを投げ続ける限り当たらない。**上位論文から
コーパス内の語彙を借りる**のがこの機構の役目。

実装で外せない3点:

- **`reranker` の前に置く。** 後ろに置くと reranker の推論が2倍になる。増えるのは索引の
  検索1回ぶんだけ。**reranker は元の質問で1回だけ**走らせる。
- **融合は `self.fuser` にそのまま任せる。** `fuser: paper_rrf` なら論文単位で混ざる。
- **anchor 本文は `ChunkStore` から `title_abstract` チャンクを引く**（`{paper_id}#c0000`）。
  ChunkStore が無い構成ではヒットしたチャンク本文で代用する。

### `rerank_blend` — reranker に順位を置き換えさせない

既定では reranker が RRF 融合後の順位を**完全に置き換える**（`retrieve/hybrid.py`）。
だが reranker は「質問に答えるか」で判定するので、質問文が名指ししないピア gold を必ず下げる。
ランキングB を reranker に通さないのはそのためだが、**ランキングA の内部では無防備なまま**
だった。`rerank_blend` を書くと順位融合になる。

    score(c) = w_orig / (k + rank_fused) + w_rerank / (k + rank_reranked)

**スコアではなく順位だけを見る**（RRF スコアと yes 確率はスケールが違って足せない）。

実装で外せない2点:

- **融合順位を `score` に書き戻す。** 下流はどこも score で並べ直すので、返り値の並び順に
  しか順位が無いと100%捨てられる。**`protect_top` も同じ理由で score に載せる**
  （最大スコア + 1 を足す）。
- **`rerank(query, fused, len(fused))` に変えても推論コストは増えない。**
  `Qwen3Reranker.rerank` は候補を全件スコアしてから `top_k` で切っているだけ。

### ランキング統合（A/B の RRF 融合）

- **A（質問→論文）**: 検索。BM25 + 埋め込み → RRF → reranker
- **B（論文→論文）**: SPECTER2 / 書誌結合 / 全文MLT の RRF 融合。**reranker には通さない**

      score(p) = w_A / (k + rank_A) + w_B / (k + related_offset + rank_B)

**`combine_rrf_k` は 10。** k は「リスト内の順位」と「両方に載っていること」のどちらを
重く見るかを決めるつまみ。k=60 だと A でも B でも r 位の論文が `2/(61+r)` を得るので、
**リスト長50本の現状では両方に載っていればどれだけ深くても A の1位に勝ってしまう**
（`2/(61+r) > 1/61 ⟺ r < 61`）。k=10 なら閾値が `r < 11` になり、「両方の**上位**に
居るときだけ勝つ」という本来の意味になる。`neighbors: 100` とセットで入れる
（片方だけでは効かない）。

**anchor 自身をランキングB の先頭に置くこと。** 各 expander は anchor を自分の近傍から
外すので、そのままだと anchor は A の `1/(k+1)` しか持てず、「A にも B にも居る」論文
（2項ぶん）に軒並み抜かれる。実測で **single_paper の候補1位だった gold が top20 から
消えた**（single の cr@20 が 1.000 -> 0.923）。論文は自分自身に最も近いので、B の1位に
置くのが定義どおりでもある。**この保護があるので A/B 融合は候補1位を入れ替えない**
（論文 Table 2 で (a) と (c) の @1 が一致するのはこのため）。

**素の RRF（`related_weight: 1.0` / `related_offset: 0`）が最良。** 重みを下げると B 単独の
論文が A の裾より下に落ちて統合の意味が消え、上げると B が候補列を占領する。

統合するときは**50本で切る前の全長**をランキングA に使う（`_build_prediction`）。
51位の論文を B が強く推していても、先に切ると押し上げようがないため。

**論文→論文展開の近さは3種類あり、`build_expander()` が RRF で併用する**
（`retrieve/paper_expander.py`）:

- `specter2`: SPECTER2(proximity) 埋め込みの近傍。構築済み索引を再利用（追加構築なし）。
- `bib_coupling`: 書誌結合。参考文献の arXiv ID 集合の Jaccard。初回のみコーパス1走査で
  索引を作りキャッシュ（29〜47秒）。GPU 不要。
- `bm25_mlt`: 論文全文の more-like-this。anchor の title+abstract をクエリにして構築済みの
  `bm25s_paper` 索引を引く。LLM 呼び出しゼロ・追加構築ゼロ。`papers.jsonl`(2.5GB) は
  **クエリ時には読まない**（BM25 本体は `mmap=True`、行番号→paper_id は初回1回だけ pickle 化）。
- `fused`: 上記を RRF 融合（agent yaml の `expansion.sources` に並べると自動でこれになる）。

**併用の根拠は「違う gold を拾う」こと**——候補圏外 gold 37本の回収は SPECTER2 15本 /
書誌結合 11本 / 全文MLT 16本で、**MLT だけが拾えた gold が2本**、既存2つだけが拾えたのが
6本、重複14本。

**書誌結合は引用グラフ（A が B を引く）ではない。** このコーパスは2024〜2025年しか無く
同時期の論文は互いに引用できないので、引用リンクはほぼ張れない（anchor から解決できた
コーパス内引用は実測1本）。共有している**古い文献**で繋ぐのが要点。`min_shared: 2` は
「共有1本だけ」を切るため（Adam や ResNet のような汎用引用で繋がってしまう）。

### `anchor_from: verdict` — 起点に読解 LLM の確認済み論文を足す

`_read_and_judge()` が返す `paper_ids` は本文を読んだうえでの判定なのに、順位付けに一度も
使っていなかった。`anchor_from: verdict` を書くと、ランキングB の起点を「候補1位」から
「候補1位 ∪ LLM 確認済み」に広げる（`agent/reading.py` の `_anchor_papers()`）。

**何を直しているのか。** multi の gold をクエリ内で順位順に並べると、1本目は解けているのに
3本目以降が沈んでいる（1本目 @5 100% / 3本目 @5 27% / 4本目 @5 8%）。anchor が1本だと
展開できるトピッククラスタも1つなので、そこが埋まらない。**伸びは @10 に集中する**
（土台5本で multi@10 が +0.030〜+0.082）。

**候補1位を必ず起点に残すこと。** LLM 確認済みだけにすると候補1位が B の先頭から外れ、
**single の cr@1 が 0.923 → 0.885 に落ちる**。和集合なら single は完全に不変。

**効いているのは精度ではなく本数。** LLM 確認済みの gold 率は 76% で候補1位の 85% より
**低い**。それでも効くのは1本の anchor では1クラスタしか展開できないため。anchor が2本
以上になるのは55件中16件で、**うち14件が multi_paper**（single に副作用が出ないのはこの偏り）。

**`anchors` は上げない。** 土台4本で測ると3本にすると悪化する。2位・3位を B の先頭に据えると、
それ自体が「A にも B にも居る」2項ぶんを得て本来上位に来る論文を押し下げる。

### 表チャンクを「論文の代表スコア」に使わない（`paper_score_skip_chunk_types: [table]`）

`to_gold_papers(agg="max")` は論文の最高スコアのチャンク1つで論文を代表させる。
**そこに表チャンクが選ばれると論文の順位が壊れる。** 表チャンクは数値と短いラベルが密なので、
BM25 も reranker も語の重なりだけで高いスコアを出しやすく、論文が質問の主題でなくても
表1枚で代表スコアが跳ね上がる。

重みを振ると **0.85 以下は完全に同値**なので、**閾値ではなく規則そのものが効いている**。
`w = 0` ＝「表チャンクは代表にしない」と書けて、**自由パラメータが無い**。

**`figure` / `equation_algorithm` を一緒に下げると悪化する。** 落とすのは `table` だけ。
`agg="sum"` にすると効果がほぼ消えるのが傍証で、**これは max 集約に固有の歪み**。

**「表しか無い論文には表スコアを使う」というフォールバックを入れてはいけない。**
親切に見えるが実測で負ける（multi@5 0.758 -> 0.720）。表しか手掛かりが無い論文が488本あり、
**それを沈めること自体が効いている**（スコア0 になるだけで候補列からは消えないので、
ランキングB が押し上げれば戻ってくる）。

**表チャンクを evidence 用途で落としてはいけない。** 重みを掛けるのは論文の代表スコアだけで、
チャンクプールは無変更。表チャンクは読解 LLM にそのまま渡り `evidence` にも出せる
（gold の `primary_evidence_type` は table が17件で最多）。実装は
`RetrievalResult.chunk_type == "table"` を見る（`chunk_id` の接頭ではなく）。

フル走行での効果（`notable` vs 表除外なし）: multi cr@20 0.839 -> **0.859** /
multi ecr@20 0.908 -> **0.925**、**single は1桁も動かない**。

### 属性フィルタ（会議名・年）

`build_retriever()` が `attribute_extractor` を渡すと、質問が明示した会議名で
検索結果を絞り込む（`retrieve/attribute_filter.py`）。**索引の改修も再構築も不要**。
`RetrievalResult.metadata` に既に venue/year が入っているので、各索引から多めに取ってから
落とすだけ。

**`max_fetch_k` を上げると faiss 検索が桁で遅くなる。** `per_index_k: 1000` に合わせて
`max_fetch_k: 40000` にしたところ、NAACL(選択率4.3%)で 34,560件の要求になり
**faiss search が 1.5秒 -> 91.1秒（61倍）に膨らんだ**（実測）。`IndexFlatIP` は全走査して
top-k を選ぶので k が効き、しかも取った件数はフィルタ後に per_index_k へ切られるので大半が
無駄になる。件数が足りなければ `min_results` の fail-open で「絞り込みなし」に戻るだけなので、
**小さく抑えるのが正しい**。

**発火条件は「会議名が一意に取れたとき」だけ。** 次の場合は抽出せず、従来と完全に同一の
コードパスを通る:

- 年しか書かれていない（コーパスは 2025 が91.3%・2024 が8.7%の2値で、絞る意味が薄い）
- `all venues` を含む
- 会議名が2種類以上見つかった（引用先の会議名に引きずられないため）

**年で絞る意味が薄いのは、コーパスで year が venue から一意に決まるから。** (venue, year) の
組は9通りしかなく、**ECCV だけが 2024 で残り8会議は全部 2025**。

**制約は元の質問から1回だけ取り、サブクエリには渡さない。** `_decompose()` が作るサブクエリは
「NAACL 2025」を落とすことがあるため、`run()` で `query.question` から抽出して反復ステップ
全体で使い回す。

検証55件での実測: 発火5件、**gold がその制約を満たす率 18/18 = 100%**。

### サブクエリ生成

**サブクエリの本数は4本固定（`SUBQUERY_COUNT`）。** `_decompose()` は task_family で件数を
振り分けない。**プロンプトで本数を頼むだけでは守られない**ので、返り値も切る
（`subquery_count`、既定4）。以前 `_refine()` は本数を書いていなかったため平均8.2〜9.3本・
最大20本まで膨らんでいた。**サブクエリ1本 = 検索1回 = reranker が `pool_k` 件を推論する量**
なので、これがそのまま走行時間になる。

膨らんだぶんは**検索力に一切効いていない**。812本を1本ずつ抜いて候補列を組み直すと、
**抜くと ecr@50 の gold が減るのは5本だけ**（0.6%。step1 の305本では 0/305）。
**増やすほど良くなる関係にはなっていない**——サブクエリを足すほど比較可能でないスコアと
裾の雑音が混ざって上位が薄まる。

**「Web検索エンジンではない」と書かないと LLM は Google 検索クエリを書く**
（`agent/reading.py` の `CORPUS_NOTE`）。実測で `_refine()` が作るサブクエリの **29〜41%** が
`site:arxiv.org` / `filetype:pdf` のような Web検索演算子付きだった。投げ先はローカルの
BM25 と faiss なのでこれらは1件もヒットせず、2周目・3周目の検索が丸ごと空振りしていた。
`CORPUS_NOTE` は分解・再分解の両方の先頭に置いてある。

**属性制約が取れたときは、サブクエリの先頭に `[NAACL 2025]` を付けさせる**
（`_constraint_note()`）。絞り込み自体は `attribute_filter` が担当するので、これは
**検索語としての**制約。title_abstract チャンクの本文は実際に `[ACL 2025] タイトル…` と
この表記で始まる（`preprocess/mineru_chunker.py`）ので、同じ表記が BM25 の語として効く。

**反復の停止条件は `_read_and_judge()` が返す LLM の `sufficient` 判定のみ。**
提出本数は候補列の先頭 `max_papers` 本で切るだけで、`task_family`（single/multi）には
依存しない——本番入力にその項目が無く、質問から推定しても正解率0.67程度で
当てにならないため（推定器は削除済み）。

### 提出論文は選定しない

**どれを提出するかを決めるのは読解チーム側の担当**なので、検索エージェントは候補列の順位を
渡すところで止める。`gold_papers` は `candidate_papers` の順位そのまま（`max_papers: 10` で
頭打ち）で、読解 LLM が返した `paper_ids` は使わない。

**それでも `_read_and_judge()` は呼ぶ。** 1回の LLM 呼び出しが返す3つのうち、選定
（`paper_ids`）以外の2つは別の役割を持っているため:

- `sufficient` … **反復の停止条件そのもの**。これが無いと `max_steps` 固定になる
- `evidence_chunk_ids` … 根拠チャンク（`evidence_f1`）
- `paper_ids` … `anchor_from: verdict` のランキングB の起点（上記）

### 回答は生成しない

`Prediction.answer` は常に空（`Answer()`）。freeform / multiple_choice / table を埋めるのは
読解チーム側の担当で、検索エージェントが渡すのは **candidate_papers と evidence まで**。

---

## 評価の作法

**目標は `candidate_recall` を上げること。提出物側の指標は既定で出さない。**
`evaluate.py` が返すのは `candidate_recall` / `evidence_candidate_recall` の系列だけで、
`paper_precision` / `paper_recall` / `paper_f1` / `evidence_*` / 回答系は
**`--metrics all` を付けたときだけ**足される。提出論文の選定も回答生成も読解チーム側の
担当なので、我々が動かせない数字を並べると、その上下を改善・悪化として読んでしまうため。

### `evidence_candidate_recall`（ecr@k）— 取りに行ける gold だけの検索力

**multi_paper の gold には、検索では原理的に取れない論文が混ざっている。** `gold_papers` に
名前はあるのに `evidence` が1件も紐づいていない論文が **gold 120本中29本（24%）**あり、
中身は「質問文が名指ししていない同トピックのピア論文」だった。q_036「TCM の batch size は？」
の gold に IMM / sCT / Consistency Models Made Easy が並び、q_039「IMM の kernel function は？」
の gold が**まったく同じ4本**、という作りになっている。質問文が求めているのは「TCM の
batch size」なので、埋め込みを大きくしても reranker を強くしてもそのクエリベクトルの近傍に
ピア論文は来ない。

実測でこの2群は当たり方がはっきり違う（micro）:

| 分母 | @10 | @20 | @50 |
|---|---|---|---|
| 根拠付き 91本 | 0.615 | 0.736 | **0.813** |
| 根拠なし 29本 | 0.103 | 0.207 | **0.345** |

**使い分け:** 索引・fuser・reranker を変えた効果を読むときは ecr を見る。gold 全体で測ると
取れない29本が常に混ざって天井が張り付き、改善が薄まって見える。

根拠付き gold が1本も無いクエリは分母が空になるので集計から除外する（`recall_at_k()` は
gold が空だと 1.0 を返す仕様なので、入れると満点が水増しされる）。

**外部チームと single/multi を突き合わせるときは `..._by_backed_...` を見る。** 同じ ecr を、
**single/multi の振り分けだけ**「根拠付き gold の本数」でやり直した系列を並べて出している。

### 実験は必ず tmux セッションの中で回す

1構成4〜5時間かかるので、端末やエージェントのセッションが切れた時点でプロセスごと落ちると
数時間が丸ごと消える。

```
mkdir -p logs
tmux new-session -d -s littrace-exp \
  "PYTHONUNBUFFERED=1 uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions_{識別子}.jsonl \
  --production-input 2>&1 | tee logs/{識別子}.log"

tmux ls                        # 生きているか
tmux attach -t littrace-exp    # 進捗を直接見る（抜けるのは Ctrl-b d）
```

**`PYTHONUNBUFFERED=1` を付ける。** stdout がパイプ（`tee`）に繋がると Python は
ブロックバッファリングになり、`N/55 完了` の進捗が数十分ぶん溜まってから一気に出る。

予測ファイルは全55件が終わってから一括で書き出される実装なので、**途中経過は
`wc -l predictions_*.jsonl` では測れない**——進捗はログの `N/55 完了` で見る。

**評価は `--production-input` を付けて回す。** `data/validation_inputs.jsonl` は55件すべてに
`task_family` が入っているが、本番入力には無い。与えたまま評価すると「正解を教えてもらった
状態」の点数になり本番と乖離する。

LLM は非決定的（温度指定を受け付けない）でクエリは55件しかないので、数ポイントの差は
ノイズの可能性がある。結論を出す前に複数回まわすこと。

**分割実行（val_a / val_b）は必ず `--merge-with` で結合してから採点する。** `evaluate.py` は
常に55件の gold と突き合わせるので、片側だけを採点すると**全 macro 指標が網羅率のぶん薄まる**。

---

## 本番データで候補列を作る

**「実験を回す」＝本番データで候補列を作ること。** 入力は `data/test_inputs.jsonl`(71件)、
成果物は `data/test_inputs_with_candidates.jsonl`。検証55件での走行は「打ち手を選ぶための
測定」であって、納品物を作る実験ではない。

```
# 1) 検索 -> 予測（tmux の中で）
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --queries data/test_inputs.jsonl \
  --output predictions_test_{識別子}.jsonl

# 2) 予測 -> 受け渡しファイル（gold が無いので --no-gold は必須）
uv run python scripts/build_candidate_handoff.py \
  --predictions predictions_test_{識別子}.jsonl \
  --inputs data/test_inputs.jsonl --no-gold \
  --output data/test_inputs_with_candidates.jsonl
```

**⚠ 本番データには gold が無い。採点してはいけない。** `run_search.py` は最後に必ず
`evaluate.py --gold data/validation.jsonl` を呼ぶが、query_id が `ltqa_*` 形式で検証の `q_*` と
1件も重ならないため、**全指標 0.0 の JSON が正常に返ってくる**（例外では落ちない）。その結果:

- `results/experiments.jsonl` に **全部 0 の行**が追記される
- `report/*.md` が1枚書かれ、LLM コメントが「全指標が壊滅的に悪化した」と書く

`--skip-eval` のようなフラグは無い。**本番走行のあとは `results/experiments.jsonl` の末尾行と
`report/` の該当ファイルを消すこと**（消し忘れると以後の `generate_comment()` がその 0 行を
「前回」として読む）。手法の良し悪しは従来どおり `data/validation_inputs.jsonl` の55件で測る。

**本番入力のフィールドは6つ**（`query_id` / `benchmark` / `question` / `answer_types` /
`multiple_choice_options` / `table_schema`）。`answer_types` は multiple_choice 50 / table 21 で
**freeform は0件**。`task_family` / `primary_evidence_type` は無いので `--production-input` は
付けても付けなくても結果が同じ。

---

## 隔離 venv が必要な前処理

MinerU は本体と依存が両立しない（transformers / torch / requires-python が衝突）。
PDF → content_list.json の変換だけを隔離 venv で先に済ませ、本体の `MinerUChunker` は
その成果物を読むだけにする。

```
bash scripts/setup_mineru_env.sh   # 初回のみ（.venv-mineru を作りモデルを取得）
.venv-mineru/bin/python scripts/run_mineru.py \
  --paths configs/paths/default.yaml --gpus 0,1,2,3
```

出力先は `pdf_dir` の兄弟 `mineru/`（構成にパスを直書きしない方針に従い、
`MinerUChunker` が自動導出する）。27,489件で 4GPU 約25時間。変換済みの論文は飛ばすので、
中断しても同じコマンドで再開できる。

---

## GPU の割り当て

**`device` / `devices` は実行時に空いているGPUへ書き換える前提の値。** yaml に書いてあるのは
書いた時点で空いていたGPUなので、埋め込み索引の構築と同居させないこと（8B系は fp16 でも
約15〜18GB占有し、RTX3090(24GB) では KV cache 分の余裕が薄い）。

**`devices` で複数GPUを指定するとスレッド並列になる**（CUDA forward が GIL を解放する）。
このとき **`torch.compile` は自動で無効化される**——compile 済みモデルを複数スレッドから
呼ぶと dynamo が落ちるため。compile は実測でほぼ無効果（188 vs 212ms）なので損失はない。

**`batch_size` ではなく `max_batch_tokens` で制御する。** 論文代表テキストは長さがばらつく
（実測 中央313〜661tok・max 2116）ため、件数固定だと長い外れ値で `batch_size × 最長` が
VRAMを食い、**batch_size=4 でピーク22GB、8/16 は即OOM**した。
