# configs/ の使い方

## コンセプト

前処理・検索手法・エージェント・共有パスは、それぞれ独立に差し替え可能な4つの軸として分離されている。
1つのyamlに全部まとめず、**4フォルダから1ファイルずつ選んで組み合わせて使う**。

```
configs/
├── paths/           共有パス（pdf_dir, index_dirのルート等）
├── process_style/    前処理（Preprocessor）
├── search_style/     検索手法（Indexer群 + Fuser + Reranker）
└── agent_style/       エージェント（Agent）
```

これは `src/littraceqa/di_pipeline/` 側のDI設計（`registry.py` で `@register(kind, name)` したクラスを
`registry.build(kind, name, **params)` で組み立てる仕組み）をそのままconfigの
ファイル単位に反映したもの。`src/littraceqa/di_pipeline/config.py` の `compose_config()` が4つの
dictを合成し、`build_pipeline()` に渡す。

## なぜ分けているか

- **本文チャンク(mineru)を図表チャンク(figure_vlm)に差し替えても、検索手法やエージェントの設定を書き直さなくていい**
- **同じ検索手法(search_style)を別の前処理(process_style)と組み合わせても、索引の保存先が衝突しない**
  - `process_style`/`search_style` のファイルには `pdf_dir`/`index_dir` を書かない
  - `compose_config()` が `paths` から `{index_dir}/{process名}/{indexer名}` のように自動導出する
  - 例: `mineru + bm25s` → `index/mineru/bm25s`、`figure_vlm + bm25s` → `index/figure_vlm/bm25s`（別物として保存される）
- 新しい手法を1つ追加したいだけなのに、既存の組み合わせファイルを全部複製・修正する必要がない

## 使い方

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries data/validation_inputs.jsonl \
  --output ~/littraceqa_data/mineru_eval/unused.jsonl \
  --artifact-root ~/littraceqa_data/mineru_eval/smoke \
  --limit 3 --max-chars-per-chunk 4000 \
  --build --build-only --resume
```

`--build` always requires `--artifact-root` and either `--paper-id` or
`--limit` (maximum 5,000). Selecting more than 200 papers additionally requires
`--paper-ids-file` and an exact `--confirm-paper-count`. `--build-only` avoids
constructing or calling an LLM agent. Prebuilt shared indexes are read-only and
normally do not need rebuilding.

`--resume` validates and reuses an atomic Chunk file for each paper. It also
loads each completed Indexer after verifying the merged-corpus, config, and
implementation signatures. A failed paper is retried without reprocessing
successful papers, and a failed later Indexer does not rebuild verified earlier
Indexers.

Search configs with `resumable_build: true` checkpoint BM25 tokenization and
scoring in bounded batches. The final index still uses one corpus-wide document
frequency and IDF definition, so it is equivalent to a normal global bm25s
index rather than a mixture of independently scored shards. A completed
generation is published through an atomic `CURRENT.json` update; an interrupted
generation remains hidden and can be resumed with the same artifact root and
settings. Concurrent builders for one BM25 index root are rejected.

Components also declare implementation dependencies, so changing method-alias
extraction or a parser helper invalidates stale output.

By default, paper checkpoints are stored below
`<artifact-root>/preprocess`. Pass a user-owned
`--preprocess-cache-root` to share those checkpoints across separate artifact
roots, for example when expanding a 3,000-paper corpus to 5,000 papers. The
cache must not overlap the read-only root or any preprocessing input root. Do
not run concurrent writers against the same shared cache. Internal cache
symlinks are rejected so writes cannot escape into a protected input tree.
Artifacts created before paper checkpoints were introduced do not contain a
reusable cache; this option does not silently trust or import legacy Chunk
files.

Use nested corpora containing the same gold papers when comparing retrieval at
different scales. The generator checks only the exact selected MinerU paths
instead of traversing the full MinerU root.

```bash
uv run python scripts/create_controlled_retrieval_corpus.py \
  --metadata data/paper_metadata.jsonl \
  --validation data/validation.jsonl \
  --mineru-root /data2/iseakira/pdfs/mineru \
  --output-root ~/littraceqa_data/mineru_eval/accuracy_ladder \
  --sizes 500 1000 2000 3000 5000
```

The generated sets are nested (`500 ⊂ 1000 ⊂ 2000 ⊂ 3000 ⊂ 5000`).
Existing manifests are not overwritten unless `--overwrite` is explicit.

Use the same shared cache for both bounded builds while keeping their indexes
and merged Chunk files separate:

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_paper_rank_seed_expansion_qwen3_reranker.yaml \
  --agent configs/agent_style/reading.yaml \
  --queries data/validation_inputs.jsonl \
  --output ~/littraceqa_data/mineru_eval/accuracy_ladder/accuracy_5000/sparse/unused.jsonl \
  --artifact-root ~/littraceqa_data/mineru_eval/accuracy_ladder/accuracy_5000/sparse \
  --preprocess-cache-root ~/littraceqa_data/mineru_eval/preprocess_cache/mineru_2000 \
  --paper-ids-file ~/littraceqa_data/mineru_eval/accuracy_ladder/accuracy_5000/paper_ids_5000.txt \
  --limit 5000 --confirm-paper-count 5000 \
  --build --build-only --resume
```

The 5,000-paper BM25 indexes are globally rescored even when 3,000 preprocessing
checkpoints are reused. Token and score checkpoints make an interrupted build
resumable within that 5,000-paper artifact root, while corpus expansion still
recomputes global statistics. This preserves corpus-wide IDF and prevents an
invalid mix of independently built BM25 shards.

The final Qwen3 search config uses a bounded paper-level embedding sidecar.
The sidecar reuses precomputed title/abstract vectors and does not load an
embedding model during retrieval. Build it once for the exact corpus paper IDs:

```bash
uv run --with faiss-cpu==1.14.3 python scripts/subset_paper_embeddings.py \
  --source-index-dir /path/to/read-only/faiss_specter2_abstract \
  --paper-ids-file /path/to/paper_ids_5000.txt \
  --output-dir /path/to/corpus/index/mineru/specter2_paper_embeddings \
  --shared-read-only-root /path/to/read-only/root \
  --expected-count 5000 \
  --max-papers 5000
```

The search config keeps positions 1-10 unchanged and applies paper similarity
only to positions 11-20. It uses both an explicitly identified method-owner
paper and the normal rank-one retrieval result as bounded seeds, deduplicates
them, and allows at most two new papers in total. It then keeps positions 1-19
fixed and uses position 20 as a conservative exploration slot. An explicit
high-confidence reciprocal check runs first: a new paper must be found within
the top 20 neighbors of one of the leading eight papers, and at least six of
those eight papers must occur in the new paper's own top 10 neighbors. At most
32 forward candidates receive that reverse lookup, which bounds the additional
full-matrix searches. An explicit method edge may otherwise fill the slot only
when an independent owner-text search ranks the linked paper in its top five.
The final fallback requires at least two of the leading three papers to agree
on the same top-15 embedding neighbor. The reciprocal thresholds were selected
on the current validation set, remain opt-in through these experimental
configs, and require held-out verification. If either supporting index is
absent or invalid, retrieval falls back to the original sparse candidate tail.

`bm25_paper_rank_seed_expansion_qwen3_reranker.yaml` fixes a 50-paper candidate
set and reranks all 50 papers only after dense-tail and consensus exploration
finish. A caller can request a smaller final result, such as the 20 papers used
by the reading agent. Qwen scores the first 2,000 characters of each paper-level
document, while the returned retrieval results retain their original evidence
chunks. The model revision and bfloat16 dtype are explicit, and the configured
rank fusion gives a slight preference to the original retrieval rank. Qwen can
reorder the original top 20, but those 20 papers remain protected from
lower-ranked candidates because the reading agent consumes 20 papers. The
weight and protection boundary were selected conservatively on the current
validation set and should be rechecked on held-out data. Run the model from a
local cache in offline mode so evaluation cannot trigger a download.

The same config also enables a guarded exploration slot for open-set
enumeration questions such as `Which ... papers` or `what ... does each
method`. Only those questions expand from the next four unique seed papers.
A new paper must occur in at least two independent searches and reach rank two
in one of them. The selected paper is inserted at rank 20 after Qwen reranking,
so the baseline's final top 19 papers remain unchanged. The gate reads only the
question text and does not use gold papers, `task_family`, or
`primary_evidence_type`. Disable this lane by setting `open_set_seed_k: 1`.

Evaluate the fixed retrieval baseline directly against the working
answer-bearing gold without calling an answer-generation agent:

```bash
uv run python scripts/eval_retrieval.py \
  --paths <user-owned-paths.yaml> \
  --process configs/process_style/mineru.yaml \
  --search configs/search_style/bm25_paper_rank_seed_expansion_qwen3_reranker.yaml \
  --queries data/validation_answer_bearing_gold_draft.jsonl \
  --ks 1,5,8,10,20,50 \
  --output <user-owned-output.json>
```

組み合わせを変えたいときは、該当する引数だけ差し替える。他の3つはそのままでよい。

```bash
# 検索手法だけColBERTに変える
  --search configs/search_style/bm25_colbert.yaml

# 前処理を図表チャンク(figure_vlm)に変える
  --process configs/process_style/figure_vlm.yaml
```

4フォルダのファイルはどう組み合わせても壊れない設計なので、新しいyamlを
書く必要があるのは「まだ存在しない前処理・検索手法・エージェント自体」を
追加するときだけ。

## 現在のファイル一覧

```
configs/
├── paths/
│   └── default.yaml
├── process_style/
│   ├── marker.yaml           : PDFをブロック単位でチャンク化
│   ├── mineru.yaml           : MinerU。事前に scripts/run_mineru.py で変換が必要（デフォルト、構築済み）
│   └── figure_vlm.yaml       : Docling+Qwen2-VLで図表をチャンク化
├── search_style/
│   ├── bm25.yaml             : BM25 単体
│   ├── bm25_qwen3.yaml       : BM25 + Qwen3-Embedding-8B
│   ├── bm25_colbert.yaml     : BM25 + ColBERT
│   ├── bm25_specter2.yaml    : BM25 + SPECTER2（全チャンク版）
│   ├── bm25_qwen3_siglip.yaml : BM25 + Qwen3-Embedding-8B + SigLIP（図表画像を直接embedding）
│   ├── bm25_paper_rank_seed_expansion_qwen3_reranker.yaml : 論文単位候補補充 + Qwen3 Reranker
│   └── abstract_specter2_body_qwen3.yaml : BM25 + SPECTER2(title_abstractのみ) +
│         Qwen3-Embedding-0.6B(本文のみ)。各モデルを設計どおりの粒度で使う（デフォルト、構築済み）
└── agent_style/
    └── reading.yaml          : 分解→読解→不足分の再検索を繰り返す唯一の本命（デフォルト）
```

推奨デフォルトの組み合わせ: `process_style/mineru.yaml` + `search_style/abstract_specter2_body_qwen3.yaml` + `agent_style/reading.yaml`

## 新しい手法を追加するとき

新しい Indexer / Preprocessor / Agent を実装したら、対応するフォルダに
設定ファイルを1つ追加する。詳しいルールは `CLAUDE.md` の
「検索手法を追加するときのルール」を参照。
