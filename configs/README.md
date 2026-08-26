# configs/ の使い方

**ここに置くのは実行環境ごとに変わる場所だけ。** 手法の設定（モデル名・k=60・
per_index_k=100 など）は `src/littraceqa/di_pipeline/pipeline.py` にある。

```
configs/
└── paths/
    ├── default.yaml   : このマシン（nlp01）用
    └── nlp02.yaml     : 分散ビルド先のマシン用
```

```yaml
pdf_dir: /data2/iseakira/pdfs/pdfs        # 論文PDF
chunks_dir: /data2/iseakira/pdfs/chunks   # 前処理が書き出すチャンク
index_dir: /data2/iseakira/pdfs/index     # 索引のルート
paper_metadata: data/paper_metadata.jsonl # 論文メタデータ（会議名・年の絞り込みに使う）
```

索引の置き場所は `Paths.index(名前)` が `{index_dir}/mineru/{名前}` に導出する。
前処理名を挟むのは、前処理を変えて作り直しても既存の索引と衝突しないため。

## なぜ手法の設定は yaml に無いのか

以前は前処理・検索手法・エージェントを4つの yaml に分け、registry で名前から実装を
引く DI 構成だった。**いろいろな手法を試すため**の作りで、実際に ablation を回すのには
役立った。いまは論文の最終構成1つを出す段階なので、差し替えの継ぎ目は
「yaml のキー → registry のキー → デコレータ → クラス」という4ホップの読みにくさしか
生まなくなった。

そこで**構成そのものを1つの関数に書き下した**（`pipeline.build_agent`）。
値からクラスへ定義ジャンプで飛べて、システム全体が上から順に読める。

差し替えて実験したくなったら `iseakira/paper-ablation` ブランチに旧構成が残っている。

## 使い方

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl \
  --production-input
  # --build は初回のみ（前処理 + 索引構築）。索引は構築済みなので通常不要

uv run python scripts/evaluate.py --gold data/validation.jsonl --pred predictions.jsonl
```
