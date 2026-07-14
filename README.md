# LitTraceQA

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).
Install Git LFS **before** cloning — `data/paper_metadata.jsonl` is stored via
LFS and will otherwise be fetched as a broken pointer file.

```bash
git lfs install
git clone git@github.com:tus-ist-nlp/littraceqa.git
cd littraceqa
uv sync
uv run pytest
```

`uv sync`は、ロック済みのPythonパッケージとテスト環境をリポジトリ内の`.venv`へ
構築します。`pyproject.toml`ではPyTorchを明示的なCPU wheel indexへ割り当てているため、
通常の同期でCUDA runtime一式を取得しません。GPU環境を使う場合は、共有設定を
そのまま変更せず、別途依存方針を確認してください。

次のデータはリポジトリや`uv sync`には含まれず、自動ダウンロードもされません。

- 論文PDF
- ビルド済みMinerU出力
- BGE-M3の固定revision snapshot
- 実験で構築したChunk、索引、ranking、評価結果

これらは実行者が読み取り可能な外部パスから明示的に渡し、出力はユーザー領域の
別ディレクトリへ保存してください。BGE-M3設定は`local_files_only: true`であり、
実行中にモデルを取得しません。

## Safe MinerU run

MinerU v1は`configs/process_style/mineru.yaml`、ページ単位のv2は
`configs/process_style/mineru_v2.yaml`で選択します。共有入力は読み取り専用として
扱い、`--paper-id`と`--limit`で対象を限定し、`--artifact-root`を入力外へ置きます。
具体的なコマンドは[configs/README.md](configs/README.md)を参照してください。

## Evaluation status

現在の数値は、正解論文を含む100論文または200論文と55問を使った
controlled diagnosticです。
27,487論文の本番規模を表す結果ではありません。この限定条件では、
BM25、論文単位BM25、BGE-M3をPaperRank RRFで融合した方式は、特に複数goldの
平均Recall@10〜20を改善しました。一方で、問題ごとの全gold回収率は一貫して
改善しないため、両方を分けて報告します。評価はgold論文数が1件の質問と
複数件の質問を分けて集計し、
`task_family`と`primary_evidence_type`を検索条件には使用しません。

詳細な設定、数値、再現条件は
[configs/search_style/README.md](configs/search_style/README.md)に記載しています。
