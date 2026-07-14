# litqa/index/

共通`Chunk`列から索引を構築し、質問に対して`RetrievalResult`を返す層です。

今回の比較で使うIndexer:

- `bm25_index.py` — `BM25Index`（`bm25s`）: Chunk単位の疎検索
- `paper_bm25.py` — `PaperBM25Index`（`paper_bm25`）: paperごとに連続する
  共通Chunkを1件の論文全文へストリーミング集約し、粗い論文検索を行う
- `bge_m3_index.py` — `BGEM3NumpyIndex`（`bge_m3_numpy`）: 観測済みの
  `title_abstract` ChunkをBGE-M3でL2正規化し、NumPy内積によるexact dense検索を行う

BGE-M3の比較入力は`common_chunk`です。モデルID、revision、最大token数、対象Chunk、
checksumを索引設定へ保存し、異なる条件の索引を誤って再利用しないようにします。
モデルsnapshotは外部入力で、`local_files_only`設定により自動取得しません。

索引全体を一括でメモリへ載せる現在の実装は100論文のcontrolled diagnostic向けです。
全コーパスへ適用する前に、batch単位の埋め込み、逐次追加、on-disk metadata、shard、
中断再開を実装する必要があります。
