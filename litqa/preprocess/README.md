# litqa/preprocess/

論文metadata 1件から共通`Chunk`列を作るPreprocessorです。

- `pypdf_chunker.py` — `PyPDFChunker`（`pypdf`）: PDF本文をページ単位の
  `title_abstract`／`text_span`へ変換
- `marker_chunker.py` — `MarkerChunker`（`marker`）: Marker出力から本文、図、表、
  数式を共通Chunkへ変換し、保存できた画像パスをmetadataへ保持
- `figure_vlm.py` — `FigureVLMChunker`（`figure_vlm`）: 図表を抽出し、VLMの説明と
  画像パスをfigure／table Chunkへ保持
- `mineru_chunker.py` — `MinerUChunker`（`mineru`／`mineru_v2`）: 既に生成済みの
  `content_list.json`またはページ単位の`content_list_v2.json`を読み込み、MinerUを
  再実行せずtext、figure、table、equationを共通Chunkへ正規化

MinerU Adapterは、元データにあるpage、modality、visible ID、captionだけを保持し、
欠損値を推測しません。画像はまとめて開かず、安全に解決できる既存パスだけを
metadataへ記録します。v1とv2は別のprocess namespaceへ保存するため、同じpaper IDを
比較しても成果物が衝突しません。

`scripts/run_search.py`からのbuildは、対象paper、最大件数、worker、batch size、
読み取り専用root、出力rootを明示するbounded実行です。論文単位のshard、失敗記録、
入力checksum付きstateを使い、条件が一致した成果物だけをresumeします。
