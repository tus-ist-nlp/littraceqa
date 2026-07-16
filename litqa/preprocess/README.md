# litqa/preprocess/

論文1件（`paper: dict`）からChunk列を作るPreprocessor。前処理ごとに担当範囲が異なり、本文と図表は別々に処理してマージする想定（`scripts/merge_chunks.py`）。

- `figure_vlm.py` — `FigureVLMChunker`（"figure_vlm"）: DoclingでPDFから図・表を抽出し、Qwen2-VLで検索用の記述文を生成してチャンク化（figure/table）。本文チャンクは返さない。切り出した画像自体もPNGとして保存し、パスをmetadata["image_path"]に記録する（`litqa/index/siglip_image.py` が画像を直接vision encoderでベクトル化する際に使う）
