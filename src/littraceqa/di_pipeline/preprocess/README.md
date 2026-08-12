# src/littraceqa/di_pipeline/preprocess/

論文1件（`paper: dict`）からChunk列を作るPreprocessor。前処理ごとに担当範囲が異なり、本文と図表は別々に処理してマージする想定（`scripts/merge_chunks.py`）。

