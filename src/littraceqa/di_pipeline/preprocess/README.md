# src/littraceqa/di_pipeline/preprocess/

Preprocessors: turn one paper (`paper: dict`) into a stream of Chunks. Where the
body text and the figures are produced separately, the results are joined with
`scripts/merge_chunks.py`.
