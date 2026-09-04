# configs/

**Only what differs per machine belongs here.** The method's settings — model
names, k=60, per_index_k=100 — live in
`src/littraceqa/search/pipeline.py`.

```
configs/
└── paths/
    ├── default.yaml   : this machine (nlp01)
    └── nlp02.yaml     : the machine the distributed build runs on
```

```yaml
pdf_dir: /data2/iseakira/pdfs/pdfs        # the paper PDFs
chunks_dir: /data2/iseakira/pdfs/chunks   # the chunks the preprocessing writes
index_dir: /data2/iseakira/pdfs/index     # the root of the indexes
paper_metadata: data/paper_metadata.jsonl # paper metadata (used by the venue/year filter)
```

Where an index lives is derived by `Paths.index(name)` as
`{index_dir}/mineru/{name}`. The preprocessor's name sits in the middle so that
rebuilding with a different preprocessor cannot collide with the existing indexes.

## Why the method's settings are not in yaml

They used to be: preprocessing, retrieval and the agent were split across four yaml
files, with a registry resolving a name to an implementation. That was built **to
try many methods**, and it earned its keep while ablations were being run. Now that
one final configuration is what ships, the seam buys nothing and costs four hops to
read — yaml key, registry key, decorator, class.

So **the configuration is written out as one function** (`pipeline.build_agent`).
Every value jumps to its class in one step, and the whole system reads top to
bottom.

The old arrangement is still on the `iseakira/paper-ablation` branch, for going back
to swapping methods.

## Usage

```bash
uv run python scripts/run_search.py \
  --paths configs/paths/default.yaml \
  --queries data/validation_inputs.jsonl \
  --output predictions.jsonl
  # --build only on the first run (preprocessing + building the indexes); the
  # indexes already exist, so it is normally unnecessary

uv run python scripts/evaluate.py --gold data/validation.jsonl --pred predictions.jsonl
```
