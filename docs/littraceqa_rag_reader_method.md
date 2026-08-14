# Evidence-grounded scientific-paper reader

## Scope

The system answers questions over a fixed collection of scientific papers.
Its inputs are a structured query, an externally produced candidate-paper
ranking, and a MinerU-derived corpus containing text, tables, figures,
equations, citation contexts, page metadata, and optional images. Retrieval and
reading are deliberately separated: the reader does not search for new papers
or revise the supplied ranking.

The implementation has two model-assisted reading stages surrounded by
deterministic validation.

## System overview

```text
query + candidate papers + extracted corpus
                  |
                  v
        input and corpus validation
                  |
                  v
       Stage 1: paper-level reading
                  |
          validated source chunks
                  |
                  v
       Stage 2: answer construction
                  |
                  v
       typed serialization + checks
```

The language model proposes relevance decisions, evidence-linked facts, and
answers. Python owns identity checks, schema validation, path safety, source-ID
validation, type normalization, deterministic calculations, checkpointing, and
the final freeze boundary.

## Input contracts

Each query contains a stable identifier, question text, requested answer types,
and the schema needed by multiple-choice or table answers. Candidate records
contain paper identifiers and ranks; optional bibliographic fields are
rehydrated from paper metadata. The loader requires exact query coverage,
unique papers, and consecutive ranks, and rejects answer- or evidence-bearing
development fields.

Corpus records retain their owning paper and source type. Locators use the
smallest source-specific identity available: page or section for text, and
visible table, figure, equation, algorithm, or citation identifiers when the
source type provides one. Corpus-supplied image paths are not trusted directly;
validated image suffixes are rebased beneath an explicit image root.

## Stage 1: paper-level reading

Stage 1 processes each query-paper pair independently. It receives the query,
paper metadata, a bounded paper context, and only images that passed local
validation. The model determines whether the paper can contribute to the
answer and identifies the source records needed by the next stage.

Long papers are compacted deterministically before inference. The compactor
preserves source boundaries and prioritizes query-matching passages and
eligible objects rather than splitting one paper into unrelated requests.
Model-returned paper IDs and chunk IDs must exist, belong to the current paper,
and be visible in the supplied context. A claimed visual observation is valid
only when the corresponding image was attached to that call.

For cross-paper tables, a paper may support only one cell fragment of a row
whose remaining cells are owned by another paper. Stage 1 preserves such
fragments as named, grounded units without fabricating a complete row; Stage 2
joins only the accepted source records when constructing the final table.

Every successful judgment is checkpointed before the coordinator proceeds.
Malformed JSON, invented identifiers, missing required images, and provider
errors remain explicit failures or repair attempts; they are never converted
into silent positive judgments.

## Stage 2: answer construction

Stage 2 receives only the accepted source records and bounded same-paper
neighbours needed to interpret them. It constructs the requested answer object
and associates each answer component with source support.

The serializer enforces the declared output types:

- multiple choice uses a declared option label and verifies its option text;
- table rows contain exactly the declared columns and native JSON types;
- number and Boolean cells remain native values;
- prompts require reported string cells to preserve the source form;
- evidence is deduplicated at the evaluator's coarse locator granularity.

Calculations are represented by named source facts and deterministic
operations. The runtime recomputes arithmetic, ratios, counts, extrema,
comparisons, and option mappings rather than accepting an unverified result
string. This separates evidence extraction from deterministic transformation
and makes disagreements between intermediate facts and final answers
detectable.

## Reproducibility

Runs store their configuration, source hashes, per-query checkpoints, provider
attempt records, reading traces, and final prediction.

We recommend reporting:

1. candidate-paper source and version;
2. corpus extraction version and image availability;
3. model deployment and prompt/runtime versions;
4. sampling settings;
5. deterministic validation failures and repairs.

## Design rationale

The system keeps model output narrow where exact structure matters and moves
stable constraints into code. This division provides three benefits. First,
paper and source identities cannot drift through free-form generation. Second,
typed output and calculations can be checked without another model call.

## Limitations

The reader depends on the recall of the external candidate ranking and the
quality of PDF extraction. Some facts are visible only in figures, scanned
tables, or supplementary files; missing images can therefore make a question
unanswerable even when nearby text is available. Equivalent evidence can also
have several valid locators, while an exact evaluator may recognize only one.

Table string answers remain sensitive to the requested representation. A source
may use a full title while a question names an acronym, or may print citation
numbers and units that are not part of the requested cell.

The optional, human-in-the-loop table verification layer is documented
separately in
[`table_verification_protocol.md`](table_verification_protocol.md).
