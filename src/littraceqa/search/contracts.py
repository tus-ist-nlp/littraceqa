"""The data contracts between pipeline stages.

LitTraceQA answers questions against a corpus of 27,487 scientific papers by
locating the supporting papers and passages. Preprocessing, indexing, retrieval,
fusion, evidence extraction and submission hand data to each other only through
the dataclasses defined here, so **reading just these boundaries is enough to
follow the whole flow** without opening any stage.

`to_dict()` comes from `_AsDict` (which delegates to `dataclasses.asdict()`).
The reverse, `from_dict()`, exists only on Query, which is the one class read from
external jsonl, and it **raises KeyError on a missing required field** rather than
quietly substituting None.

The classes in pipeline order:
    Query            -- one input to the system (a question about the corpus).
    Chunk            -- the preprocessing → indexing boundary.
    RetrievalResult  -- the indexing → fusion boundary.
    EvidenceLocator  -- where one piece of evidence sits (fields vary by source_type).
    Evidence         -- one piece of submitted evidence.
    Answer           -- one submitted answer (freeform / multiple_choice / table).
    Prediction       -- the submission record for one ``Query``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


class _AsDict:
    """Supplies `to_dict()`; the work is `dataclasses.asdict()`.

    These classes used to hand-assemble the dict field by field, which meant
    adding one field required edits in three places (the definition, to_dict and
    from_dict) and was easy to miss. `asdict()` expands every field recursively in
    declaration order, so the result is **identical down to key order**.

    It also expands nested dataclasses (Evidence's locator, Prediction's evidence
    and answer) and **copies** dicts and lists. The hand-written version returned
    `metadata` by reference, so mutating the result corrupted the original object.
    """

    def to_dict(self) -> dict:
        return asdict(self)


# The two values `task_family` takes in the gold data. **It is not on Query** —
# retrieval never reads it; only scoring (scripts/evaluate.py) uses it, and reads it
# from the gold record, to break results down into single/multi.
SINGLE = "hidden_source_single_paper"
MULTI = "multi_paper"


# 1. Query -- one input to the system (a question).
#
# **These four fields are the whole input**, and they are exactly the ones a
# production record and a validation record share. The validation data also carries
# task_family, primary_evidence_type and (joined in from gold) the multiple_choice
# options; none of them are here, because nothing in retrieval ever read them. Which
# papers to submit is simply the top max_papers candidates, and generating the
# answer belongs to the reading team.
@dataclass
class Query(_AsDict):
    query_id: str
    question: str
    answer_types: list[str]
    # Column spec, present only for table answers:
    # [{"name": ..., "type": ..., "is_row_key": bool}]
    table_schema: list[dict] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        """Build a Query from one input jsonl record, ignoring any other field."""
        return cls(
            query_id=d["query_id"],
            question=d["question"],
            answer_types=d.get("answer_types") or [],
            table_schema=d.get("table_schema"),
        )


# 2. Chunk -- the preprocessing → indexing boundary.
@dataclass
class Chunk(_AsDict):
    chunk_id: str  # of the form "{paper_id}#c{idx:04d}"
    paper_id: str
    text: str  # of the form "[{venue} {year}] {title}\n{body}"
    # observed: "title_abstract" / "text_span" / "table" / "figure" / "equation_algorithm" / "citation_context"
    chunk_type: str
    metadata: dict


# 3. RetrievalResult -- the indexing → fusion boundary.
@dataclass
class RetrievalResult(_AsDict):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    chunk_type: str
    metadata: dict
    source: str = ""  # e.g. "bm25s" / "bm25s_paper" / "faiss_qwen3"


# 4. EvidenceLocator -- where one piece of evidence sits. Which fields apply
# depends on source_type, so every field is optional.
@dataclass
class EvidenceLocator(_AsDict):
    page: int | None = None
    # table
    table_id: str | None = None
    row: str | None = None
    column: str | None = None
    # text_span
    section: str | None = None
    paragraph_id: str | None = None
    sentence_start: int | None = None
    sentence_end: int | None = None
    # figure
    figure_id: str | None = None
    region: str | None = None
    # equation_algorithm
    equation_id: str | None = None
    # citation_context
    citation_id: str | None = None
    cited_paper: str | None = None


# 5. Evidence -- one piece of submitted evidence.
@dataclass
class Evidence(_AsDict):
    paper_id: str
    source_type: str
    locator: EvidenceLocator
    evidence_text_or_value: str | None = None  # kept for reference; not required for submission


# 6. Answer -- one submitted answer. Which field applies depends on answer_types.
@dataclass
class Answer(_AsDict):
    freeform: dict | None = None  # e.g. {"text": "14.70"}
    multiple_choice: dict | None = None  # e.g. {"options": {"A": "...", "B": "..."}, "gold": "C"}
    table: dict | None = None  # e.g. {"schema": [{"name": ..., "type": ..., "is_row_key": True}], "rows": [...]}


# 7. Prediction -- the submission record for one Query.
@dataclass
class Prediction(_AsDict):
    query_id: str
    gold_papers: list[dict[str, str]]  # submission format: [{"paper_id": ...}]
    evidence: list[Evidence]
    answer: Answer
    trace: list[dict] = field(default_factory=list)  # debug log, not scored
    # Every paper retrieval collected, in score order (not scored). gold_papers is
    # cut down to a handful, so on its own it cannot tell you afterwards whether
    # retrieval ever had the gold paper in hand. Keeping the top 50 lets us compute
    # recall@5/10/20/50 without re-running (a re-run costs a lot of LLM calls).
    candidate_papers: list[str] = field(default_factory=list)


# ---- which chunk_types an index takes in ------------------------------------
#
# The same set of chunks is handed to every index; this lets each one keep the
# granularity its model was built for. **A model taken off its design granularity
# does not perform:**
#
# * SPECTER2's proximity adapter was trained on title+abstract — a **whole-paper**
#   model. Embedding body fragments, tables or equations separately takes the input
#   off that distribution, so it is given `title_abstract` only.
# * BM25 matches on vocabulary, so granularity does not matter to it. LitTraceQA's
#   questions are full of method, dataset and metric names (PointLoRA, ModelNet40,
#   ECM-XL), which is exactly where exact rare-term matching wins — every chunk.
# * Passage-level dense vectors (Qwen3 and the like) go on body chunks.
#
# **This lives here, with the Chunk contract, rather than beside the indexes**, so
# that faiss_qwen3.py can use it without importing indexes.py — which would drag
# bm25s and the SPECTER2 adapters into the distributed build that only needs torch.

# The chunk_type values MinerUChunker actually emits.
ALL_CHUNK_TYPES = (
    "title_abstract",
    "text_span",
    "table",
    "figure",
    "equation_algorithm",
    "citation_context",
)


def filter_chunk_types(
    chunks: Iterable[Chunk], chunk_types: list[str] | None
) -> list[Chunk]:
    """Keep only the chunks whose chunk_type is listed; None keeps everything."""
    if not chunk_types:
        return list(chunks)

    unknown = set(chunk_types) - set(ALL_CHUNK_TYPES)
    if unknown:
        raise ValueError(
            f"unknown chunk_types: {sorted(unknown)} "
            f"(expected a subset of {list(ALL_CHUNK_TYPES)})"
        )

    wanted = set(chunk_types)
    return [chunk for chunk in chunks if chunk.chunk_type in wanted]


# ---- reading and writing a chunks jsonl -------------------------------------
#
# **The format is defined here and nowhere else.** Every index writes a copy of its
# chunks beside itself, because bm25s and faiss both store vectors or terms and none
# of a Chunk's metadata; that sidecar, in the same row order, is what turns a hit
# back into a Chunk. **Row i of the file must be row i of the index**, so the two
# halves cannot be allowed to drift apart — which is what having six copies of them
# invited.
#
# It lives with the Chunk contract rather than beside the indexes so that
# faiss_qwen3.py can use it without importing indexes.py, which would drag bm25s and
# the SPECTER2 adapters into the distributed build.


def read_chunks_jsonl(path: str | Path) -> list[Chunk]:
    """Read a chunks jsonl. Blank lines are skipped; a bad line names itself."""
    path = Path(path)
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            chunks.append(Chunk(**record))
    return chunks


def write_chunks_jsonl(path: str | Path, chunks: Iterable[Chunk]) -> None:
    """Write a chunks jsonl, one Chunk per line, in the order given."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


# ---- facts more than one module has to agree on -----------------------------
#
# **The writer and the reader of an index directory live in different modules**,
# so a name changed on one side and not the other fails silently: the reader simply
# does not find the file, or finds a stale one. These used to be private constants
# repeated in indexes.py, expander.py and faiss_qwen3.py.

# The faiss index itself: written by Specter2FAISSIndex and Qwen3FAISSIndex,
# opened directly by Specter2PaperExpander.
INDEX_FILENAME = "index.faiss"
# The chunk sidecar (see read_chunks_jsonl): written by BM25Index and by both faiss
# indexes, read back by Specter2PaperExpander.
CHUNKS_FILENAME = "chunks.jsonl"
# The one-document-per-paper sidecar: written by BM25PaperIndex, read by
# BM25MLTExpander as its more-like-this source.
PAPERS_FILENAME = "papers.jsonl"

# How many candidates a Prediction keeps. **scripts/evaluate.py's recall curve
# stops here**, so the two have to agree; importing it is what makes that true
# rather than merely intended. It is a recording limit, not retrieval's — inside a
# run the ranking goes much deeper.
CANDIDATE_PAPERS_LIMIT = 50
