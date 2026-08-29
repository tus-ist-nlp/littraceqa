"""The data contracts between pipeline stages.

LitTraceQA answers questions against a corpus of 27,487 scientific papers by
locating the supporting papers and passages. Preprocessing, indexing, retrieval,
fusion, evidence extraction and submission hand data to each other only through
the dataclasses defined here, so **reading just these boundaries is enough to
follow the whole flow** without opening any stage.

`to_dict()` comes from `_AsDict` (which delegates to `dataclasses.asdict()`).
The reverse, `from_dict()`, exists only on Query and PaperMeta because those read
external jsonl, and it **raises KeyError on a missing required field** rather than
quietly substituting None.

The classes in pipeline order:
    Query            -- one input to the system (a question about the corpus).
    PaperMeta        -- one record of ``data/paper_metadata.jsonl``.
    Chunk            -- the preprocessing → indexing boundary.
    RetrievalResult  -- the indexing → fusion boundary.
    EvidenceLocator  -- where one piece of evidence sits (fields vary by source_type).
    Evidence         -- one piece of submitted evidence.
    Answer           -- one submitted answer (freeform / multiple_choice / table).
    Prediction       -- the submission record for one ``Query``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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


# The two values `task_family` takes in the gold data. **Absent from production
# input**, so retrieval never uses it; scoring (scripts/evaluate.py) uses it to
# break results down into single/multi.
SINGLE = "hidden_source_single_paper"
MULTI = "multi_paper"


# 1. Query -- one input to the system (a question).
#
# Production input carries only four fields: query_id / question / answer_types /
# table_schema. task_family and primary_evidence_type are not provided (the local
# validation_inputs.jsonl has them, production does not), so both are optional and
# retrieval uses neither (submission is simply the top max_papers candidates).
@dataclass
class Query(_AsDict):
    query_id: str
    question: str
    answer_types: list[str]
    # Column spec, present only for table answers:
    # [{"name": ..., "type": ..., "is_row_key": bool}]
    table_schema: list[dict] | None = None
    # multiple_choice options {"A": "...", "B": "..."}. In the validation data these
    # live on the gold side only, so this is populated only when the options are
    # joined in explicitly. **Answer generation belongs to the reading team**, so
    # the retrieval agent never reads it.
    options: dict | None = None
    # The next two are absent from production input; only the local validation data has them.
    task_family: str | None = None  # SINGLE / MULTI (used only to break down scores)
    primary_evidence_type: str | None = None  # observed: "table" / "figure" / "text_span" / "citation_context" / "equation_algorithm"

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        """Build a Query from one input jsonl record; fields absent in production become None."""
        return cls(
            query_id=d["query_id"],
            question=d["question"],
            answer_types=d.get("answer_types") or [],
            table_schema=d.get("table_schema"),
            options=d.get("options"),
            task_family=d.get("task_family"),
            primary_evidence_type=d.get("primary_evidence_type"),
        )


# 2. PaperMeta -- one record of data/paper_metadata.jsonl.
@dataclass
class PaperMeta(_AsDict):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    venue: str
    year: int
    track: str | None
    award: str | None
    source_url: str | None
    pdf_url: str | None
    arxiv_id: str | None
    doi: str | None
    openreview_id: str | None
    anthology_id: str | None

    @classmethod
    def from_dict(cls, d: dict) -> PaperMeta:
        return cls(
            paper_id=d["paper_id"],
            title=d["title"],
            authors=d["authors"],
            abstract=d["abstract"],
            venue=d["venue"],
            year=d["year"],
            track=d.get("track"),
            award=d.get("award"),
            source_url=d.get("source_url"),
            pdf_url=d.get("pdf_url"),
            arxiv_id=d.get("arxiv_id"),
            doi=d.get("doi"),
            openreview_id=d.get("openreview_id"),
            anthology_id=d.get("anthology_id"),
        )


# 3. Chunk -- the preprocessing → indexing boundary.
@dataclass
class Chunk(_AsDict):
    chunk_id: str  # of the form "{paper_id}#c{idx:04d}"
    paper_id: str
    text: str  # of the form "[{venue} {year}] {title}\n{body}"
    # observed: "title_abstract" / "text_span" / "table" / "figure" / "equation_algorithm" / "citation_context"
    chunk_type: str
    metadata: dict


# 4. RetrievalResult -- the indexing → fusion boundary.
@dataclass
class RetrievalResult(_AsDict):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    chunk_type: str
    metadata: dict
    source: str = ""  # e.g. "bm25s" / "bm25s_paper" / "faiss_qwen3"


# 5. EvidenceLocator -- where one piece of evidence sits. Which fields apply
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


# 6. Evidence -- one piece of submitted evidence.
@dataclass
class Evidence(_AsDict):
    paper_id: str
    source_type: str
    locator: EvidenceLocator
    evidence_text_or_value: str | None = None  # kept for reference; not required for submission


# 7. Answer -- one submitted answer. Which field applies depends on answer_types.
@dataclass
class Answer(_AsDict):
    freeform: dict | None = None  # e.g. {"text": "14.70"}
    multiple_choice: dict | None = None  # e.g. {"options": {"A": "...", "B": "..."}, "gold": "C"}
    table: dict | None = None  # e.g. {"schema": [{"name": ..., "type": ..., "is_row_key": True}], "rows": [...]}


# 8. Prediction -- the submission record for one Query.
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
