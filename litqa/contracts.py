"""LitTraceQA の各パイプライン段の入出力契約（dataclass）を定義するモジュール。

LitTraceQA は 27,487 件の科学論文コーパスに対する質問について、根拠となる
論文・箇所を検索するシステム。前処理・索引・検索・融合・根拠抽出・回答構築・
提出という各段はここで定義する dataclass を介してデータをやり取りする。
契約を固定することで、各段を互いに独立して実装・差し替えできるようにする。

各クラスの役割（パイプライン順）:
    Query            -- システムへの入力1件（コーパスに対する質問）。
    PaperMeta        -- ``data/paper_metadata.jsonl`` の1レコード。
    Chunk            -- 前処理 → 索引 の境界。
    RetrievalResult  -- 索引 → 融合 の境界。
    EvidenceLocator  -- 根拠1件の位置情報（source_type ごとに使うフィールドが変わる）。
    Evidence         -- 提出用の根拠1件。
    Answer           -- 提出用の回答1件（freeform / multiple_choice / table）。
    Prediction       -- ``Query`` 1件に対する提出レコード。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# 1. Query -- システムへの入力1件（質問）。
@dataclass
class Query:
    query_id: str
    question: str
    answer_types: list[str]
    table_schema: list[dict] = field(default_factory=list)
    multiple_choice_options: dict | list | None = None
    # Legacy development labels are retained for input compatibility only.
    task_family: str | None = None
    primary_evidence_type: str | None = None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "task_family": self.task_family,
            "primary_evidence_type": self.primary_evidence_type,
            "question": self.question,
            "answer_types": self.answer_types,
            "table_schema": self.table_schema,
            "multiple_choice_options": self.multiple_choice_options,
        }


# 2. PaperMeta -- data/paper_metadata.jsonl の1レコード。
@dataclass
class PaperMeta:
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

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "venue": self.venue,
            "year": self.year,
            "track": self.track,
            "award": self.award,
            "source_url": self.source_url,
            "pdf_url": self.pdf_url,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "openreview_id": self.openreview_id,
            "anthology_id": self.anthology_id,
        }

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


# 3. Chunk -- 前処理 → 索引 の境界。
@dataclass
class Chunk:
    chunk_id: str  # "{paper_id}#c{idx:04d}" の形式
    paper_id: str
    text: str  # "[{venue} {year}] {title}\n{body}" の形式
    # 観測値: "title_abstract" / "text_span" / "table" / "figure" / "equation_algorithm" / "citation_context"
    chunk_type: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
        }


# 4. RetrievalResult -- 索引 → 融合 の境界。
@dataclass
class RetrievalResult:
    chunk_id: str
    paper_id: str
    score: float
    text: str
    chunk_type: str
    metadata: dict
    source: str = ""  # "bm25s" / "faiss" / "colbert" など

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "score": self.score,
            "text": self.text,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata,
            "source": self.source,
        }


# 5. EvidenceLocator -- 根拠1件の位置情報。source_type によって使うフィールドが変わる。
# 全フィールド Optional で定義する。
@dataclass
class EvidenceLocator:
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

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "table_id": self.table_id,
            "row": self.row,
            "column": self.column,
            "section": self.section,
            "paragraph_id": self.paragraph_id,
            "sentence_start": self.sentence_start,
            "sentence_end": self.sentence_end,
            "figure_id": self.figure_id,
            "region": self.region,
            "equation_id": self.equation_id,
            "citation_id": self.citation_id,
            "cited_paper": self.cited_paper,
        }


# 6. Evidence -- 提出用の根拠1件。
@dataclass
class Evidence:
    paper_id: str
    source_type: str
    locator: EvidenceLocator
    evidence_text_or_value: str | None = None  # 参照用（提出には不要だが保持）

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "source_type": self.source_type,
            "locator": self.locator.to_dict(),
            "evidence_text_or_value": self.evidence_text_or_value,
        }


# 7. Answer -- 提出用の回答1件。answer_types に応じて使うフィールドが変わる。
@dataclass
class Answer:
    freeform: dict | None = None  # 例: {"text": "14.70"}
    multiple_choice: dict | None = None  # 例: {"options": {"A": "...", "B": "..."}, "gold": "C"}
    table: dict | None = None  # 例: {"schema": [{"name": ..., "type": ..., "is_row_key": True}], "rows": [...]}

    def to_dict(self) -> dict:
        return {
            "freeform": self.freeform,
            "multiple_choice": self.multiple_choice,
            "table": self.table,
        }


# 8. Prediction -- Query 1件に対する提出レコード。
@dataclass
class Prediction:
    query_id: str
    gold_papers: list[dict[str, str]]  # 提出フォーマット: [{"paper_id": ...}]
    evidence: list[Evidence]
    answer: Answer
    trace: list[dict] = field(default_factory=list)  # デバッグ用ログ（採点対象外）

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "gold_papers": self.gold_papers,
            "evidence": [item.to_dict() for item in self.evidence],
            "answer": self.answer.to_dict(),
            "trace": self.trace,
        }

    @classmethod
    def from_query(cls, query: Query) -> Prediction:
        return cls(
            query_id=query.query_id,
            gold_papers=[],
            evidence=[],
            answer=Answer(),
            trace=[],
        )
