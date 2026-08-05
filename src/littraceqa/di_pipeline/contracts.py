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
#
# 現行の公式入力は query_id / benchmark / question / answer_types と、回答形式に応じた
# multiple_choice_options / table_schema から成る。task_family と
# primary_evidence_type は開発用 validation にしか無く、本番では与えられない。
# そのため後二者は後方互換のためだけに Optional で保持し、本番入力を直列化する
# ``to_dict`` には含めない。
@dataclass
class Query:
    query_id: str
    question: str
    answer_types: list[str]
    # 回答が table 型のときだけ与えられる列定義: [{"name": ..., "type": ..., "is_row_key": bool}]
    table_schema: list[dict] | None = None
    # 内部では参照しやすい label -> text の順序付き辞書に正規化する。公式 JSONL は
    # ``multiple_choice_options: [{"label": "A", "text": "..."}, ...]`` 形式。
    # 選択肢数は4個とは限らず、E などの label も有効。
    options: dict[str, str] | None = None
    benchmark: str = "LitTraceQA"
    # 以下2つは本番入力には無い。手元の検証データにだけ入っている。
    task_family: str | None = None  # 観測値: "hidden_source_single_paper" / "multi_paper"
    primary_evidence_type: str | None = None  # 観測値: "table" / "figure" / "text_span" / "citation_context" / "equation_algorithm"

    def __post_init__(self) -> None:
        if self.options is not None:
            self.options = _normalize_multiple_choice_options(
                self.options, field_name="options"
            )

    @property
    def option_labels(self) -> tuple[str, ...]:
        """Return valid multiple-choice labels in the released input order."""

        return tuple(self.options or ())

    def to_dict(self) -> dict:
        """Serialize the released participant-input shape.

        Development-only classifier hints and the ambiguous legacy ``options``
        mapping are deliberately omitted.
        """

        output = {
            "query_id": self.query_id,
            "benchmark": self.benchmark,
            "question": self.question,
            "answer_types": self.answer_types,
        }
        if self.options is not None:
            output["multiple_choice_options"] = [
                {"label": label, "text": text}
                for label, text in self.options.items()
            ]
        if self.table_schema:
            output["table_schema"] = self.table_schema
        return output

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        """Create a query from current official or legacy input records.

        The current official list form is canonical.  A top-level legacy
        ``options`` mapping remains accepted so old validation scripts can be
        migrated without copying gold answers into prompts.
        """

        official_options = d.get("multiple_choice_options")
        legacy_options = d.get("options")
        if official_options is not None and legacy_options is not None:
            normalized_official = _normalize_multiple_choice_options(
                official_options, field_name="multiple_choice_options"
            )
            normalized_legacy = _normalize_multiple_choice_options(
                legacy_options, field_name="options"
            )
            if normalized_official != normalized_legacy:
                raise ValueError(
                    "multiple_choice_options and legacy options disagree"
                )
            options = normalized_official
        else:
            raw_options = (
                official_options if official_options is not None else legacy_options
            )
            options = (
                _normalize_multiple_choice_options(
                    raw_options,
                    field_name=(
                        "multiple_choice_options"
                        if official_options is not None
                        else "options"
                    ),
                )
                if raw_options is not None
                else None
            )
        return cls(
            query_id=d["query_id"],
            question=d["question"],
            answer_types=d.get("answer_types") or [],
            table_schema=d.get("table_schema"),
            options=options,
            benchmark=str(d.get("benchmark") or "LitTraceQA"),
            task_family=d.get("task_family"),
            primary_evidence_type=d.get("primary_evidence_type"),
        )


def _normalize_multiple_choice_options(
    raw_options: object, *, field_name: str
) -> dict[str, str]:
    """Normalize official list and legacy mapping option shapes."""

    if isinstance(raw_options, dict):
        items = list(raw_options.items())
    elif isinstance(raw_options, list):
        items = []
        for position, item in enumerate(raw_options, start=1):
            if not isinstance(item, dict) or set(item) != {"label", "text"}:
                raise TypeError(
                    f"{field_name}[{position}] must contain only label and text"
                )
            items.append((item["label"], item["text"]))
    else:
        raise TypeError(f"{field_name} must be a list or mapping")

    normalized: dict[str, str] = {}
    for raw_label, raw_text in items:
        if not isinstance(raw_label, str) or not raw_label.isascii():
            raise TypeError(f"{field_name} labels must be strings")
        label = raw_label.strip().upper()
        if field_name == "multiple_choice_options" and raw_label != label:
            raise ValueError(
                "multiple_choice_options labels must already be uppercase A-Z "
                f"letters: {raw_label!r}"
            )
        if len(label) != 1 or not ("A" <= label <= "Z"):
            raise ValueError(
                f"{field_name} label must be one uppercase A-Z letter: {raw_label!r}"
            )
        if label in normalized:
            raise ValueError(f"duplicate multiple-choice label: {label}")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"{field_name} option {label} has empty text")
        normalized[label] = raw_text.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


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
    algorithm_id: str | None = None
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
            "algorithm_id": self.algorithm_id,
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
    # 検索が集めた候補論文をスコア順に並べたもの（採点対象外）。gold_papers は LLM の
    # 選定と cutoff で数本に絞られるため、これだけでは「検索が gold を候補に拾えて
    # いたか」を後から測れない。上位50本を残せば、予測を作り直さずに
    # recall@5/10/20/50 まで計算できる（再実行は LLM コストが高いので多めに持つ）。
    candidate_papers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "gold_papers": self.gold_papers,
            "evidence": [item.to_dict() for item in self.evidence],
            "answer": self.answer.to_dict(),
            "trace": self.trace,
            "candidate_papers": self.candidate_papers,
        }

    @classmethod
    def from_query(cls, query: Query) -> Prediction:
        return cls(
            query_id=query.query_id,
            gold_papers=[],
            evidence=[],
            answer=Answer(),
            trace=[],
            candidate_papers=[],
        )
