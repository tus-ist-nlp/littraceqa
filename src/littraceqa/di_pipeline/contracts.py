"""LitTraceQA の各パイプライン段の入出力契約（dataclass）を定義するモジュール。

LitTraceQA は 27,487 件の科学論文コーパスに対する質問について、根拠となる
論文・箇所を検索するシステム。前処理・索引・検索・融合・根拠抽出・提出という
各段は、ここで定義する dataclass を介してデータをやり取りする。段をまたぐのは
この形だけなので、**各段の中身を読まずに境界だけを見れば全体の流れが追える**。

`to_dict()` は `_AsDict` が配る（中身は `dataclasses.asdict()`）。逆向きの
`from_dict()` は外部の jsonl を読む Query / PaperMeta だけが持ち、
**必須フィールドが欠けたら KeyError で落ちる**（黙って None を入れない）。

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

from dataclasses import asdict, dataclass, field


class _AsDict:
    """`to_dict()` を配るだけの土台。中身は `dataclasses.asdict()` に任せる。

    フィールドを手で並べ直した辞書を返していたが、フィールドを1つ足すたびに
    定義・to_dict・（あれば）from_dict の3箇所を直す必要があり、実際に漏れる。
    `asdict()` は定義順にすべてのフィールドを再帰的に展開するので、
    **キーの順序も含めて手書き版と同じ**辞書になる。

    `asdict()` はネストした dataclass（Evidence の locator、Prediction の
    evidence / answer）も展開し、dict や list は**複製する**。手書き版は
    `metadata` などを参照のまま返していたので、返り値を書き換えると元の
    オブジェクトが壊れた。
    """

    def to_dict(self) -> dict:
        return asdict(self)


# gold の `task_family` に入る2値。**本番入力には無い**ので検索側は使わないが、
# 採点（scripts/evaluate.py）が single/multi の内訳を出すのに使う。
SINGLE = "hidden_source_single_paper"
MULTI = "multi_paper"


# 1. Query -- システムへの入力1件（質問）。
#
# 本番の入力に実際に入っているのは query_id / question / answer_types / table_schema の
# 4つだけで、task_family と primary_evidence_type は与えられない（手元の
# validation_inputs.jsonl にはこの2つが入っているが、本番では欠ける）。
# そのため両者は Optional とし、検索側はどちらも使わない（提出本数は max_papers 本で切る）。
@dataclass
class Query(_AsDict):
    query_id: str
    question: str
    answer_types: list[str]
    # 回答が table 型のときだけ与えられる列定義: [{"name": ..., "type": ..., "is_row_key": bool}]
    table_schema: list[dict] | None = None
    # multiple_choice の選択肢 {"A": "...", "B": "..."}。検証データでは gold 側にしか
    # 無いので、ここが埋まるのは選択肢を結合して渡したときだけ。**回答生成は読解チーム側の
    # 担当**なので、検索エージェントはこれを使わない。
    options: dict | None = None
    # 以下2つは本番入力には無い。手元の検証データにだけ入っている。
    task_family: str | None = None  # SINGLE / MULTI（採点の内訳にだけ使う）
    primary_evidence_type: str | None = None  # 観測値: "table" / "figure" / "text_span" / "citation_context" / "equation_algorithm"

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        """入力 jsonl の1レコードから Query を作る。本番に無いフィールドは None になる。"""
        return cls(
            query_id=d["query_id"],
            question=d["question"],
            answer_types=d.get("answer_types") or [],
            table_schema=d.get("table_schema"),
            options=d.get("options"),
            task_family=d.get("task_family"),
            primary_evidence_type=d.get("primary_evidence_type"),
        )


# 2. PaperMeta -- data/paper_metadata.jsonl の1レコード。
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


# 3. Chunk -- 前処理 → 索引 の境界。
@dataclass
class Chunk(_AsDict):
    chunk_id: str  # "{paper_id}#c{idx:04d}" の形式
    paper_id: str
    text: str  # "[{venue} {year}] {title}\n{body}" の形式
    # 観測値: "title_abstract" / "text_span" / "table" / "figure" / "equation_algorithm" / "citation_context"
    chunk_type: str
    metadata: dict


# 4. RetrievalResult -- 索引 → 融合 の境界。
@dataclass
class RetrievalResult(_AsDict):
    chunk_id: str
    paper_id: str
    score: float
    text: str
    chunk_type: str
    metadata: dict
    source: str = ""  # "bm25s" / "faiss" / "colbert" など


# 5. EvidenceLocator -- 根拠1件の位置情報。source_type によって使うフィールドが変わる。
# 全フィールド Optional で定義する。
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


# 6. Evidence -- 提出用の根拠1件。
@dataclass
class Evidence(_AsDict):
    paper_id: str
    source_type: str
    locator: EvidenceLocator
    evidence_text_or_value: str | None = None  # 参照用（提出には不要だが保持）


# 7. Answer -- 提出用の回答1件。answer_types に応じて使うフィールドが変わる。
@dataclass
class Answer(_AsDict):
    freeform: dict | None = None  # 例: {"text": "14.70"}
    multiple_choice: dict | None = None  # 例: {"options": {"A": "...", "B": "..."}, "gold": "C"}
    table: dict | None = None  # 例: {"schema": [{"name": ..., "type": ..., "is_row_key": True}], "rows": [...]}


# 8. Prediction -- Query 1件に対する提出レコード。
@dataclass
class Prediction(_AsDict):
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
