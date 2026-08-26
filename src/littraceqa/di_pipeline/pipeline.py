"""LitTraceQA 検索システムの構成そのもの。**ここを読めば全体が分かる**ようにしてある。

質問1件が候補論文50本になるまでの流れ:

    質問
     ├─ 会議名・年の制約を1回だけ抽出（AttributeExtractor）
     ├─ LLM がサブクエリ4本に分解                     ┐
     │   各サブクエリごとに:                           │
     │     3索引を引く（chunk BM25 / paper BM25 / Qwen3-8B）
     │     論文単位RRF で融合（1論文1票）              │ 反復
     │     Seed Expansion（1位論文の語彙で引き直す）    │ 最大3周
     │     Qwen3-Reranker-8B で採点し、順位融合         │
     ├─ 上位20論文を読解 LLM に読ませ、根拠と充足を判定  │
     └─ 足りなければ不足点から再分解して戻る            ┘
        ↓
    ランキングA（質問→論文）と B（論文→論文展開）を RRF 統合 → 候補50本

各機構の実測と選定理由は CLAUDE.md、実装は各モジュールの docstring にある。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

from littraceqa.chunk_store import ChunkStore
from littraceqa.common import ROOT
from littraceqa.di_pipeline.agent.reading import CombineConfig, ReadingAgent, ReadingConfig
from littraceqa.di_pipeline.index.bm25_index import BM25Index
from littraceqa.di_pipeline.index.bm25_paper_index import BM25PaperIndex
from littraceqa.di_pipeline.index.faiss_qwen3 import (
    INDEX_NAME as QWEN3_INDEX_NAME,
)
from littraceqa.di_pipeline.index.faiss_qwen3 import (
    PRODUCTION_PARAMS as QWEN3_PARAMS,
)
from littraceqa.di_pipeline.index.faiss_qwen3 import Qwen3FAISSIndex
from littraceqa.di_pipeline.llm.azure_openai import AzureOpenAILLM
from littraceqa.di_pipeline.llm.base import LLMClient
from littraceqa.di_pipeline.preprocess.mineru_chunker import MinerUChunker
from littraceqa.di_pipeline.retrieve.attribute_filter import AttributeExtractor
from littraceqa.di_pipeline.retrieve.hybrid import HybridRetriever, RerankBlend, SeedExpansion
from littraceqa.di_pipeline.retrieve.paper_expander import (
    BibCouplingExpander,
    BM25MLTExpander,
    FusedPaperExpander,
    Specter2PaperExpander,
)
from littraceqa.di_pipeline.retrieve.paper_rrf import PaperRRFFuser
from littraceqa.di_pipeline.retrieve.reranker import Qwen3Reranker

# APIキー等はリポジトリ直下の .env から読む（コードにも yaml にも書かない）。
# 既に export されている環境変数は上書きしない。
load_dotenv(ROOT / ".env")

# 前処理の名前。索引ディレクトリの中間パスに入る（`{index_dir}/mineru/bm25s`）ので、
# 別の前処理で作り直しても既存の索引と衝突しない。
PROCESS = "mineru"


@dataclass(frozen=True)
class Paths:
    """実行環境ごとに変わる場所だけ。**手法の設定はここに書かない。**

    `configs/paths/*.yaml` から読む（マシンによって置き場所が違うため yaml に出してある）。
    """

    pdf_dir: Path
    chunks_dir: Path
    index_dir: Path
    paper_metadata: Path

    @classmethod
    def load(cls, path: str | Path) -> Paths:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            pdf_dir=Path(raw["pdf_dir"]),
            chunks_dir=Path(raw["chunks_dir"]),
            index_dir=Path(raw["index_dir"]),
            paper_metadata=Path(raw["paper_metadata"]),
        )

    @property
    def chunks(self) -> Path:
        """前処理が書き出したチャンク（索引と読解の両方が読む）。"""
        return self.chunks_dir / f"{PROCESS}_chunks.jsonl"

    def index(self, name: str) -> Path:
        """索引1本の置き場所。`name` は索引ごとに固有にする（上書き事故を防ぐ）。"""
        return self.index_dir / PROCESS / name


def build_preprocessor(paths: Paths) -> MinerUChunker:
    """PDF → チャンク。`scripts/run_mineru.py` が作った content_list.json を読むだけ。"""
    return MinerUChunker(pdf_dir=str(paths.pdf_dir), max_chars_per_chunk=2000)


def build_indexers(paths: Paths) -> list:
    """索引3本。`--build` はこれと前処理だけを使う（reranker も ChunkStore も要らない）。"""
    return [
        # 質問語が1チャンクに揃うとき強い。
        BM25Index(index_dir=str(paths.index("bm25s"))),
        # 論文全体を1ドキュメントとして引く。質問語が論文内の離れた場所に分散していると
        # chunk 側は弱くなる（1チャンクに語が揃わない）ので併用する。
        BM25PaperIndex(index_dir=str(paths.index("bm25s_paper"))),
        # 語彙が一致しない言い換えを拾う。モデル設定は index/faiss_qwen3.py に
        # 置いてある（分散ビルドのスクリプトと共有するため）。検索時は devices[0]
        # しか使わないので1枚だけ指定し、残りを reranker に空ける。
        Qwen3FAISSIndex(
            index_dir=str(paths.index(QWEN3_INDEX_NAME)),
            devices="cuda:0",
            **QWEN3_PARAMS,
        ),
    ]


def build_retriever(paths: Paths) -> HybridRetriever:
    """質問1本 → チャンクの順位。3索引を論文単位RRFで融合し、reranker と順位融合する。

    **チャンクが既にある前提**（`anchor_store` が読む）。索引構築の実行では
    `build_indexers()` を直接使う。
    """
    return HybridRetriever(
        indexers=build_indexers(paths),
        # **1論文1票。** チャンク単位で融合すると、長い論文・表が多い論文が
        # チャンク数だけで上位を占有する（評価は論文単位なので指標に直接効く）。
        fuser=PaperRRFFuser(k=60, chunks_per_paper=3),
        # マルチGPU 指定時は torch.compile が自動で無効になる（compile 済みモデルを
        # 複数スレッドから呼ぶと dynamo が落ちるため）。モデルの読み込みは遅延。
        reranker=Qwen3Reranker(
            model="Qwen/Qwen3-Reranker-8B",
            devices="cuda:1,cuda:2",
            fp16=True,
            max_batch_tokens=2048,
            batch_size=4,
            max_tokens=2048,
        ),
        per_index_k=100,
        pool_k=200,
        # 1位論文の語彙を質問に足して引き直す。**reranker の前**なので推論は増えない。
        seed_expansion=SeedExpansion(query_chars=512),
        anchor_store=ChunkStore(str(paths.chunks)),
        # reranker に順位を置き換えさせず、融合前の順位と混ぜる。
        rerank_blend=RerankBlend(
            original_weight=0.6, rerank_weight=0.4, rrf_k=60, protect_top=20
        ),
        # 質問が会議名を明示したときだけ発火する。取れなければ従来のコードパス。
        attribute_extractor=AttributeExtractor(paths.paper_metadata),
        fetch_safety=1.5,
        # **上げてはいけない。** per_index_k に合わせて 40000 にしたところ、
        # NAACL(選択率4.3%) で faiss search が 1.5秒 -> 91.1秒（61倍）に膨らんだ。
        max_fetch_k=3000,
        min_filtered_results=10,
    )


def build_expander(paths: Paths) -> FusedPaperExpander:
    """論文→論文の近さ（ランキングB）。**3つは違う gold を拾うので併用する。**

    候補圏外 gold 37本の回収は SPECTER2 15本 / 書誌結合 11本 / 全文MLT 16本で、
    MLT だけが拾えた gold が2本、既存2つだけが拾えたのが6本、重複14本。
    """
    return FusedPaperExpander(
        sources=[
            # SPECTER2(proximity) の近傍。構築済み索引を読むだけ（追加構築なし）。
            Specter2PaperExpander(
                index_dir=str(paths.index("faiss_specter2_abstract")), neighbors=100
            ),
            # 参考文献の arXiv ID 集合の Jaccard。**引用グラフではない**——コーパスは
            # 2024〜2025年しか無く同時期の論文は互いに引用できないので、
            # 共有している古い文献で繋ぐ。min_shared=2 は汎用引用（Adam 等）を切るため。
            BibCouplingExpander(
                chunks=str(paths.chunks),
                cache_path=str(paths.index("bib_coupling") / "refs.pkl"),
                neighbors=100,
                min_shared=2,
            ),
            # 論文全文の more-like-this。構築済みの bm25s_paper 索引を引く。
            BM25MLTExpander(
                index_dir=str(paths.index("bm25s_paper")),
                cache_path=str(paths.index("bm25_mlt") / "anchor_text.pkl"),
                neighbors=100,
                query_chars=1200,
            ),
        ],
        neighbors=100,
        rrf_k=60,
    )


def build_agent(paths: Paths, llm: LLMClient | None = None) -> ReadingAgent:
    """分解 → 読解 → 不足分の再検索 を繰り返し、最後に A/B を統合して候補50本を出す。"""
    return ReadingAgent(
        build_retriever(paths),
        llm=llm or AzureOpenAILLM(reasoning_effort="medium"),
        paper_expander=build_expander(paths),
        # **k=10。** k=60 だとリスト長50本の現状では「A にも B にも載っていれば
        # どれだけ深くても A の1位に勝つ」（2/(61+r) > 1/61 ⟺ r < 61）。
        combine=CombineConfig(
            rrf_k=10,
            related_weight=1.0,
            related_offset=0,
            anchors=1,
            anchor_from="verdict",
        ),
        config=ReadingConfig(
            max_steps=3,
            retrieve_top_k=20,
            max_candidates=20,
            chunks_per_paper=2,
            snippet_chars=1800,
            paper_cutoff="llm",
            max_papers=10,
            # 表チャンクは数値と短いラベルが密で、論文が質問の主題でなくても
            # 表1枚で代表スコアが跳ね上がる。**読解には従来どおり渡る。**
            paper_score_skip_chunk_types=("table",),
        ),
    )
