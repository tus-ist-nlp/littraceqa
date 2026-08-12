"""Qwen3-VL-Reranker による、図表画像も見る reranker。

reranker.py の Qwen3Reranker（テキスト専用）との違い:

* Qwen3Reranker は RetrievalResult.text しか見ない。図表チャンクの text は MinerU が
  起こした表のテキストや図キャプションなので、**図そのものは判断材料に入らない**。
* こちらは metadata["image_path"] を持つチャンクに対して画像も一緒に渡す。
  「Figure 4 のサブ図は何個か」のように図を見ないと答えられない質問で効くはず。

CrossEncoder の predict() は document として
  - str                     … テキストのみ
  - {"text": ..., "image": ...} … テキスト＋画像
の両方を受け付ける（公式 README の使用例）。融合後の候補プールには本文チャンクと
図表チャンクが混在するので、チャンクごとに形を出し分ける。

**隔離 venv (.venv-vl) が必要**: 本体 .venv は pylate(ColBERT) が
sentence-transformers==5.3.0 を固定しているが、Qwen3-VL 系は 5.4.0 以降を要求する。
そのため sentence_transformers の import は _ensure_model() 内に置いてあり、
本体 venv から config.py 経由で import しても壊れない。

コストの注意: reranker は索引構築と違い**クエリのたびに pool_k 件を推論する**。
8B かつ画像込みなので 1 クエリあたりの時間は text 版より大きく伸びる。
pool_k を上げるときは実行時間とセットで見ること。
"""

from __future__ import annotations

from pathlib import Path

from littraceqa.di_pipeline.contracts import RetrievalResult
from littraceqa.di_pipeline.registry import register

# 公式 README の例に準じた既定 prompt。テキストと画像の両方を候補に含める用途なので
# 「images or text」と明示する。LitTraceQA 向けに振るならここがチューニング対象。
_DEFAULT_PROMPT = (
    "Given a scientific question, retrieve figures, tables, or passages from "
    "research papers that help identify or support the answer."
)


@register("reranker", "qwen3_vl")
class Qwen3VLReranker:
    name = "qwen3_vl"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-Reranker-8B",
        device: str = "cuda",
        fp16: bool = True,
        batch_size: int = 4,
        prompt: str = _DEFAULT_PROMPT,
        # 画像を渡す上限。図表チャンクが多いと 8B×画像の推論が支配的になるので、
        # 上位いくつまで画像込みで見るかを絞れるようにしておく（0 なら全件、
        # テキストのみで良ければ use_images=False）。
        use_images: bool = True,
        max_image_docs: int = 0,
    ):
        self.model_name = model
        self.device = device
        self.fp16 = fp16 and str(device).startswith("cuda")
        self.batch_size = batch_size
        self.prompt = prompt
        self.use_images = use_images
        self.max_image_docs = max_image_docs

        self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        self._ensure_model()

        documents = [
            self._as_document(c, rank) for rank, c in enumerate(candidates)
        ]
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, prompt=self.prompt, show_progress_bar=False
        )
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [candidate for candidate, _ in ranked[:top_k]]

    def _as_document(self, result: RetrievalResult, rank: int):
        """候補1件を CrossEncoder に渡す形にする。

        画像を持つチャンクは {"text":..., "image":...}、それ以外は素の str。
        画像ファイルが消えている場合はテキストだけに落とす（実行を止めない）。
        """
        text = result.text
        if not self.use_images:
            return text
        if self.max_image_docs and rank >= self.max_image_docs:
            return text
        image_path = (result.metadata or {}).get("image_path")
        if not image_path or not Path(image_path).exists():
            return text
        return {"text": text, "image": image_path}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # 本体 venv (sentence-transformers 5.3.0) では読めないのでここで import する。
        import torch
        from sentence_transformers import CrossEncoder

        model_kwargs = {"dtype": torch.float16} if self.fp16 else {}
        self._model = CrossEncoder(
            self.model_name, device=self.device, model_kwargs=model_kwargs
        )
