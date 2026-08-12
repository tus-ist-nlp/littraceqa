#!/usr/bin/env python3
"""BM25 単体のパラメータ総当りスイープ。

巨大コーパス（~250万チャンク・3.8GB）を毎回ディスク索引化すると chunks.jsonl が
索引ごとに丸ごと複製されて破綻するので、ここではコーパスを **トークナイズ1回** だけ
メモリ上で行い、スコアリング系パラメータ（method / k1 / b）は同じトークン列を使い
回して BM25 を貼り直すだけで総当りする。トークナイズ系の軸（stopwords / stemmer）が
変わるときだけ再トークナイズする。

評価は scripts/eval_retrieval.py と同じく agent を通さない生のランキングに対する
recall@k / precision@k / hit_rate@k（gold paper 単位）。チャンク→論文の集約は
to_gold_papers と同じ max 集約に加えて sum 集約も併記する。

    uv run --with pystemmer python scripts/sweep_bm25_params.py \
      --chunks /data2/iseakira/pdfs/chunks/mineru_chunks.jsonl \
      --queries data/validation.jsonl \
      --out results/bm25_sweep.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import bm25s

try:
    import Stemmer  # PyStemmer
except ImportError:  # stemmer 系の変種は Stemmer が無ければ自動でスキップする
    Stemmer = None

KS = [5, 10, 20, 50]
RETRIEVE_K = 300  # recall@50(paper) がチャンク切り詰めで過小評価されない程度に多めに取る


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    paper_ids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            paper_ids.append(rec["paper_id"])
    return texts, paper_ids


def load_queries(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold = {p["paper_id"] for p in rec.get("gold_papers", []) if p.get("paper_id")}
            out.append({"question": rec["question"], "gold": gold})
    return out


def rank_papers(doc_ids, scores, paper_ids: list[str], agg: str) -> list[str]:
    paper_score: dict[str, float] = {}
    for doc_id, score in zip(doc_ids, scores):
        pid = paper_ids[int(doc_id)]
        s = float(score)
        if agg == "max":
            paper_score[pid] = max(paper_score.get(pid, s), s)
        else:  # sum
            paper_score[pid] = paper_score.get(pid, 0.0) + s
    return sorted(paper_score, key=lambda p: paper_score[p], reverse=True)


def evaluate(retriever, query_tokens_list, queries, paper_ids, agg: str) -> dict:
    recall = {k: 0.0 for k in KS}
    precision = {k: 0.0 for k in KS}
    hit = {k: 0 for k in KS}
    n = len(queries)
    for qtok, q in zip(query_tokens_list, queries):
        doc_ids, scores = retriever.retrieve(qtok, k=RETRIEVE_K, show_progress=False)
        ranked = rank_papers(doc_ids[0], scores[0], paper_ids, agg)
        gold = q["gold"]
        for k in KS:
            topk = set(ranked[:k])
            inter = gold & topk
            recall[k] += (len(inter) / len(gold)) if gold else 1.0
            precision[k] += len(inter) / k
            if inter:
                hit[k] += 1
    return {
        "recall": {k: recall[k] / n for k in KS},
        "precision": {k: precision[k] / n for k in KS},
        "hit_rate": {k: hit[k] / n for k in KS},
    }


def tokenization_variants() -> list[dict]:
    variants = [
        {"tag": "en", "stopwords": "en", "stemmer": None},
        {"tag": "none", "stopwords": None, "stemmer": None},
    ]
    if Stemmer is not None:
        variants.append({"tag": "en+stem", "stopwords": "en", "stemmer": "english"})
    return variants


def scoring_combos() -> list[dict]:
    combos: list[dict] = []
    # 1) method=lucene 固定で k1×b を掃く（デフォルトは k1=1.5, b=0.75）
    for k1 in (0.9, 1.2, 1.5, 2.0):
        for b in (0.4, 0.75, 1.0):
            combos.append({"method": "lucene", "k1": k1, "b": b})
    # 2) k1=1.5, b=0.75 固定で method を掃く
    for method in ("atire", "bm25+", "bm25l", "robertson"):
        combos.append({"method": method, "k1": 1.5, "b": 0.75})
    return combos


def fmt(metrics: dict) -> str:
    r = metrics["recall"]
    return " ".join(f"R@{k}={r[k]:.3f}" for k in KS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--queries", default="data/validation.jsonl")
    ap.add_argument("--out", default="results/bm25_sweep.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("コーパス読み込み中...", flush=True)
    t0 = time.time()
    texts, paper_ids = load_corpus(Path(args.chunks))
    print(f"  {len(texts):,} チャンク / {time.time() - t0:.0f}s", flush=True)
    queries = load_queries(Path(args.queries))
    print(f"  {len(queries)} クエリ（gold付き）", flush=True)

    header = (
        f"{'tokenize':>8} | {'method':>9} | {'k1':>4} | {'b':>4} | {'agg':>3} | "
        + " | ".join(f"R@{k}" for k in KS)
    )

    for tv in tokenization_variants():
        stemmer = Stemmer.Stemmer(tv["stemmer"]) if tv["stemmer"] else None
        print(f"\n=== tokenize: {tv['tag']} ===  トークナイズ中...", flush=True)
        t0 = time.time()
        corpus_tokens = bm25s.tokenize(
            texts, stopwords=tv["stopwords"], stemmer=stemmer, show_progress=False
        )
        query_tokens_list = [
            bm25s.tokenize(
                [q["question"]], stopwords=tv["stopwords"], stemmer=stemmer,
                show_progress=False,
            )
            for q in queries
        ]
        print(f"  トークナイズ {time.time() - t0:.0f}s", flush=True)
        print(header, flush=True)
        print("-" * len(header), flush=True)

        for combo in scoring_combos():
            retriever = bm25s.BM25(method=combo["method"], k1=combo["k1"], b=combo["b"])
            try:
                retriever.index(corpus_tokens, show_progress=False)
            except Exception as exc:  # robertson は非負でない idf で不安定なことがある
                print(f"  {combo} -> index失敗: {exc}", flush=True)
                continue
            for agg in ("max", "sum"):
                metrics = evaluate(retriever, query_tokens_list, queries, paper_ids, agg)
                row = {"tokenize": tv["tag"], **combo, "agg": agg, **metrics}
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                r = metrics["recall"]
                print(
                    f"{tv['tag']:>8} | {combo['method']:>9} | {combo['k1']:>4} | "
                    f"{combo['b']:>4} | {agg:>3} | "
                    + " | ".join(f"{r[k]:.3f}" for k in KS),
                    flush=True,
                )
            del retriever
            gc.collect()

        del corpus_tokens, query_tokens_list
        gc.collect()

    print(f"\n完了。全結果は {out_path}", flush=True)


if __name__ == "__main__":
    main()
