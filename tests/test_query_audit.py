"""クエリ品質監査（docs/query_audit_spec.md）のテスト。

判定側（audit_queries.py）は FakeLLM で回し、仕様の要点を固定する:

- relevance = no_evidence は LLM の返答に関わらず「evidence_id が存在しない」という
  データセットの事実から機械的に確定する（spec 3.2 / 検討事項）
- noise_type は supporting / partial では必ず null（spec 3.2）
- クエリラベルは gold_paper 単位の判定から機械的に導出する（spec 3.3）
- paper_recall_macro_clean は分母を supporting/partial に絞り、
  分母 0 のクエリをマクロ平均から除外する（spec 4.2）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_queries import (
    ChunkStore,
    audit_query,
    normalize_judgment,
    rank_map_of,
)
from audit_report import (
    build_html,
    corrected_metrics,
    group_by_query,
    merge_split_runs,
    query_label,
    summarize,
    viewer_payload,
)
from littraceqa.di_pipeline.llm.fake import FakeLLM


def _write_chunks(tmp_path: Path) -> ChunkStore:
    chunks = [
        {
            "chunk_id": "p_a#c0000",
            "paper_id": "p_a",
            "text": "[ACL 2025] Paper A\nAbstract of paper A about compression.",
            "chunk_type": "title_abstract",
            "metadata": {},
        },
        {
            "chunk_id": "p_a#c0001",
            "paper_id": "p_a",
            "text": "[ACL 2025] Paper A\nThe F1 score improves by 15.66 points on NaturalQ.",
            "chunk_type": "text_span",
            "metadata": {},
        },
        {
            "chunk_id": "p_b#c0000",
            "paper_id": "p_b",
            "text": "[ACL 2025] Paper B\nAbstract of paper B about a different topic.",
            "chunk_type": "title_abstract",
            "metadata": {},
        },
        {
            "chunk_id": "p_b#c0001",
            "paper_id": "p_b",
            "text": "[ACL 2025] Paper B\nBody text of paper B.",
            "chunk_type": "text_span",
            "metadata": {},
        },
    ]
    path = tmp_path / "mineru_chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )
    return ChunkStore(path)


def _gold() -> dict:
    return {
        "query_id": "q_001",
        "task_family": "multi_paper",
        "question": "How much does A improve F1?",
        "answer_types": ["freeform"],
        "gold_papers": [
            {"paper_id": "p_a", "title": "Paper A", "venue": "ACL", "year": 2025},
            {"paper_id": "p_b", "title": "Paper B", "venue": "ACL", "year": 2025},
        ],
        "evidence": [
            {
                "evidence_id": "ev_001",
                "paper_id": "p_a",
                "source_type": "text_span",
                "evidence_text_or_value": "15.66",
                "locator": {"page": 6},
            }
        ],
        "answer": {"freeform": {"text": "15.66"}},
    }


def _llm_response() -> str:
    # p_b は evidence が無いのに LLM が supporting と誤答するケース。
    # 機械的な確定（no_evidence）が LLM に勝つことを確認する。
    return json.dumps(
        {
            "papers": [
                {
                    "paper_id": "p_a",
                    "relevance": "supporting",
                    "evidence_role": "F1改善幅15.66を直接支持する",
                    "noise_type": "annotation_error",
                    "relation_to_gold": "正解論文そのもの",
                    "misleading_risk": None,
                    "body_supports_answer": None,
                    "confidence": "high",
                },
                {
                    "paper_id": "p_b",
                    "relevance": "supporting",
                    "evidence_role": "同トピックのピア論文",
                    "noise_type": "same_topic_different_finding",
                    "relation_to_gold": "Paper A と同じベンチマークを使う",
                    "misleading_risk": None,
                    "body_supports_answer": False,
                    "confidence": "medium",
                },
            ]
        }
    )


def test_audit_query_mechanical_rules(tmp_path: Path) -> None:
    store = _write_chunks(tmp_path)
    llm = FakeLLM([_llm_response()])
    records = audit_query(
        llm, _gold(), store, rank_of={"p_a": 3}, judge_model="fake"
    )
    by_id = {r["paper_id"]: r for r in records}

    # evidence がある p_a: LLM の supporting が通り、noise_type は機械的に null。
    assert by_id["p_a"]["relevance"] == "supporting"
    assert by_id["p_a"]["noise_type"] is None
    assert by_id["p_a"]["evidence_ids"] == ["ev_001"]
    assert by_id["p_a"]["retrieval"] == {"rank": 3, "score": None}
    # evidence 周辺の抜粋が同梱されている（ビューアの自己完結性）。
    assert "15.66" in by_id["p_a"]["context"]["evidence_excerpts"][0]["chunk_excerpt"]

    # evidence が無い p_b: LLM が supporting と言っても no_evidence に上書き。
    assert by_id["p_b"]["relevance"] == "no_evidence"
    assert by_id["p_b"]["noise_type"] == "same_topic_different_finding"
    assert by_id["p_b"]["body_supports_answer"] is False
    assert by_id["p_b"]["retrieval"] == {"rank": None, "score": None}
    # no_evidence の論文には本文冒頭の抜粋が付く。
    assert "Body text" in by_id["p_b"]["context"]["body_excerpt"]


def test_audit_query_retries_on_bad_json(tmp_path: Path) -> None:
    store = _write_chunks(tmp_path)
    llm = FakeLLM(["not json at all", _llm_response()])
    records = audit_query(llm, _gold(), store, rank_of={}, judge_model="fake")
    assert len(llm.calls) == 2
    assert {r["paper_id"] for r in records} == {"p_a", "p_b"}


def test_normalize_judgment_invalid_enums() -> None:
    judged = normalize_judgment(
        {"relevance": "banana", "noise_type": "banana", "confidence": "high"},
        has_evidence=True,
    )
    assert judged["relevance"] is None
    assert judged["noise_type"] is None
    assert judged["confidence"] == "low"
    assert "judge_error" in judged


def test_rank_map_of() -> None:
    pred = {"candidate_papers": ["p_a", {"paper_id": "p_b"}, "p_a"]}
    assert rank_map_of(pred) == {"p_a": 1, "p_b": 2}


def _record(query_id: str, paper_id: str, relevance: str | None, **extra) -> dict:
    return {
        "query_id": query_id,
        "paper_id": paper_id,
        "task_family": "multi_paper",
        "relevance": relevance,
        "evidence_role": "",
        "noise_type": None,
        "relation_to_gold": "",
        "misleading_risk": None,
        "body_supports_answer": None,
        "confidence": "high",
        "evidence_ids": [],
        "retrieval": {"rank": None, "score": None},
        "question": "q?",
        "answer_text": "a",
        "paper": {"title": paper_id},
        "context": {"abstract": "", "evidence_excerpts": [], "body_excerpt": ""},
        **extra,
    }


def test_query_label_rules() -> None:
    # 全 supporting -> good
    assert query_label([_record("q", "a", "supporting")]) == "good"
    # 全論文 evidence 持ちだが supporting 以外を含む -> fair
    assert (
        query_label([_record("q", "a", "supporting"), _record("q", "b", "partial")])
        == "fair"
    )
    # 判定失敗（relevance None）は supporting と認めず fair 側に倒す
    assert query_label([_record("q", "a", None)]) == "fair"
    # no_evidence を1本でも含めば noisy（他が supporting でも）
    assert (
        query_label([_record("q", "a", "supporting"), _record("q", "b", "no_evidence")])
        == "noisy"
    )


def test_corrected_metrics_clean_denominator() -> None:
    grouped = group_by_query(
        [
            # q1: gold {a: supporting, b: no_evidence}, 提出 {a}
            #   従来 recall 0.5 / clean recall 1.0
            _record("q1", "a", "supporting"),
            _record("q1", "b", "no_evidence"),
            # q2: 全部 no_evidence -> clean の分母 0 なのでマクロ平均から除外
            _record("q2", "c", "no_evidence"),
        ]
    )
    pred_by_id = {
        "q1": {"query_id": "q1", "gold_papers": [{"paper_id": "a"}]},
        "q2": {"query_id": "q2", "gold_papers": []},
    }
    metrics = corrected_metrics(grouped, pred_by_id)
    assert metrics["paper_recall_macro"] == (0.5 + 0.0) / 2
    assert metrics["paper_recall_macro_clean"] == 1.0
    assert metrics["clean_excluded_queries"] == 1


def test_summarize_flags() -> None:
    grouped = group_by_query(
        [
            _record("q1", "a", "supporting"),  # rank None -> 未検出 supporting
            _record("q1", "b", "contradicting"),
            _record(
                "q2", "c", "no_evidence", body_supports_answer=True
            ),  # アノテ漏れ疑い
        ]
    )
    summary = summarize(grouped, pred_by_id=None)
    assert summary["label_counts"] == {"fair": 1, "noisy": 1}
    assert summary["contradicting_queries"] == ["q1"]
    assert ["q1", "a"] in summary["unretrieved_supporting"]
    assert summary["missing_annotation_suspects"] == [["q2", "c"]]
    assert "metrics" not in summary  # pred なしでは補正指標を出さない


def _exp(name: str, ts: str, config: str | None, covered: set[str],
         coverage_full: bool = False, explicit: bool = False) -> dict:
    return {
        "name": name, "explicit": explicit, "timestamp": ts, "config": config,
        "pred_by_id": {q: {"query_id": q, "gold_papers": [], "candidate_papers": ["x"]}
                       for q in covered},
        "covered": covered, "coverage": len(covered), "metrics": {},
    }


def test_merge_split_runs() -> None:
    grouped = group_by_query(
        [_record(q, "a", "supporting") for q in ("q1", "q2", "q3", "q4")]
    )
    experiments = [
        # 同一構成の分割実行 -> 1本に統合される
        _exp("run_a", "2026-07-25", "cfg_split", {"q1", "q2"}),
        _exp("run_b", "2026-07-26", "cfg_split", {"q3", "q4"}),
        # フル実行に包含される片割れ -> 落ちる
        _exp("half", "2026-07-20", "cfg_full", {"q1", "q2"}),
        _exp("full", "2026-07-21", "cfg_full", {"q1", "q2", "q3", "q4"}),
    ]
    merged = merge_split_runs(experiments, grouped)
    names = [e["name"] for e in merged]
    assert "half" not in names
    assert "full" in names
    combined = next(e for e in merged if "結合" in e["name"])
    assert combined["coverage"] == 4
    assert combined["covered"] == {"q1", "q2", "q3", "q4"}
    assert "run_b" in combined["name"]  # 新しい方の名前を引き継ぐ
    assert len(merged) == 2


def test_build_html_self_contained() -> None:
    grouped = group_by_query(
        [_record("q1", "a", "supporting", evidence_role="</script>を含む記述")]
    )
    summary = summarize(grouped, pred_by_id=None)
    page = build_html(viewer_payload(grouped, None), summary)
    assert "q1" in page
    # データ内の </script> で script 要素が閉じない。
    assert "</script>を含む" not in page
    assert "<\\/script>" in page
    # 外部リソースへの参照が無い（単一ファイル・外部依存なし）。
    assert "http://" not in page and "https://" not in page


def test_chunk_store_with_offsets(tmp_path: Path) -> None:
    store = _write_chunks(tmp_path)
    raw = store.path.read_bytes()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    # p_a は先頭2行、p_b は残り。
    split = len((lines[0] + lines[1]).encode("utf-8"))
    offsets = {
        "version": 1,
        "offsets": {"p_a": [0, split], "p_b": [split, len(raw) - split]},
    }
    Path(f"{store.path}.offsets.json").write_text(json.dumps(offsets), encoding="utf-8")
    indexed = ChunkStore(store.path)
    assert [c["chunk_id"] for c in indexed.paper_chunks("p_a")] == ["p_a#c0000", "p_a#c0001"]
    assert [c["chunk_id"] for c in indexed.paper_chunks("p_b")] == ["p_b#c0000", "p_b#c0001"]
    assert indexed.paper_chunks("missing") == []
