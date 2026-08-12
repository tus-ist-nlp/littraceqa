"""タイトル言及グラフ / 共言及グラフと、その土台の識別子辞書のテスト。

守りたい性質は3つで、どれも実データで踏んだ地雷そのもの:

* **MinerU の分かち書き崩れを吸収する**（``M o RE`` と ``MoRE`` が同じ名前）
* **大文字小文字を捨てない**（``MoRE`` が本文の ``more`` に当たってはいけない）
* **ハブになった名前を落とす**（``HTML`` は123本から「名指し」されていた）
"""

from __future__ import annotations

import json

from littraceqa.di_pipeline.retrieve import paper_titles
from littraceqa.di_pipeline.retrieve.relation_graph import (
    MethodCoMentionExpander,
    TitleMentionExpander,
    build_relations,
)


def _chunk(paper_id: str, kind: str, text: str, section: str | None = None) -> str:
    return json.dumps(
        {
            "chunk_id": f"{paper_id}#{kind}",
            "paper_id": paper_id,
            "text": text,
            "chunk_type": kind,
            "metadata": {"title": text.split("\n")[0], "section": section},
        }
    )


def _corpus(tmp_path, rows: list[tuple[str, str, list[str]]], sections=None):
    """rows: (paper_id, title, [本文…])。"""
    lines = []
    for paper_id, title, bodies in rows:
        lines.append(_chunk(paper_id, "title_abstract", title))
        for i, body in enumerate(bodies):
            section = (sections or {}).get((paper_id, i))
            lines.append(_chunk(paper_id, "text_span", body, section))
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---- 識別子の作り方 --------------------------------------------------------


def test_broken_spacing_normalizes_to_the_same_key():
    """MinerU は大文字の前でスペースを入れる。連結すれば本文側と一致する。"""
    assert paper_titles.alnum("M o RE") == paper_titles.alnum("MoRE") == "MoRE"
    assert paper_titles.alnum("UI - E 2 I -Synth") == "UIE2ISynth"


def test_colon_head_becomes_an_identifier():
    keys = paper_titles.identifiers_of("D-FINE: Redefine Regression Task in DETRs")
    assert "DFINE" in keys


def test_plain_capitalized_head_is_not_an_identifier():
    """"Harmony:" のような普通の単語は名前と区別できないので採らない。"""
    assert "Harmony" not in paper_titles.identifiers_of("Harmony: A Study of Things")


def test_generic_names_are_dropped():
    assert "BERT" not in paper_titles.identifiers_of("BERT: Pre-training of Deep ...")


# ---- 本文との照合 ----------------------------------------------------------


def test_lowercase_english_word_does_not_match_an_allcaps_name(tmp_path):
    """``MoRE`` が本文の ``more`` に当たってはいけない（実データで壊れた点）。"""
    path = _corpus(
        tmp_path,
        [
            ("p1", "M o RE : A Mixture of Low-Rank Experts", ["irrelevant body"]),
            ("p2", "Some Other Paper About Things", ["we want more accuracy than before"]),
        ],
    )
    payload = build_relations(path)
    assert payload["mentions"].get("p2") is None


def test_exact_case_name_matches_across_broken_spacing(tmp_path):
    """タイトル側が崩れていても、本文が正しく書いていれば繋がる。"""
    path = _corpus(
        tmp_path,
        [
            ("p1", "M o RE : A Mixture of Low-Rank Experts", ["irrelevant body"]),
            ("p2", "Another Paper", ["we compare against MoRE on all benchmarks"]),
        ],
    )
    payload = build_relations(path)
    assert payload["mentions"]["p2"] == {"p1"}
    assert payload["mentioned_by"]["p1"] == {"p2"}


def test_references_section_is_ignored(tmp_path):
    """参考文献に名前が並んでいるだけでは辺を張らない。"""
    path = _corpus(
        tmp_path,
        [
            ("p1", "D-FINE: Redefine Regression Task in DETRs", ["body"]),
            ("p2", "Another Paper", ["D-FINE: Redefine Regression Task in DETRs"]),
        ],
        sections={("p2", 0): "References"},
    )
    assert build_relations(path)["mentions"].get("p2") is None


def test_hub_names_are_dropped(tmp_path):
    """多くの論文から「名指し」される名前は弁別に使えないので捨てる。"""
    rows = [("p0", "HTML : Hierarchical Topology Multi-task Learning", ["body"])]
    rows += [
        (f"p{i}", f"Paper Number {i} About Things", ["we output HTML tables"])
        for i in range(1, 6)
    ]
    path = _corpus(tmp_path, rows)
    # 5本から「名指し」される。しきい値がそれより上なら辺は残り……
    assert build_relations(path, max_key_degree=10)["mentions"] == {
        f"p{i}": {"p0"} for i in range(1, 6)
    }
    # ……下回れば名前ごと捨てられる。
    assert build_relations(path, max_key_degree=3)["mentions"] == {}
    assert "HTML" in dict(build_relations(path, max_key_degree=3)["dropped_hubs"])


def test_a_paper_does_not_mention_itself(tmp_path):
    path = _corpus(
        tmp_path,
        [("p1", "D-FINE: Redefine Regression Task in DETRs", ["D-FINE is our method"])],
    )
    assert build_relations(path)["mentions"].get("p1") is None


# ---- expander -------------------------------------------------------------


def _built(tmp_path, rows, **kwargs):
    path = _corpus(tmp_path, rows)
    return path, tmp_path / "cache.pkl"


def test_title_mention_returns_both_directions(tmp_path):
    """A が B を挙げていれば、A から見ても B から見ても近い。"""
    path, cache = _built(
        tmp_path,
        [
            ("p1", "D-FINE: Redefine Regression Task in DETRs", ["our body"]),
            ("p2", "Another Interesting Paper", ["we compare with D-FINE here"]),
        ],
    )
    expander = TitleMentionExpander(str(path), str(cache), neighbors=5)
    assert expander.rank(["p2"]) == ["p1"]
    assert expander.rank(["p1"]) == ["p2"]


def test_method_comention_links_papers_that_name_the_same_work(tmp_path):
    """互いを挙げていなくても、同じ2本を挙げていれば仲間として繋がる。

    ピア gold（質問文が名指ししない同トピック論文）に届くのはこの辺。
    """
    path, cache = _built(
        tmp_path,
        [
            ("owner1", "D-FINE: Redefine Regression Task in DETRs", ["our body"]),
            ("owner2", "SECRET-X: Semi-supervised Clinical Search", ["our body"]),
            ("peerA", "First Peer Paper On Detection", ["we use D-FINE and SECRET-X"]),
            ("peerB", "Second Peer Paper On Detection", ["both D-FINE and SECRET-X help"]),
        ],
    )
    expander = MethodCoMentionExpander(str(path), str(cache), neighbors=5, min_shared=2)
    assert expander.rank(["peerA"]) == ["peerB"]
    assert expander.rank(["peerB"]) == ["peerA"]
    # 名指しされた側どうしは共言及していないので繋がらない
    assert expander.rank(["owner1"]) == []


def test_rank_pools_keeps_anchors_separate(tmp_path):
    """consensus 用。anchor ごとのランキングを潰さずに返す。"""
    path, cache = _built(
        tmp_path,
        [
            ("p1", "D-FINE: Redefine Regression Task in DETRs", ["body"]),
            ("p2", "SECRET-X: Semi-supervised Clinical Search", ["body"]),
            ("p3", "Third Paper Here", ["we use D-FINE"]),
            ("p4", "Fourth Paper Here", ["we use SECRET-X"]),
        ],
    )
    expander = TitleMentionExpander(str(path), str(cache), neighbors=5, anchors=2)
    pools = expander.rank_pools(["p3", "p4"])
    assert pools == [["p1"], ["p2"]]
