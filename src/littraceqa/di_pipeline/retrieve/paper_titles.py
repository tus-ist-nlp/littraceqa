"""論文の「名前」をコーパス横断で引くための識別子辞書。

2箇所から使う:

- ``relation_graph``: A の本文に B のタイトルが出てくるか
  （= コーパス内論文どうしの明示的な関係）
- ``agent/reading.py`` の名指し保護: 質問文が論文を名指ししているか

## 分かち書きの崩れを吸収する

**MinerU のタイトルは分かち書きが壊れている。** 実データを見ると
``M o RE : A Mixture of Low-Rank Experts``、``T oken S hapley: ...``、
``D e F ine: ...``、``AIMSC heck: ...`` のように大文字の前でスペースが入る
（3,000件サンプルで頻出）。一方、本文中で言及されるときは ``MoRE`` と
正しく書かれていることが多い。

そこで**英数字以外を全部落として連結した文字列**をキーにする。

    "M o RE"      -> "MoRE"
    "MoRE"        -> "MoRE"
    "UI - E 2 I -Synth" -> "UIE2ISynth"

本文側も同じ正規化で1本の文字列に潰してから部分文字列で照合するので、
どちらに崩れがあっても一致する。

## 大文字小文字を捨ててはいけない

**小文字化して照合すると壊れる。** 実測（コーパスの21%・5,738論文）で、
``MoRE`` / ``MoST`` / ``DeFine`` / ``DIFFER`` / ``RANGE`` / ``STEP`` /
``SCALE`` / ``GOAL`` / ``CLEAR`` / ``MUST`` といった手法名が、本文中の
**普通の英単語** more / most / define / differ / range / step … に当たり、
5,738論文中5,719本が ``MoRE`` を「名指ししている」ことになった。

そこで**短い識別子は大文字小文字も含めて完全一致**させる
（``MoRE`` は ``MoRE`` にだけ当たり、``more`` や ``More`` には当たらない）。
長い識別子（``CASE_SENSITIVE_MAX_LEN`` 以上、実質すべての正式タイトル）は
偶然の衝突が無視できるので小文字化して照合する——引用時に大文字小文字が
変わることがあるため、こちらは緩くしたい。

## 照合の前置フィルタ（これが無いと総当たりで終わらない）

27,489論文 × 27,489識別子の総当たりは不可能なので、本文側に
「トークン境界から6文字」の集合を作って絞る。本文連結文字列はトークンを
順に繋いだものなので、**識別子の出現位置は必ずトークン境界**になる
（識別子は元テキストで語頭から始まるため）。よって6文字プレフィックスの
集合に当たらない識別子は、その論文には出現しない。

境界のうち**名前らしい形のトークン**（大文字・数字・ハイフンを含む）だけを
起点にする。小文字だけの語から始まる名前は無いので、探索量が数分の1になる。
"""

from __future__ import annotations

import re

# 語の切り出し。ハイフンとアポストロフィは語内文字として拾ってから正規化で落とす。
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")
# 名前らしい形（大文字・数字・ハイフンのどれかを含む）。小文字だけの語は名前の
# 先頭になりえないので、本文側の照合起点から外す。
_NAME_SHAPE = re.compile(r"[A-Z0-9\-]")
# 識別子側の条件。CamelCase / ALLCAPS / 数字 / ハイフンのどれかを持つこと。
# 先頭だけ大文字の普通の単語（"Harmony"）は名前と区別できないので採らない。
_NAME_LIKE = re.compile(r"(?:[A-Za-z][A-Z]|[0-9\-])")

# 前置フィルタに使うプレフィックス長。
GATE_LEN = 6

# 部分文字列で照合してよい最短の長さ。これ未満は連結文字列の中で偶然一致しうる
# （"tcm" が "...t cm..." に当たる類）ので、**トークン完全一致だけ**を認める。
MIN_SUBSTRING_LEN = 6

# これ未満の長さの識別子は大文字小文字も含めて一致させる（上の節を参照）。
CASE_SENSITIVE_MAX_LEN = 12

# 一意性フィルタ（同じ識別子を複数論文が持てば捨てる）でほとんど落ちるが、
# 「その名前を冠した論文がコーパスに1本だけある」汎用語は生き残ってしまう。
# 汎用語は本文のどこにでも出るので、関係グラフを一気にノイズで埋める。
_GENERIC = frozenset(
    """
    bert rag llm llms gpt gpt4 clip vit lora sft rlhf dpo ppo moe cnn rnn lstm
    gan vae bleu rouge mmlu coco imagenet transformer transformers attention
    resnet adam sgd nlp cv ai ml dl rl vlm mllm sam yolo unet diffusion
    """.split()
)


def alnum(text: str) -> str:
    """英数字以外を落とす（**大文字小文字は保つ**）。"""
    return _NON_ALNUM.sub("", text)


def normalize(text: str) -> str:
    """英数字以外を落として小文字化する（節名の判定など、照合以外で使う）。"""
    return alnum(text).lower()


def tokenize(text: str) -> tuple[list[str], list[bool]]:
    """正規化済みトークン列（大文字小文字は保つ）と、名前らしい形かのフラグ。"""
    tokens: list[str] = []
    shapes: list[bool] = []
    for raw in _TOKEN_RE.findall(text):
        token = alnum(raw)
        if not token:
            continue
        tokens.append(token)
        shapes.append(bool(_NAME_SHAPE.search(raw)))
    return tokens, shapes


class Mentions:
    """1論文ぶんの本文を照合できる形に落としたもの。"""

    __slots__ = ("concat", "lowered", "gate", "gate_lower", "tokens", "lower_tokens")

    def __init__(self, text: str) -> None:
        tokens, shapes = tokenize(text)
        offsets: list[int] = []
        position = 0
        for token in tokens:
            offsets.append(position)
            position += len(token)
        self.concat = "".join(tokens)
        self.lowered = self.concat.lower()
        # 名前らしい形のトークンだけを照合の起点にする。
        starts = [o for o, shape in zip(offsets, shapes) if shape]
        self.gate = {self.concat[o : o + GATE_LEN] for o in starts}
        self.gate_lower = {self.lowered[o : o + GATE_LEN] for o in starts}
        self.tokens = {t for t, shape in zip(tokens, shapes) if shape}
        self.lower_tokens = {t.lower() for t in self.tokens}


def identifiers_of(title: str) -> list[str]:
    """タイトルから識別子キーを作る（大文字小文字を保った連結形）。

    * 正式タイトル全文
    * コロン前の見出し部（``D-FINE: Redefine Regression...`` -> ``DFINE``）。
      コーパスの65%のタイトルがコロンを持ち、その前が手法名になっている。
    """
    keys = []
    full = alnum(title)
    # 正式タイトルは十分に長いものだけ（短いタイトルは名前と区別が付かない）。
    if len(full) >= CASE_SENSITIVE_MAX_LEN:
        keys.append(full)
    head, sep, _ = title.partition(":")
    if sep:
        head_key = alnum(head)
        # 4文字未満は偶然一致しやすく、40文字超はもう「名前」ではない。
        # 名前らしい形（CamelCase / ALLCAPS / 数字 / ハイフン）を必須にする。
        if (
            4 <= len(head_key) <= 40
            and head_key.lower() not in _GENERIC
            and _NAME_LIKE.search(head_key)
        ):
            keys.append(head_key)
    return list(dict.fromkeys(keys))


class TitleIndex:
    """識別子キー -> paper_id。**複数論文が持つキーは捨てる**（曖昧な名前）。

    捨てる判断を固定リストではなく一意性でやるのが要点。``BERT`` や ``RAG`` が
    危ないのは「汎用語だから」ではなく「どの論文を指すか決まらないから」で、
    コーパスを見れば機械的に判定できる。一意性は**小文字化して**判定する
    （``MoRE`` と ``More`` が別の論文の名前なら、どちらも曖昧なので捨てる）。
    """

    def __init__(self) -> None:
        # 短い識別子: 大文字小文字も一致させる
        self.cs_keys: dict[str, str] = {}
        self.cs_by_prefix: dict[str, list[str]] = {}
        # 長い識別子: 小文字化して照合する
        self.ci_keys: dict[str, str] = {}
        self.ci_by_prefix: dict[str, list[str]] = {}
        # MIN_SUBSTRING_LEN 未満: トークン完全一致でだけ拾う（大文字小文字も一致）
        self.short_keys: dict[str, str] = {}
        # 上3つを1本にまとめた「キー -> 提案元論文」。長さで空間が分かれている
        # （short < MIN_SUBSTRING_LEN <= cs < CASE_SENSITIVE_MAX_LEN <= ci）ので
        # 衝突しない。ハブ除去はキー単位でやるため、この対応が要る。
        self.owner: dict[str, str] = {}
        self.titles: dict[str, str] = {}

    def add(self, paper_id: str, title: str, owners: dict[str, set[str]]) -> None:
        self.titles.setdefault(paper_id, title)
        for key in identifiers_of(title):
            owners.setdefault(key, set()).add(paper_id)

    def finalize(self, owners: dict[str, set[str]]) -> None:
        """一意なキーだけを採用してプレフィックス索引を張る。"""
        # 一意性は小文字化して見る（大文字違いの同名を両方捨てるため）。
        by_lower: dict[str, set[str]] = {}
        for key, papers in owners.items():
            by_lower.setdefault(key.lower(), set()).update(papers)

        for key, papers in owners.items():
            if len(by_lower[key.lower()]) != 1 or len(papers) != 1:
                continue
            paper_id = next(iter(papers))
            if len(key) < MIN_SUBSTRING_LEN:
                self.short_keys[key] = paper_id
                self.owner[key] = paper_id
            elif len(key) < CASE_SENSITIVE_MAX_LEN:
                self.cs_keys[key] = paper_id
                self.cs_by_prefix.setdefault(key[:GATE_LEN], []).append(key)
                self.owner[key] = paper_id
            else:
                lowered = key.lower()
                self.ci_keys[lowered] = paper_id
                self.ci_by_prefix.setdefault(lowered[:GATE_LEN], []).append(lowered)
                self.owner[lowered] = paper_id

    def __len__(self) -> int:
        return len(self.cs_keys) + len(self.ci_keys) + len(self.short_keys)

    def lookup_keys(self, mentions: Mentions) -> set[str]:
        """この本文に出現した識別子キーの集合。

        ``gate`` に当たったキーだけを部分文字列で確認する（総当たりを避ける）。
        論文IDではなくキーを返すのは、ハブになった名前（ALLCAPS の普通の英単語が
        手法名になっている類）をキー単位で落とせるようにするため。
        """
        found: set[str] = set()
        for prefix in mentions.gate & self.cs_by_prefix.keys():
            for key in self.cs_by_prefix[prefix]:
                if key in mentions.concat:
                    found.add(key)
        for prefix in mentions.gate_lower & self.ci_by_prefix.keys():
            for key in self.ci_by_prefix[prefix]:
                if key in mentions.lowered:
                    found.add(key)
        found |= mentions.tokens & self.short_keys.keys()
        return found

    def lookup(self, mentions: Mentions, skip: frozenset[str] = frozenset()) -> set[str]:
        """この本文が名指ししている論文の集合。``skip`` のキーは無視する。"""
        return {self.owner[key] for key in self.lookup_keys(mentions) - skip}

    def lookup_text(self, text: str, skip: frozenset[str] = frozenset()) -> set[str]:
        """生のテキスト（質問文など）が名指ししている論文。短文向けの入口。"""
        return self.lookup(Mentions(text), skip)
