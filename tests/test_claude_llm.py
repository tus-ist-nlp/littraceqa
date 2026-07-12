"""ClaudeLLM が Claude API を正しく叩き、応答からテキストだけを取り出すことのテスト。

実際の API は呼ばず、anthropic.Anthropic をスタブに差し替えて検証する。
"""

from __future__ import annotations

import pytest

import litqa.llm.claude as claude_module
from litqa.llm.claude import ClaudeLLM
from litqa.registry import build


class _Block:
    def __init__(self, type: str, text: str = ""):
        self.type = type
        self.text = text


class _Response:
    def __init__(self, content):
        self.content = content


class _StubMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _StubClient:
    """資格情報が解決できた状態の Anthropic クライアントを模したスタブ。"""

    def __init__(self, response, api_key="sk-ant-test", **init_kwargs):
        self.messages = _StubMessages(response)
        self.api_key = api_key
        self.auth_token = None
        self.credentials = None
        self.init_kwargs = init_kwargs


def _make_factory(created, response, **client_kwargs):
    def factory(**kwargs):
        client = _StubClient(response, **client_kwargs)
        created.append(client)
        return client

    return factory


@pytest.fixture
def stub_anthropic(monkeypatch):
    """anthropic.Anthropic を差し替え、生成されたスタブクライアントを返す。"""
    created: list[_StubClient] = []
    response = _Response(
        [
            _Block("thinking", ""),
            _Block("text", '{"task_family": '),
            _Block("text", '"multi_paper"}'),
        ]
    )
    monkeypatch.setattr(
        claude_module.anthropic, "Anthropic", _make_factory(created, response)
    )
    return created


def test_returns_only_text_blocks(stub_anthropic):
    """thinking ブロックは捨て、text ブロックだけを連結して返す。"""
    llm = ClaudeLLM()
    assert llm("質問") == '{"task_family": "multi_paper"}'


def test_sends_adaptive_thinking_and_effort(stub_anthropic):
    """Opus 4.8 は thinking を省くと思考なしで走るので、明示的に adaptive を送る。"""
    ClaudeLLM(effort="low")("質問")
    call = stub_anthropic[0].messages.calls[0]
    assert call["model"] == "claude-opus-4-8"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "low"}
    assert call["messages"] == [{"role": "user", "content": "質問"}]


def test_thinking_can_be_disabled(stub_anthropic):
    ClaudeLLM(thinking=False)("質問")
    assert stub_anthropic[0].messages.calls[0]["thinking"] == {"type": "disabled"}


def test_does_not_send_sampling_params(stub_anthropic):
    """temperature / top_p / top_k は Opus 4.8 では 400 になるので送らない。"""
    ClaudeLLM()("質問")
    call = stub_anthropic[0].messages.calls[0]
    for param in ("temperature", "top_p", "top_k"):
        assert param not in call


def test_missing_credentials_fails_at_construction(monkeypatch):
    """資格情報が無いことは実行中ではなく構築時に検出する。

    SDK は資格情報が無くても構築だけは成功し、最初の API 呼び出しで初めて落ちる。
    だがエージェントは LLM 呼び出しを try/except で握りつぶしてフォールバックするので、
    そのままでは「LLMが動いていないのに静かに劣化する」状態になってしまう。
    """
    monkeypatch.setattr(
        claude_module.anthropic,
        "Anthropic",
        _make_factory([], _Response([]), api_key=None),
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeLLM()


def test_registered_as_claude(stub_anthropic):
    """agent_style の yaml から llm: { name: claude } で呼べる。"""
    assert isinstance(build("llm", "claude"), ClaudeLLM)
