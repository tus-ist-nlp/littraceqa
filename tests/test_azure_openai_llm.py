from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from littraceqa.di_pipeline.llm import azure_openai as azure_openai_module


class _FakeClock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now = 0.0
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        with self._lock:
            self.sleep_calls.append(seconds)
            self._now += seconds


class _Completions:
    def __init__(
        self,
        clock: _FakeClock,
        *,
        invocation_barrier: threading.Barrier | None = None,
    ) -> None:
        self.clock = clock
        self.invocation_barrier = invocation_barrier
        self._lock = threading.Lock()
        self.launches: list[float] = []
        self.requests: list[dict] = []

    def create(self, **kwargs):
        launched_at = self.clock.monotonic()
        with self._lock:
            self.launches.append(launched_at)
            self.requests.append(dict(kwargs))
        if self.invocation_barrier is not None:
            # Every provider call must be able to overlap. This would time out
            # if the pacer retained either of its locks during the API call.
            self.invocation_barrier.wait(timeout=2)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="{}"), finish_reason="stop"
                )
            ],
            usage=None,
            model="test-model",
            id="test-request",
        )


class _RawCompletions(_Completions):
    @property
    def with_raw_response(self):
        def create(**kwargs):
            parsed = self.create(**kwargs)
            return SimpleNamespace(
                headers={
                    "x-ratelimit-limit-requests": "600",
                    "x-ratelimit-remaining-tokens": "123456",
                    "retry-after-ms": "250",
                    "authorization": "must-not-be-saved",
                },
                parse=lambda: parsed,
            )

        return SimpleNamespace(create=create)


def _build_llm(
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeClock,
    completions: _Completions,
    **kwargs,
) -> azure_openai_module.AzureOpenAILLM:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        azure_openai_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep),
    )
    monkeypatch.setattr(
        azure_openai_module,
        "AzureOpenAI",
        lambda **_client_kwargs: client,
    )
    return azure_openai_module.AzureOpenAILLM(
        endpoint="https://example.openai.azure.com",
        api_key="test-key",
        api_version="test-version",
        deployment="test-deployment",
        **kwargs,
    )


def test_min_request_interval_spaces_concurrent_provider_launches(monkeypatch):
    call_count = 6
    clock = _FakeClock()
    completions = _Completions(
        clock,
        invocation_barrier=threading.Barrier(call_count),
    )
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        min_request_interval_seconds=1.0,
    )
    start_barrier = threading.Barrier(call_count)

    def invoke(index: int) -> str:
        start_barrier.wait(timeout=2)
        return llm.complete(f"request {index}")

    with ThreadPoolExecutor(max_workers=call_count) as executor:
        responses = list(executor.map(invoke, range(call_count)))

    launches = sorted(completions.launches)
    assert responses == ["{}"] * call_count
    assert len(launches) == call_count
    assert all(
        later - earlier >= 1.0
        for earlier, later in zip(launches, launches[1:])
    )


def test_zero_request_interval_disables_pacer_sleep(monkeypatch):
    clock = _FakeClock()
    completions = _Completions(clock)
    llm = _build_llm(monkeypatch, clock, completions)

    assert llm.complete("first") == "{}"
    assert llm.complete("second") == "{}"
    assert completions.launches == [0.0, 0.0]
    assert clock.sleep_calls == []


def test_target_tpm_prepays_mixed_short_and_long_request_slots(monkeypatch):
    clock = _FakeClock()
    completions = _Completions(clock)
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        system="",
        max_completion_tokens=100,
        target_tpm=6_000,
    )

    short = llm.complete_with_metadata("s" * 320)
    long = llm.complete_with_metadata("l" * 3_200)

    assert short["estimated_reserved_tokens"] == 200
    assert long["estimated_reserved_tokens"] == 1_100
    assert short["launch_interval_seconds"] == pytest.approx(2.0)
    assert long["launch_interval_seconds"] == pytest.approx(11.0)
    assert completions.launches == pytest.approx([2.0, 13.0])
    assert (
        (200 + 1_100) * 60 / completions.launches[-1]
        <= llm.target_tpm
    )


def test_target_tpm_is_shared_by_concurrent_provider_calls(monkeypatch):
    call_count = 4
    clock = _FakeClock()
    completions = _Completions(
        clock,
        invocation_barrier=threading.Barrier(call_count),
    )
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        system="",
        max_completion_tokens=100,
        target_tpm=6_000,
    )
    start_barrier = threading.Barrier(call_count)

    def invoke(index: int) -> str:
        start_barrier.wait(timeout=2)
        return llm.complete(str(index) * 320)

    with ThreadPoolExecutor(max_workers=call_count) as executor:
        responses = list(executor.map(invoke, range(call_count)))

    assert responses == ["{}"] * call_count
    assert sorted(completions.launches) == pytest.approx([2.0, 4.0, 6.0, 8.0])


def test_target_tpm_uses_effective_per_call_completion_override(monkeypatch):
    clock = _FakeClock()
    completions = _Completions(clock)
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        system="",
        max_completion_tokens=1_200,
        target_tpm=6_000,
    )

    overridden = llm.complete_with_metadata(
        "", max_completion_tokens=100
    )
    defaulted = llm.complete_with_metadata("")

    assert overridden["estimated_reserved_tokens"] == 100
    assert defaulted["estimated_reserved_tokens"] == 1_200
    assert completions.launches == pytest.approx([1.0, 13.0])


def test_target_tpm_image_allowance_is_independent_of_encoded_file_size(
    monkeypatch,
):
    clock = _FakeClock()
    completions = _Completions(clock)
    monkeypatch.setattr(
        azure_openai_module,
        "_image_data_url",
        lambda _path: "data:image/png;base64,AA==",
    )
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        system="",
        max_completion_tokens=1,
        target_tpm=60,
    )

    result = llm.complete_with_metadata("", image_paths=["tiny.png"])

    assert result["estimated_reserved_tokens"] == 513
    assert result["launch_interval_seconds"] == pytest.approx(513.0)


def test_target_tpm_keeps_fixed_interval_as_microburst_floor(monkeypatch):
    clock = _FakeClock()
    completions = _Completions(clock)
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        system="",
        max_completion_tokens=1,
        target_tpm=1_000_000_000,
        min_request_interval_seconds=0.075,
    )

    llm.complete("")
    llm.complete("")

    assert completions.launches == pytest.approx([0.075, 0.15])


def test_per_call_completion_limit_overrides_default(monkeypatch):
    clock = _FakeClock()
    completions = _Completions(clock)
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        max_completion_tokens=12_000,
    )

    result = llm.complete_with_metadata(
        "stage one", max_completion_tokens=1_024
    )

    assert completions.requests[-1]["max_completion_tokens"] == 1_024
    assert result["max_completion_tokens"] == 1_024
    assert result["finish_reason"] == "stop"
    assert llm.max_completion_tokens == 12_000


def test_concurrent_completion_limits_do_not_bleed_between_calls(monkeypatch):
    limits = [1_024, 12_000] * 3
    clock = _FakeClock()
    completions = _Completions(
        clock, invocation_barrier=threading.Barrier(len(limits))
    )
    llm = _build_llm(
        monkeypatch,
        clock,
        completions,
        max_completion_tokens=12_000,
    )

    def invoke(limit: int) -> dict:
        return llm.complete_with_metadata(
            "mixed stage", max_completion_tokens=limit
        )

    with ThreadPoolExecutor(max_workers=len(limits)) as executor:
        results = list(executor.map(invoke, limits))

    assert sorted(
        request["max_completion_tokens"] for request in completions.requests
    ) == sorted(limits)
    assert [result["max_completion_tokens"] for result in results] == limits
    assert llm.max_completion_tokens == 12_000


def test_raw_response_persists_only_safe_rate_limit_headers(monkeypatch):
    clock = _FakeClock()
    completions = _RawCompletions(clock)
    llm = _build_llm(monkeypatch, clock, completions)

    result = llm.complete_with_metadata("quota telemetry")

    assert result["rate_limit"] == {
        "retry-after-ms": "250",
        "x-ratelimit-limit-requests": "600",
        "x-ratelimit-remaining-tokens": "123456",
    }
    assert "authorization" not in result["rate_limit"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_per_call_completion_limit_is_rejected(monkeypatch, value):
    clock = _FakeClock()
    completions = _Completions(clock)
    llm = _build_llm(monkeypatch, clock, completions)

    with pytest.raises(ValueError, match="max_completion_tokens"):
        llm.complete_with_metadata("invalid", max_completion_tokens=value)

    assert completions.requests == []


@pytest.mark.parametrize("interval", [-0.01, float("nan"), float("inf")])
def test_invalid_request_interval_is_rejected(monkeypatch, interval):
    monkeypatch.setattr(
        azure_openai_module,
        "AzureOpenAI",
        lambda **_kwargs: pytest.fail("client must not be constructed"),
    )

    with pytest.raises(ValueError, match="min_request_interval_seconds"):
        azure_openai_module.AzureOpenAILLM(
            endpoint="https://example.openai.azure.com",
            api_key="test-key",
            api_version="test-version",
            deployment="test-deployment",
            min_request_interval_seconds=interval,
        )


@pytest.mark.parametrize(
    "target_tpm",
    [0, -1, True, float("nan"), float("inf"), float("-inf"), "invalid"],
)
def test_invalid_target_tpm_is_rejected_before_client_construction(
    monkeypatch, target_tpm
):
    monkeypatch.setattr(
        azure_openai_module,
        "AzureOpenAI",
        lambda **_kwargs: pytest.fail("client must not be constructed"),
    )

    with pytest.raises(ValueError, match="target_tpm"):
        azure_openai_module.AzureOpenAILLM(
            endpoint="https://example.openai.azure.com",
            api_key="test-key",
            api_version="test-version",
            deployment="test-deployment",
            target_tpm=target_tpm,
        )
