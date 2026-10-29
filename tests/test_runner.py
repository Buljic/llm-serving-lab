import json
from pathlib import Path

import pytest

from bench.metrics import SloSpec
from bench.runner import ConfigError, RunConfig, collect_environment, run

PROMPTS = Path(__file__).resolve().parent.parent / "prompts" / "prompts.jsonl"


def make_config(**overrides):
    params = dict(
        name="test-run",
        model="stub-model",
        request_count=6,
        arrival_rate_rps=50.0,
        slo=SloSpec(max_ttft_s=10.0, max_tpot_s=1.0),
        warmup_requests=0,
        repetitions=1,
    )
    params.update(overrides)
    return RunConfig(**params)


class QuickStubTransport:
    """One immediate token, no waiting; keeps runs fast and deterministic."""

    def stream(self, payload, arrival_s=None):
        yield json.dumps(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
        )
        yield "[DONE]"


def stub_factory():
    return lambda: QuickStubTransport()


def test_zero_request_count_is_refused(tmp_path):
    config = make_config(request_count=0)
    with pytest.raises(ConfigError, match="at least one request"):
        run(config, tmp_path, prompts_path=PROMPTS, transport_factory=stub_factory())


def test_missing_slo_is_refused(tmp_path):
    config = make_config()
    config.slo = None
    with pytest.raises(ConfigError, match="SLO"):
        run(config, tmp_path, prompts_path=PROMPTS, transport_factory=stub_factory())


def test_bad_rate_and_concurrency_are_refused(tmp_path):
    for overrides in ({"arrival_rate_rps": 0.0}, {"max_concurrent": 0}, {"repetitions": 0}):
        config = make_config(**overrides)
        with pytest.raises(ConfigError):
            run(config, tmp_path, prompts_path=PROMPTS, transport_factory=stub_factory())


def test_run_writes_rows_and_metadata(tmp_path):
    config = make_config(request_count=8, repetitions=2)
    out = run(config, tmp_path, prompts_path=PROMPTS, transport_factory=stub_factory())
    rows = [json.loads(line) for line in open(out / "requests.jsonl", encoding="utf-8")]
    assert len(rows) == 16
    assert all(r["finish_reason"] == "stop" for r in rows)
    reps = {r["request_id"].split("-")[2] for r in rows}
    assert reps == {"r0", "r1"}
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["requests"] == 16
    assert meta["environment"]["fixture"] is False
    assert meta["config"]["slo"] == {"max_ttft_s": 10.0, "max_tpot_s": 1.0}
    assert meta["config"]["api_key"] is None


def test_api_key_is_redacted_in_metadata(tmp_path):
    config = make_config(api_key="supersecret")
    out = run(config, tmp_path, prompts_path=PROMPTS, transport_factory=stub_factory())
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["api_key"] == "***"
    assert "supersecret" not in (out / "run.json").read_text(encoding="utf-8")


def test_collect_environment_reports_no_gpu_as_none():
    env = collect_environment()
    assert env["python"]
    assert env["platform"]
    assert "fixture" in env


class RecordingStubTransport:
    def __init__(self):
        self.payloads = []

    def stream(self, payload, arrival_s=None):
        self.payloads.append(payload)
        yield json.dumps(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
        )
        yield "[DONE]"


def test_max_tokens_cap_limits_request_budget(tmp_path):
    recorder = RecordingStubTransport()
    config = make_config(request_count=3, max_tokens_cap=10)
    run(config, tmp_path, prompts_path=PROMPTS, transport_factory=lambda: recorder)
    assert len(recorder.payloads) == 3
    assert all(p["max_tokens"] == 10 for p in recorder.payloads)


def test_max_tokens_cap_never_raises_budget(tmp_path):
    recorder = RecordingStubTransport()
    config = make_config(request_count=3, max_tokens_cap=500)
    run(config, tmp_path, prompts_path=PROMPTS, transport_factory=lambda: recorder)
    assert all(p["max_tokens"] <= 500 for p in recorder.payloads)


def test_max_tokens_cap_must_be_positive(tmp_path):
    config = make_config(max_tokens_cap=0)
    with pytest.raises(ConfigError):
        run(config, tmp_path, prompts_path=PROMPTS, transport_factory=stub_factory())
