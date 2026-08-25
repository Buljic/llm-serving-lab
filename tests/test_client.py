from bench.client import RequestResult, StreamingClient


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt

    def set(self, t):
        if t > self.now:
            self.now = t


class ScriptTransport:
    """Yields scripted SSE chunks, advancing the fake clock between chunks.

    script: list of (delay_s, chunk_dict or "[DONE]").
    """

    def __init__(self, script, clock):
        self._script = script
        self._clock = clock

    def stream(self, payload, arrival_s=None):
        for delay, chunk in self._script:
            self._clock.advance(delay)
            if chunk == "[DONE]":
                yield "[DONE]"
            else:
                import json

                yield json.dumps(chunk)


def content_chunk(text, finish=None):
    return {"choices": [{"delta": {"content": text}, **({"finish_reason": finish} if finish else {})}]}


def usage_chunk(tokens):
    return {"choices": [], "usage": {"completion_tokens": tokens}}


def test_ttft_and_tpot_on_scripted_stream():
    clock = FakeClock()
    script = [
        (0.30, content_chunk("Hello")),  # ttft: 0.30
        (0.05, content_chunk(" world")),  # inter-token: 0.05
        (0.07, content_chunk("!", finish="stop")),  # inter-token: 0.07
        (0.01, usage_chunk(3)),
        (0.0, "[DONE]"),
    ]
    client = StreamingClient(model="m", transport=ScriptTransport(script, clock), clock=clock)
    result = client.run("r1", "hi", 16, prompt_id="p1")
    assert isinstance(result, RequestResult)
    assert abs(result.ttft_s - 0.30) < 1e-9
    assert abs(result.tpot_s - 0.06) < 1e-9
    assert abs(result.total_s - 0.43) < 1e-9
    assert result.output_tokens == 3
    assert result.finish_reason == "stop"
    assert result.error is None


def test_single_token_reply_has_zero_tpot():
    clock = FakeClock()
    script = [
        (0.2, content_chunk("done", finish="stop")),
        (0.0, usage_chunk(1)),
        (0.0, "[DONE]"),
    ]
    client = StreamingClient(model="m", transport=ScriptTransport(script, clock), clock=clock)
    result = client.run("r2", "hi", 16)
    assert result.tpot_s == 0.0
    assert result.output_tokens == 1
    assert result.finish_reason == "stop"


def test_empty_reply_is_reported_not_crashed():
    clock = FakeClock()
    script = [(0.4, usage_chunk(0)), (0.0, "[DONE]")]
    client = StreamingClient(model="m", transport=ScriptTransport(script, clock), clock=clock)
    result = client.run("r3", "hi", 16)
    assert result.output_tokens == 0
    assert result.finish_reason == "no_content"


def test_transport_error_is_recorded():
    class Broken:
        def stream(self, payload, arrival_s=None):
            raise ConnectionError("refused")

    clock = FakeClock()
    client = StreamingClient(model="m", transport=Broken(), clock=clock)
    result = client.run("r4", "hi", 16, prompt_id="p9")
    assert result.finish_reason == "error"
    assert "ConnectionError" in result.error
    assert result.prompt_id == "p9"


def test_finish_reason_without_usage_counts_chunks():
    clock = FakeClock()
    script = [
        (0.1, content_chunk("a")),
        (0.1, content_chunk("b")),
        (0.1, content_chunk("c", finish="length")),
        (0.0, "[DONE]"),
    ]
    client = StreamingClient(model="m", transport=ScriptTransport(script, clock), clock=clock)
    result = client.run("r5", "hi", 2)
    assert result.output_tokens == 3
    assert result.finish_reason == "length"


def test_to_row_round_trips():
    clock = FakeClock()
    script = [(0.1, content_chunk("x", finish="stop")), (0.0, "[DONE]")]
    client = StreamingClient(model="m", transport=ScriptTransport(script, clock), clock=clock)
    row = client.run("r6", "hi", 8).to_row()
    assert row["request_id"] == "r6"
    assert row["finish_reason"] == "stop"
    assert set(row) == {
        "request_id",
        "prompt_id",
        "ttft_s",
        "tpot_s",
        "total_s",
        "output_tokens",
        "finish_reason",
        "error",
    }
