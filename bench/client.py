"""Streaming client for OpenAI-compatible chat completion endpoints.

Records per request: time to first token, mean inter-token time, total latency,
output token count and finish reason. The clock and the transport are injectable
so tests can replay a scripted server without network or sleeping.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    prompt_id: str
    ttft_s: float
    tpot_s: float
    total_s: float
    output_tokens: int
    finish_reason: str
    error: str | None = None

    def to_row(self) -> dict:
        return asdict(self)


class Transport(Protocol):
    """Yields raw SSE payload strings for one request.

    arrival_s is the request's scheduled arrival on the bench clock when the
    runner drives a virtual schedule; None means "arrived now".
    """

    def stream(self, payload: dict, arrival_s: float | None = None) -> Iterable[str]: ...


class HttpTransport:
    """SSE stream against an OpenAI-compatible server."""

    def __init__(self, base_url: str, api_key: str | None, timeout_s: float = 600.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s

    def stream(self, payload: dict, arrival_s: float | None = None) -> Iterable[str]:
        import httpx

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        with httpx.Client(timeout=self._timeout_s) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        yield line[len("data:"):].strip()


class StreamingClient:
    def __init__(
        self,
        model: str,
        transport: Transport,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._model = model
        self._transport = transport
        self._clock = clock

    def run(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        prompt_id: str = "",
        arrival_s: float | None = None,
    ) -> RequestResult:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = self._clock() if arrival_s is None else arrival_s
        first_token_at: float | None = None
        last_token_at = start
        inter_token_total = 0.0
        inter_token_count = 0
        content_chunks = 0
        finish_reason = ""
        usage_tokens: int | None = None

        try:
            for raw in self._transport.stream(payload, arrival_s=arrival_s):
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                now = self._clock()
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        if first_token_at is None:
                            first_token_at = now
                        else:
                            inter_token_total += now - last_token_at
                            inter_token_count += 1
                        last_token_at = now
                        content_chunks += 1
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    usage_tokens = int(usage["completion_tokens"])
        except Exception as exc:  # recorded, not raised: a bench run must survive one bad request
            end = max(self._clock(), start)
            return RequestResult(
                request_id=request_id,
                prompt_id=prompt_id,
                ttft_s=0.0,
                tpot_s=0.0,
                total_s=end - start,
                output_tokens=content_chunks,
                finish_reason="error",
                error=f"{type(exc).__name__}: {exc}",
            )

        end = max(self._clock(), start)
        if first_token_at is None:
            return RequestResult(
                request_id=request_id,
                prompt_id=prompt_id,
                ttft_s=end - start,
                tpot_s=0.0,
                total_s=end - start,
                output_tokens=0,
                finish_reason=finish_reason or "no_content",
            )
        return RequestResult(
            request_id=request_id,
            prompt_id=prompt_id,
            ttft_s=first_token_at - start,
            tpot_s=inter_token_total / inter_token_count if inter_token_count else 0.0,
            total_s=end - start,
            output_tokens=usage_tokens if usage_tokens is not None else content_chunks,
            finish_reason=finish_reason or "stop",
        )
