"""A deterministic stand-in server for pipeline checks and the fixture run.

It owns a single serving slot. A request reserves the slot from its arrival
(or when the slot frees, whichever is later), pays a prefill cost proportional
to the prompt length, then streams tokens at a fixed decode rate. The absolute
numbers are not model behavior; the queueing shape under offered load is, which
is all the fixture run needs to prove the measurement pipeline end to end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from bench.client import Transport


def _word_count(text: str) -> int:
    return max(1, len(text.split()))


@dataclass
class ServerParams:
    prefill_tokens_per_s: float = 3000.0
    decode_tokens_per_s: float = 120.0
    output_word_overhead: int = 20


class SimulatedServer:
    """Single-slot server on a shared virtual clock."""

    def __init__(self, clock, params: ServerParams | None = None):
        self._clock = clock
        self._params = params or ServerParams()
        self._slot_free_at = 0.0

    def transport(self) -> Transport:
        return _SimulatedTransport(self)

    def reserve(self, arrival_s: float, prompt_text: str, max_tokens: int) -> dict:
        p = self._params
        start = max(arrival_s, self._slot_free_at)
        prompt_words = _word_count(prompt_text)
        # short prompts decode to short answers; cap by the requested budget
        target_output = min(max_tokens, prompt_words * 3 + p.output_word_overhead)
        prefill_s = prompt_words / p.prefill_tokens_per_s
        decode_s = target_output / p.decode_tokens_per_s
        self._slot_free_at = start + prefill_s + decode_s
        return {
            "queue_wait_s": start - arrival_s,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "output_tokens": int(target_output),
        }


class _SimulatedTransport:
    def __init__(self, server: SimulatedServer):
        self._server = server

    def stream(self, payload, arrival_s=None):
        prompt = payload["messages"][0]["content"]
        arrival = self._server._clock() if arrival_s is None else arrival_s
        plan = self._server.reserve(arrival, prompt, int(payload["max_tokens"]))
        clock = self._server._clock
        first_token_at = arrival + plan["queue_wait_s"] + plan["prefill_s"]
        clock.set(first_token_at)
        yield json.dumps({"choices": [{"delta": {"content": "x"}}]})
        tokens = plan["output_tokens"]
        per_token = plan["decode_s"] / tokens
        for i in range(tokens - 1):
            clock.advance(per_token)
            finish = "stop" if i == tokens - 2 else None
            yield json.dumps(
                {"choices": [{"delta": {"content": "x"}, **({"finish_reason": finish} if finish else {})}]}
            )
        yield json.dumps({"choices": [], "usage": {"completion_tokens": tokens}})
        yield "[DONE]"
