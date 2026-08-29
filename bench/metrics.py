"""Metrics over per-request records.

Percentiles use linear interpolation on the sorted samples (numpy default). A
request meets the SLO when every stated bound it has a value for holds. Goodput
is the rate of SLO-meeting requests, not the raw completion rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SloSpec:
    max_ttft_s: float | None = None
    max_tpot_s: float | None = None

    def validate(self) -> None:
        for value in (self.max_ttft_s, self.max_tpot_s):
            if value is not None and value <= 0:
                raise ValueError("SLO bounds must be positive")

    def met_by(self, ttft_s: float, tpot_s: float) -> bool:
        if self.max_ttft_s is not None and ttft_s > self.max_ttft_s:
            return False
        if self.max_tpot_s is not None and tpot_s > self.max_tpot_s:
            return False
        return True


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("percentile of empty sample")
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    return float(np.percentile(np.asarray(values, dtype=float), p))


@dataclass(frozen=True)
class Summary:
    requests: int
    errors: int
    wall_s: float
    ttft_p50_s: float
    ttft_p95_s: float
    ttft_p99_s: float
    tpot_p50_s: float
    tpot_p95_s: float
    tpot_p99_s: float
    mean_output_tokens: float
    output_tokens_per_s: float
    requests_per_s: float
    goodput_requests_per_s: float
    goodput_fraction: float
    slo: SloSpec
    cost_per_million_output_tokens: float | None

    def to_row(self) -> dict:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "wall_s": round(self.wall_s, 3),
            "ttft_p50_ms": round(self.ttft_p50_s * 1000, 2),
            "ttft_p95_ms": round(self.ttft_p95_s * 1000, 2),
            "ttft_p99_ms": round(self.ttft_p99_s * 1000, 2),
            "tpot_p50_ms": round(self.tpot_p50_s * 1000, 2),
            "tpot_p95_ms": round(self.tpot_p95_s * 1000, 2),
            "tpot_p99_ms": round(self.tpot_p99_s * 1000, 2),
            "mean_output_tokens": round(self.mean_output_tokens, 1),
            "output_tokens_per_s": round(self.output_tokens_per_s, 1),
            "requests_per_s": round(self.requests_per_s, 3),
            "goodput_requests_per_s": round(self.goodput_requests_per_s, 3),
            "goodput_fraction": round(self.goodput_fraction, 4),
            "cost_per_million_output_tokens": (
                round(self.cost_per_million_output_tokens, 2)
                if self.cost_per_million_output_tokens is not None
                else None
            ),
        }


def summarize(
    records: list[dict],
    wall_s: float,
    slo: SloSpec,
    gpu_cost_per_hour_usd: float | None = None,
) -> Summary:
    """Aggregate per-request rows (see bench.client.RequestResult.to_row)."""
    slo.validate()
    if wall_s <= 0:
        raise ValueError("wall_s must be positive")
    ok = [r for r in records if r.get("finish_reason") != "error"]
    if not ok:
        raise ValueError("no successful requests to summarize")
    errors = len(records) - len(ok)

    ttft = [float(r["ttft_s"]) for r in ok]
    tpot = [float(r["tpot_s"]) for r in ok if int(r["output_tokens"]) > 1]
    tokens = [int(r["output_tokens"]) for r in ok]
    total_tokens = sum(tokens)
    slo_hits = sum(1 for r in ok if slo.met_by(float(r["ttft_s"]), float(r["tpot_s"])))

    cost = None
    if gpu_cost_per_hour_usd is not None:
        if gpu_cost_per_hour_usd < 0:
            raise ValueError("gpu cost per hour must not be negative")
        cost = (wall_s / 3600.0) * gpu_cost_per_hour_usd / (total_tokens / 1e6)

    return Summary(
        requests=len(ok),
        errors=errors,
        wall_s=wall_s,
        ttft_p50_s=percentile(ttft, 50),
        ttft_p95_s=percentile(ttft, 95),
        ttft_p99_s=percentile(ttft, 99),
        tpot_p50_s=percentile(tpot, 50) if tpot else 0.0,
        tpot_p95_s=percentile(tpot, 95) if tpot else 0.0,
        tpot_p99_s=percentile(tpot, 99) if tpot else 0.0,
        mean_output_tokens=total_tokens / len(ok),
        output_tokens_per_s=total_tokens / wall_s,
        requests_per_s=len(ok) / wall_s,
        goodput_requests_per_s=slo_hits / wall_s,
        goodput_fraction=slo_hits / len(ok),
        slo=slo,
        cost_per_million_output_tokens=cost,
    )
