"""Regenerate the committed fixture under results/fixture.

The fixture runs the full pipeline (workload -> client -> rows -> metadata)
against the single-slot simulated server on a virtual clock. It exists so the
report and its tests run without a GPU; it is not model measurement, and every
run.json it writes says fixture: true.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bench.metrics import SloSpec
from bench.runner import RunConfig, run
from bench.stub import SimulatedServer


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

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "results" / "fixture"

LOADS = (0.5, 1.0, 2.0, 4.0)
SYNTHETIC_DATE = "2026-09-05T19:30:00+00:00"


def main() -> None:
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    for load in LOADS:
        name = f"stub-load-{load:g}rps"
        clock = FakeClock()
        server = SimulatedServer(clock)
        config = RunConfig(
            name=name,
            model="stub-single-slot",
            request_count=24,
            arrival_rate_rps=load,
            warmup_requests=2,
            repetitions=2,
            seed=11,
            slo=SloSpec(max_ttft_s=0.75, max_tpot_s=0.10),
            gpu_cost_per_hour_usd=0.60,
        )
        out = run(
            config,
            FIXTURE_DIR,
            prompts_path=ROOT / "prompts" / "prompts.jsonl",
            transport_factory=server.transport,
            clock=clock,
            virtual=True,
            fixture=True,
            started_at=SYNTHETIC_DATE,
        )
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
