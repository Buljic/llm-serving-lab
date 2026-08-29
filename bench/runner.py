"""Run configurations against a serving endpoint and record raw rows.

Each run writes <output_dir>/<run_name>/requests.jsonl with one row per request
and run.json with the environment and the exact configuration, so a table in a
report can be traced back to the machine it came from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from bench.client import HttpTransport, StreamingClient
from bench.metrics import SloSpec
from bench.workload import WorkloadSpec, build_requests, load_prompts


class ConfigError(ValueError):
    """Raised before a run starts, never mid-run."""


@dataclass
class RunConfig:
    name: str
    model: str
    request_count: int
    arrival_rate_rps: float
    slo: SloSpec
    max_concurrent: int = 8
    warmup_requests: int = 3
    repetitions: int = 1
    seed: int = 7
    class_weights: dict[str, float] = field(
        default_factory=lambda: {"short": 0.5, "medium": 0.3, "long": 0.2}
    )
    gpu_cost_per_hour_usd: float | None = None
    kv_cache_fraction: float | None = None
    base_url: str = "http://localhost:8000"
    api_key: str | None = None
    prompt_file: str = "prompts/prompts.jsonl"

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("run name must not be empty")
        if self.request_count <= 0:
            raise ConfigError(
                f"run {self.name}: request_count is {self.request_count}; a run must send at least one request"
            )
        if self.arrival_rate_rps <= 0:
            raise ConfigError(f"run {self.name}: arrival_rate_rps must be positive")
        if self.slo is None:
            raise ConfigError(f"run {self.name}: an SLO is required so goodput is defined")
        self.slo.validate()
        if self.max_concurrent <= 0:
            raise ConfigError(f"run {self.name}: max_concurrent must be positive")
        if self.repetitions <= 0:
            raise ConfigError(f"run {self.name}: repetitions must be positive")
        if self.warmup_requests < 0:
            raise ConfigError(f"run {self.name}: warmup_requests must not be negative")
        if self.gpu_cost_per_hour_usd is not None and self.gpu_cost_per_hour_usd < 0:
            raise ConfigError(f"run {self.name}: gpu_cost_per_hour_usd must not be negative")
        if self.kv_cache_fraction is not None and not 0.0 < self.kv_cache_fraction <= 1.0:
            raise ConfigError(f"run {self.name}: kv_cache_fraction must be in (0, 1]")

    def workload(self) -> WorkloadSpec:
        return WorkloadSpec(
            seed=self.seed,
            arrival_rate_rps=self.arrival_rate_rps,
            request_count=self.request_count,
            class_weights=self.class_weights,
        )


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _nvidia_smi() -> dict | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    name, _, driver = out.partition(",")
    return {"gpu": name.strip(), "driver": driver.strip()}


def collect_environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cuda": None,
        "vllm": None,
        "fixture": False,
    }
    nvidia = _nvidia_smi()
    if nvidia:
        env["gpu"] = nvidia["gpu"]
        env["driver"] = nvidia["driver"]
    for dist in ("vllm", "torch"):
        try:
            from importlib.metadata import version

            env[dist] = version(dist)
        except Exception:
            pass
    try:
        import torch

        env["cuda"] = torch.version.cuda
    except Exception:
        pass
    return env


async def _drive_once(
    config: RunConfig,
    requests,
    transport_factory,
    clock,
    rep: int,
    virtual: bool,
) -> list[dict]:
    t0 = clock()

    if virtual:
        # Arrivals only order the tasks; the clock moves inside the
        # transports, which set it to absolute service times.
        async def sleep(offset_s: float) -> None:
            await asyncio.sleep(0)
    else:
        async def sleep(offset_s: float) -> None:
            await asyncio.sleep(max(0.0, t0 + offset_s - clock()))

    async def one(seq, arrival_offset_s, prompt):
        await sleep(arrival_offset_s)
        client = StreamingClient(
            model=config.model,
            transport=transport_factory(),
            clock=clock,
        )
        result = client.run(
            request_id=f"{config.name}-r{rep}-{seq}",
            prompt=prompt.prompt,
            max_tokens=prompt.max_tokens,
            prompt_id=prompt.id,
            arrival_s=(t0 + arrival_offset_s) if virtual else None,
        )
        return result.to_row()

    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def bounded(*args):
        async with semaphore:
            return await one(*args)

    tasks = [
        asyncio.create_task(bounded(r.seq, r.arrival_offset_s, r.prompt))
        for r in requests
    ]
    return list(await asyncio.gather(*tasks))


def run(
    config: RunConfig,
    output_dir: str | Path,
    prompts_path: str | Path | None = None,
    transport_factory=None,
    clock=time.monotonic,
    virtual: bool = False,
    fixture: bool = False,
    started_at: str | None = None,
) -> Path:
    """Run one configuration and write requests.jsonl plus run.json.

    transport_factory builds a fresh Transport per request (httpx streams are
    not safe to share). Tests pass a factory that returns scripted transports.
    With virtual=True, arrivals advance the injected clock instead of sleeping,
    which is how the fixture runs without waiting for wall time.
    """
    config.validate()
    prompts = load_prompts(prompts_path or config.prompt_file)
    if transport_factory is None:
        transport_factory = lambda: HttpTransport(config.base_url, config.api_key)

    out = Path(output_dir) / config.name
    out.mkdir(parents=True, exist_ok=True)
    if started_at is None:
        started_at = datetime.now(timezone.utc).isoformat()
    wall_start = clock()
    if config.warmup_requests > 0:
        warmup_spec = WorkloadSpec(
            seed=config.seed,
            arrival_rate_rps=config.arrival_rate_rps,
            request_count=config.warmup_requests,
            class_weights=config.class_weights,
        )
        asyncio.run(
            _drive_once(
                config,
                build_requests(warmup_spec, prompts),
                transport_factory,
                clock,
                rep=-1,
                virtual=virtual,
            )
        )
    rows: list[dict] = []
    for rep in range(config.repetitions):
        rows.extend(
            asyncio.run(
                _drive_once(
                    config,
                    build_requests(config.workload(), prompts),
                    transport_factory,
                    clock,
                    rep,
                    virtual,
                )
            )
        )
    wall_s = max(clock() - wall_start, 1e-9)

    with open(out / "requests.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    metadata = {
        "name": config.name,
        "model": config.model,
        "base_url": config.base_url,
        "started_at": started_at,
        "wall_s": round(wall_s, 6),
        "fixture": fixture,
        "config": {
            **asdict(config),
            "slo": asdict(config.slo),
            "api_key": "***" if config.api_key else None,
        },
        "environment": {**collect_environment(), "fixture": fixture},
        "requests": len(rows),
    }
    with open(out / "run.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return out


def load_sweep(path: str | Path) -> list[RunConfig]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    shared = raw.get("shared", {})
    configs = []
    for entry in raw.get("runs", []):
        merged = {**shared, **entry}
        merged["slo"] = SloSpec(**merged["slo"])
        configs.append(RunConfig(**merged))
    return configs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="run one bench configuration or a sweep")
    parser.add_argument("--config", required=True, help="YAML sweep file")
    parser.add_argument("--only", help="run a single named entry from the sweep")
    parser.add_argument("--output", default="results/campaign")
    parser.add_argument("--prompt-file", default=None)
    args = parser.parse_args(argv)

    configs = load_sweep(args.config)
    if args.only:
        configs = [c for c in configs if c.name == args.only]
        if not configs:
            raise SystemExit(f"no run named {args.only!r} in {args.config}")
    for config in configs:
        out = run(config, args.output, prompts_path=args.prompt_file)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
