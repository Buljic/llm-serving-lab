"""Smoke check against a live local server. The only GPU-touching target.

Requires nvidia-smi on PATH and a vLLM server answering at --base-url. Without
either, it prints why it is skipping and exits 0, so `make smoke` is safe on a
laptop or CI box with no GPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.client import HttpTransport, StreamingClient
from bench.metrics import SloSpec, summarize


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--requests", type=int, default=5)
    args = parser.parse_args(argv)

    if shutil.which("nvidia-smi") is None:
        print("SKIP: no NVIDIA GPU present (nvidia-smi not found)")
        return 0

    health = f"{args.base_url}/health"
    try:
        with urllib.request.urlopen(health, timeout=5) as response:
            if response.status != 200:
                raise OSError(f"HTTP {response.status}")
    except Exception as exc:
        print(f"SKIP: no server at {args.base_url} ({type(exc).__name__}: {exc})")
        print("start one with: vllm serve <model> --port 8000")
        return 0

    model = args.model or _default_model(args.base_url)
    if model is None:
        print("SKIP: server did not list a model; pass --model")
        return 0

    client = StreamingClient(model=model, transport=HttpTransport(args.base_url, api_key=None))
    records = []
    start = time.monotonic()
    for i in range(args.requests):
        result = client.run(f"smoke-{i}", "Say OK.", 8)
        records.append(result.to_row())
    wall = time.monotonic() - start

    failed = [r for r in records if r["finish_reason"] == "error"]
    if failed:
        print(f"FAIL: {len(failed)} of {len(records)} requests errored: {failed[0]['error']}")
        return 1
    summary = summarize(records, wall_s=wall, slo=SloSpec(max_ttft_s=30.0))
    row = summary.to_row()
    print(f"OK: {row['requests']} requests, model {model}")
    print(
        f"ttft p50 {row['ttft_p50_ms']} ms, tpot p50 {row['tpot_p50_ms']} ms, "
        f"{row['output_tokens_per_s']} output tok/s"
    )
    return 0


def _default_model(base_url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
        return payload["data"][0]["id"]
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
