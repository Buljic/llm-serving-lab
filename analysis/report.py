"""Regenerate the headline table and figures from a results directory.

Reads every <dir>/<run>/requests.jsonl plus its run.json, summarizes each run,
and writes a markdown table plus up to three PNG figures into --out.
With --check, the table is compared against the block embedded in REPORT.md
and the exit code is non-zero on mismatch.

Only committed fixture data is in the repo, so `make report` runs offline with
no GPU. A real campaign writes to results/campaign (gitignored) and the same
command regenerates its report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.metrics import SloSpec, summarize  # noqa: E402

REPORT_MD = ROOT / "REPORT.md"
TABLE_RE = re.compile(r"<!-- report-table -->\n(.*?)\n<!-- /report-table -->", re.DOTALL)

COLUMNS = [
    "run",
    "offered_rps",
    "requests",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "output_tokens_per_s",
    "goodput_rps",
    "goodput_fraction",
    "cost_per_million_output_tokens",
]


def load_run(run_dir: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return {"rows": rows, "meta": meta}


def table_line(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def format_row(run: dict) -> dict:
    meta = run["meta"]
    slo = SloSpec(**meta["config"]["slo"])
    summary = summarize(
        run["rows"],
        wall_s=meta["wall_s"],
        slo=slo,
        gpu_cost_per_hour_usd=meta["config"].get("gpu_cost_per_hour_usd"),
    ).to_row()
    return {
        "run": meta["name"],
        "offered_rps": meta["config"]["arrival_rate_rps"],
        "requests": summary["requests"],
        "ttft_p50_ms": summary["ttft_p50_ms"],
        "ttft_p95_ms": summary["ttft_p95_ms"],
        "tpot_p50_ms": summary["tpot_p50_ms"],
        "tpot_p95_ms": summary["tpot_p95_ms"],
        "output_tokens_per_s": summary["output_tokens_per_s"],
        "goodput_rps": summary["goodput_requests_per_s"],
        "goodput_fraction": summary["goodput_fraction"],
        "cost_per_million_output_tokens": summary["cost_per_million_output_tokens"],
    }


def build_table(results_dir: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if (run_dir / "requests.jsonl").exists() and (run_dir / "run.json").exists():
            runs.append(format_row(load_run(run_dir)))
    if not runs:
        raise SystemExit(f"{results_dir}: no runs found")
    return runs


def render_markdown_table(rows: list[dict]) -> str:
    lines = [table_line(COLUMNS), table_line(["---"] + ["---:"] * (len(COLUMNS) - 1))]
    for row in rows:
        cells = []
        for col in COLUMNS:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}".rstrip("0").rstrip("."))
            else:
                cells.append(str(value))
        lines.append(table_line(cells))
    return "\n".join(lines) + "\n"


def write_figures(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = [r["offered_rps"] for r in rows]
    series = [
        ("ttft_p95_ms", "ttft_p95_vs_load.png", "TTFT p95 (ms) vs offered load"),
        ("tpot_p95_ms", "tpot_p95_vs_load.png", "TPOT p95 (ms) vs offered load"),
        ("goodput_rps", "goodput_vs_load.png", "goodput (rps) vs offered load"),
    ]
    for key, filename, title in series:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, [r[key] for r in rows], marker="o")
        ax.set_xlabel("offered load (requests per second)")
        ax.set_ylabel(title.split(" vs")[0])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results/fixture")
    parser.add_argument("--out", default="analysis/out")
    parser.add_argument("--check", action="store_true", help="fail if the table changed")
    args = parser.parse_args(argv)

    rows = build_table(Path(args.results))
    table = render_markdown_table(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "report_table.md"
    if args.check:
        match = TABLE_RE.search(REPORT_MD.read_text(encoding="utf-8"))
        if not match:
            print("REPORT.md has no <!-- report-table --> block", file=sys.stderr)
            return 1
        if match.group(1).strip() + "\n" != table:
            print("report table does not match the committed results", file=sys.stderr)
            return 1
        print("report table matches the committed results")
        return 0
    table_path.write_text(table, encoding="utf-8")
    write_figures(rows, out_dir)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
