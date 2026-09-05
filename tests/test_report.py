from pathlib import Path

import pytest

from analysis.report import ROOT, build_table, render_markdown_table

FIXTURE = Path(__file__).resolve().parent.parent / "results" / "fixture"
REPORT = Path(__file__).resolve().parent.parent / "REPORT.md"
BEGIN = "<!-- report-table -->"
END = "<!-- /report-table -->"


def embedded_table() -> str:
    text = REPORT.read_text(encoding="utf-8")
    assert BEGIN in text and END in text, "REPORT.md must embed the generated table"
    section = text.split(BEGIN, 1)[1].split(END, 1)[0]
    lines = [line for line in section.strip().splitlines() if line.startswith("|")]
    return "\n".join(lines) + "\n"


def test_fixture_table_matches_report():
    regenerated = render_markdown_table(build_table(FIXTURE))
    assert embedded_table() == regenerated


def test_fixture_declares_itself_as_fixture():
    for run_dir in FIXTURE.iterdir():
        meta = __import__("json").loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert meta["fixture"] is True, f"{run_dir.name} is not marked as a fixture"
        assert meta["environment"]["fixture"] is True


def test_fixture_loads_increase_monotonically():
    rows = build_table(FIXTURE)
    loads = [r["offered_rps"] for r in rows]
    assert loads == sorted(loads)
    assert len(rows) >= 3


def test_offering_more_than_the_server_serves_breaks_ttft():
    rows = {r["run"]: r for r in build_table(FIXTURE)}
    lowest = min(rows.values(), key=lambda r: r["offered_rps"])
    highest = max(rows.values(), key=lambda r: r["offered_rps"])
    assert highest["ttft_p95_ms"] > lowest["ttft_p95_ms"]
