from pathlib import Path

import pytest

from analysis.report import ROOT, TABLE_RE, build_table, render_markdown_table

FIXTURE = Path(__file__).resolve().parent.parent / "results" / "fixture"
CAMPAIGN = Path(__file__).resolve().parent.parent / "results" / "qwen32-1b-cpu"
REPORT = Path(__file__).resolve().parent.parent / "REPORT.md"
README = Path(__file__).resolve().parent.parent / "README.md"


def annotated_blocks(doc: Path):
    text = doc.read_text(encoding="utf-8")
    blocks = []
    for match in TABLE_RE.finditer(text):
        rel = match.group(1)
        lines = "\n".join(line for line in match.group(2).strip().splitlines() if line.startswith("|")) + "\n"
        blocks.append((rel, lines))
    return blocks


@pytest.mark.parametrize("doc", [REPORT, README])
def test_embedded_tables_match_regeneration(doc):
    blocks = annotated_blocks(doc)
    assert blocks, f"{doc.name} embeds no report tables"
    for rel, embedded in blocks:
        results_dir = (ROOT / rel) if rel else FIXTURE
        assert embedded == render_markdown_table(build_table(results_dir)), f"{doc.name}: {rel} drifted"


def test_fixture_declares_itself_as_fixture():
    import json

    for run_dir in FIXTURE.iterdir():
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert meta["fixture"] is True, f"{run_dir.name} is not marked as a fixture"
        assert meta["environment"]["fixture"] is True


def test_campaign_is_a_real_run_not_a_fixture():
    import json

    metas = []
    for run_dir in CAMPAIGN.iterdir():
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert meta["fixture"] is False, f"{run_dir.name} must not be marked as a fixture"
        metas.append(meta)
    assert len(metas) >= 2


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


def test_campaign_goodput_is_a_fraction():
    for row in build_table(CAMPAIGN):
        assert 0.0 <= row["goodput_fraction"] <= 1.0
        assert row["requests"] > 0
