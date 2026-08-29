import pytest

from bench.metrics import SloSpec, percentile, summarize


def record(ttft, tpot, tokens=5, finish="stop"):
    return {
        "request_id": "r",
        "prompt_id": "p",
        "ttft_s": ttft,
        "tpot_s": tpot,
        "total_s": ttft + tpot * tokens,
        "output_tokens": tokens,
        "finish_reason": finish,
        "error": None,
    }


def test_percentile_known_answers_linear_interpolation():
    # numpy linear interpolation: rank = p/100 * (n-1)
    assert percentile([1.0], 95) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    assert percentile(list(range(1, 101)), 50) == 50.5
    assert percentile(list(range(1, 101)), 99) == pytest.approx(99.01)


def test_percentile_rejects_empty_and_bad_p():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_slo_both_bounds_checked():
    slo = SloSpec(max_ttft_s=0.5, max_tpot_s=0.05)
    assert slo.met_by(0.4, 0.04)
    assert not slo.met_by(0.6, 0.04)
    assert not slo.met_by(0.4, 0.06)


def test_slo_partial_bounds():
    slo = SloSpec(max_tpot_s=0.05)
    assert slo.met_by(99.0, 0.01)
    assert not slo.met_by(0.01, 0.5)


def test_slo_validation():
    with pytest.raises(ValueError):
        SloSpec(max_ttft_s=0.0).validate()
    with pytest.raises(ValueError):
        SloSpec(max_tpot_s=-1.0).validate()


def test_summarize_rates_and_goodput():
    records = [record(0.1, 0.01, tokens=10) for _ in range(9)]
    records.append(record(2.0, 0.5, tokens=10))  # one SLO violation
    slo = SloSpec(max_ttft_s=0.5, max_tpot_s=0.05)
    summary = summarize(records, wall_s=20.0, slo=slo)
    assert summary.requests == 10
    assert summary.goodput_fraction == pytest.approx(0.9)
    assert summary.goodput_requests_per_s == pytest.approx(0.45)
    assert summary.requests_per_s == pytest.approx(0.5)
    assert summary.output_tokens_per_s == pytest.approx(100 / 20.0)
    assert summary.mean_output_tokens == 10.0


def test_summarize_counts_errors_out_of_denominator():
    records = [record(0.1, 0.01) for _ in range(4)]
    records.append(record(0.0, 0.0, tokens=0, finish="error"))
    summary = summarize(records, wall_s=10.0, slo=SloSpec(max_ttft_s=1.0))
    assert summary.requests == 4
    assert summary.errors == 1
    assert summary.goodput_fraction == 1.0


def test_summarize_cost_known_answer():
    # 1,000,000 tokens over a 1 hour wall at 1.20 USD/h -> 1.20 USD per 1M
    records = [record(0.1, 0.01, tokens=100000) for _ in range(10)]
    summary = summarize(records, wall_s=3600.0, slo=SloSpec(), gpu_cost_per_hour_usd=1.2)
    assert summary.cost_per_million_output_tokens == pytest.approx(1.2)


def test_summarize_rejects_bad_inputs():
    slo = SloSpec(max_ttft_s=1.0)
    with pytest.raises(ValueError):
        summarize([], wall_s=10.0, slo=slo)
    with pytest.raises(ValueError):
        summarize([record(0.1, 0.01)], wall_s=0.0, slo=slo)
    with pytest.raises(ValueError):
        summarize([record(0.1, 0.01)], wall_s=10.0, slo=slo, gpu_cost_per_hour_usd=-1.0)


def test_single_token_requests_do_not_break_tpot():
    records = [record(0.1, 0.0, tokens=1) for _ in range(3)]
    summary = summarize(records, wall_s=5.0, slo=SloSpec(max_ttft_s=1.0))
    assert summary.tpot_p50_s == 0.0


def test_to_row_round_trips():
    records = [record(0.1, 0.01, tokens=10) for _ in range(5)]
    row = summarize(records, wall_s=10.0, slo=SloSpec(max_ttft_s=1.0)).to_row()
    assert row["requests"] == 5
    assert row["cost_per_million_output_tokens"] is None
    assert "ttft_p95_ms" in row
