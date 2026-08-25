from pathlib import Path

import numpy as np
import pytest

from bench.workload import (
    WorkloadSpec,
    build_requests,
    load_prompts,
    poisson_arrivals,
)

PROMPTS = Path(__file__).resolve().parent.parent / "prompts" / "prompts.jsonl"


def make_spec(**overrides):
    params = dict(
        seed=7,
        arrival_rate_rps=2.0,
        request_count=50,
        class_weights={"short": 0.5, "medium": 0.3, "long": 0.2},
    )
    params.update(overrides)
    return WorkloadSpec(**params)


def test_same_seed_gives_same_schedule():
    a = poisson_arrivals(make_spec())
    b = poisson_arrivals(make_spec())
    np.testing.assert_array_equal(a, b)


def test_different_seed_gives_different_schedule():
    a = poisson_arrivals(make_spec(seed=1))
    b = poisson_arrivals(make_spec(seed=2))
    assert not np.array_equal(a, b)


def test_inter_arrival_mean_matches_rate():
    spec = make_spec(arrival_rate_rps=4.0, request_count=20000)
    arrivals = poisson_arrivals(spec)
    inter = np.diff(arrivals)
    assert abs(inter.mean() - 0.25) < 0.01


def test_request_count_is_honored():
    reqs = build_requests(make_spec(request_count=120), load_prompts(PROMPTS))
    assert len(reqs) == 120
    assert [r.seq for r in reqs] == list(range(120))
    assert all(a.arrival_offset_s <= b.arrival_offset_s for a, b in zip(reqs, reqs[1:]))


def test_class_mix_follows_weights():
    spec = make_spec(
        request_count=4000,
        class_weights={"short": 0.8, "medium": 0.1, "long": 0.1},
    )
    reqs = build_requests(spec, load_prompts(PROMPTS))
    counts = {}
    for r in reqs:
        counts[r.prompt.prompt_class] = counts.get(r.prompt.prompt_class, 0) + 1
    assert counts["short"] / len(reqs) > 0.7


def test_zero_weight_class_never_appears():
    spec = make_spec(class_weights={"short": 1.0, "medium": 0.0, "long": 0.0})
    reqs = build_requests(spec, load_prompts(PROMPTS))
    assert all(r.prompt.prompt_class == "short" for r in reqs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"arrival_rate_rps": 0.0},
        {"arrival_rate_rps": -1.0},
        {"request_count": 0},
        {"class_weights": {"short": 0.0, "medium": 0.0, "long": 0.0}},
        {"class_weights": {"weird": 1.0}},
    ],
)
def test_invalid_specs_raise(overrides):
    with pytest.raises(ValueError):
        poisson_arrivals(make_spec(**overrides))


def test_prompt_file_loads_and_classes_cover_all():
    items = load_prompts(PROMPTS)
    assert len(items) >= 15
    classes = {i.prompt_class for i in items}
    assert classes == {"short", "medium", "long"}
    assert all(i.max_tokens > 0 for i in items)
