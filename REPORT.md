# Measurement report

## The question

At what offered load does a single-GPU LLM serving stack stop meeting an
interactive SLO, and which limit shows up first: queueing delay, decode
contention, or KV cache pressure? Serving benchmarks usually answer a
different question (peak throughput on a saturated batch), so the interesting
part of the curve, where a user would already notice, stays unmeasured.

## Setup

Target: one OpenAI-compatible server (vLLM) on a single GPU, model class
7B-instruct, CPU-only client machine. Arrivals are a Poisson process with a
fixed seed over a committed prompt set with short, medium and long exchanges
(50/30/20 percent). Each configuration runs a warmup, then three repetitions of
a fixed request count, and every run records the machine it ran on: GPU,
driver, CUDA and vLLM versions when those are present.

SLO for goodput: TTFT at most 0.75 s and TPOT at most 0.10 s per request.
Cost is computed as GPU-hours times a stated hourly rate, divided by output
tokens.

## Method

Per request the client records time to first token, per-token inter-token
times, total latency, output tokens and finish reason. Aggregation uses linear
interpolation percentiles over successful requests. Goodput is the rate of
requests meeting the SLO, which is a harsher and more honest number than raw
completion rate. The runner refuses to start a configuration with zero
requests or no SLO, because both would produce numbers that look meaningful
and are not.

## Results

The table below is generated from `results/fixture/` by `make report`; the
test suite compares it against a regeneration and fails on drift.

<!-- report-table -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stub-load-0.5rps | 0.5 | 48 | 259.44 | 4386.13 | 8.33 | 8.33 | 61.5 | 0.304 | 0.625 | 2.71 |
| stub-load-1rps | 1 | 48 | 3847.05 | 6325.7 | 8.33 | 8.33 | 104.2 | 0.172 | 0.2083 | 1.6 |
| stub-load-2rps | 2 | 48 | 7087.05 | 12701.48 | 8.33 | 8.33 | 115.6 | 0.19 | 0.2083 | 1.44 |
| stub-load-4rps | 4 | 48 | 9221.77 | 18320.91 | 8.33 | 8.33 | 116 | 0.114 | 0.125 | 1.44 |
<!-- /report-table -->

Important: these rows come from the single-slot simulated server, not from a
model. Every `run.json` in the fixture carries `"fixture": true`. What the
fixture proves is the measurement pipeline: schedule generation, streaming
capture, aggregation, and report regeneration all run and agree. It also shows
the shape the real campaign is looking for: as offered load approaches what
the server can decode, TTFT grows through queueing while TPOT stays flat,
goodput peaks and then falls, and cost per token stops improving long before
throughput stops growing.

## Mechanism

An LLM server can be read as two queues. Prompts wait for a free slot and
prefill (TTFT), then tokens stream out at a rate set by batch composition and
KV cache residency (TPOT). Under open-loop Poisson arrivals the first queue
saturates at a predictable point: when the arrival rate times mean service time
approaches one, waits grow linearly with backlog. The second queue degrades
later, when larger batches trade per-request decode speed for total
throughput. Continuous batching engines push that tradeoff further, which is
why peak-throughput numbers hide it. The fixture models only the first queue
on purpose: it is the one an interactive user hits first, and the one a
closed-loop benchmark never measures.

## Limits

- The fixture is a single-slot server with fixed rates; it models queueing, not
  batching, preemption, or KV cache eviction. Its absolute numbers are not
  model behavior.
- The real campaign has not been run yet. No claim in this report is made
  about any actual model or GPU.
- Cost per token depends on the hourly rate assumption (0.60 USD/h here for
  the fixture), not on measured energy or cloud billing.
- Percentiles use linear interpolation on modest sample counts (48 requests
  per run here); p99 on 48 samples is an extrapolation, not a measurement.

## Reproduction

```sh
make test          # offline: workload, client, metrics, runner guards, report checks
make fixture       # regenerates results/fixture from the simulated server
make report        # regenerates this table and the three figures
make report-check  # fails if the table above drifts from the fixture
```

The GPU campaign, when the card is available:

```sh
pip install -r requirements-serving.txt
vllm serve <model> --port 8000
cp configs/sweep.example.yaml configs/sweep.yaml  # adjust
.venv/bin/python -m bench.runner --config configs/sweep.yaml --output results/campaign
PYTHONPATH=. .venv/bin/python analysis/report.py --results results/campaign --out analysis/out
```
