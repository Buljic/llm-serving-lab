# Measurement report

## The question

At what offered load does a single-GPU LLM serving stack stop meeting an
interactive SLO, and which limit shows up first: queueing delay, decode
contention, or KV cache pressure? Serving benchmarks usually answer a
different question (peak throughput on a saturated batch), so the interesting
part of the curve, where a user would already notice, stays unmeasured.

## Setup

Target: one OpenAI-compatible server. The GPU campaign (pending, blocked by
the research lease) targets vLLM on a single GPU, model class 7B-instruct.
The CPU campaign below used llama.cpp with llama3.2:1b on 8 cores of a shared
12-core box. Arrivals are a Poisson process with a fixed seed over a committed
prompt set with short, medium and long exchanges (50/30/20 percent; the CPU
campaign excluded the long class and capped answers at 120 tokens). Each
configuration runs a warmup, then a fixed number of repetitions (three in the
example sweep, two in the CPU campaign), and every run records the machine it
ran on: GPU, driver, CUDA and vLLM versions when those are present, plus a
free-form device note.

SLO for goodput is stated per campaign: TTFT at most 0.75 s and TPOT at most
0.10 s for the GPU target; 20 s and 2.5 s for the CPU run, where the numbers
are CPU-bound. Cost is computed as GPU-hours times a stated hourly rate,
divided by output tokens.

## Method

Per request the client records time to first token, per-token inter-token
times, total latency, output tokens and finish reason. Aggregation uses linear
interpolation percentiles over successful requests. Goodput is the rate of
requests meeting the SLO, which is a harsher and more honest number than raw
completion rate. The runner refuses to start a configuration with zero
requests or no SLO, because both would produce numbers that look meaningful
and are not.

## Results

### A real run: llama3.2:1b on CPU, 2026-10-29

The harness has been run against a live model. llama3.2:1b was served by
llama.cpp on 8 CPU cores of a shared 12-core box (niced, 4 parallel slots,
CPU-only build; the GPU was under a research lease and was not used). The
sweep file is `configs/sweep.cpu.yaml`: 4 offered loads from 0.1 to 1.0
requests per second, 16 requests per configuration with 2 repetitions, the
seeded Poisson schedule, short and medium prompts with answers capped at 120
tokens. The SLO was relaxed to 20 s TTFT and 2.5 s TPOT because those are
CPU numbers, not interactive targets. Raw rows and machine metadata are
committed under `results/qwen32-1b-cpu/`.

<!-- report-table:results/qwen32-1b-cpu -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu-load-0.1rps | 0.1 | 16 | 84.01 | 324.31 | 76.78 | 107.04 | 8.7 | 0.09 | 1 | 19.13 |
| cpu-load-0.25rps | 0.25 | 16 | 108.74 | 154.61 | 79.67 | 95.5 | 11.1 | 0.118 | 1 | 15 |
| cpu-load-0.5rps | 0.5 | 16 | 82.75 | 145.1 | 67.7 | 80.7 | 13.1 | 0.135 | 1 | 12.68 |
| cpu-load-1rps | 1 | 16 | 104.3 | 145.72 | 79.33 | 89.5 | 11.7 | 0.123 | 1 | 14.2 |
<!-- /report-table -->

Three things are visible. First, per-request latency barely moves with offered
load: TTFT p50 stays around 84-108 ms and TPOT around 68-80 ms across all four
configurations. Second, aggregate throughput plateaus at roughly 12-13 output
tokens per second, which is the machine's real capacity for this model under
contention. Third, the wall clock stretches far beyond the arrival window at
higher loads (the 1 rps configuration offered its 16 requests inside 16
seconds but took over two minutes to drain). The client concurrency cap of 4
is why latency stays flat: requests past the cap wait client-side before they
are even sent, so server-side queueing is masked and the saturation shows up
in throughput and drain time instead. All 64 requests completed with zero
errors and every request met the relaxed SLO.

### Pipeline fixture

The same pipeline is also proven on a simulated server, committed under
`results/fixture/`. The stub owns one serving slot, so queueing behavior
emerges from the schedule. Its rows are labeled `fixture: true` and are not
model measurements.

<!-- report-table:results/fixture -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stub-load-0.5rps | 0.5 | 48 | 259.44 | 4386.13 | 8.33 | 8.33 | 61.5 | 0.304 | 0.625 | 2.71 |
| stub-load-1rps | 1 | 48 | 3847.05 | 6325.7 | 8.33 | 8.33 | 104.2 | 0.172 | 0.2083 | 1.6 |
| stub-load-2rps | 2 | 48 | 7087.05 | 12701.48 | 8.33 | 8.33 | 115.6 | 0.19 | 0.2083 | 1.44 |
| stub-load-4rps | 4 | 48 | 9221.77 | 18320.91 | 8.33 | 8.33 | 116 | 0.114 | 0.125 | 1.44 |
<!-- /report-table -->

The fixture shows the shape the GPU campaign is looking for: as offered load
approaches what the server can decode, TTFT grows through queueing while TPOT
stays flat, goodput peaks and then falls, and cost per token stops improving
long before throughput stops growing. Whether that holds on an actual 7B-class
model on one consumer GPU is exactly what the pending GPU campaign is for.

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

- The CPU campaign used a 1B model on a shared, contended box; its numbers
  validate the harness, not any serving configuration. The SLO was relaxed to
  match (20 s TTFT, 2.5 s TPOT).
- Long prompts were excluded from the CPU campaign because their
  multi-hundred-token answers take minutes at CPU speeds; the GPU campaign
  must include them.
- The client concurrency cap masks server-side queueing at high offered loads;
  latency percentiles alone understate saturation, which only shows in
  aggregate throughput and drain time.
- The GPU campaign with a 7B-class model, continuous batching and KV cache
  pressure has not been run yet; the research lease on the card decides when.
- Cost per token uses a stated hourly assumption (0.60 USD/h), not measured
  energy or cloud billing.
- Percentiles use linear interpolation on modest sample counts (16 requests
  per configuration); tails are indicative, not precise.

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
