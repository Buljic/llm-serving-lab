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
requests meeting the SLO, which is a harsher number than raw
completion rate. The runner refuses to start a configuration with zero
requests or no SLO, because both would produce numbers that look meaningful
and are not.

## Results

### GPU: Qwen2.5-7B-AWQ on one RTX 3090

The campaign this report was written for. vLLM 0.10.2 served
Qwen2.5-7B-Instruct-AWQ with max-model-len 4096 and up to 32 concurrent
sequences; the client sent the committed prompt mix as a seeded Poisson process
at 1, 2, 4 and 8 offered requests per second, 120 requests per configuration
with two repetitions and a warmup, on 2026-11-01. The interactive SLO applies
throughout: 0.75 s TTFT and 0.10 s per output token. 1200 requests, zero
errors.

<!-- report-table:results/qwen25-7b-awq-3090 -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-load-1rps | 1 | 240 | 20.61 | 26.9 | 8.21 | 8.73 | 169.9 | 0.953 | 1 | 0.65 |
| gpu-load-2rps | 2 | 240 | 21.92 | 27.71 | 8.45 | 9.07 | 326 | 1.833 | 1 | 0.34 |
| gpu-load-4rps | 4 | 240 | 24.62 | 30.48 | 8.9 | 10.2 | 603.8 | 3.402 | 1 | 0.18 |
| gpu-load-8rps | 8 | 240 | 31.44 | 42.04 | 13.2 | 15.13 | 1050.5 | 5.896 | 1 | 0.11 |
| gpu-load-8rps-kv0.45 | 8 | 240 | 30.83 | 44.25 | 13.4 | 15.13 | 1023.5 | 5.743 | 1 | 0.11 |
<!-- /report-table -->

Read across the load column: aggregate output throughput rises 170 to 1050
tokens per second almost linearly while p95 TTFT stays inside 27-42 ms and p95
TPOT inside 8.7-15.1 ms; every completed request meets the SLO. The achieved
request rate at 8 offered is 5.9, so the queue is growing even though latency
has not broken yet; push further and TTFT is the first thing to go, which is
what the fixture models.

The fifth row repeats the 8 rps load with `--gpu-memory-utilization 0.45`
instead of 0.9, roughly halving the KV cache. Throughput differs by 2.5
percent, inside run-to-run noise. At 4096 context and this prompt mix the cache
never fills, so the eviction question stays open by design, not by accident.

### CPU: llama3.2:1b on llama.cpp

The fallback run from 2026-10-29, re-measured on 2026-11-01 after the runner
fixed a real defect: the synchronous client blocked the event loop and
serialized every request, so the earlier CPU numbers measured a serial client
rather than the server. With each request on its own thread the four slots
actually overlap.

<!-- report-table:results/qwen32-1b-cpu -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu-load-0.1rps | 0.1 | 16 | 69.88 | 279.46 | 31.47 | 34.33 | 11.8 | 0.121 | 1 | 14.09 |
| cpu-load-0.25rps | 0.25 | 16 | 53.54 | 70.49 | 31.56 | 33.88 | 26.4 | 0.271 | 1 | 6.31 |
| cpu-load-0.5rps | 0.5 | 16 | 60.88 | 111.02 | 32.76 | 58.5 | 44.2 | 0.458 | 1 | 3.77 |
| cpu-load-1rps | 1 | 16 | 67.93 | 172.54 | 33.51 | 40.97 | 68.3 | 0.701 | 1 | 2.44 |
<!-- /report-table -->

Throughput scales 11.8 to 68.3 output tokens per second as concurrency grows,
and the achieved rate falls behind offered load at 1 rps (0.70 achieved), which
is the box capacity for this model. This run uses the relaxed CPU SLO (20 s
TTFT, 2.5 s TPOT).

### Pipeline fixture

The simulated server under `results/fixture/` proves the measurement chain with
no model at all: single serving slot, prefill by prompt length, decode by
token. Its rows are labeled `fixture: true` and are not measurements.

<!-- report-table:results/fixture -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stub-load-0.5rps | 0.5 | 48 | 259.44 | 4386.13 | 8.33 | 8.33 | 61.5 | 0.304 | 0.625 | 2.71 |
| stub-load-1rps | 1 | 48 | 3847.05 | 6325.7 | 8.33 | 8.33 | 104.2 | 0.172 | 0.2083 | 1.6 |
| stub-load-2rps | 2 | 48 | 7087.05 | 12701.48 | 8.33 | 8.33 | 115.6 | 0.19 | 0.2083 | 1.44 |
| stub-load-4rps | 4 | 48 | 9221.77 | 18320.91 | 8.33 | 8.33 | 116 | 0.114 | 0.125 | 1.44 |
<!-- /report-table -->

## Limits

- One GPU, one model, one quantisation: a single RTX 3090, Qwen2.5-7B-AWQ, vLLM
  0.10.2. A different engine or quantisation can move these numbers, so the
  table is a measurement of this stack, not a ranking.
- KV cache pressure was not reached. At 4096 context with the halved cache the
  throughput difference was inside noise; eviction and preemption need long
  contexts and more sequences than this campaign used.
- The client holds at most 32 requests in flight, so above the server's
  capacity the offered load shows up as a growing client-side queue; achieved
  request rate is the saturation signal, not latency alone.
- The CPU campaign used a 1B model on a shared, contended box with a relaxed
  SLO; it validates the harness and the box capacity, not a serving config.
- Long prompts were excluded from the CPU run because their answers take
  minutes at CPU speeds; the GPU campaign includes them.
- Cost per token uses stated hourly assumptions (0.40 USD/h on GPU, 0.60 USD/h
  on CPU), not measured energy or cloud billing.
- Percentiles use linear interpolation on 240 requests per GPU configuration
  and 16 per CPU configuration; the GPU tails are the firmer ones.

## Reproduction

```sh
make test          # offline: workload, client, metrics, runner guards, report checks
make fixture       # regenerates results/fixture from the simulated server
make report        # regenerates the tables and the figures from committed runs
make report-check  # fails if any table in this file drifts from its run
```

The GPU campaign, when the card is available:

```sh
pip install -r requirements-serving.txt
vllm serve <model> --port 8000
cp configs/sweep.example.yaml configs/sweep.yaml  # adjust
.venv/bin/python -m bench.runner --config configs/sweep.yaml --output results/campaign
PYTHONPATH=. .venv/bin/python analysis/report.py --results results/campaign --out analysis/out
```
