# llm-serving-lab

Benchmarks tell you a server does 2000 tokens per second. What they usually do
not tell you is when your users start waiting. Under real traffic the first
token is late long before the average throughput drops, because requests queue
behind each other and the KV cache fills up. I wanted a small harness that
measures that breakdown instead of quoting a peak number, and that keeps the
raw per-request rows so any number in the report can be recomputed.

## What this is

A single-GPU measurement study of an OpenAI-compatible serving stack (vLLM is
the target). A workload generator produces seeded Poisson arrivals over a fixed
prompt set with short, medium and long exchanges. A streaming client records
time to first token, inter-token time, total latency, output tokens and finish
reason per request. The runner repeats each configuration after a warmup and
writes the raw rows plus a metadata file with the GPU, driver, CUDA and vLLM
versions. Analysis turns the rows into p50/p95/p99 percentiles, goodput against
a stated SLO, and cost per million output tokens.

## Does it work?

Yes, and it has been run for real on both sides of the question: a full sweep on
one GPU with vLLM, and a second sweep on the CPU-only fallback. Every row below
is regenerated from committed raw rows, and the test suite fails if the tables
drift from them.

### The GPU campaign

Qwen2.5-7B-Instruct-AWQ on one RTX 3090 with vLLM 0.10.2, max-model-len 4096,
up to 32 concurrent sequences, 120 requests per configuration with 2
repetitions, the interactive SLO from REPORT.md (0.75 s TTFT, 0.10 s per output
token), five configurations including one with the KV budget halved. 1200
requests total, zero errors, on 2026-11-01.

<!-- report-table:results/qwen25-7b-awq-3090 -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-load-1rps | 1 | 240 | 20.61 | 26.9 | 8.21 | 8.73 | 169.9 | 0.953 | 1 | 0.65 |
| gpu-load-2rps | 2 | 240 | 21.92 | 27.71 | 8.45 | 9.07 | 326 | 1.833 | 1 | 0.34 |
| gpu-load-4rps | 4 | 240 | 24.62 | 30.48 | 8.9 | 10.2 | 603.8 | 3.402 | 1 | 0.18 |
| gpu-load-8rps | 8 | 240 | 31.44 | 42.04 | 13.2 | 15.13 | 1050.5 | 5.896 | 1 | 0.11 |
| gpu-load-8rps-kv0.45 | 8 | 240 | 30.83 | 44.25 | 13.4 | 15.13 | 1023.5 | 5.743 | 1 | 0.11 |
<!-- /report-table -->

What it shows: from 1 to 8 offered requests per second the server scales from
170 to 1050 output tokens per second while p95 time-to-first-token moves from 27
to 42 ms and p95 inter-token time from 8.7 to 15.1 ms. Even at 8 offered rps
every completed request met the interactive SLO; what falls is the achieved
rate, 5.9 rps against 8 offered, which is the saturation signal an open-loop
client with 32 slots can show without violating latency. The halved-KV
configuration lands within noise of the full one (1024 vs 1050 tokens per
second), so at 4096 context this workload never pressures the cache; that
comparison needs long contexts to say anything.

### The CPU campaign

llama3.2:1b served by llama.cpp on 8 CPU cores of a shared 12-core box, niced,
4 parallel slots, CPU-only build. Four offered loads, short and medium prompts,
answers capped at 120 tokens, the relaxed CPU SLO (20 s TTFT, 2.5 s TPOT).

<!-- report-table:results/qwen32-1b-cpu -->
| run | offered_rps | requests | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | output_tokens_per_s | goodput_rps | goodput_fraction | cost_per_million_output_tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu-load-0.1rps | 0.1 | 16 | 69.88 | 279.46 | 31.47 | 34.33 | 11.8 | 0.121 | 1 | 14.09 |
| cpu-load-0.25rps | 0.25 | 16 | 53.54 | 70.49 | 31.56 | 33.88 | 26.4 | 0.271 | 1 | 6.31 |
| cpu-load-0.5rps | 0.5 | 16 | 60.88 | 111.02 | 32.76 | 58.5 | 44.2 | 0.458 | 1 | 3.77 |
| cpu-load-1rps | 1 | 16 | 67.93 | 172.54 | 33.51 | 40.97 | 68.3 | 0.701 | 1 | 2.44 |
<!-- /report-table -->

Throughput scales 11.8 to 68.3 output tokens per second as the four slots fill,
then the achieved rate stops keeping up with offered load (0.70 rps achieved at
1 rps offered), which is the capacity limit of this box.

### The pipeline fixture

The same harness also runs against a simulated server, committed under
`results/fixture/`, so the report and its tests work with no model at all. Its
rows are labeled `fixture: true` and are not measurements.

## How to run

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
make test      # offline, no GPU, no network
make report    # regenerates the table and figures from results/fixture
make smoke     # touches the GPU; skips with a message when none is present
```

Running against a real server needs the serving extras
(`pip install -r requirements-serving.txt`), a running `vllm serve`, and a
sweep file shaped like `configs/sweep.example.yaml`:

```sh
.venv/bin/python -m bench.runner --config configs/sweep.yaml --output results/campaign
PYTHONPATH=. .venv/bin/python analysis/report.py --results results/campaign --out analysis/out
```

Raw campaign output stays gitignored; only the report artifacts and the fixture
are committed.

## What is not done yet

- KV cache pressure is untested: at 4096 context the halved cache changed
  nothing, so the next campaign needs long contexts, more sequences, and a
  preemption count to say anything about eviction.
- One model, one GPU. No multi-GPU tensor parallelism, no quantisation
  comparison (AWQ only), no prefix caching experiment.
- The runner drives one endpoint at a time; it does not sweep server-side
  flags like `--gpu-memory-utilization` itself, the KV cache fraction is
  recorded per run and the server is restarted between runs by hand.
- The client-side concurrency cap turns high offered loads partly
  closed-loop; a true open-loop mode that keeps sending at the offered rate
  regardless of in-flight count would expose server queueing directly.
