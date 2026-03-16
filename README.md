# Perfsmith (ServingOps)

Perfsmith finds the cheapest self-hosted serving configuration that meets your p99 latency SLO for a given workload, then produces a reproducible SLO Pack report.

Project site: https://servingops.com

## Why It Exists

vLLM can benchmark. Engineers can tweak flags.
The hard part is making a repeatable decision:

- Which config meets my p99 targets?
- What max concurrency can it sustain before p99 blows up?
- What is the cost per token on my GPU?
- How do I rerun this after a model or engine update to catch regressions?

Perfsmith turns raw benchmark artifacts into a decision plus proof.

## What Perfsmith Outputs

Given a workload and SLO tier, Perfsmith produces:

- Winner config: must pass the selected SLO tier and verification requirements
- Top alternatives: tradeoffs across throughput and tail latency
- SLO Pack (Markdown):
  - exact `vllm serve`-style configuration values
  - measured TTFT and ITL percentiles from the selected candidate
  - throughput and estimated cost per 1M tokens
  - reproduction information from the saved decision artifact

## What Perfsmith Measures (v0)

Perfsmith treats serving as a constraint problem:

- p99 TTFT: time to first token
- p99 ITL: inter-token latency
- throughput: tokens per second, used as a cost proxy
- failure rate

Current default SLO tiers:

- `strict`: p99 TTFT <= 500 ms, p99 ITL <= 60 ms
- `balanced`: p99 TTFT <= 650 ms, p99 ITL <= 80 ms

## Quickstart (No GPU Required)

Perfsmith can work purely from existing benchmark artifacts, either as a zip file or a directory.

```bash
python3 -m pip install -e .
perfsmith summarize --input fixtures/runs/qwen3_4b_5090_short_smoke.raw.zip --out artifacts/atlas.csv
perfsmith optimize --atlas artifacts/atlas.csv --workload fixtures/workloads/short_balanced.json --sla-tier balanced --gpu-cost-per-hour 1.75 --out artifacts/decision.json
perfsmith report --decision artifacts/decision.json --out artifacts/slo_pack.md
```

If the commands above produce `artifacts/slo_pack.md`, the decision pipeline is working locally.

## Running Real Benchmarks (GPU)

Perfsmith owns the run loop for a small vLLM config grid.

```bash
perfsmith run \
  --model Qwen/Qwen3-4B \
  --workload recipes/vllm/workload.short.json \
  --grid recipes/vllm/grid.v0.json \
  --out artifacts/qwen3-4b-short \
  --gpu-cost-per-hour 0.322
```

That command starts `vllm serve` for each config, runs `vllm bench serve`, saves raw benchmark JSON, summarizes the run, emits decision artifacts for strict and balanced tiers, and writes markdown SLO Packs.

## Tool-Aware Replay (Early)

Perfsmith includes an early trace replay mode for agent-style workflows where tool latency dominates the tail.

- tool calls can be stubbed with configurable p50, p95, p99 latency and failure rate
- output includes a simple decomposition across model, tool, and orchestration time

```bash
perfsmith replay-trace --trace fixtures/trace/agent_trace.json --stub fixtures/trace/tool_stub.json
```

## Status

Active development.

Current v0 focus:

- vLLM-first ingestion and deterministic decision/reporting
- strict and balanced SLO tiers with verification flow
- workload specs that reflect real length distributions
- reproducible run recipes that can be lifted onto GPU hosts

Planned next:

- automated execution adapters for GPU providers
- better workload trace intake without requiring prompt text
- regression gating against a last-known-good baseline

## License

MIT. See [LICENSE](LICENSE).
