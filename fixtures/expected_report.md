# Perfsmith SLA Pack

- Run ID: `fixture-short-balanced`
- Created At (UTC): `2026-03-06T12:15:51+00:00`
- Workload: `short_qwen3`
- Benchmark: `short`
- SLA Tier: `balanced`

## Winner

- Candidate ID: `short|256|128|250|inf|1.0|0.94|1024|128|16384|Qwen/Qwen3-8B|23`
- Status: `verified`
- Max Concurrency: `23`
- Total Token Throughput: `2744.493` tok/s
- p99 TTFT: `643.934` ms
- p99 ITL: `26.127` ms

## Serve Flags

```bash
vllm serve Qwen/Qwen3-8B --gpu-memory-utilization 0.94 --max-model-len 1024 --max-num-seqs 128 --max-num-batched-tokens 16384
```

## SLA Compliance

| Tier | Threshold | Pass |
| --- | --- | --- |
| strict | p99_ttft<=500, p99_itl<=60 | no |
| balanced | p99_ttft<=650, p99_itl<=80 | yes |

## Verification Variance

- Verification runs: `2`
- p99 TTFT stddev: `0.000` ms
- p99 ITL stddev: `0.000` ms
- Throughput stddev: `0.000` tok/s

## Cost Estimate

- Cost per 1M tokens: `0.177` USD

## Repro Commands

```bash
perfsmith report --run-id fixture-short-balanced
perfsmith report --run-id fixture-short-balanced
# Artifact: artifacts/runs/fixture-short-balanced.json
```
