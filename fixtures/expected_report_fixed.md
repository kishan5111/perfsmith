# Perfsmith SLO Pack

- Decision ID: `decision`
- Generated At (UTC): `2026-03-06T00:00:00+00:00`
- Workload: `short_qwen3`
- Benchmark: `short`
- SLA Tier: `balanced`
- Atlas: `fixtures/expected_summary.csv`

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

- Verification runs: `1`
- p99 TTFT stddev: `0.000` ms
- p99 ITL stddev: `0.000` ms
- Throughput stddev: `0.000` tok/s

## Cost Estimate

- Cost per 1M tokens: `0.177` USD

## Alternatives

- `short|256|128|250|inf|1.0|0.94|1024|128|16384|Qwen/Qwen3-8B|17`: concurrency `17`, throughput `2160.508` tok/s, p99 TTFT `494.400` ms, p99 ITL `24.260` ms
- `short|256|128|250|inf|1.0|0.94|1024|128|16384|Qwen/Qwen3-8B|1`: concurrency `1`, throughput `173.962` tok/s, p99 TTFT `41.351` ms, p99 ITL `18.775` ms

## Repro Commands

```bash
perfsmith report --decision decision.json --out <report.md>
# Decision artifact: decision.json
```
