# Perfsmith SLO Pack

- Decision ID: `decision`
- Generated At (UTC): `2026-03-15T00:00:00+00:00`
- Workload: `short_qwen3_4b`
- Benchmark: `short`
- SLA Tier: `balanced`
- Atlas: `fixtures/atlas/optimizer_tiered.csv`

## Winner

- Candidate ID: `short|256|128|400|inf|1.0|0.92|1024|128|8192|Qwen/Qwen3-4B|2`
- Status: `verified`
- Max Concurrency: `2`
- Total Token Throughput: `900.000` tok/s
- p99 TTFT: `620.000` ms
- p99 ITL: `50.000` ms

## Serve Flags

```bash
vllm serve Qwen/Qwen3-4B --gpu-memory-utilization 0.92 --max-model-len 1024 --max-num-seqs 128 --max-num-batched-tokens 8192
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

- Cost per 1M tokens: `0.099` USD

## Alternatives

- `short|256|128|400|inf|1.0|0.9|1024|32|8192|Qwen/Qwen3-4B|2`: concurrency `2`, throughput `815.152` tok/s, p99 TTFT `88.296` ms, p99 ITL `7.242` ms
- `short|256|128|400|inf|1.0|0.92|1024|128|8192|Qwen/Qwen3-4B|1`: concurrency `1`, throughput `468.849` tok/s, p99 TTFT `33.057` ms, p99 ITL `6.637` ms
- `short|256|128|400|inf|1.0|0.9|1024|32|8192|Qwen/Qwen3-4B|1`: concurrency `1`, throughput `405.903` tok/s, p99 TTFT `58.130` ms, p99 ITL `6.543` ms

## Repro Commands

```bash
perfsmith report --decision decision.json --out <report.md>
# Decision artifact: decision.json
```
