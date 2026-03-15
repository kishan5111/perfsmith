# vLLM Recipe

This recipe is the minimal reproducible path for running Perfsmith on a GPU host.

## Files

- `workload.short.json`: short synthetic workload used for the first run
- `grid.v0.json`: conservative vLLM serve config grid
- `run_matrix.sh`: thin wrapper around `perfsmith run`
- `Dockerfile`: optional container image pinned to the vLLM runtime

## Run From Repo Root

```bash
GPU_COST_PER_HOUR=0.322 ./recipes/vllm/run_matrix.sh
```

Override the model or output directory if needed:

```bash
MODEL_ID=Qwen/Qwen3-4B \
OUT=artifacts/qwen3-4b-short \
GPU_COST_PER_HOUR=0.322 \
./recipes/vllm/run_matrix.sh
```

This wrapper calls:

```bash
perfsmith run \
  --model "$MODEL_ID" \
  --workload recipes/vllm/workload.short.json \
  --grid recipes/vllm/grid.v0.json \
  --out "$OUT" \
  --gpu-cost-per-hour "$GPU_COST_PER_HOUR"
```
