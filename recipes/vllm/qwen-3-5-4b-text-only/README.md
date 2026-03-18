# Qwen3.5-4B Text-Only Recipe

This recipe is for `Qwen/Qwen3.5-4B` served in text-only mode.

Important:
- pass `--language-model-only` to `vllm serve`
- keep `max_model_len` derived from the workload, not swept in the main search

## Stage A: Saturation Cliff Finder

This folder gives you two runnable searches:

- `short`: `256 in / 128 out`, fixed `max_model_len=768`
- `long_prefill`: `2048 in / 128 out`, fixed `max_model_len=3072`

The search is intentionally focused on:
- `gpu_memory_utilization`
- `max_num_batched_tokens`

while holding `max_num_seqs` near a realistic concurrency ceiling for each workload.

## Run Short Search

```bash
GPU_COST_PER_HOUR=0.65 ./recipes/vllm/qwen-3-5-4b-text-only/run_short_search.sh
```

## Run Long-Prefill Search

```bash
GPU_COST_PER_HOUR=0.65 ./recipes/vllm/qwen-3-5-4b-text-only/run_long_prefill_search.sh
```

## Direct CLI Form

Short:

```bash
perfsmith run \
  --model Qwen/Qwen3.5-4B \
  --workload recipes/vllm/qwen-3-5-4b-text-only/workload.short.search.v1.json \
  --grid recipes/vllm/qwen-3-5-4b-text-only/grid.short.search.v1.json \
  --out artifacts/qwen3_5_4b_text_only/short_search_v1 \
  --gpu-cost-per-hour "$GPU_COST_PER_HOUR" \
  --server-extra-args "--language-model-only --no-enable-prefix-caching"
```

Long-prefill:

```bash
perfsmith run \
  --model Qwen/Qwen3.5-4B \
  --workload recipes/vllm/qwen-3-5-4b-text-only/workload.long_prefill.search.v1.json \
  --grid recipes/vllm/qwen-3-5-4b-text-only/grid.long_prefill.search.v1.json \
  --out artifacts/qwen3_5_4b_text_only/long_prefill_search_v1 \
  --gpu-cost-per-hour "$GPU_COST_PER_HOUR" \
  --server-extra-args "--language-model-only --no-enable-prefix-caching"
```

## Why These Grids

Short search:
- `max_model_len=768`
- `max_num_seqs=48`
- sweep `gpu_memory_utilization in {0.90,0.94,0.97,0.99}`
- sweep `max_num_batched_tokens in {4096,8192,12288,16384}`

Long-prefill search:
- `max_model_len=3072`
- `max_num_seqs=8`
- sweep `gpu_memory_utilization in {0.90,0.94,0.97,0.99}`
- sweep `max_num_batched_tokens in {4096,8192,16384,32768}`

This is the first-stage KV/batch frontier search. After these runs, use the winners to design the second-stage `max_num_seqs` sweep.
