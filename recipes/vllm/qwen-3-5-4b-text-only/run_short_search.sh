#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_COST_PER_HOUR="${GPU_COST_PER_HOUR:?set GPU_COST_PER_HOUR}"
OUT="${OUT:-$ROOT_DIR/artifacts/qwen3_5_4b_text_only/short_search_v1}"
NUM_RUNS="${NUM_RUNS:-1}"

cd "$ROOT_DIR"
perfsmith run \
  --model Qwen/Qwen3.5-4B \
  --workload "$ROOT_DIR/recipes/vllm/qwen-3-5-4b-text-only/workload.short.search.v1.json" \
  --grid "$ROOT_DIR/recipes/vllm/qwen-3-5-4b-text-only/grid.short.search.v1.json" \
  --out "$OUT" \
  --gpu-cost-per-hour "$GPU_COST_PER_HOUR" \
  --num-runs "$NUM_RUNS" \
  --server-extra-args "--language-model-only --no-enable-prefix-caching --enforce-eager"
