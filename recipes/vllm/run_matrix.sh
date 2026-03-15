#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
OUT="${OUT:-$ROOT_DIR/artifacts/qwen3-4b-short}"
GPU_COST_PER_HOUR="${GPU_COST_PER_HOUR:?set GPU_COST_PER_HOUR}"
NUM_RUNS="${NUM_RUNS:-1}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:---no-enable-prefix-caching --enforce-eager}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
RAW_ARTIFACT_URL="${RAW_ARTIFACT_URL:-}"

cd "$ROOT_DIR"
cmd=(
  perfsmith run
  --model "$MODEL_ID"
  --workload "$ROOT_DIR/recipes/vllm/workload.short.json"
  --grid "$ROOT_DIR/recipes/vllm/grid.v0.json"
  --out "$OUT"
  --gpu-cost-per-hour "$GPU_COST_PER_HOUR"
  --num-runs "$NUM_RUNS"
  --server-extra-args "$SERVER_EXTRA_ARGS"
  --bench-extra-args "$BENCH_EXTRA_ARGS"
)

if [[ -n "$RAW_ARTIFACT_URL" ]]; then
  cmd+=(--raw-artifact-url "$RAW_ARTIFACT_URL")
fi

"${cmd[@]}"
