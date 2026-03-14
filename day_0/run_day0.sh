#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/artifacts/day_0}"
RAW_DIR="$OUTPUT_ROOT/raw"
REPORT_DIR="$OUTPUT_ROOT/reports"
GPU_COST_PER_HOUR="${GPU_COST_PER_HOUR:-0}"
NUM_RUNS="${NUM_RUNS:-2}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"

mkdir -p "$RAW_DIR" "$REPORT_DIR"

for cmd in python3 vllm perfsmith nvidia-smi; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: $cmd" >&2
    exit 1
  fi
done

python3 - "$ROOT_DIR/day_0/serve_params.json" "$OUTPUT_ROOT/serve_params.resolved.json" "$MODEL_ID" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
model_id = sys.argv[3]
payload = json.loads(source_path.read_text(encoding="utf-8"))
for row in payload:
    row["model"] = model_id
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model_id=$MODEL_ID"
  echo "gpu_cost_per_hour=$GPU_COST_PER_HOUR"
  echo "num_runs=$NUM_RUNS"
  echo "python=$(python3 --version 2>&1)"
  echo "vllm=$(python3 -m pip show vllm 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')"
  echo "perfsmith=$(python3 -m pip show perfsmith 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')"
  echo "gpu_info="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$OUTPUT_ROOT/system_info.txt"

echo "[day_0] running vLLM sweep into $RAW_DIR"
vllm bench sweep serve_sla \
  --serve-cmd "vllm serve" \
  --bench-cmd "vllm bench serve" \
  --serve-params "$OUTPUT_ROOT/serve_params.resolved.json" \
  --bench-params "$ROOT_DIR/day_0/bench_params.json" \
  --sla-params "$ROOT_DIR/day_0/sla_params.json" \
  --sla-variable max_concurrency \
  --num-runs "$NUM_RUNS" \
  --server-ready-timeout 600 \
  --output-dir "$RAW_DIR"

echo "[day_0] summarizing raw runs"
perfsmith summarize --input "$RAW_DIR" --out "$OUTPUT_ROOT/summary.csv"

python3 - "$OUTPUT_ROOT/summary.csv" "$OUTPUT_ROOT/workload_short.json" "$GPU_COST_PER_HOUR" "$NUM_RUNS" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
gpu_cost = float(sys.argv[3])
num_runs = int(sys.argv[4])
rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8", newline="")))
if not rows:
    raise SystemExit("summary.csv has no rows")
first = rows[0]
payload = {
    "name": "day0_short",
    "benchmark_name": first.get("benchmark_name", ""),
    "expected_max_input_tokens": 256,
    "expected_max_output_tokens": 128,
    "summary_table": str(summary_path),
    "gpu_cost_per_hour": gpu_cost,
    "top_k_verify": 3,
    "prune_top_n": 10,
    "verification_min_runs": max(1, num_runs),
    "model_id": first.get("model_id") or None,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

run_optimize() {
  local tier="$1"
  local run_id="day0-${tier}"

  if perfsmith optimize \
    --workload "$OUTPUT_ROOT/workload_short.json" \
    --sla-tier "$tier" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$run_id" > "$OUTPUT_ROOT/${run_id}.stdout.json"; then
    perfsmith report \
      --run-id "$run_id" \
      --output-root "$OUTPUT_ROOT" \
      --output "$REPORT_DIR/${run_id}.md"
  else
    echo "optimize failed for tier=$tier" > "$REPORT_DIR/${run_id}.failed.txt"
  fi
}

run_optimize strict
run_optimize balanced

echo "[day_0] completed"
find "$OUTPUT_ROOT" -maxdepth 3 -type f | sort
