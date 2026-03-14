#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/artifacts/day_0}"
RAW_DIR="$OUTPUT_ROOT/raw"
REPORT_DIR="$OUTPUT_ROOT/reports"
LOG_DIR="$OUTPUT_ROOT/logs"
GPU_COST_PER_HOUR="${GPU_COST_PER_HOUR:-0}"
NUM_RUNS="${NUM_RUNS:-1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
BENCHMARK_NAME="${BENCHMARK_NAME:-short}"
NUM_PROMPTS="${NUM_PROMPTS:-400}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-256}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
BURSTINESS="${BURSTINESS:-1.0}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1,2,4,8}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:---no-enable-prefix-caching --enforce-eager --disable-log-requests}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "$RAW_DIR" "$REPORT_DIR" "$LOG_DIR"

for cmd in python3 vllm perfsmith nvidia-smi curl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: $cmd" >&2
    exit 1
  fi
done

python3 - "$ROOT_DIR/day_0/serve_params.json" "$OUTPUT_ROOT/serve_params.resolved.json" "$MODEL_ID" <<'PY2'
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
PY2

server_pid=""
server_log=""

cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local timeout="$1"
  local started_at
  started_at=$(date +%s)

  while true; do
    if curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi

    if [[ -n "${server_pid:-}" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
      echo "server exited before becoming ready" >&2
      if [[ -n "${server_log:-}" ]] && [[ -f "$server_log" ]]; then
        tail -n 80 "$server_log" >&2 || true
      fi
      return 1
    fi

    if (( $(date +%s) - started_at >= timeout )); then
      echo "server did not become ready within ${timeout}s" >&2
      if [[ -n "${server_log:-}" ]] && [[ -f "$server_log" ]]; then
        tail -n 80 "$server_log" >&2 || true
      fi
      return 1
    fi

    sleep 2
  done
}

stop_server() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
}

start_server() {
  local gpu_mem="$1"
  local max_model_len="$2"
  local max_num_seqs="$3"
  local max_num_batched_tokens="$4"
  local log_path="$5"

  stop_server

  local -a cmd
  local -a extra_args
  read -r -a extra_args <<< "$SERVER_EXTRA_ARGS"

  cmd=(
    vllm serve "$MODEL_ID"
    --host "$SERVER_HOST"
    --port "$SERVER_PORT"
    --gpu-memory-utilization "$gpu_mem"
    --max-model-len "$max_model_len"
  )

  if [[ -n "$max_num_seqs" ]] && [[ "$max_num_seqs" != "None" ]]; then
    cmd+=(--max-num-seqs "$max_num_seqs")
  fi
  if [[ -n "$max_num_batched_tokens" ]] && [[ "$max_num_batched_tokens" != "None" ]]; then
    cmd+=(--max-num-batched-tokens "$max_num_batched_tokens")
  fi
  if (( ${#extra_args[@]} > 0 )); then
    cmd+=("${extra_args[@]}")
  fi

  server_log="$log_path"
  (
    exec "${cmd[@]}"
  ) >"$server_log" 2>&1 &
  server_pid=$!

  wait_for_server "$SERVER_READY_TIMEOUT"
}

enrich_result_json() {
  local result_path="$1"
  local run_number="$2"
  local run_date="$3"
  local gpu_mem="$4"
  local max_model_len="$5"
  local max_num_seqs="$6"
  local max_num_batched_tokens="$7"
  local max_concurrency="$8"

  python3 - "$result_path" "$BENCHMARK_NAME" "$MODEL_ID" "$run_number" "$run_date" "$gpu_mem" "$max_model_len" "$max_num_seqs" "$max_num_batched_tokens" "$max_concurrency" "$NUM_PROMPTS" "$REQUEST_RATE" "$BURSTINESS" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" <<'PY2'
import json
import sys
from pathlib import Path

(
    result_path,
    benchmark_name,
    model_id,
    run_number,
    run_date,
    gpu_mem,
    max_model_len,
    max_num_seqs,
    max_num_batched_tokens,
    max_concurrency,
    num_prompts,
    request_rate,
    burstiness,
    random_input_len,
    random_output_len,
) = sys.argv[1:]

path = Path(result_path)
payload = json.loads(path.read_text(encoding="utf-8"))
if isinstance(payload, list):
    payload = payload[0] if payload else {}
if not isinstance(payload, dict):
    raise SystemExit(f"unexpected benchmark JSON shape in {path}")

metadata = payload.get("metadata")
if isinstance(metadata, dict):
    for key, value in metadata.items():
        payload.setdefault(key, value)


def first(*keys, default=None):
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return default


def to_int(value, default=0):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default

payload["_benchmark_name"] = first("_benchmark_name", default=benchmark_name)
payload["model_id"] = first("model_id", "model", default=model_id)
payload["tokenizer_id"] = first("tokenizer_id", "tokenizer", default=model_id)
payload["backend"] = first("backend", default="vllm")
payload["endpoint_type"] = first("endpoint_type", default="vllm")
payload["run_number"] = to_int(first("run_number", default=run_number), default=int(run_number))
payload["date"] = first("date", default=run_date)
payload["num_prompts"] = to_int(first("num_prompts", default=num_prompts), default=int(num_prompts))
payload["request_rate"] = first("request_rate", default=request_rate)
payload["burstiness"] = to_float(first("burstiness", default=burstiness), default=float(burstiness))
payload["max_concurrency"] = to_int(first("max_concurrency", "concurrency", default=max_concurrency), default=int(max_concurrency))
payload["gpu_memory_utilization"] = to_float(first("gpu_memory_utilization", default=gpu_mem), default=float(gpu_mem))
payload["max_model_len"] = to_int(first("max_model_len", default=max_model_len), default=int(max_model_len))
payload["max_num_seqs"] = to_int(first("max_num_seqs", default=max_num_seqs), default=int(max_num_seqs))
payload["max_num_batched_tokens"] = to_int(first("max_num_batched_tokens", default=max_num_batched_tokens), default=int(max_num_batched_tokens))
payload["random_input_len"] = to_int(first("random_input_len", default=random_input_len), default=int(random_input_len))
payload["random_output_len"] = to_int(first("random_output_len", default=random_output_len), default=int(random_output_len))
payload["completed"] = to_int(first("completed", "successful_requests", default=0))
payload["failed"] = to_int(first("failed", "failed_requests", default=0))
payload["duration"] = to_float(first("duration", "benchmark_duration", default=0.0))
payload["request_throughput"] = to_float(first("request_throughput", default=0.0))
payload["output_throughput"] = to_float(first("output_throughput", "output_token_throughput", default=0.0))
input_tp = to_float(first("input_throughput", "input_token_throughput", default=0.0))
payload["total_token_throughput"] = to_float(
    first("total_token_throughput", default=input_tp + payload["output_throughput"]),
    default=input_tp + payload["output_throughput"],
)
payload["mean_ttft_ms"] = to_float(first("mean_ttft_ms", default=0.0))
payload["p99_ttft_ms"] = to_float(first("p99_ttft_ms", default=0.0))
payload["mean_itl_ms"] = to_float(first("mean_itl_ms", default=0.0))
payload["p99_itl_ms"] = to_float(first("p99_itl_ms", default=0.0))
payload["mean_e2el_ms"] = to_float(first("mean_e2el_ms", default=0.0))
payload["p99_e2el_ms"] = to_float(first("p99_e2el_ms", default=0.0))

path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY2
}

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model_id=$MODEL_ID"
  echo "gpu_cost_per_hour=$GPU_COST_PER_HOUR"
  echo "num_runs=$NUM_RUNS"
  echo "benchmark_name=$BENCHMARK_NAME"
  echo "num_prompts=$NUM_PROMPTS"
  echo "request_rate=$REQUEST_RATE"
  echo "burstiness=$BURSTINESS"
  echo "random_input_len=$RANDOM_INPUT_LEN"
  echo "random_output_len=$RANDOM_OUTPUT_LEN"
  echo "concurrency_values=$CONCURRENCY_VALUES"
  echo "server_extra_args=$SERVER_EXTRA_ARGS"
  echo "bench_extra_args=$BENCH_EXTRA_ARGS"
  echo "python=$(python3 --version 2>&1)"
  echo "vllm=$(python3 -m pip show vllm 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')"
  echo "perfsmith=$(python3 -m pip show perfsmith 2>/dev/null | awk -F': ' '/^Version:/ {print $2}')"
  echo "gpu_info="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "$OUTPUT_ROOT/system_info.txt"

mapfile -t CONFIG_ROWS < <(
  python3 - "$OUTPUT_ROOT/serve_params.resolved.json" <<'PY2'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for idx, row in enumerate(rows):
    print("\t".join([
        str(idx),
        str(row.get("gpu_memory_utilization", "")),
        str(row.get("max_model_len", "")),
        str(row.get("max_num_seqs", "")),
        str(row.get("max_num_batched_tokens", "")),
    ]))
PY2
)

IFS=',' read -r -a CONCURRENCY_ARRAY <<< "$CONCURRENCY_VALUES"

successful_runs=0
for config_row in "${CONFIG_ROWS[@]}"; do
  IFS=$'\t' read -r config_index gpu_mem max_model_len max_num_seqs max_num_batched_tokens <<< "$config_row"

  combo_dir="$RAW_DIR/$RUN_STAMP/SERVE--gpu_memory_utilization=${gpu_mem}-max_model_len=${max_model_len}-max_num_seqs=${max_num_seqs}-max_num_batched_tokens=${max_num_batched_tokens}-BENCH--random_input_len=${RANDOM_INPUT_LEN}-random_output_len=${RANDOM_OUTPUT_LEN}"
  mkdir -p "$combo_dir"

  current_server_log="$LOG_DIR/server_config_${config_index}.log"
  echo "[day_0] starting server for config ${config_index}: gpu_memory_utilization=${gpu_mem}, max_model_len=${max_model_len}, max_num_seqs=${max_num_seqs}, max_num_batched_tokens=${max_num_batched_tokens}"
  if ! start_server "$gpu_mem" "$max_model_len" "$max_num_seqs" "$max_num_batched_tokens" "$current_server_log"; then
    echo "[day_0] failed to start server for config ${config_index}; see $current_server_log" >&2
    continue
  fi

  for concurrency in "${CONCURRENCY_ARRAY[@]}"; do
    concurrency="${concurrency// /}"
    [[ -z "$concurrency" ]] && continue

    run_dir="$combo_dir/max_concurrency=${concurrency}"
    mkdir -p "$run_dir"

    for ((run_index=0; run_index<NUM_RUNS; run_index++)); do
      run_path="$run_dir/run=${run_index}.json"
      bench_log="$LOG_DIR/bench_config_${config_index}_c${concurrency}_run${run_index}.log"
      run_date=$(date -u +%Y%m%d-%H%M%S)

      local_bench_cmd=(
        vllm bench serve
        --backend vllm
        --base-url "http://${SERVER_HOST}:${SERVER_PORT}"
        --endpoint /v1/completions
        --model "$MODEL_ID"
        --dataset-name random
        --num-prompts "$NUM_PROMPTS"
        --random-input-len "$RANDOM_INPUT_LEN"
        --random-output-len "$RANDOM_OUTPUT_LEN"
        --request-rate "$REQUEST_RATE"
        --max-concurrency "$concurrency"
        --save-result
        --result-dir "$run_dir"
        --result-filename "run=${run_index}.json"
        --metadata "_benchmark_name=${BENCHMARK_NAME}" "burstiness=${BURSTINESS}"
      )

      if [[ -n "$BENCH_EXTRA_ARGS" ]]; then
        read -r -a bench_extra_array <<< "$BENCH_EXTRA_ARGS"
        local_bench_cmd+=("${bench_extra_array[@]}")
      fi

      echo "[day_0] benchmarking config ${config_index}, concurrency=${concurrency}, run=${run_index}"
      if "${local_bench_cmd[@]}" > >(tee "$bench_log") 2>&1; then
        if [[ -f "$run_path" ]]; then
          enrich_result_json "$run_path" "$run_index" "$run_date" "$gpu_mem" "$max_model_len" "$max_num_seqs" "$max_num_batched_tokens" "$concurrency"
          successful_runs=$((successful_runs + 1))
        else
          echo "[day_0] benchmark reported success but result file missing: $run_path" >&2
        fi
      else
        echo "[day_0] benchmark failed for config ${config_index}, concurrency=${concurrency}, run=${run_index}; see $bench_log" >&2
        if [[ -f "$run_path" ]]; then
          enrich_result_json "$run_path" "$run_index" "$run_date" "$gpu_mem" "$max_model_len" "$max_num_seqs" "$max_num_batched_tokens" "$concurrency"
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
          echo "[day_0] server died during benchmark; stopping config ${config_index}" >&2
          break 2
        fi
      fi
    done
  done

  stop_server
  sleep 2

done

stop_server

echo "manual_server_bench" > "$OUTPUT_ROOT/sweep_mode.txt"

echo "[day_0] successful benchmark files: $successful_runs"
if [[ "$successful_runs" -eq 0 ]]; then
  echo "[day_0] no successful run files were produced" >&2
  exit 1
fi

echo "[day_0] summarizing raw runs"
perfsmith summarize --input "$RAW_DIR" --out "$OUTPUT_ROOT/summary.csv"

python3 - "$OUTPUT_ROOT/summary.csv" "$OUTPUT_ROOT/workload_short.json" "$GPU_COST_PER_HOUR" "$NUM_RUNS" <<'PY2'
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
    "top_k_verify": 2,
    "prune_top_n": 6,
    "verification_min_runs": max(1, num_runs),
    "model_id": first.get("model_id") or None,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY2

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
find "$OUTPUT_ROOT" -maxdepth 4 -type f | sort
