# Day 0

This folder gives you one reproducible starting point for the first Perfsmith experiment:

- one model: `Qwen/Qwen3-4B`
- one workload: `256` input tokens, `128` output tokens, `400` prompts
- one search space: three conservative vLLM serving configs
- two SLA tiers: `strict` and `balanced`

The goal is not to find the global optimum. The goal is to produce one clean dataset you can bring back into Perfsmith.

## Files

- `Dockerfile`: builds a GPU-ready image from the official `vllm/vllm-openai` base image and installs the local Perfsmith CLI.
- `serve_params.json`: a conservative three-config vLLM sweep tuned for stability on day 0.
- `bench_params.json`: the short synthetic workload definition used by `vllm bench serve` with `400` prompts.
- `sla_params.json`: the strict and balanced p99 limits.
- `run_day0.sh`: starts one server per config, runs a benchmark concurrency ladder, saves raw `run=*.json` files, then summarizes and reports.


## Build the image

From the repo root:

```bash
docker build -f day_0/Dockerfile -t perfsmith-day0 .
```

The image is pinned to `vllm/vllm-openai:v0.17.1`.

## Run it on a GPU machine

From the repo root:

Make sure the host already has the NVIDIA Container Toolkit available; otherwise `--gpus all` will not work.

```bash
docker run --gpus all --rm -it \
  -v "$PWD":/workspace/perfsmith \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  perfsmith-day0
```

Why mount the repo with `-v "$PWD":/workspace/perfsmith`?

- your outputs land directly in the cloned repo
- you can edit scripts without rebuilding the image
- artifacts are still there after the container exits

## Run the experiment

Inside the container:

```bash
cd /workspace/perfsmith
GPU_COST_PER_HOUR=0.65 ./day_0/run_day0.sh
```

By default, the script now runs `NUM_RUNS=1` across the concurrency ladder `1,2,4,8`. Once the smoke pass works, increase the repeat count or extend the ladder:

```bash
NUM_RUNS=2 GPU_COST_PER_HOUR=0.65 ./day_0/run_day0.sh
```

The default server command disables prefix caching, enforces eager execution, and disables request logging for this synthetic benchmark.

Optional override if you want to explore more concurrency levels:

```bash
CONCURRENCY_VALUES=1,2,4,8,12,16 GPU_COST_PER_HOUR=0.65 ./day_0/run_day0.sh
```

Optional override if you want to test a different model later:

```bash
MODEL_ID=Qwen/Qwen3-4B GPU_COST_PER_HOUR=0.65 ./day_0/run_day0.sh
```

Replace `0.65` with the actual hourly rate from Vast.ai.

## What the script does

1. Writes a resolved `serve_params` file with your chosen `MODEL_ID`.
2. Captures system metadata into `artifacts/day_0/system_info.txt`.
3. Starts `vllm serve` in the background for one config at a time and waits for `/v1/models` to become ready.
4. Runs `vllm bench serve` against that live server across a fixed concurrency ladder and saves `run=*.json` artifacts.
5. Enriches each saved result with the serve config and benchmark metadata Perfsmith expects.
6. Converts the raw `run=*.json` artifacts into `artifacts/day_0/summary.csv`.
7. Generates a workload spec for the short benchmark and requires the same repeat count during verification.
8. Runs `perfsmith optimize` for `strict` and `balanced` tiers.
9. Writes markdown reports into `artifacts/day_0/reports/`.

## Expected outputs

After a successful run, you should have:

- `artifacts/day_0/raw/`
- `artifacts/day_0/summary.csv`
- `artifacts/day_0/workload_short.json`
- `artifacts/day_0/runs/day0-strict.json`
- `artifacts/day_0/runs/day0-balanced.json`
- `artifacts/day_0/reports/day0-strict.md`
- `artifacts/day_0/reports/day0-balanced.md`

If strict fails, you may only get the balanced report. That is still useful.

## What to bring back here

Bring back:

- `artifacts/day_0/raw/` or a zip of it
- `artifacts/day_0/summary.csv`
- `artifacts/day_0/system_info.txt`
- both JSON optimize artifacts under `artifacts/day_0/runs/`
- both markdown reports under `artifacts/day_0/reports/`

## Important limitation

`verification_min_runs` follows `NUM_RUNS`, which defaults to `1` for the first smoke pass. Once the run is stable, use `NUM_RUNS=2` for a more defensible result.
