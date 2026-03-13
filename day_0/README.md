# Day 0

This folder gives you one reproducible starting point for the first Perfsmith experiment:

- one model: `Qwen/Qwen3.5-4B`
- one workload: `256` input tokens, `128` output tokens, `250` prompts
- one search space: four vLLM serving configs
- two SLA tiers: `strict` and `balanced`

The goal is not to find the global optimum. The goal is to produce one clean dataset you can bring back into Perfsmith.

## Files

- `Dockerfile`: builds a GPU-ready image from the official `vllm/vllm-openai` base image and installs the local Perfsmith CLI.
- `serve_params.json`: the small vLLM config sweep for day 0.
- `bench_params.json`: the short synthetic workload definition used by `vllm bench sweep serve_sla`.
- `sla_params.json`: the strict and balanced p99 limits.
- `run_day0.sh`: runs the sweep, summarizes results, and generates Perfsmith optimize/report artifacts.

## Docker concepts you should know

### Image vs container

- An image is a template.
- A container is a running instance of that template.

You build an image once, then run containers from it.

### Why `FROM`

`FROM vllm/vllm-openai:v0.16.0` means "start from an image that already knows how to run vLLM on GPUs." That saves you from installing CUDA and vLLM by hand, while keeping the runtime pinned.

### Why `WORKDIR`

`WORKDIR /workspace/perfsmith` sets the default directory inside the container. Every later command runs there unless you change it.

### Why `COPY`

`COPY` moves files from your repo into the image at build time. Here we copy only what day 0 needs: the package, fixtures, and this folder.

### Why `RUN`

`RUN` executes during image build. We use it to install the local Perfsmith package so the `perfsmith` CLI is available inside the container.

### Why `ENTRYPOINT`

`ENTRYPOINT ["/bin/bash"]` makes the container start in a shell. For day 0 this is simpler than hiding everything behind a fixed entrypoint.

### Why `.dockerignore`

`.dockerignore` keeps the build context small. Without it, Docker would send `.git`, raw data, tests, and other unnecessary files into the build.

## Build the image

From the repo root:

```bash
docker build -f day_0/Dockerfile -t perfsmith-day0 .
```

The image is pinned to `vllm/vllm-openai:v0.16.0`.

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

Optional override if you want to test a different model later:

```bash
MODEL_ID=Qwen/Qwen3.5-4B GPU_COST_PER_HOUR=0.65 ./day_0/run_day0.sh
```

Replace `0.65` with the actual hourly rate from Vast.ai.

## What the script does

1. Writes a resolved `serve_params` file with your chosen `MODEL_ID`.
2. Captures system metadata into `artifacts/day_0/system_info.txt`.
3. Runs `vllm bench sweep serve_sla`.
4. Converts raw `run=*.json` artifacts into `artifacts/day_0/summary.csv`.
5. Generates a workload spec for the short benchmark.
6. Runs `perfsmith optimize` for `strict` and `balanced` tiers.
7. Writes markdown reports into `artifacts/day_0/reports/`.

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

`verification_min_runs` is set to `1` in the generated workload spec so day 0 stays cheap. That means the first winner is directional, not final. Once we see the day-0 frontier, we should rerun the best candidate with more repeats.
