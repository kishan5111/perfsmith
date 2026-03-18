"""Execute a reproducible vLLM config grid and emit Perfsmith artifacts."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .optimize import load_workload_spec, optimize_atlas
from .report import write_report
from .summarize import summarize_to_table
from .types import ServeConfig, WorkloadSpec
from .utils import ensure_parent, now_utc_iso


def load_serve_grid(path: Path, model_id: str | None = None) -> list[ServeConfig]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("configs", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Serve grid must be a list or an object with 'configs': {path}")
    return [ServeConfig.from_dict(row, model_id=model_id) for row in rows]


def _capture_cmd_output(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    text = (completed.stdout or completed.stderr or "").strip()
    return text


def _parse_version_from_pip_show(package_name: str) -> str:
    output = _capture_cmd_output(["python3", "-m", "pip", "show", package_name])
    for line in output.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return ""


def collect_system_info(model_id: str, gpu_cost_per_hour: float, num_runs: int, workload: WorkloadSpec) -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc_iso(),
        "model_id": model_id,
        "gpu_cost_per_hour": gpu_cost_per_hour,
        "num_runs": num_runs,
        "python": _capture_cmd_output(["python3", "--version"]),
        "vllm_version": _parse_version_from_pip_show("vllm"),
        "perfsmith_version": _parse_version_from_pip_show("perfsmith"),
        "gpu_info": _capture_cmd_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "workload": {
            "benchmark_name": workload.benchmark_name,
            "num_prompts": workload.num_prompts,
            "request_rate": workload.request_rate,
            "burstiness": workload.burstiness,
            "expected_max_input_tokens": workload.expected_max_input_tokens,
            "expected_max_output_tokens": workload.expected_max_output_tokens,
            "concurrency_values": workload.concurrency_values,
        },
    }


def _wait_for_server(host: str, port: int, timeout: int, process: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/v1/models"
    while time.time() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"Server exited before becoming ready. Log tail:\n{tail}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        time.sleep(2)
    tail = log_path.read_text(encoding="utf-8")[-4000:] if log_path.exists() else ""
    raise RuntimeError(f"Server did not become ready within {timeout}s. Log tail:\n{tail}")


def _stop_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return

    pid = getattr(process, "pid", None)
    if pid is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        return

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def _start_server(
    model_id: str,
    cfg: ServeConfig,
    host: str,
    port: int,
    extra_args: list[str],
    timeout: int,
    log_path: Path,
) -> subprocess.Popen[Any]:
    ensure_parent(log_path)
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-model-len",
        str(cfg.max_model_len),
    ]
    if cfg.max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(cfg.max_num_seqs)]
    if cfg.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(cfg.max_num_batched_tokens)]
    cmd += extra_args

    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _wait_for_server(host, port, timeout, process, log_path)
    return process


def _enrich_result_json(
    result_path: Path,
    benchmark_name: str,
    model_id: str,
    run_number: int,
    run_date: str,
    workload: WorkloadSpec,
    cfg: ServeConfig,
    max_concurrency: int,
) -> None:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected benchmark JSON shape in {result_path}")

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            payload.setdefault(key, value)

    payload.setdefault("_benchmark_name", benchmark_name)
    payload.setdefault("model_id", model_id)
    payload.setdefault("tokenizer_id", model_id)
    payload.setdefault("backend", workload.backend)
    payload.setdefault("endpoint_type", workload.backend)
    payload.setdefault("run_number", run_number)
    payload.setdefault("date", run_date)
    payload.setdefault("num_prompts", workload.num_prompts)
    payload.setdefault("request_rate", workload.request_rate)
    payload.setdefault("burstiness", workload.burstiness)
    payload.setdefault("max_concurrency", max_concurrency)
    payload.setdefault("gpu_memory_utilization", cfg.gpu_memory_utilization)
    payload.setdefault("max_model_len", cfg.max_model_len)
    payload.setdefault("max_num_seqs", cfg.max_num_seqs)
    payload.setdefault("max_num_batched_tokens", cfg.max_num_batched_tokens)
    payload.setdefault("random_input_len", workload.expected_max_input_tokens)
    payload.setdefault("random_output_len", workload.expected_max_output_tokens)
    payload.setdefault("completed", payload.get("successful_requests", 0))
    payload.setdefault("failed", payload.get("failed_requests", 0))
    payload.setdefault("duration", payload.get("benchmark_duration", 0.0))
    payload.setdefault("output_throughput", payload.get("output_token_throughput", 0.0))
    payload.setdefault(
        "total_token_throughput",
        float(payload.get("input_token_throughput", 0.0) or 0.0) + float(payload.get("output_token_throughput", 0.0) or 0.0),
    )
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    model_id: str,
    workload_path: Path,
    grid_path: Path,
    output_root: Path,
    gpu_cost_per_hour: float,
    *,
    num_runs: int = 1,
    raw_artifact_url: str | None = None,
    server_host: str = "127.0.0.1",
    server_port: int = 8000,
    server_ready_timeout: int = 600,
    server_extra_args: str = "--no-enable-prefix-caching",
    bench_extra_args: str = "",
    run_stamp: str | None = None,
) -> dict[str, Any]:
    workload = load_workload_spec(workload_path)
    workload.model_id = model_id
    grid = load_serve_grid(grid_path, model_id=model_id)
    run_stamp = run_stamp or time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    raw_dir = output_root / "raw"
    logs_dir = output_root / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    copied_workload = output_root / "workload.json"
    copied_grid = output_root / "serve_grid.json"
    ensure_parent(copied_workload)
    copied_workload.write_text(json.dumps({**json.loads(workload_path.read_text(encoding="utf-8")), "model_id": model_id}, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(grid_path, copied_grid)

    successful_runs = 0
    server_process: subprocess.Popen[Any] | None = None
    server_args = shlex.split(server_extra_args) if server_extra_args else []
    bench_args = shlex.split(bench_extra_args) if bench_extra_args else []

    try:
        for config_index, cfg in enumerate(grid):
            combo_dir = raw_dir / run_stamp / (
                f"SERVE--gpu_memory_utilization={cfg.gpu_memory_utilization}-"
                f"max_model_len={cfg.max_model_len}-"
                f"max_num_seqs={cfg.max_num_seqs}-"
                f"max_num_batched_tokens={cfg.max_num_batched_tokens}-"
                f"BENCH--random_input_len={workload.expected_max_input_tokens}-"
                f"random_output_len={workload.expected_max_output_tokens}"
            )
            combo_dir.mkdir(parents=True, exist_ok=True)
            server_log = logs_dir / f"server_config_{config_index}.log"
            server_process = _start_server(
                model_id=model_id,
                cfg=cfg,
                host=server_host,
                port=server_port,
                extra_args=server_args,
                timeout=server_ready_timeout,
                log_path=server_log,
            )

            try:
                for concurrency in workload.concurrency_values:
                    run_dir = combo_dir / f"max_concurrency={concurrency}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    for run_index in range(num_runs):
                        run_path = run_dir / f"run={run_index}.json"
                        bench_log = logs_dir / f"bench_config_{config_index}_c{concurrency}_run{run_index}.log"
                        bench_cmd = [
                            "vllm",
                            "bench",
                            "serve",
                            "--backend",
                            workload.backend,
                            "--base-url",
                            f"http://{server_host}:{server_port}",
                            "--endpoint",
                            workload.endpoint,
                            "--model",
                            model_id,
                            "--dataset-name",
                            workload.dataset_name,
                            "--num-prompts",
                            str(workload.num_prompts),
                            "--random-input-len",
                            str(workload.expected_max_input_tokens),
                            "--random-output-len",
                            str(workload.expected_max_output_tokens),
                            "--request-rate",
                            workload.request_rate,
                            "--max-concurrency",
                            str(concurrency),
                            "--save-result",
                            "--result-dir",
                            str(run_dir),
                            "--result-filename",
                            f"run={run_index}.json",
                            "--metadata",
                            f"_benchmark_name={workload.benchmark_name}",
                            f"burstiness={workload.burstiness}",
                        ] + bench_args

                        ensure_parent(bench_log)
                        with bench_log.open("w", encoding="utf-8") as handle:
                            completed = subprocess.run(bench_cmd, check=False, stdout=handle, stderr=subprocess.STDOUT, text=True)

                        if run_path.exists():
                            _enrich_result_json(
                                run_path,
                                benchmark_name=workload.benchmark_name,
                                model_id=model_id,
                                run_number=run_index,
                                run_date=time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
                                workload=workload,
                                cfg=cfg,
                                max_concurrency=concurrency,
                            )
                            successful_runs += 1

                        if completed.returncode != 0:
                            if server_process.poll() is not None:
                                break
                            break
                    if server_process.poll() is not None:
                        break
            finally:
                _stop_server(server_process)
                server_process = None
                time.sleep(1)
    finally:
        _stop_server(server_process)

    if successful_runs == 0:
        raise RuntimeError("No successful benchmark files were produced")

    summary_path = output_root / "summary.csv"
    summarize_to_table(raw_dir, summary_path)

    strict_decision_path = output_root / "decision.strict.json"
    balanced_decision_path = output_root / "decision.balanced.json"
    strict_report_path = output_root / "slo_pack.strict.md"
    balanced_report_path = output_root / "slo_pack.balanced.md"

    decisions: dict[str, str | None] = {"strict": None, "balanced": None}
    reports: dict[str, str | None] = {"strict": None, "balanced": None}
    errors: dict[str, str] = {}

    for tier, decision_path, report_path in [
        ("strict", strict_decision_path, strict_report_path),
        ("balanced", balanced_decision_path, balanced_report_path),
    ]:
        try:
            optimize_atlas(
                atlas_path=summary_path,
                workload_spec=workload,
                sla_tier=tier,
                gpu_cost_per_hour=gpu_cost_per_hour,
                out_path=decision_path,
            )
            write_report(decision_path, report_path)
            decisions[tier] = str(decision_path)
            reports[tier] = str(report_path)
        except Exception as exc:
            errors[tier] = str(exc)

    manifest = {
        "generated_at_utc": now_utc_iso(),
        "model_id": model_id,
        "gpu_cost_per_hour": gpu_cost_per_hour,
        "raw_artifact_url": raw_artifact_url,
        "workload_path": str(copied_workload),
        "serve_grid_path": str(copied_grid),
        "summary_path": str(summary_path),
        "decisions": decisions,
        "reports": reports,
        "errors": errors,
        "system": collect_system_info(model_id, gpu_cost_per_hour, num_runs, workload),
        "run": {
            "num_runs": num_runs,
            "server_host": server_host,
            "server_port": server_port,
            "server_ready_timeout": server_ready_timeout,
            "server_extra_args": server_extra_args,
            "bench_extra_args": bench_extra_args,
            "run_stamp": run_stamp,
        },
    }
    manifest_path = output_root / "manifest.json"
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "summary": str(summary_path),
        "decisions": decisions,
        "reports": reports,
        "manifest": str(manifest_path),
        "errors": errors,
    }
