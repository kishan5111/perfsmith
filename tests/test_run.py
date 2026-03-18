from __future__ import annotations

import json
import subprocess
from pathlib import Path

from perfsmith.run import run_experiment


class FakePopen:
    def __init__(self, cmd, stdout=None, stderr=None, text=None, **kwargs):
        self.cmd = cmd
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


def test_run_writes_manifest_and_reports(tmp_path: Path, monkeypatch) -> None:
    def fake_wait_for_server(host, port, timeout, process, log_path):
        return None

    def fake_collect_system_info(model_id, gpu_cost_per_hour, num_runs, workload):
        return {
            "timestamp_utc": "2026-03-14T00:00:00+00:00",
            "model_id": model_id,
            "gpu_cost_per_hour": gpu_cost_per_hour,
            "num_runs": num_runs,
            "gpu_info": "RTX 5090, 32768 MiB, 570.0",
        }

    def fake_run(cmd, check=False, stdout=None, stderr=None, text=None, capture_output=False):
        if cmd[:3] == ["vllm", "bench", "serve"]:
            result_dir = Path(cmd[cmd.index("--result-dir") + 1])
            result_name = cmd[cmd.index("--result-filename") + 1]
            concurrency = int(cmd[cmd.index("--max-concurrency") + 1])
            payload = {
                "successful_requests": 20,
                "failed_requests": 0,
                "benchmark_duration": 10.0,
                "input_token_throughput": 40.0,
                "output_token_throughput": 80.0 + concurrency,
                "p99_ttft_ms": 350.0 + concurrency,
                "p99_itl_ms": 35.0 + concurrency,
                "mean_ttft_ms": 300.0,
                "mean_itl_ms": 30.0,
                "p99_e2el_ms": 800.0,
                "mean_e2el_ms": 700.0,
            }
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / result_name).write_text(json.dumps(payload), encoding="utf-8")
            if stdout is not None and hasattr(stdout, "write"):
                stdout.write("ok\n")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("perfsmith.run.subprocess.Popen", FakePopen)
    monkeypatch.setattr("perfsmith.run.subprocess.run", fake_run)
    monkeypatch.setattr("perfsmith.run._wait_for_server", fake_wait_for_server)
    monkeypatch.setattr("perfsmith.run.collect_system_info", fake_collect_system_info)
    monkeypatch.setattr("perfsmith.run.time.sleep", lambda _: None)

    out_dir = tmp_path / "run"
    result = run_experiment(
        model_id="Qwen/Qwen3-4B",
        workload_path=Path("fixtures/workloads/short_balanced.json"),
        grid_path=Path("fixtures/grids/short_v0.json"),
        output_root=out_dir,
        gpu_cost_per_hour=0.322,
        num_runs=1,
        raw_artifact_url="https://example.com/raw.zip",
    )

    assert Path(result["summary"]).exists()
    assert Path(result["manifest"]).exists()
    assert Path(result["decisions"]["strict"]).exists()
    assert Path(result["decisions"]["balanced"]).exists()
    assert Path(result["reports"]["strict"]).exists()
    assert Path(result["reports"]["balanced"]).exists()

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["raw_artifact_url"] == "https://example.com/raw.zip"
    assert manifest["model_id"] == "Qwen/Qwen3-4B"
