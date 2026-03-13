"""Tool-aware trace replay with latency decomposition."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from .utils import ensure_parent, quantile


@dataclass
class ToolStubServer:
    """Simple local HTTP stub for tool endpoints."""

    host: str
    port: int
    profile: dict[str, Any]
    seed: int = 42

    def __post_init__(self) -> None:
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        rng = random.Random(self.seed)
        profile = self.profile

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                tool_name = self.path.strip("/") or "default"
                tool_cfg = profile.get("tools", {}).get(tool_name, profile.get("default", {}))
                latency = _sample_latency_ms(tool_cfg, rng)
                fail = rng.random() < float(tool_cfg.get("failure_rate", 0.0) or 0.0)
                time.sleep(max(0.0, latency) / 1000.0)
                self.send_response(500 if fail else 200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = {
                    "tool": tool_name,
                    "latency_ms": latency,
                    "failed": fail,
                }
                self.wfile.write(json.dumps(payload).encode("utf-8"))

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        self._server = HTTPServer((self.host, self.port), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _sample_latency_ms(profile: dict[str, Any], rng: random.Random) -> float:
    p50 = float(profile.get("p50_ms", profile.get("latency_ms", 50.0)))
    p95 = float(profile.get("p95_ms", p50 * 2.0))
    p99 = float(profile.get("p99_ms", max(p95, p50 * 3.0)))

    u = rng.random()
    if u <= 0.50:
        lo, hi = p50 * 0.5, p50
    elif u <= 0.95:
        lo, hi = p50, p95
    elif u <= 0.99:
        lo, hi = p95, p99
    else:
        lo, hi = p99, p99 * 1.5
    return rng.uniform(lo, hi)


def _step_latency_ms(step: dict[str, Any], stub_profile: dict[str, Any], rng: random.Random) -> tuple[float, bool, str]:
    step_type = str(step.get("type", "model"))

    if step_type == "tool":
        tool_name = str(step.get("tool", "default"))
        tool_cfg = stub_profile.get("tools", {}).get(tool_name, stub_profile.get("default", {}))
        latency = float(step.get("latency_ms") or _sample_latency_ms(tool_cfg, rng))
        failed = rng.random() < float(tool_cfg.get("failure_rate", 0.0) or 0.0)
        return latency, failed, "tool"

    if step_type == "orchestration":
        latency = float(step.get("latency_ms", 5.0))
        return latency, False, "orchestration"

    latency = float(step.get("latency_ms", 30.0))
    return latency, False, "model"


def replay_trace(trace_path: Path, stub_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    with trace_path.open("r", encoding="utf-8") as handle:
        trace = json.load(handle)
    with stub_path.open("r", encoding="utf-8") as handle:
        stub_profile = json.load(handle)

    seed = int(stub_profile.get("seed", 42))
    rng = random.Random(seed)

    requests = trace.get("requests", [])
    iterations = int(trace.get("iterations", 1))

    e2e_values: list[float] = []
    model_values: list[float] = []
    tool_values: list[float] = []
    orchestration_values: list[float] = []

    failed_requests = 0

    for _ in range(iterations):
        for req in requests:
            model_ms = 0.0
            tool_ms = 0.0
            orchestration_ms = 0.0
            failed = False

            for step in req.get("steps", []):
                latency, did_fail, category = _step_latency_ms(step, stub_profile, rng)
                if category == "tool":
                    tool_ms += latency
                elif category == "orchestration":
                    orchestration_ms += latency
                else:
                    model_ms += latency
                if did_fail:
                    failed = True

            total = model_ms + tool_ms + orchestration_ms
            e2e_values.append(total)
            model_values.append(model_ms)
            tool_values.append(tool_ms)
            orchestration_values.append(orchestration_ms)
            if failed:
                failed_requests += 1

    total_requests = len(e2e_values)
    result = {
        "trace_name": trace.get("name", "trace"),
        "requests_simulated": total_requests,
        "failed_requests": failed_requests,
        "error_rate": (failed_requests / total_requests) if total_requests else 0.0,
        "latency_ms": {
            "p50": quantile(e2e_values, 0.50),
            "p95": quantile(e2e_values, 0.95),
            "p99": quantile(e2e_values, 0.99),
        },
        "decomposition_ms": {
            "model_mean": (sum(model_values) / len(model_values)) if model_values else 0.0,
            "tool_mean": (sum(tool_values) / len(tool_values)) if tool_values else 0.0,
            "orchestration_mean": (sum(orchestration_values) / len(orchestration_values))
            if orchestration_values
            else 0.0,
            "e2e_mean": (sum(e2e_values) / len(e2e_values)) if e2e_values else 0.0,
            "model_p99": quantile(model_values, 0.99),
            "tool_p99": quantile(tool_values, 0.99),
            "orchestration_p99": quantile(orchestration_values, 0.99),
        },
    }

    if output_path is not None:
        ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    return result
