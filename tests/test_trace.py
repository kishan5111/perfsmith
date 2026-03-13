from __future__ import annotations

from pathlib import Path

from perfsmith.replay_trace import replay_trace


def test_trace_replay_decomposition_consistent(tmp_path: Path) -> None:
    result = replay_trace(
        trace_path=Path("fixtures/trace/agent_trace.json"),
        stub_path=Path("fixtures/trace/tool_stub.json"),
        output_path=tmp_path / "trace_result.json",
    )

    assert result["requests_simulated"] == 200
    assert result["latency_ms"]["p99"] >= result["latency_ms"]["p95"] >= result["latency_ms"]["p50"]

    decomp = result["decomposition_ms"]
    summed = decomp["model_mean"] + decomp["tool_mean"] + decomp["orchestration_mean"]
    assert abs(summed - decomp["e2e_mean"]) < 1e-6
