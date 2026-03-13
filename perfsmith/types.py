"""Core schema types for Perfsmith."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkloadSpecV0:
    """Workload specification for optimization."""

    name: str
    benchmark_name: str
    expected_max_input_tokens: int
    expected_max_output_tokens: int
    summary_table: str
    gpu_cost_per_hour: float = 0.0
    top_k_verify: int = 3
    prune_top_n: int = 12
    verification_min_runs: int = 1
    model_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkloadSpecV0":
        return cls(
            name=str(payload.get("name", payload.get("benchmark_name", "workload"))),
            benchmark_name=str(payload.get("benchmark_name", payload.get("name", ""))),
            expected_max_input_tokens=int(payload["expected_max_input_tokens"]),
            expected_max_output_tokens=int(payload["expected_max_output_tokens"]),
            summary_table=str(payload["summary_table"]),
            gpu_cost_per_hour=float(payload.get("gpu_cost_per_hour", 0.0)),
            top_k_verify=int(payload.get("top_k_verify", 3)),
            prune_top_n=int(payload.get("prune_top_n", 12)),
            verification_min_runs=int(payload.get("verification_min_runs", 1)),
            model_id=payload.get("model_id"),
        )


@dataclass
class CandidateEvaluation:
    """Scored candidate entry."""

    candidate_id: str
    status: str
    max_concurrency: int
    total_token_throughput: float
    p99_ttft_ms: float
    p99_itl_ms: float
    meets_tier: bool
    verification_runs: int
    cost_per_1m_tokens_usd: float | None
    serve_config: dict[str, Any] = field(default_factory=dict)
    run_paths: list[str] = field(default_factory=list)
    metrics_samples: list[dict[str, float]] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Optimization output payload."""

    run_id: str
    workload_name: str
    sla_tier: str
    winner: CandidateEvaluation
    alternatives: list[CandidateEvaluation]
    screened_count: int
    pruned_count: int
    discarded_guardrail_count: int
    surrogate: dict[str, Any]
    workload_spec: dict[str, Any]


@dataclass
class ToolStubProfile:
    """Latency and error profile for a tool endpoint."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    failure_rate: float = 0.0


@dataclass
class AgentTraceSpec:
    """Agent trace replay specification."""

    requests: list[dict[str, Any]]
    iterations: int = 1
