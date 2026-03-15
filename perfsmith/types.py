"""Core schema types for Perfsmith."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkloadSpec:
    """Workload specification shared by run/optimize flows."""

    name: str
    benchmark_name: str
    expected_max_input_tokens: int
    expected_max_output_tokens: int
    num_prompts: int = 400
    request_rate: str = "inf"
    burstiness: float = 1.0
    concurrency_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dataset_name: str = "random"
    backend: str = "vllm"
    endpoint: str = "/v1/completions"
    top_k_verify: int = 3
    prune_top_n: int = 12
    verification_min_runs: int = 1
    model_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkloadSpec":
        concurrency_values = payload.get("concurrency_values", [1, 2, 4, 8])
        return cls(
            name=str(payload.get("name", payload.get("benchmark_name", "workload"))),
            benchmark_name=str(payload.get("benchmark_name", payload.get("name", "workload"))),
            expected_max_input_tokens=int(payload["expected_max_input_tokens"]),
            expected_max_output_tokens=int(payload["expected_max_output_tokens"]),
            num_prompts=int(payload.get("num_prompts", 400)),
            request_rate=str(payload.get("request_rate", "inf")),
            burstiness=float(payload.get("burstiness", 1.0)),
            concurrency_values=[int(value) for value in concurrency_values],
            dataset_name=str(payload.get("dataset_name", "random")),
            backend=str(payload.get("backend", "vllm")),
            endpoint=str(payload.get("endpoint", "/v1/completions")),
            top_k_verify=int(payload.get("top_k_verify", 3)),
            prune_top_n=int(payload.get("prune_top_n", 12)),
            verification_min_runs=int(payload.get("verification_min_runs", 1)),
            model_id=payload.get("model_id"),
        )


@dataclass
class ServeConfig:
    """One serving configuration candidate."""

    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    model_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], model_id: str | None = None) -> "ServeConfig":
        return cls(
            gpu_memory_utilization=float(payload["gpu_memory_utilization"]),
            max_model_len=int(payload["max_model_len"]),
            max_num_seqs=(None if payload.get("max_num_seqs") is None else int(payload["max_num_seqs"])),
            max_num_batched_tokens=(
                None
                if payload.get("max_num_batched_tokens") is None
                else int(payload["max_num_batched_tokens"])
            ),
            model_id=str(payload.get("model") or payload.get("model_id") or model_id) if (payload.get("model") or payload.get("model_id") or model_id) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }


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
