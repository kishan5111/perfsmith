"""Decision engine with screening + verification passes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .constants import DEFAULT_PRUNE_TOP_N, SLA_TIERS
from .summarize import load_table
from .surrogate import prune_with_surrogate
from .types import CandidateEvaluation, WorkloadSpecV0
from .utils import build_run_id, coerce_float, coerce_int, ensure_parent, now_utc_iso, stddev


def _tier_field(sla_tier: str) -> str:
    if sla_tier not in SLA_TIERS:
        raise ValueError(f"Unsupported SLA tier: {sla_tier}")
    return f"meets_{sla_tier}"


def _guardrail_filter(rows: list[dict[str, Any]], envelope_tokens: int) -> tuple[list[dict[str, Any]], int]:
    filtered: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        model_len = coerce_int(row.get("max_model_len"), None)
        if model_len is not None and model_len < envelope_tokens:
            removed += 1
            continue
        filtered.append(row)
    return filtered, removed


def _cost_per_1m_tokens(gpu_cost_per_hour: float, throughput_tok_s: float) -> float | None:
    if gpu_cost_per_hour <= 0 or throughput_tok_s <= 0:
        return None
    return (gpu_cost_per_hour / 3600.0) * 1_000_000.0 / throughput_tok_s


def _candidate_config(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": row.get("model_id"),
        "gpu_memory_utilization": row.get("gpu_memory_utilization"),
        "max_model_len": row.get("max_model_len"),
        "max_num_seqs": row.get("max_num_seqs"),
        "max_num_batched_tokens": row.get("max_num_batched_tokens"),
        "max_concurrency": row.get("max_concurrency"),
        "benchmark_name": row.get("benchmark_name"),
    }


def _aggregate_candidate(
    rows: list[dict[str, Any]],
    candidate_id: str,
    tier_name: str,
    status: str,
    gpu_cost_per_hour: float,
) -> CandidateEvaluation:
    ttfts = [coerce_float(row.get("p99_ttft_ms"), 0.0) or 0.0 for row in rows]
    itls = [coerce_float(row.get("p99_itl_ms"), 0.0) or 0.0 for row in rows]
    tps = [coerce_float(row.get("total_token_throughput"), 0.0) or 0.0 for row in rows]

    mean_ttft = sum(ttfts) / len(ttfts)
    mean_itl = sum(itls) / len(itls)
    mean_tp = sum(tps) / len(tps)

    limits = SLA_TIERS[tier_name]
    meets_tier = all(
        (coerce_int(row.get("failed"), 0) or 0) == 0
        and (coerce_int(row.get("completed"), 0) or 0) > 0
        and (coerce_float(row.get("p99_ttft_ms"), 0.0) or 0.0) <= limits["p99_ttft_ms"]
        and (coerce_float(row.get("p99_itl_ms"), 0.0) or 0.0) <= limits["p99_itl_ms"]
        for row in rows
    )

    sample_metrics = [
        {
            "p99_ttft_ms": coerce_float(row.get("p99_ttft_ms"), 0.0) or 0.0,
            "p99_itl_ms": coerce_float(row.get("p99_itl_ms"), 0.0) or 0.0,
            "total_token_throughput": coerce_float(row.get("total_token_throughput"), 0.0) or 0.0,
        }
        for row in rows
    ]

    return CandidateEvaluation(
        candidate_id=candidate_id,
        status=status,
        max_concurrency=coerce_int(rows[0].get("max_concurrency"), 0) or 0,
        total_token_throughput=mean_tp,
        p99_ttft_ms=mean_ttft,
        p99_itl_ms=mean_itl,
        meets_tier=meets_tier,
        verification_runs=len(rows),
        cost_per_1m_tokens_usd=_cost_per_1m_tokens(gpu_cost_per_hour, mean_tp),
        serve_config=_candidate_config(rows[0]),
        run_paths=[str(row.get("source_path", "")) for row in rows],
        metrics_samples=sample_metrics,
    )


def _pareto_front(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    result: list[CandidateEvaluation] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other.candidate_id == cand.candidate_id:
                continue
            other_cost = other.cost_per_1m_tokens_usd if other.cost_per_1m_tokens_usd is not None else float("inf")
            cand_cost = cand.cost_per_1m_tokens_usd if cand.cost_per_1m_tokens_usd is not None else float("inf")
            no_worse = (
                other.total_token_throughput >= cand.total_token_throughput
                and other.p99_ttft_ms <= cand.p99_ttft_ms
                and other.p99_itl_ms <= cand.p99_itl_ms
                and other_cost <= cand_cost
            )
            strictly_better = (
                other.total_token_throughput > cand.total_token_throughput
                or other.p99_ttft_ms < cand.p99_ttft_ms
                or other.p99_itl_ms < cand.p99_itl_ms
                or other_cost < cand_cost
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(cand)

    result.sort(key=lambda c: (c.total_token_throughput, c.meets_tier, -c.p99_ttft_ms, -c.p99_itl_ms), reverse=True)
    return result


def optimize_workload(
    workload_spec: WorkloadSpecV0,
    sla_tier: str,
    output_root: Path | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    tier_field = _tier_field(sla_tier)
    table_path = Path(workload_spec.summary_table)
    rows = load_table(table_path)

    rows = [row for row in rows if str(row.get("benchmark_name", "")) == workload_spec.benchmark_name]
    if workload_spec.model_id:
        rows = [row for row in rows if str(row.get("model_id", "")) == workload_spec.model_id]
    if not rows:
        raise ValueError("No rows matched workload filters")

    envelope = workload_spec.expected_max_input_tokens + workload_spec.expected_max_output_tokens
    rows, removed = _guardrail_filter(rows, envelope)
    if not rows:
        raise ValueError("No rows remain after max_model_len guardrail filtering")

    pruned_rows, surrogate_meta = prune_with_surrogate(
        rows,
        tier_field=tier_field,
        top_n=max(workload_spec.prune_top_n, DEFAULT_PRUNE_TOP_N),
    )

    screened_pass = [
        row for row in pruned_rows if bool(row.get(tier_field, False)) and (coerce_int(row.get("failed"), 0) or 0) == 0
    ]
    screened_pass.sort(
        key=lambda row: (
            coerce_float(row.get("total_token_throughput"), 0.0) or 0.0,
            coerce_int(row.get("max_concurrency"), 0) or 0,
        ),
        reverse=True,
    )

    screened_unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in screened_pass:
        key = str(row.get("candidate_key", ""))
        if key in seen:
            continue
        seen.add(key)
        screened_unique.append(row)

    if not screened_unique:
        raise ValueError("No screened candidates passed SLA tier")

    verify_count = min(workload_spec.top_k_verify, len(screened_unique))

    verified: list[CandidateEvaluation] = []
    screened_only: list[CandidateEvaluation] = []

    for idx, screened_row in enumerate(screened_unique):
        candidate_id = str(screened_row.get("candidate_key", ""))
        candidate_rows = [row for row in rows if str(row.get("candidate_key", "")) == candidate_id]
        if idx < verify_count and len(candidate_rows) >= workload_spec.verification_min_runs:
            verified.append(
                _aggregate_candidate(
                    candidate_rows,
                    candidate_id,
                    tier_name=sla_tier,
                    status="verified",
                    gpu_cost_per_hour=workload_spec.gpu_cost_per_hour,
                )
            )
        else:
            screened_only.append(
                _aggregate_candidate(
                    [screened_row],
                    candidate_id,
                    tier_name=sla_tier,
                    status="screened",
                    gpu_cost_per_hour=workload_spec.gpu_cost_per_hour,
                )
            )

    verified_pass = [cand for cand in verified if cand.meets_tier]
    if not verified_pass:
        raise ValueError(
            "Mandatory verify-run enforcement: no verified candidate meets SLA. "
            "Lower verification_min_runs or provide more verification repeats."
        )

    winner = max(
        verified_pass,
        key=lambda cand: (cand.total_token_throughput, cand.max_concurrency, -cand.p99_ttft_ms, -cand.p99_itl_ms),
    )

    pareto = _pareto_front([cand for cand in (verified + screened_only) if cand.meets_tier])
    alternatives = [cand for cand in pareto if cand.candidate_id != winner.candidate_id][:3]

    output_root = output_root or Path("artifacts")
    runs_dir = output_root / "runs"
    run_id = run_id or build_run_id(f"{workload_spec.name}-{sla_tier}")
    created_at = created_at or now_utc_iso()
    artifact_path = runs_dir / f"{run_id}.json"
    ensure_parent(artifact_path)

    result = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "workload_name": workload_spec.name,
        "benchmark_name": workload_spec.benchmark_name,
        "sla_tier": sla_tier,
        "sla_thresholds": SLA_TIERS[sla_tier],
        "screened_count": len(screened_unique),
        "pruned_count": len(pruned_rows),
        "discarded_guardrail_count": removed,
        "surrogate": surrogate_meta,
        "workload_spec": asdict(workload_spec),
        "winner": asdict(winner),
        "alternatives": [asdict(cand) for cand in alternatives],
        "screened_candidates": [asdict(cand) for cand in screened_only],
        "verified_candidates": [asdict(cand) for cand in verified],
        "verification": {
            "verification_min_runs": workload_spec.verification_min_runs,
            "winner_variance": {
                "p99_ttft_stddev_ms": stddev(sample["p99_ttft_ms"] for sample in winner.metrics_samples),
                "p99_itl_stddev_ms": stddev(sample["p99_itl_ms"] for sample in winner.metrics_samples),
                "throughput_stddev": stddev(sample["total_token_throughput"] for sample in winner.metrics_samples),
            },
        },
        "commands": {
            "report": f"perfsmith report --run-id {run_id}",
        },
    }

    with artifact_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    return {"artifact_path": str(artifact_path), "result": result}


def load_workload_spec(path: Path) -> WorkloadSpecV0:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return WorkloadSpecV0.from_dict(payload)
