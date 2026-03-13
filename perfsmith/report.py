"""SLA Pack markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import SLA_TIERS
from .utils import ensure_parent, format_float


def _load_artifact(run_id: str, output_root: Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(run_id)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as handle:
            return candidate, json.load(handle)

    artifact_path = output_root / "runs" / f"{run_id}.json"
    with artifact_path.open("r", encoding="utf-8") as handle:
        return artifact_path, json.load(handle)


def _as_vllm_flags(winner: dict[str, Any]) -> str:
    cfg = winner.get("serve_config", {})
    flags: list[str] = []
    if cfg.get("gpu_memory_utilization") is not None:
        flags.append(f"--gpu-memory-utilization {cfg['gpu_memory_utilization']}")
    if cfg.get("max_model_len") is not None:
        flags.append(f"--max-model-len {int(cfg['max_model_len'])}")
    if cfg.get("max_num_seqs") is not None:
        flags.append(f"--max-num-seqs {int(cfg['max_num_seqs'])}")
    if cfg.get("max_num_batched_tokens") is not None:
        flags.append(f"--max-num-batched-tokens {int(cfg['max_num_batched_tokens'])}")
    model = cfg.get("model_id") or "<model-id>"
    return f"vllm serve {model} {' '.join(flags)}".strip()


def _tier_pass(ttft: float, itl: float, tier_name: str) -> bool:
    limits = SLA_TIERS[tier_name]
    return ttft <= limits["p99_ttft_ms"] and itl <= limits["p99_itl_ms"]


def render_report(payload: dict[str, Any], artifact_path: Path) -> str:
    winner = payload["winner"]
    verification = payload.get("verification", {})
    variance = verification.get("winner_variance", {})

    ttft = float(winner.get("p99_ttft_ms", 0.0))
    itl = float(winner.get("p99_itl_ms", 0.0))

    strict_pass = _tier_pass(ttft, itl, "strict")
    balanced_pass = _tier_pass(ttft, itl, "balanced")

    lines: list[str] = []
    lines.append("# Perfsmith SLA Pack")
    lines.append("")
    lines.append(f"- Run ID: `{payload['run_id']}`")
    lines.append(f"- Created At (UTC): `{payload.get('created_at_utc', 'n/a')}`")
    lines.append(f"- Workload: `{payload.get('workload_name', 'n/a')}`")
    lines.append(f"- Benchmark: `{payload.get('benchmark_name', 'n/a')}`")
    lines.append(f"- SLA Tier: `{payload.get('sla_tier', 'n/a')}`")
    lines.append("")

    lines.append("## Winner")
    lines.append("")
    lines.append(f"- Candidate ID: `{winner['candidate_id']}`")
    lines.append(f"- Status: `{winner['status']}`")
    lines.append(f"- Max Concurrency: `{winner['max_concurrency']}`")
    lines.append(f"- Total Token Throughput: `{format_float(winner.get('total_token_throughput'), 3)}` tok/s")
    lines.append(f"- p99 TTFT: `{format_float(winner.get('p99_ttft_ms'), 3)}` ms")
    lines.append(f"- p99 ITL: `{format_float(winner.get('p99_itl_ms'), 3)}` ms")
    lines.append("")

    lines.append("## Serve Flags")
    lines.append("")
    lines.append("```bash")
    lines.append(_as_vllm_flags(winner))
    lines.append("```")
    lines.append("")

    lines.append("## SLA Compliance")
    lines.append("")
    lines.append("| Tier | Threshold | Pass |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| strict | p99_ttft<=500, p99_itl<=60 | {'yes' if strict_pass else 'no'} |")
    lines.append(f"| balanced | p99_ttft<=650, p99_itl<=80 | {'yes' if balanced_pass else 'no'} |")
    lines.append("")

    lines.append("## Verification Variance")
    lines.append("")
    lines.append(f"- Verification runs: `{winner.get('verification_runs', 0)}`")
    lines.append(
        f"- p99 TTFT stddev: `{format_float(variance.get('p99_ttft_stddev_ms'), 3)}` ms"
    )
    lines.append(f"- p99 ITL stddev: `{format_float(variance.get('p99_itl_stddev_ms'), 3)}` ms")
    lines.append(
        f"- Throughput stddev: `{format_float(variance.get('throughput_stddev'), 3)}` tok/s"
    )
    lines.append("")

    lines.append("## Cost Estimate")
    lines.append("")
    lines.append(
        f"- Cost per 1M tokens: `{format_float(winner.get('cost_per_1m_tokens_usd'), 3)}` USD"
    )
    lines.append("")

    lines.append("## Repro Commands")
    lines.append("")
    lines.append("```bash")
    lines.append(f"perfsmith report --run-id {payload['run_id']}")
    lines.append(payload.get("commands", {}).get("report", ""))
    lines.append(f"# Artifact: {artifact_path}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def write_report(run_id: str, output_root: Path | None = None, output_path: Path | None = None) -> Path:
    output_root = output_root or Path("artifacts")
    artifact_path, payload = _load_artifact(run_id, output_root)
    report_text = render_report(payload, artifact_path)

    if output_path is None:
        output_path = output_root / "reports" / f"{payload['run_id']}.md"

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(report_text)
    return output_path
