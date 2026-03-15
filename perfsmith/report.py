"""SLO Pack markdown report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import SLA_TIERS
from .utils import ensure_parent, format_float


def load_decision(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def render_report(payload: dict[str, Any], decision_path: Path) -> str:
    winner = payload["winner"]
    verification = payload.get("verification", {})
    variance = verification.get("winner_variance", {})
    workload = payload.get("workload", {})

    ttft = float(winner.get("p99_ttft_ms", 0.0))
    itl = float(winner.get("p99_itl_ms", 0.0))

    strict_pass = _tier_pass(ttft, itl, "strict")
    balanced_pass = _tier_pass(ttft, itl, "balanced")

    lines: list[str] = []
    lines.append("# Perfsmith SLO Pack")
    lines.append("")
    lines.append(f"- Decision ID: `{payload.get('decision_id', decision_path.stem)}`")
    lines.append(f"- Generated At (UTC): `{payload.get('generated_at_utc', 'n/a')}`")
    lines.append(f"- Workload: `{workload.get('name', 'n/a')}`")
    lines.append(f"- Benchmark: `{workload.get('benchmark_name', 'n/a')}`")
    lines.append(f"- SLA Tier: `{payload.get('sla_tier', 'n/a')}`")
    lines.append(f"- Atlas: `{payload.get('atlas_path', 'n/a')}`")
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
    lines.append(f"- p99 TTFT stddev: `{format_float(variance.get('p99_ttft_stddev_ms'), 3)}` ms")
    lines.append(f"- p99 ITL stddev: `{format_float(variance.get('p99_itl_stddev_ms'), 3)}` ms")
    lines.append(f"- Throughput stddev: `{format_float(variance.get('throughput_stddev'), 3)}` tok/s")
    lines.append("")

    lines.append("## Cost Estimate")
    lines.append("")
    lines.append(f"- Cost per 1M tokens: `{format_float(winner.get('cost_per_1m_tokens_usd'), 3)}` USD")
    lines.append("")

    if payload.get("alternatives"):
        lines.append("## Alternatives")
        lines.append("")
        for alt in payload["alternatives"]:
            lines.append(
                f"- `{alt['candidate_id']}`: concurrency `{alt['max_concurrency']}`, throughput `{format_float(alt.get('total_token_throughput'), 3)}` tok/s, p99 TTFT `{format_float(alt.get('p99_ttft_ms'), 3)}` ms, p99 ITL `{format_float(alt.get('p99_itl_ms'), 3)}` ms"
            )
        lines.append("")

    lines.append("## Repro Commands")
    lines.append("")
    lines.append("```bash")
    decision_label = decision_path.name
    lines.append(f"perfsmith report --decision {decision_label} --out <report.md>")
    lines.append(f"# Decision artifact: {decision_label}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def write_report(decision_path: Path, output_path: Path) -> Path:
    payload = load_decision(decision_path)
    report_text = render_report(payload, decision_path)
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(report_text)
    return output_path
