from __future__ import annotations

from pathlib import Path

import pytest

from perfsmith.optimize import load_workload_spec, optimize_atlas


ATLAS = Path("fixtures/expected_summary.csv")


def test_optimize_balanced_and_strict_winners_differ(tmp_path: Path) -> None:
    spec = load_workload_spec(Path("fixtures/workloads/short_balanced.json"))

    balanced = optimize_atlas(
        atlas_path=ATLAS,
        workload_spec=spec,
        sla_tier="balanced",
        gpu_cost_per_hour=1.75,
        out_path=tmp_path / "balanced.json",
        created_at="2026-03-06T00:00:00+00:00",
    )["result"]

    strict = optimize_atlas(
        atlas_path=ATLAS,
        workload_spec=spec,
        sla_tier="strict",
        gpu_cost_per_hour=1.75,
        out_path=tmp_path / "strict.json",
        created_at="2026-03-06T00:00:00+00:00",
    )["result"]

    assert balanced["winner"]["status"] == "verified"
    assert strict["winner"]["status"] == "verified"
    assert balanced["winner"]["max_concurrency"] == 23
    assert strict["winner"]["max_concurrency"] == 17
    assert len(balanced["alternatives"]) >= 1
    assert len(balanced["screened_candidates"]) >= 1


def test_guardrail_rejects_incompatible_model_len(tmp_path: Path) -> None:
    spec = load_workload_spec(Path("fixtures/workloads/short_guardrail_fail.json"))

    with pytest.raises(ValueError, match="guardrail"):
        optimize_atlas(
            atlas_path=ATLAS,
            workload_spec=spec,
            sla_tier="balanced",
            gpu_cost_per_hour=1.75,
            out_path=tmp_path / "guardrail.json",
            created_at="2026-03-06T00:00:00+00:00",
        )


def test_mandatory_verify_run_enforced(tmp_path: Path) -> None:
    spec = load_workload_spec(Path("fixtures/workloads/short_verify_fail.json"))

    with pytest.raises(ValueError, match="Mandatory verify-run enforcement"):
        optimize_atlas(
            atlas_path=ATLAS,
            workload_spec=spec,
            sla_tier="balanced",
            gpu_cost_per_hour=1.75,
            out_path=tmp_path / "verify_fail.json",
            created_at="2026-03-06T00:00:00+00:00",
        )
