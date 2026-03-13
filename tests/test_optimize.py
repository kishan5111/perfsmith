from __future__ import annotations

from pathlib import Path

import pytest

from perfsmith.optimize import load_workload_spec, optimize_workload


def test_optimize_balanced_and_strict_winners_differ(tmp_path: Path) -> None:
    spec = load_workload_spec(Path("fixtures/workloads/short_balanced.json"))

    balanced = optimize_workload(
        workload_spec=spec,
        sla_tier="balanced",
        output_root=tmp_path / "artifacts",
        run_id="balanced-test",
        created_at="2026-03-06T00:00:00+00:00",
    )["result"]

    strict = optimize_workload(
        workload_spec=spec,
        sla_tier="strict",
        output_root=tmp_path / "artifacts",
        run_id="strict-test",
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
        optimize_workload(
            workload_spec=spec,
            sla_tier="balanced",
            output_root=tmp_path / "artifacts",
            run_id="guardrail-fail",
            created_at="2026-03-06T00:00:00+00:00",
        )


def test_mandatory_verify_run_enforced(tmp_path: Path) -> None:
    spec = load_workload_spec(Path("fixtures/workloads/short_verify_fail.json"))

    with pytest.raises(ValueError, match="Mandatory verify-run enforcement"):
        optimize_workload(
            workload_spec=spec,
            sla_tier="balanced",
            output_root=tmp_path / "artifacts",
            run_id="verify-fail",
            created_at="2026-03-06T00:00:00+00:00",
        )
