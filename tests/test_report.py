from __future__ import annotations

from pathlib import Path

from perfsmith.optimize import load_workload_spec, optimize_atlas
from perfsmith.report import write_report


EXPECTED = Path("fixtures/reports/optimizer_tiered.balanced.md")
ATLAS = Path("fixtures/atlas/optimizer_tiered.csv")


def test_report_matches_snapshot(tmp_path: Path) -> None:
    spec = load_workload_spec(Path("fixtures/workloads/short_balanced.json"))
    decision_path = tmp_path / "decision.json"

    optimize_atlas(
        atlas_path=ATLAS,
        workload_spec=spec,
        sla_tier="balanced",
        gpu_cost_per_hour=0.322,
        out_path=decision_path,
        created_at="2026-03-15T00:00:00+00:00",
    )

    report_path = write_report(decision_path, tmp_path / "slo_pack.md")
    assert report_path.read_text(encoding="utf-8") == EXPECTED.read_text(encoding="utf-8")
