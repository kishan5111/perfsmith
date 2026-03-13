from __future__ import annotations

from pathlib import Path

from perfsmith.optimize import load_workload_spec, optimize_workload
from perfsmith.report import write_report


EXPECTED = Path("fixtures/expected_report_fixed.md")


def test_report_matches_snapshot() -> None:
    spec = load_workload_spec(Path("fixtures/workloads/short_balanced.json"))

    optimize_workload(
        workload_spec=spec,
        sla_tier="balanced",
        output_root=Path("artifacts"),
        run_id="fixture-short-balanced-fixed",
        created_at="2026-03-06T00:00:00+00:00",
    )

    report_path = write_report(
        run_id="fixture-short-balanced-fixed",
        output_root=Path("artifacts"),
        output_path=Path("artifacts/reports/fixture-short-balanced-fixed.md"),
    )

    assert report_path.read_text(encoding="utf-8") == EXPECTED.read_text(encoding="utf-8")
