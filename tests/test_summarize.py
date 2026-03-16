from __future__ import annotations

import csv
from pathlib import Path

from perfsmith.summarize import summarize_to_table


FIXTURE_ZIP = Path("fixtures/runs/qwen3_4b_5090_short_smoke.raw.zip")
EXPECTED_SUMMARY = Path("fixtures/atlas/qwen3_4b_5090_short_smoke.summary.csv")


def test_summarize_matches_golden(tmp_path: Path) -> None:
    out = tmp_path / "summary.csv"
    rows = summarize_to_table(FIXTURE_ZIP, out)

    assert len(rows) == 6
    assert out.read_text(encoding="utf-8") == EXPECTED_SUMMARY.read_text(encoding="utf-8")


def test_derived_sla_fields_are_deterministic(tmp_path: Path) -> None:
    out = tmp_path / "summary.csv"
    summarize_to_table(FIXTURE_ZIP, out)

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        short_rows = [row for row in reader if row["benchmark_name"] == "short"]

    strict_max = {int(row["max_concurrency_under_strict"]) for row in short_rows}
    balanced_max = {int(row["max_concurrency_under_balanced"]) for row in short_rows}

    assert strict_max == {2}
    assert balanced_max == {2}
