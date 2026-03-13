from __future__ import annotations

from pathlib import Path

from perfsmith.summarize import load_table
from perfsmith.surrogate import prune_with_surrogate


def test_surrogate_pruning_keeps_top_balanced_candidates() -> None:
    rows = load_table(Path("fixtures/expected_summary.csv"))
    pruned, metadata = prune_with_surrogate(rows, tier_field="meets_balanced", top_n=3)

    passing = [row for row in rows if row["meets_balanced"]]
    passing.sort(key=lambda row: row["total_token_throughput"], reverse=True)
    top_two_ids = {passing[0]["candidate_key"], passing[1]["candidate_key"]}
    pruned_ids = {row["candidate_key"] for row in pruned}

    assert top_two_ids.issubset(pruned_ids)
    assert metadata["pruned_from"] == len(rows)
    assert metadata["pruned_to"] == 3
    assert metadata["model_type"] in {"sklearn_gbdt", "empirical_knn"}
