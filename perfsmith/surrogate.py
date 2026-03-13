"""Surrogate model for search pruning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .utils import coerce_float, coerce_int


@dataclass
class SurrogateScore:
    candidate_key: str
    pass_probability: float
    predicted_throughput: float
    uncertainty: float
    combined_score: float


class SurrogateModel:
    """Predictive model used to prune candidate evaluations."""

    def __init__(self, tier_field: str) -> None:
        self.tier_field = tier_field
        self.model_type = "empirical_knn"
        self._sklearn_cls = None
        self._sklearn_reg = None
        self._x_train: list[list[float]] = []
        self._y_pass: list[float] = []
        self._y_tp: list[float] = []
        self._feature_scale: list[float] = []

    @staticmethod
    def _features(row: dict[str, Any]) -> list[float]:
        return [
            float(coerce_int(row.get("max_concurrency"), 0) or 0),
            float(coerce_int(row.get("max_model_len"), 0) or 0),
            float(coerce_float(row.get("gpu_memory_utilization"), 0.0) or 0.0),
            float(coerce_int(row.get("max_num_seqs"), 0) or 0),
            float(coerce_int(row.get("max_num_batched_tokens"), 0) or 0),
            float(coerce_int(row.get("random_input_len"), 0) or 0),
            float(coerce_int(row.get("random_output_len"), 0) or 0),
            float(coerce_int(row.get("num_prompts"), 0) or 0),
        ]

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._x_train = [self._features(row) for row in rows]
        self._y_pass = [1.0 if bool(row.get(self.tier_field, False)) else 0.0 for row in rows]
        self._y_tp = [float(coerce_float(row.get("total_token_throughput"), 0.0) or 0.0) for row in rows]
        if self._x_train:
            self._feature_scale = []
            dims = len(self._x_train[0])
            for i in range(dims):
                col = [x[i] for x in self._x_train]
                scale = max(col) - min(col)
                self._feature_scale.append(scale if scale > 0 else 1.0)
        else:
            self._feature_scale = [1.0] * 8

        # Prefer a true tabular GBDT when sklearn exists.
        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # type: ignore

            cls = GradientBoostingClassifier(random_state=42)
            reg = GradientBoostingRegressor(random_state=42)
            cls.fit(self._x_train, self._y_pass)
            reg.fit(self._x_train, self._y_tp)
            self._sklearn_cls = cls
            self._sklearn_reg = reg
            self.model_type = "sklearn_gbdt"
        except Exception:
            self._sklearn_cls = None
            self._sklearn_reg = None
            self.model_type = "empirical_knn"

    def _distance(self, a: list[float], b: list[float]) -> float:
        total = 0.0
        for i, (ai, bi) in enumerate(zip(a, b)):
            denom = self._feature_scale[i] if i < len(self._feature_scale) else 1.0
            total += ((ai - bi) / denom) ** 2
        return math.sqrt(total)

    def _predict_empirical(self, x: list[float]) -> tuple[float, float, float]:
        if not self._x_train:
            return 0.0, 0.0, 1.0

        distances = [(self._distance(x, train_x), idx) for idx, train_x in enumerate(self._x_train)]
        distances.sort(key=lambda item: item[0])
        k = min(5, len(distances))
        nearest = distances[:k]

        weighted_pass = 0.0
        weighted_tp = 0.0
        weight_total = 0.0
        avg_dist = 0.0

        for dist, idx in nearest:
            weight = 1.0 / (dist + 1e-6)
            weight_total += weight
            weighted_pass += self._y_pass[idx] * weight
            weighted_tp += self._y_tp[idx] * weight
            avg_dist += dist

        pass_prob = weighted_pass / weight_total if weight_total else 0.0
        pred_tp = weighted_tp / weight_total if weight_total else 0.0
        avg_dist = avg_dist / k if k else 1.0
        uncertainty = min(1.0, avg_dist)
        return pass_prob, pred_tp, uncertainty

    def score(self, rows: list[dict[str, Any]]) -> list[SurrogateScore]:
        scores: list[SurrogateScore] = []
        for row in rows:
            x = self._features(row)
            if self._sklearn_cls is not None and self._sklearn_reg is not None:
                pass_prob = float(self._sklearn_cls.predict_proba([x])[0][1])
                pred_tp = float(self._sklearn_reg.predict([x])[0])
                uncertainty = 1.0 - abs((pass_prob - 0.5) * 2.0)
            else:
                pass_prob, pred_tp, uncertainty = self._predict_empirical(x)

            combined_score = pass_prob * max(pred_tp, 0.0) * (1.0 - 0.25 * uncertainty)
            scores.append(
                SurrogateScore(
                    candidate_key=str(row.get("candidate_key", "")),
                    pass_probability=pass_prob,
                    predicted_throughput=pred_tp,
                    uncertainty=uncertainty,
                    combined_score=combined_score,
                )
            )
        return scores


def prune_with_surrogate(
    rows: list[dict[str, Any]],
    tier_field: str,
    top_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"model_type": "none", "pruned_from": 0, "pruned_to": 0}

    surrogate = SurrogateModel(tier_field)
    surrogate.fit(rows)
    scores = surrogate.score(rows)

    by_key = {score.candidate_key: score for score in scores}
    ranked = sorted(
        rows,
        key=lambda row: (
            by_key[str(row.get("candidate_key", ""))].combined_score,
            by_key[str(row.get("candidate_key", ""))].pass_probability,
            by_key[str(row.get("candidate_key", ""))].predicted_throughput,
        ),
        reverse=True,
    )

    selected = ranked[: max(1, min(top_n, len(ranked)))]
    metadata = {
        "model_type": surrogate.model_type,
        "pruned_from": len(rows),
        "pruned_to": len(selected),
        "tier_field": tier_field,
        "top_n": top_n,
    }
    return selected, metadata
