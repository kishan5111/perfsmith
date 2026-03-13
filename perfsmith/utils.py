"""Utility helpers."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import pstdev
from typing import Iterable


def coerce_int(value: object, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def coerce_float(value: object, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.lower() == "inf":
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def quantile(values: Iterable[float], q: float) -> float:
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    if q <= 0:
        return data[0]
    if q >= 1:
        return data[-1]
    idx = (len(data) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    frac = idx - lo
    return data[lo] * (1 - frac) + data[hi] * frac


def stddev(values: Iterable[float]) -> float:
    data = [float(v) for v in values]
    if len(data) <= 1:
        return 0.0
    return float(pstdev(data))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{prefix}"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"
