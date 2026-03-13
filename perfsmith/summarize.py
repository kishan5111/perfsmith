"""Deterministic summarizer from raw run artifacts to normalized table."""

from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import SLA_TIERS
from .utils import coerce_bool, coerce_float, coerce_int, ensure_parent

NORMALIZED_FIELDS = [
    "source_input",
    "source_path",
    "run_number",
    "date",
    "benchmark_name",
    "model_id",
    "tokenizer_id",
    "backend",
    "endpoint_type",
    "num_prompts",
    "request_rate",
    "burstiness",
    "max_concurrency",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "random_input_len",
    "random_output_len",
    "completed",
    "failed",
    "error_rate",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "p99_ttft_ms",
    "p99_itl_ms",
    "p99_e2el_ms",
    "mean_ttft_ms",
    "mean_itl_ms",
    "mean_e2el_ms",
    "workload_fingerprint",
    "engine_fingerprint",
    "hardware_fingerprint",
    "serve_config_fingerprint",
    "candidate_key",
    "meets_strict",
    "meets_balanced",
    "max_concurrency_under_strict",
    "max_concurrency_under_balanced",
]

NUMERIC_FIELDS = {
    "run_number",
    "num_prompts",
    "burstiness",
    "max_concurrency",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "random_input_len",
    "random_output_len",
    "completed",
    "failed",
    "error_rate",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "p99_ttft_ms",
    "p99_itl_ms",
    "p99_e2el_ms",
    "mean_ttft_ms",
    "mean_itl_ms",
    "mean_e2el_ms",
    "max_concurrency_under_strict",
    "max_concurrency_under_balanced",
}

BOOL_FIELDS = {
    "meets_strict",
    "meets_balanced",
}


def _parse_kv_blob(blob: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for token in blob.split("-"):
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        if raw == "":
            continue
        if raw.lower() in {"none", "null"}:
            values[key] = None
            continue
        intval = coerce_int(raw, None)
        flt = coerce_float(raw, None)
        if intval is not None and flt is not None and float(intval) == flt and "." not in raw:
            values[key] = intval
        elif flt is not None:
            values[key] = flt
        else:
            values[key] = raw
    return values


def _path_metadata(source_path: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for part in source_path.split("/"):
        if part.startswith("SERVE--"):
            payload = part[len("SERVE--") :]
            if "-BENCH--" in payload:
                serve_blob, bench_blob = payload.split("-BENCH--", 1)
                meta.update(_parse_kv_blob(serve_blob))
                meta.update(_parse_kv_blob(bench_blob))
            else:
                meta.update(_parse_kv_blob(payload))
        elif part.startswith("BENCH--"):
            meta.update(_parse_kv_blob(part[len("BENCH--") :]))
        elif part.startswith("max_concurrency="):
            meta["max_concurrency"] = coerce_int(part.split("=", 1)[1], None)
    return meta


def _iter_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _iter_run_payloads(input_path: Path) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path, "r") as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".json") or "/run=" not in name:
                    continue
                with archive.open(name, "r") as handle:
                    payload = json.load(handle)
                for row in _iter_json_records(payload):
                    entries.append((name, row))
        return entries

    for path in sorted(input_path.rglob("run=*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rel = str(path.relative_to(input_path))
        for row in _iter_json_records(payload):
            entries.append((rel, row))
    return entries


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def _build_fingerprints(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    workload_fp = "|".join(
        [
            _safe_text(row.get("benchmark_name")),
            _safe_text(row.get("random_input_len")),
            _safe_text(row.get("random_output_len")),
            _safe_text(row.get("num_prompts")),
            _safe_text(row.get("request_rate")),
            _safe_text(row.get("burstiness")),
        ]
    )
    engine_fp = "|".join(
        [
            _safe_text(row.get("backend")),
            _safe_text(row.get("endpoint_type")),
            _safe_text(row.get("model_id")),
            _safe_text(row.get("tokenizer_id")),
            _safe_text(row.get("vllm_version")),
            _safe_text(row.get("torch_version")),
            _safe_text(row.get("cuda_version")),
        ]
    )
    hardware_fp = "|".join(
        [
            _safe_text(row.get("gpu_name", "unknown_gpu")),
            _safe_text(row.get("gpu_memory_gb", "unknown_vram")),
            _safe_text(row.get("host", "unknown_host")),
        ]
    )
    serve_fp = "|".join(
        [
            _safe_text(row.get("gpu_memory_utilization")),
            _safe_text(row.get("max_model_len")),
            _safe_text(row.get("max_num_seqs")),
            _safe_text(row.get("max_num_batched_tokens")),
            _safe_text(row.get("model_id")),
        ]
    )
    candidate_key = "|".join([workload_fp, serve_fp, _safe_text(row.get("max_concurrency"))])
    return workload_fp, engine_fp, hardware_fp, serve_fp, candidate_key


def _normalize_record(source_input: Path, source_path: str, raw: dict[str, Any]) -> dict[str, Any]:
    metadata = _path_metadata(source_path)

    completed = coerce_int(raw.get("completed"), 0) or 0
    failed = coerce_int(raw.get("failed"), 0) or 0
    total = completed + failed
    error_rate = (failed / total) if total > 0 else (1.0 if failed > 0 else 0.0)

    record: dict[str, Any] = {
        "source_input": str(source_input),
        "source_path": source_path,
        "run_number": coerce_int(raw.get("run_number"), 0),
        "date": raw.get("date", ""),
        "benchmark_name": raw.get("_benchmark_name", metadata.get("benchmark_name", "")),
        "model_id": raw.get("model_id", ""),
        "tokenizer_id": raw.get("tokenizer_id", ""),
        "backend": raw.get("backend", ""),
        "endpoint_type": raw.get("endpoint_type", ""),
        "num_prompts": coerce_int(raw.get("num_prompts"), 0),
        "request_rate": raw.get("request_rate", ""),
        "burstiness": coerce_float(raw.get("burstiness"), 0.0),
        "max_concurrency": coerce_int(raw.get("max_concurrency", metadata.get("max_concurrency")), 0),
        "gpu_memory_utilization": coerce_float(
            raw.get("gpu_memory_utilization", metadata.get("gpu_memory_utilization")), None
        ),
        "max_model_len": coerce_int(raw.get("max_model_len", metadata.get("max_model_len")), None),
        "max_num_seqs": coerce_int(raw.get("max_num_seqs", metadata.get("max_num_seqs")), None),
        "max_num_batched_tokens": coerce_int(
            raw.get("max_num_batched_tokens", metadata.get("max_num_batched_tokens")), None
        ),
        "random_input_len": coerce_int(raw.get("random_input_len", metadata.get("random_input_len")), None),
        "random_output_len": coerce_int(raw.get("random_output_len", metadata.get("random_output_len")), None),
        "completed": completed,
        "failed": failed,
        "error_rate": error_rate,
        "duration": coerce_float(raw.get("duration"), 0.0),
        "request_throughput": coerce_float(raw.get("request_throughput"), 0.0),
        "output_throughput": coerce_float(raw.get("output_throughput"), 0.0),
        "total_token_throughput": coerce_float(raw.get("total_token_throughput"), 0.0),
        "p99_ttft_ms": coerce_float(raw.get("p99_ttft_ms"), 0.0),
        "p99_itl_ms": coerce_float(raw.get("p99_itl_ms"), 0.0),
        "p99_e2el_ms": coerce_float(raw.get("p99_e2el_ms"), 0.0),
        "mean_ttft_ms": coerce_float(raw.get("mean_ttft_ms"), 0.0),
        "mean_itl_ms": coerce_float(raw.get("mean_itl_ms"), 0.0),
        "mean_e2el_ms": coerce_float(raw.get("mean_e2el_ms"), 0.0),
    }

    workload_fp, engine_fp, hardware_fp, serve_fp, candidate_key = _build_fingerprints(record)
    record["workload_fingerprint"] = workload_fp
    record["engine_fingerprint"] = engine_fp
    record["hardware_fingerprint"] = hardware_fp
    record["serve_config_fingerprint"] = serve_fp
    record["candidate_key"] = candidate_key

    strict = SLA_TIERS["strict"]
    balanced = SLA_TIERS["balanced"]
    record["meets_strict"] = (
        record["failed"] == 0
        and record["completed"] > 0
        and (record["p99_ttft_ms"] or 0.0) <= strict["p99_ttft_ms"]
        and (record["p99_itl_ms"] or 0.0) <= strict["p99_itl_ms"]
    )
    record["meets_balanced"] = (
        record["failed"] == 0
        and record["completed"] > 0
        and (record["p99_ttft_ms"] or 0.0) <= balanced["p99_ttft_ms"]
        and (record["p99_itl_ms"] or 0.0) <= balanced["p99_itl_ms"]
    )

    record["max_concurrency_under_strict"] = 0
    record["max_concurrency_under_balanced"] = 0
    return record


def _attach_max_concurrency(rows: list[dict[str, Any]]) -> None:
    maxima: dict[str, dict[str, int]] = defaultdict(lambda: {"strict": 0, "balanced": 0})
    for row in rows:
        key = f"{row['workload_fingerprint']}|{row['serve_config_fingerprint']}"
        conc = coerce_int(row.get("max_concurrency"), 0) or 0
        if coerce_bool(row.get("meets_strict")):
            maxima[key]["strict"] = max(maxima[key]["strict"], conc)
        if coerce_bool(row.get("meets_balanced")):
            maxima[key]["balanced"] = max(maxima[key]["balanced"], conc)

    for row in rows:
        key = f"{row['workload_fingerprint']}|{row['serve_config_fingerprint']}"
        row["max_concurrency_under_strict"] = maxima[key]["strict"]
        row["max_concurrency_under_balanced"] = maxima[key]["balanced"]


def summarize_records(input_path: Path) -> list[dict[str, Any]]:
    rows = [_normalize_record(input_path, src, payload) for src, payload in _iter_run_payloads(input_path)]
    rows.sort(key=lambda item: (item["source_path"], item["run_number"], item["date"]))
    _attach_max_concurrency(rows)
    return rows


def write_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=NORMALIZED_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in NORMALIZED_FIELDS})
        return

    if suffix == ".parquet":
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Parquet output requires pyarrow") from exc
        table = pa.Table.from_pylist([{field: row.get(field) for field in NORMALIZED_FIELDS} for row in rows])
        pq.write_table(table, output_path)
        return

    raise ValueError(f"Unsupported output format for {output_path}. Use .csv or .parquet")


def summarize_to_table(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = summarize_records(input_path)
    write_table(rows, output_path)
    return rows


def load_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized: dict[str, Any] = {}
                for key, value in row.items():
                    if key in NUMERIC_FIELDS:
                        normalized[key] = coerce_float(value, 0.0)
                        if key in {
                            "run_number",
                            "num_prompts",
                            "max_concurrency",
                            "max_model_len",
                            "max_num_seqs",
                            "max_num_batched_tokens",
                            "random_input_len",
                            "random_output_len",
                            "completed",
                            "failed",
                            "max_concurrency_under_strict",
                            "max_concurrency_under_balanced",
                        }:
                            normalized[key] = coerce_int(value, 0)
                    elif key in BOOL_FIELDS:
                        normalized[key] = coerce_bool(value)
                    else:
                        normalized[key] = value
                rows.append(normalized)
        return rows

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reading parquet requires pyarrow") from exc
        table = pq.read_table(path)
        for row in table.to_pylist():
            rows.append(row)
        return rows

    raise ValueError(f"Unsupported input table format: {path}")
