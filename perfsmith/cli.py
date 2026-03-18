"""Perfsmith CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .optimize import load_workload_spec, optimize_atlas
from .replay_trace import replay_trace
from .report import write_report
from .run import run_experiment
from .summarize import summarize_to_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perfsmith")
    sub = parser.add_subparsers(dest="command", required=True)

    summarize = sub.add_parser("summarize", help="Summarize run artifacts into a normalized atlas")
    summarize.add_argument("--input", required=True, help="Input run directory or zip")
    summarize.add_argument("--out", required=True, help="Output CSV/Parquet path")

    run = sub.add_parser("run", help="Run a vLLM grid, summarize results, and emit decision artifacts")
    run.add_argument("--model", required=True)
    run.add_argument("--workload", required=True)
    run.add_argument("--grid", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--gpu-cost-per-hour", required=True, type=float)
    run.add_argument("--num-runs", type=int, default=1)
    run.add_argument("--raw-artifact-url", default=None)
    run.add_argument("--server-host", default="127.0.0.1")
    run.add_argument("--server-port", type=int, default=8000)
    run.add_argument("--server-ready-timeout", type=int, default=600)
    run.add_argument("--server-extra-args", default="--no-enable-prefix-caching")
    run.add_argument("--bench-extra-args", default="")

    optimize = sub.add_parser("optimize", help="Pick a winner from a summarized atlas")
    optimize.add_argument("--atlas", required=True)
    optimize.add_argument("--workload", required=True)
    optimize.add_argument("--sla-tier", required=True, choices=["strict", "balanced"])
    optimize.add_argument("--gpu-cost-per-hour", required=True, type=float)
    optimize.add_argument("--out", required=True)

    report = sub.add_parser("report", help="Generate a markdown SLO Pack from a decision artifact")
    report.add_argument("--decision", required=True)
    report.add_argument("--out", required=True)

    replay = sub.add_parser("replay-trace", help="Replay tool-aware agent traces")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--stub", required=True)
    replay.add_argument("--output", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "summarize":
        rows = summarize_to_table(Path(args.input), Path(args.out))
        print(json.dumps({"rows": len(rows), "output": args.out}, indent=2, sort_keys=True))
        return

    if args.command == "run":
        result = run_experiment(
            model_id=args.model,
            workload_path=Path(args.workload),
            grid_path=Path(args.grid),
            output_root=Path(args.out),
            gpu_cost_per_hour=args.gpu_cost_per_hour,
            num_runs=args.num_runs,
            raw_artifact_url=args.raw_artifact_url,
            server_host=args.server_host,
            server_port=args.server_port,
            server_ready_timeout=args.server_ready_timeout,
            server_extra_args=args.server_extra_args,
            bench_extra_args=args.bench_extra_args,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "optimize":
        workload = load_workload_spec(Path(args.workload))
        result = optimize_atlas(
            atlas_path=Path(args.atlas),
            workload_spec=workload,
            sla_tier=args.sla_tier,
            gpu_cost_per_hour=args.gpu_cost_per_hour,
            out_path=Path(args.out),
        )
        winner = result["result"]["winner"]
        print(
            json.dumps(
                {
                    "artifact_path": result["artifact_path"],
                    "winner": {
                        "candidate_id": winner["candidate_id"],
                        "status": winner["status"],
                        "max_concurrency": winner["max_concurrency"],
                        "total_token_throughput": winner["total_token_throughput"],
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.command == "report":
        output_path = write_report(Path(args.decision), Path(args.out))
        print(json.dumps({"report": str(output_path)}, indent=2, sort_keys=True))
        return

    if args.command == "replay-trace":
        result = replay_trace(
            trace_path=Path(args.trace),
            stub_path=Path(args.stub),
            output_path=Path(args.output) if args.output else None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
