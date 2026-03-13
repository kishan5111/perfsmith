"""Perfsmith CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .optimize import load_workload_spec, optimize_workload
from .replay_trace import replay_trace
from .report import write_report
from .summarize import summarize_to_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perfsmith")
    sub = parser.add_subparsers(dest="command", required=True)

    summarize = sub.add_parser("summarize", help="Summarize run artifacts into normalized table")
    summarize.add_argument("--input", required=True, help="Input run directory or zip")
    summarize.add_argument("--out", required=True, help="Output CSV/Parquet path")

    optimize = sub.add_parser("optimize", help="Optimize configuration from summarized runs")
    optimize.add_argument("--workload", required=True, help="Workload spec JSON")
    optimize.add_argument("--sla-tier", required=True, choices=["strict", "balanced"])
    optimize.add_argument("--output-root", default="artifacts", help="Artifacts root directory")
    optimize.add_argument("--run-id", default=None)

    report = sub.add_parser("report", help="Generate markdown SLA pack from optimize artifact")
    report.add_argument("--run-id", required=True)
    report.add_argument("--output-root", default="artifacts")
    report.add_argument("--output", default=None)

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

    if args.command == "optimize":
        spec = load_workload_spec(Path(args.workload))
        output = optimize_workload(
            workload_spec=spec,
            sla_tier=args.sla_tier,
            output_root=Path(args.output_root),
            run_id=args.run_id,
        )
        summary = {
            "artifact_path": output["artifact_path"],
            "winner": {
                "candidate_id": output["result"]["winner"]["candidate_id"],
                "status": output["result"]["winner"]["status"],
                "total_token_throughput": output["result"]["winner"]["total_token_throughput"],
                "max_concurrency": output["result"]["winner"]["max_concurrency"],
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "report":
        output_path = write_report(
            run_id=args.run_id,
            output_root=Path(args.output_root),
            output_path=Path(args.output) if args.output else None,
        )
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
