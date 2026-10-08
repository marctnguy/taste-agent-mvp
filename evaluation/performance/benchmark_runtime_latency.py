from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "I want something English from the 80s"
DEFAULT_RUNS = 3
OUTPUT_DIR = Path("evaluation/performance/runtime_benchmark")
PROFILE_SCRIPT = Path("evaluation/performance/profile_runtime_latency.py")
KEY_STAGES = [
    "qualification_total",
    "candidate_pool_generation",
    "catalog_retrieval",
    "request_understanding",
]


def _stage_total(payload: dict[str, Any], stage: str) -> float | None:
    for row in payload.get("aggregate", []):
        if row.get("stage") == stage:
            return float(row.get("total_seconds", 0.0))
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated end-to-end Taste Agent latency profiles and report medians."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help=f'Default: "{DEFAULT_PROMPT}"',
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of isolated runs. Default: {DEFAULT_RUNS}",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LANGSMITH_TRACING"] = "false"
    env["LANGCHAIN_TRACING_V2"] = "false"
    env["PYTHONPATH"] = env.get("PYTHONPATH") or "."

    payloads: list[dict[str, Any]] = []

    print("\n" + "=" * 88)
    print("TASTE AGENT ISOLATED RUNTIME BENCHMARK")
    print("=" * 88)
    print(f"Prompt: {args.prompt}")
    print(f"Runs:   {args.runs}")

    for index in range(1, args.runs + 1):
        output_path = OUTPUT_DIR / f"run_{index}.json"
        print(f"\n--- Run {index}/{args.runs} ---")
        command = [
            sys.executable,
            str(PROFILE_SCRIPT),
            args.prompt,
            "--output",
            str(output_path),
        ]
        subprocess.run(command, check=True, env=env)
        payloads.append(json.loads(output_path.read_text()))

    totals = [float(item["recommendation_total_seconds"]) for item in payloads]
    init_times = [float(item["service_initialization_seconds"]) for item in payloads]

    print("\n" + "=" * 88)
    print("BENCHMARK SUMMARY")
    print("=" * 88)
    print(f"{'run':>4} {'total':>10} {'init':>9} " + " ".join(f"{stage[:18]:>18}" for stage in KEY_STAGES))
    print("-" * 88)

    for index, payload in enumerate(payloads, start=1):
        row = [
            f"{index:>4}",
            f"{float(payload['recommendation_total_seconds']):>9.2f}s",
            f"{float(payload['service_initialization_seconds']):>8.2f}s",
        ]
        row.extend(f"{_fmt(_stage_total(payload, stage)):>18}" for stage in KEY_STAGES)
        print(" ".join(row))

    print("-" * 88)
    print(f"Median recommendation total: {statistics.median(totals):.2f}s")
    print(f"Mean recommendation total:   {statistics.mean(totals):.2f}s")
    print(f"Min / max:                   {min(totals):.2f}s / {max(totals):.2f}s")
    print(f"Median service init:         {statistics.median(init_times):.2f}s")

    print("\nMedian key stages:")
    for stage in KEY_STAGES:
        values = [
            value
            for value in (_stage_total(payload, stage) for payload in payloads)
            if value is not None
        ]
        print(f"  {stage:28} {statistics.median(values):.2f}s" if values else f"  {stage:28} n/a")

    baseline_checks = [payload.get("matches_accepted_baseline_slate") for payload in payloads]
    selected_titles = [payload.get("selected_titles") for payload in payloads]
    print("\nOutput preservation:")
    for index, (matched, titles) in enumerate(zip(baseline_checks, selected_titles), start=1):
        print(f"  Run {index}: baseline_match={matched} titles={titles}")

    summary = {
        "prompt": args.prompt,
        "runs": args.runs,
        "recommendation_totals": totals,
        "median_recommendation_total_seconds": statistics.median(totals),
        "mean_recommendation_total_seconds": statistics.mean(totals),
        "min_recommendation_total_seconds": min(totals),
        "max_recommendation_total_seconds": max(totals),
        "median_service_initialization_seconds": statistics.median(init_times),
        "median_key_stages": {
            stage: statistics.median(
                [
                    value
                    for value in (_stage_total(payload, stage) for payload in payloads)
                    if value is not None
                ]
            )
            for stage in KEY_STAGES
            if any(_stage_total(payload, stage) is not None for payload in payloads)
        },
        "baseline_matches": baseline_checks,
        "selected_titles": selected_titles,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved benchmark summary to: {summary_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
