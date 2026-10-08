from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_PROFILE = Path("evaluation/performance/runtime_latency_profile.json")


def _extract_tmdb_id(detail: str) -> str | None:
    match = re.search(r"tmdb_id=(\d+)", str(detail or ""))
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze duplicate TMDB movie_details calls from an existing latency profile without running the recommender again."
    )
    parser.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE),
        help=f"Existing profiler JSON. Default: {DEFAULT_PROFILE}",
    )
    args = parser.parse_args()

    path = Path(args.profile)
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    payload = json.loads(path.read_text())
    events = payload.get("events", [])
    tmdb_events = [event for event in events if event.get("label") == "tmdb_movie_details"]

    ids = [_extract_tmdb_id(event.get("detail", "")) for event in tmdb_events]
    ids = [movie_id for movie_id in ids if movie_id is not None]
    counts = Counter(ids)

    unique_ids = len(counts)
    duplicate_calls = max(0, len(ids) - unique_ids)
    repeated_ids = {movie_id: count for movie_id, count in counts.items() if count > 1}
    repeated_call_volume = sum(repeated_ids.values())

    durations_by_id: dict[str, float] = {}
    for event in tmdb_events:
        movie_id = _extract_tmdb_id(event.get("detail", ""))
        if movie_id is None:
            continue
        durations_by_id[movie_id] = durations_by_id.get(movie_id, 0.0) + float(event.get("elapsed_seconds", 0.0) or 0.0)

    duplicate_time_upper_bound = sum(
        durations_by_id[movie_id] * ((count - 1) / count)
        for movie_id, count in repeated_ids.items()
        if count > 0
    )

    print("\nTMDB DETAIL REUSE ANALYSIS")
    print("=" * 72)
    print(f"Profile:                   {path}")
    print(f"Total movie_details calls: {len(tmdb_events)}")
    print(f"Calls with parsed TMDB ID: {len(ids)}")
    print(f"Unique TMDB IDs:           {unique_ids}")
    print(f"Duplicate calls:           {duplicate_calls}")
    print(f"Duplicate-call share:      {(duplicate_calls / len(ids) * 100.0) if ids else 0.0:.1f}%")
    print(f"IDs called >1 time:        {len(repeated_ids)}")
    print(f"Repeated-call volume:      {repeated_call_volume}")
    print(f"Duplicate time upper bound:{duplicate_time_upper_bound:8.2f}s")

    if repeated_ids:
        print("\nMost repeated TMDB IDs:")
        for movie_id, count in sorted(repeated_ids.items(), key=lambda item: (-item[1], item[0]))[:20]:
            print(f"  {movie_id:>10}  {count:>3} calls")

    print("\nDecision guidance:")
    if duplicate_calls >= 20 or (ids and duplicate_calls / len(ids) >= 0.10):
        print("  Strong in-request cache opportunity: repeated detail lookups are material.")
    elif duplicate_calls > 0:
        print("  Some reuse exists, but expected wall-clock gain is probably modest.")
    else:
        print("  No reuse opportunity in this profile: detail calls are all unique.")
    print("=" * 72)


if __name__ == "__main__":
    main()
