from __future__ import annotations

import functools
import time
from collections import defaultdict
from typing import Any, Callable

import mvp.src.candidates as candidates_module
from mvp.src.prepare import TMDBClient


class Profiler:
    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []

    def wrap(self, obj: Any, attribute: str, label: str) -> None:
        original = getattr(obj, attribute)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.events.append((label, time.perf_counter() - started))

        setattr(obj, attribute, wrapped)

    def aggregate(self) -> list[dict[str, Any]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for label, elapsed in self.events:
            buckets[label].append(elapsed)
        rows = []
        for label, durations in buckets.items():
            rows.append(
                {
                    "stage": label,
                    "calls": len(durations),
                    "total": sum(durations),
                    "mean": sum(durations) / len(durations),
                    "max": max(durations),
                }
            )
        return sorted(rows, key=lambda row: row["total"], reverse=True)


def main() -> None:
    profiler = Profiler()

    # High-level candidate generation stages.
    profiler.wrap(candidates_module, "build_watched_exclusions", "watched_exclusions")
    profiler.wrap(candidates_module, "_discover_popular", "discover_popular")
    profiler.wrap(candidates_module, "_discover_top_rated", "discover_top_rated")
    profiler.wrap(candidates_module, "_discover_recent", "discover_recent")
    profiler.wrap(candidates_module, "_discover_genre_diverse", "discover_genre_diverse")
    profiler.wrap(candidates_module, "_enrich_candidate_records", "candidate_enrichment")

    # Nested network calls, useful for deciding whether discovery concurrency is worth it.
    profiler.wrap(candidates_module, "_tmdb_request", "tmdb_discovery_request")
    profiler.wrap(TMDBClient, "search_movie", "tmdb_search_movie")
    profiler.wrap(TMDBClient, "movie_details", "tmdb_movie_details")

    started = time.perf_counter()
    frame, report = candidates_module.generate_candidate_pool(
        candidate_limit=300,
        include_seeded_recommendations=False,
    )
    total = time.perf_counter() - started

    print("\nCANDIDATE GENERATION PROFILE")
    print("=" * 78)
    print(f"Total candidate generation: {total:.2f}s")
    print(f"Final candidates:           {len(frame)}")
    print("\nInclusive stage timings (nested rows can overlap):")
    print("-" * 78)
    print(f"{'stage':30} {'calls':>7} {'total':>10} {'mean':>10} {'max':>10}")
    print("-" * 78)
    for row in profiler.aggregate():
        print(
            f"{row['stage'][:30]:30} "
            f"{row['calls']:>7} "
            f"{row['total']:>9.2f}s "
            f"{row['mean']:>9.2f}s "
            f"{row['max']:>9.2f}s"
        )
    print("-" * 78)
    print("Candidate report:")
    print(f"  before_deduplication: {report.get('before_deduplication')}")
    print(f"  after_deduplication:  {report.get('after_deduplication')}")
    print(f"  removed_watched:      {report.get('removed_watched')}")
    print(f"  after_exclusion:      {report.get('after_exclusion')}")
    print(f"  enrichment_failures:  {report.get('enrichment_failures')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
