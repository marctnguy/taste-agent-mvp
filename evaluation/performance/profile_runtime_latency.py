from __future__ import annotations

import argparse
import functools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import mvp.src.generative_v4.qualification_chain as qualification_module
import mvp.src.retrieval.catalog_retrieval as catalog_module
import mvp.src.watchlist_personalization.reversible_history_watchlist_service as service_module
from mvp.src.prepare import TMDBClient


DEFAULT_PROMPT = "I want something English from the 80s"
OUTPUT_PATH = Path("evaluation/performance/runtime_latency_profile.json")
ACCEPTED_BASELINE_TITLES = [
    "Dead Poets Society",
    "The Shining",
    "This Is Spinal Tap",
    "Maurice",
    "Stop Making Sense",
]


class Profiler:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, label: str, elapsed: float, detail: str | None = None) -> None:
        self.events.append(
            {
                "label": label,
                "elapsed_seconds": round(float(elapsed), 4),
                "detail": detail or "",
            }
        )

    def wrap(
        self,
        obj: Any,
        attribute: str,
        label: str,
        detail_fn: Callable[[tuple[Any, ...], dict[str, Any]], str | None] | None = None,
    ) -> None:
        original = getattr(obj, attribute)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            detail = None
            if detail_fn is not None:
                try:
                    detail = detail_fn(args, kwargs)
                except Exception:
                    detail = None

            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.record(label, time.perf_counter() - started, detail)

        setattr(obj, attribute, wrapped)

    def aggregate(self) -> list[dict[str, Any]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for event in self.events:
            buckets[event["label"]].append(float(event["elapsed_seconds"]))

        rows = []
        for label, durations in buckets.items():
            rows.append(
                {
                    "stage": label,
                    "calls": len(durations),
                    "total_seconds": round(sum(durations), 4),
                    "mean_seconds": round(sum(durations) / len(durations), 4),
                    "max_seconds": round(max(durations), 4),
                }
            )

        return sorted(rows, key=lambda row: row["total_seconds"], reverse=True)


def _frame_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if not args:
        return None
    value = args[0]
    if isinstance(value, pd.DataFrame):
        return f"rows={len(value)}"
    return None


def _catalog_detail(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if not args:
        return None
    request = args[0]
    query = getattr(request, "query", None)
    return str(query) if query else None


def _qualification_detail(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if len(args) < 2:
        return None
    frame = args[1]
    if isinstance(frame, pd.DataFrame):
        return f"candidates={len(frame)}"
    if isinstance(frame, list):
        return f"candidates={len(frame)}"
    return None


def _tmdb_detail(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if len(args) >= 2:
        return f"tmdb_id={args[1]}"
    return None


def _extract_recommendations(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    response = result.get("response") or {}
    recommendations = response.get("recommendations") or []
    if not isinstance(recommendations, list):
        return []
    return [item for item in recommendations if isinstance(item, dict)]


def install_instrumentation(profiler: Profiler) -> None:
    profiler.wrap(
        service_module,
        "understand_request",
        "request_understanding",
        lambda args, kwargs: str(args[0])[:120] if args else None,
    )
    profiler.wrap(
        service_module,
        "generate_candidate_pool",
        "candidate_pool_generation",
    )
    profiler.wrap(
        service_module,
        "discover_catalog_for_request",
        "catalog_retrieval",
        _catalog_detail,
    )
    profiler.wrap(
        service_module,
        "qualify_candidates",
        "qualification_total",
        _qualification_detail,
    )

    profiler.wrap(
        catalog_module,
        "embed_documents",
        "candidate_document_embedding",
        _frame_size,
    )
    profiler.wrap(
        catalog_module,
        "embed_text",
        "request_text_embedding",
        lambda args, kwargs: f"chars={len(str(args[0]))}" if args else None,
    )

    profiler.wrap(
        qualification_module,
        "_qualify_with_llm",
        "qualification_llm_batch",
        _qualification_detail,
    )

    profiler.wrap(
        TMDBClient,
        "movie_details",
        "tmdb_movie_details",
        _tmdb_detail,
    )


def print_report(
    *,
    prompt: str,
    service_init_seconds: float,
    recommend_seconds: float,
    profiler: Profiler,
    selected_titles: list[str],
    selected_ids: list[str],
    matches_accepted_baseline: bool | None,
) -> None:
    print("\n" + "=" * 78)
    print("TASTE AGENT RUNTIME LATENCY PROFILE")
    print("=" * 78)
    print(f"Prompt: {prompt}")
    print(f"Service initialization: {service_init_seconds:.2f}s")
    print(f"Recommendation total:   {recommend_seconds:.2f}s")
    print("\nInclusive stage timings (nested rows can overlap):")
    print("-" * 78)
    print(f"{'stage':34} {'calls':>7} {'total':>10} {'mean':>10} {'max':>10}")
    print("-" * 78)

    for row in profiler.aggregate():
        print(
            f"{row['stage'][:34]:34} "
            f"{row['calls']:>7} "
            f"{row['total_seconds']:>9.2f}s "
            f"{row['mean_seconds']:>9.2f}s "
            f"{row['max_seconds']:>9.2f}s"
        )

    print("-" * 78)
    print(
        "NOTE: stage totals are inclusive. For example, catalog_retrieval includes "
        "embedding/TMDB work performed inside it, and qualification_total includes "
        "its qualification_llm_batch calls."
    )
    print("\nOutput preservation check:")
    print(f"Selected titles: {selected_titles}")
    print(f"Selected IDs:    {selected_ids}")
    if matches_accepted_baseline is not None:
        print(f"Matches accepted baseline slate: {matches_accepted_baseline}")
    else:
        print("Matches accepted baseline slate: n/a for this prompt")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one real Taste Agent recommendation request without changing "
            "runtime behavior."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help=f'Recommendation prompt. Default: "{DEFAULT_PROMPT}"',
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=f"JSON output path. Default: {OUTPUT_PATH}",
    )
    args = parser.parse_args()

    profiler = Profiler()
    install_instrumentation(profiler)

    init_started = time.perf_counter()
    service = service_module.load_reversible_history_watchlist_service()
    service_init_seconds = time.perf_counter() - init_started

    recommend_started = time.perf_counter()
    result = service.recommend(
        args.prompt,
        scenario_label="full_watchlist",
        variant="A",
        requested_count=5,
    )
    recommend_seconds = time.perf_counter() - recommend_started

    recommendations = _extract_recommendations(result)
    selected_titles = [str(item.get("title")) for item in recommendations]
    selected_ids = [
        str(item.get("candidate_id") or item.get("source_id") or "")
        for item in recommendations
    ]
    normalized_prompt = " ".join(args.prompt.lower().split())
    normalized_default = " ".join(DEFAULT_PROMPT.lower().split())
    matches_accepted_baseline = (
        selected_titles == ACCEPTED_BASELINE_TITLES
        if normalized_prompt == normalized_default
        else None
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "prompt": args.prompt,
        "service_version": service_module.SERVICE_VERSION,
        "service_initialization_seconds": round(service_init_seconds, 4),
        "recommendation_total_seconds": round(recommend_seconds, 4),
        "aggregate": profiler.aggregate(),
        "events": profiler.events,
        "recommendation_count": len(recommendations),
        "selected_titles": selected_titles,
        "selected_ids": selected_ids,
        "accepted_baseline_titles": ACCEPTED_BASELINE_TITLES
        if matches_accepted_baseline is not None
        else None,
        "matches_accepted_baseline_slate": matches_accepted_baseline,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(
        prompt=args.prompt,
        service_init_seconds=service_init_seconds,
        recommend_seconds=recommend_seconds,
        profiler=profiler,
        selected_titles=selected_titles,
        selected_ids=selected_ids,
        matches_accepted_baseline=matches_accepted_baseline,
    )
    print(f"\nSaved JSON profile to: {output_path}")


if __name__ == "__main__":
    main()
