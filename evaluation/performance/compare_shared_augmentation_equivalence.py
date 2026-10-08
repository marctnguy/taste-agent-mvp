from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import mvp.src.retrieval.catalog_retrieval as catalog_module
import mvp.src.watchlist_personalization.reversible_history_watchlist_service as service_module
from mvp.src.candidates import generate_candidate_pool


PROMPT = "I want something English from the 80s"
OUTPUT_PATH = Path("evaluation/performance/shared_augmentation_equivalence.json")
COMPARE_FIELDS = [
    "source_id",
    "tmdb_id",
    "title",
    "year",
    "release_year",
    "tmdb_genres",
    "tmdb_original_language",
    "tmdb_production_countries",
    "tmdb_overview",
    "candidate_sources",
    "candidate_source_ranks",
    "rank",
    "raw_rank",
    "request_relevance",
    "request_relevance_score",
    "request_relevance_rank",
    "predicted_preference",
    "b3_applicability",
    "b3_applicability_distance",
]


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _equal(left: Any, right: Any) -> bool:
    left = _normalize(left)
    right = _normalize(right)
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _compare_frames(
    legacy: pd.DataFrame,
    optimized: pd.DataFrame,
) -> dict[str, Any]:
    legacy_ids = legacy["source_id"].astype(str).tolist() if not legacy.empty else []
    optimized_ids = (
        optimized["source_id"].astype(str).tolist() if not optimized.empty else []
    )
    differences: list[dict[str, Any]] = []

    if len(legacy) != len(optimized):
        differences.append(
            {
                "reason": "row_count_mismatch",
                "legacy_count": len(legacy),
                "optimized_count": len(optimized),
            }
        )
    else:
        for index, (left_row, right_row) in enumerate(
            zip(legacy.to_dict(orient="records"), optimized.to_dict(orient="records"))
        ):
            field_diffs: dict[str, Any] = {}
            for field in COMPARE_FIELDS:
                left = left_row.get(field)
                right = right_row.get(field)
                if not _equal(left, right):
                    field_diffs[field] = {
                        "legacy": _normalize(left),
                        "optimized": _normalize(right),
                    }
            if field_diffs:
                differences.append(
                    {
                        "index": index,
                        "source_id": str(
                            left_row.get("source_id")
                            or right_row.get("source_id")
                            or ""
                        ),
                        "title": left_row.get("title") or right_row.get("title"),
                        "differences": field_diffs,
                    }
                )

    return {
        "same_ordered_ids": legacy_ids == optimized_ids,
        "legacy_count": len(legacy),
        "optimized_count": len(optimized),
        "field_difference_count": len(differences),
        "differences": differences,
    }


def _fixed_copy(
    augmentation: catalog_module.QueryAwareAugmentation,
) -> catalog_module.QueryAwareAugmentation:
    return catalog_module.QueryAwareAugmentation(
        frame=augmentation.frame.copy(deep=True),
        report=dict(augmentation.report),
    )


def main() -> None:
    if not hasattr(catalog_module, "QueryAwareAugmentation"):
        raise RuntimeError(
            "Shared augmentation patch is not applied locally. Run "
            "evaluation/performance/apply_shared_query_augmentation.py first."
        )

    service = service_module.load_reversible_history_watchlist_service()
    request = service_module.understand_request(PROMPT, client=service.tmdb_client)

    request_pool, _ = generate_candidate_pool(
        candidate_limit=300,
        include_seeded_recommendations=False,
    )

    positive = service.historical.get("positive", pd.DataFrame())
    if isinstance(positive, pd.DataFrame) and not positive.empty:
        seed_frame = positive.copy().reset_index(drop=True)
        if "rating" in seed_frame.columns:
            seed_frame["rating"] = pd.to_numeric(
                seed_frame["rating"], errors="coerce"
            )
        history_pool = service_module._select_diverse_history_seeds(
            seed_frame,
            limit=min(24, len(seed_frame)),
        )
    else:
        history_pool = pd.DataFrame()

    watched_keys = set().union(
        service.consumed_keys.get("source", set()),
        service.consumed_keys.get("tmdb", set()),
        service.consumed_keys.get("title_year", set()),
    )

    fixed_augmentation = catalog_module.build_query_aware_augmentation(
        request, service.tmdb_client
    )

    # Legacy-control path: discover_catalog_for_request asks its builder for
    # augmentation independently. We freeze that builder to the same captured
    # request-only result so this comparison isolates reuse mechanics rather than
    # live TMDB drift between calls.
    original_builder = catalog_module.build_query_aware_augmentation

    def frozen_builder(_request, _client):
        return _fixed_copy(fixed_augmentation)

    catalog_module.build_query_aware_augmentation = frozen_builder
    try:
        legacy_request = catalog_module.discover_catalog_for_request(
            request,
            client=service.tmdb_client,
            candidate_pool=request_pool,
            watched_ids=watched_keys,
        )
        legacy_history = catalog_module.discover_catalog_for_request(
            request,
            client=service.tmdb_client,
            candidate_pool=history_pool,
            watched_ids=watched_keys,
        )
    finally:
        catalog_module.build_query_aware_augmentation = original_builder

    optimized_request = catalog_module.discover_catalog_for_request(
        request,
        client=service.tmdb_client,
        candidate_pool=request_pool,
        watched_ids=watched_keys,
        precomputed_augmentation=_fixed_copy(fixed_augmentation),
    )
    optimized_history = catalog_module.discover_catalog_for_request(
        request,
        client=service.tmdb_client,
        candidate_pool=history_pool,
        watched_ids=watched_keys,
        precomputed_augmentation=_fixed_copy(fixed_augmentation),
    )

    request_comparison = _compare_frames(
        legacy_request.catalog_frame,
        optimized_request.catalog_frame,
    )
    history_comparison = _compare_frames(
        legacy_history.catalog_frame,
        optimized_history.catalog_frame,
    )

    payload = {
        "prompt": PROMPT,
        "augmentation_candidate_rows": int(len(fixed_augmentation.frame)),
        "request_route": request_comparison,
        "history_route": history_comparison,
        "equivalent": bool(
            request_comparison["same_ordered_ids"]
            and request_comparison["field_difference_count"] == 0
            and history_comparison["same_ordered_ids"]
            and history_comparison["field_difference_count"] == 0
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )

    print("\nSHARED QUERY AUGMENTATION EQUIVALENCE CHECK")
    print("=" * 78)
    print(f"Prompt: {PROMPT}")
    print(f"Captured augmentation rows: {len(fixed_augmentation.frame)}")
    print("Request route:")
    print(f"  Same ordered IDs:     {request_comparison['same_ordered_ids']}")
    print(f"  Field differences:    {request_comparison['field_difference_count']}")
    print("History route:")
    print(f"  Same ordered IDs:     {history_comparison['same_ordered_ids']}")
    print(f"  Field differences:    {history_comparison['field_difference_count']}")
    print(f"Overall equivalent:     {payload['equivalent']}")
    print(f"Saved details to:       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
