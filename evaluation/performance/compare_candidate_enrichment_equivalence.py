from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import mvp.src.candidates as candidates_module


OUTPUT_PATH = Path("evaluation/performance/candidate_enrichment_equivalence.json")
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
]


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _compare_rows(
    serial_rows: list[dict[str, Any]],
    parallel_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if len(serial_rows) != len(parallel_rows):
        differences.append(
            {
                "reason": "row_count_mismatch",
                "serial_count": len(serial_rows),
                "parallel_count": len(parallel_rows),
            }
        )
        return differences

    for index, (left, right) in enumerate(zip(serial_rows, parallel_rows)):
        field_diffs = {}
        for field in COMPARE_FIELDS:
            lval = _normalize(left.get(field))
            rval = _normalize(right.get(field))
            if lval != rval:
                field_diffs[field] = {"serial": lval, "parallel": rval}
        if field_diffs:
            differences.append(
                {
                    "index": index,
                    "source_id": str(left.get("source_id") or right.get("source_id") or ""),
                    "title": left.get("title") or right.get("title"),
                    "differences": field_diffs,
                }
            )
    return differences


def main() -> None:
    if not hasattr(candidates_module, "_enrich_candidate_records"):
        raise RuntimeError(
            "Parallel candidate enrichment patch is not applied locally. "
            "Run evaluation/performance/apply_parallel_candidate_enrichment.py first."
        )

    # Build one real candidate pool once so discovery/exclusion are fixed for this check.
    # We then reconstruct the exact selected rows and feed those identical inputs to
    # serial vs parallel enrichment. This avoids conflating TMDB discovery drift with
    # enrichment concurrency.
    enriched_pool, report = candidates_module.generate_candidate_pool(
        candidate_limit=300,
        include_seeded_recommendations=False,
    )

    fixed_rows: list[dict[str, Any]] = []
    for row in enriched_pool.to_dict(orient="records"):
        fixed_rows.append(
            {
                "source_id": str(row.get("source_id")),
                "tmdb_id": row.get("tmdb_id"),
                "canonical_id": str(row.get("source_id")),
                "title": row.get("title"),
                "year": row.get("year"),
                "release_year": row.get("release_year"),
                "candidate_sources": row.get("candidate_sources", []),
                "candidate_source_ranks": row.get("candidate_source_ranks", []),
            }
        )

    keys = candidates_module.get_api_keys()
    if not keys.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required for the equivalence check.")
    client = candidates_module.TMDBClient(api_key=keys.tmdb_api_key)

    serial_rows, serial_failures = candidates_module._enrich_candidate_records(
        client,
        fixed_rows,
        max_workers=1,
    )
    parallel_rows, parallel_failures = candidates_module._enrich_candidate_records(
        client,
        fixed_rows,
        max_workers=8,
    )

    differences = _compare_rows(serial_rows, parallel_rows)
    serial_ids = [str(row.get("source_id")) for row in serial_rows]
    parallel_ids = [str(row.get("source_id")) for row in parallel_rows]

    payload = {
        "candidate_count": len(fixed_rows),
        "source_candidate_report": report,
        "same_ordered_ids": serial_ids == parallel_ids,
        "serial_failures": int(serial_failures),
        "parallel_failures": int(parallel_failures),
        "metadata_difference_count": len(differences),
        "metadata_differences": differences,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    print("\nCANDIDATE ENRICHMENT EQUIVALENCE CHECK")
    print("=" * 78)
    print(f"Candidate count:           {len(fixed_rows)}")
    print(f"Same ordered IDs:          {payload['same_ordered_ids']}")
    print(f"Serial enrichment failures:{serial_failures}")
    print(f"Parallel failures:         {parallel_failures}")
    print(f"Metadata differences:      {len(differences)}")
    print(f"Saved details to:          {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
