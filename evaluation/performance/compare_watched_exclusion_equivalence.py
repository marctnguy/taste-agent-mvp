from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mvp.src.candidates import (
    build_watched_exclusions,
    load_condition_a_enriched,
    load_consumption_history,
    load_watched_override_registry,
)
from mvp.src.config import get_api_keys
from mvp.src.prepare import TMDBClient


OUTPUT_PATH = Path("evaluation/performance/watched_exclusion_equivalence.json")
COMPARE_FIELDS = [
    "source_id",
    "title",
    "year",
    "tmdb_id",
    "match_method",
    "normalized_title",
    "normalized_title_year",
    "exclusion_key",
]


def _normalize(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _compare_frames(serial: pd.DataFrame, parallel: pd.DataFrame) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if len(serial) != len(parallel):
        differences.append(
            {
                "reason": "row_count_mismatch",
                "serial_count": len(serial),
                "parallel_count": len(parallel),
            }
        )
        return differences

    for index, (left, right) in enumerate(
        zip(serial.to_dict(orient="records"), parallel.to_dict(orient="records"))
    ):
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
    keys = get_api_keys()
    if not keys.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required for the equivalence check.")

    client = TMDBClient(api_key=keys.tmdb_api_key)
    consumed = load_consumption_history()
    enriched = load_condition_a_enriched()
    overrides = load_watched_override_registry()

    serial_frame, serial_report = build_watched_exclusions(
        consumed,
        enriched,
        client=client,
        watched_overrides=overrides,
        search_max_workers=1,
    )
    parallel_frame, parallel_report = build_watched_exclusions(
        consumed,
        enriched,
        client=client,
        watched_overrides=overrides,
        search_max_workers=8,
    )

    differences = _compare_frames(serial_frame, parallel_frame)
    same_ordered_keys = (
        serial_frame.get("exclusion_key", pd.Series(dtype=str)).astype(str).tolist()
        == parallel_frame.get("exclusion_key", pd.Series(dtype=str)).astype(str).tolist()
    )
    same_report = serial_report == parallel_report

    payload = {
        "serial_count": int(len(serial_frame)),
        "parallel_count": int(len(parallel_frame)),
        "same_ordered_exclusion_keys": bool(same_ordered_keys),
        "same_report": bool(same_report),
        "field_difference_count": int(len(differences)),
        "serial_report": serial_report,
        "parallel_report": parallel_report,
        "differences": differences,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    print("\nWATCHED EXCLUSION EQUIVALENCE CHECK")
    print("=" * 78)
    print(f"Serial exclusion rows:       {len(serial_frame)}")
    print(f"Parallel exclusion rows:     {len(parallel_frame)}")
    print(f"Same ordered exclusion keys: {same_ordered_keys}")
    print(f"Same report:                 {same_report}")
    print(f"Field differences:           {len(differences)}")
    print(f"Overall equivalent:          {same_ordered_keys and same_report and not differences}")
    print(f"Saved details to:            {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
