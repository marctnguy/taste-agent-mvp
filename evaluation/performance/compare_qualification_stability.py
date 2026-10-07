from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import mvp.src.generative_v4.qualification_chain as qualification_module
import mvp.src.watchlist_personalization.reversible_history_watchlist_service as service_module


PROMPT = "I want something English from the 80s"
OUTPUT_PATH = Path("evaluation/performance/qualification_stability.json")
COMPARE_FIELDS = [
    "qualification_status",
    "supported_required_aspects",
    "unsupported_required_aspects",
    "supported_preferred_aspects",
    "unsupported_preferred_aspects",
    "violated_semantic_exclusions",
    "request_match",
    "caveat",
]


def _records_by_id(records: list[Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.model_dump() if hasattr(record, "model_dump") else dict(record)
        result[str(payload.get("candidate_id"))] = payload
    return result


def _selected_view(
    qualified: pd.DataFrame,
    route_ids: dict[str, set[str]],
    request: Any,
    *,
    requested_count: int = 5,
) -> pd.DataFrame:
    allowed_ids = service_module._scenario_allowed_ids(
        route_ids,
        "full_watchlist",
        variant="A",
    )
    scenario_qualified = qualified.loc[
        qualified["source_id"].astype(str).isin(allowed_ids)
    ].copy().reset_index(drop=True)
    scenario_qualified = service_module._attach_selection_percentiles(
        scenario_qualified
    )
    return service_module._select_view(
        scenario_qualified,
        request_mode=request.request_mode,
        variant="A",
        requested_count=requested_count,
        novelty_requested=request.novelty_requested,
    )


def _run(
    label: str,
    workers: int,
    request: Any,
    qualification_pool: pd.DataFrame,
    route_ids: dict[str, set[str]],
) -> dict[str, Any]:
    qualification_module.SEMANTIC_QUALIFICATION_MAX_WORKERS = workers
    qualified, records = qualification_module.qualify_candidates(
        request,
        qualification_pool,
        max_candidates=48,
    )
    selected = _selected_view(qualified, route_ids, request)
    return {
        "label": label,
        "workers": workers,
        "records": _records_by_id(records),
        "selected_ids": selected["source_id"].astype(str).tolist()
        if not selected.empty
        else [],
        "selected_titles": selected["title"].astype(str).tolist()
        if not selected.empty
        else [],
    }


def _compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_records = left["records"]
    right_records = right["records"]
    all_ids = sorted(set(left_records) | set(right_records))
    candidate_differences: list[dict[str, Any]] = []
    field_counts: Counter[str] = Counter()
    status_change_count = 0

    for candidate_id in all_ids:
        lrow = left_records.get(candidate_id)
        rrow = right_records.get(candidate_id)
        if lrow is None or rrow is None:
            candidate_differences.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "candidate_missing_from_one_run",
                    "left_present": lrow is not None,
                    "right_present": rrow is not None,
                }
            )
            field_counts["candidate_presence"] += 1
            continue

        differences: dict[str, Any] = {}
        for field in COMPARE_FIELDS:
            if lrow.get(field) != rrow.get(field):
                differences[field] = {
                    "left": lrow.get(field),
                    "right": rrow.get(field),
                }
                field_counts[field] += 1
                if field == "qualification_status":
                    status_change_count += 1

        if differences:
            candidate_differences.append(
                {
                    "candidate_id": candidate_id,
                    "differences": differences,
                }
            )

    return {
        "left": left["label"],
        "right": right["label"],
        "same_selected_ids": left["selected_ids"] == right["selected_ids"],
        "same_selected_titles": left["selected_titles"] == right["selected_titles"],
        "left_selected_titles": left["selected_titles"],
        "right_selected_titles": right["selected_titles"],
        "candidate_difference_count": len(candidate_differences),
        "qualification_status_change_count": status_change_count,
        "field_difference_counts": dict(field_counts),
        "candidate_differences": candidate_differences,
    }


def main() -> None:
    service = service_module.load_reversible_history_watchlist_service()
    request = service_module.understand_request(PROMPT, client=service.tmdb_client)
    master, _, route_ids = service._build_candidate_pool(
        request,
        profile=None,
        watchlist=None,
    )
    qualification_pool, _ = service_module._build_qualification_pool(
        master,
        route_ids,
        limit=48,
    )

    sequential_a = _run(
        "sequential_a",
        1,
        request,
        qualification_pool,
        route_ids,
    )
    sequential_b = _run(
        "sequential_b",
        1,
        request,
        qualification_pool,
        route_ids,
    )
    parallel = _run(
        "parallel",
        4,
        request,
        qualification_pool,
        route_ids,
    )

    comparisons = [
        _compare(sequential_a, sequential_b),
        _compare(sequential_a, parallel),
        _compare(sequential_b, parallel),
    ]

    payload = {
        "prompt": PROMPT,
        "qualification_pool_count": int(len(qualification_pool)),
        "runs": [
            {
                "label": run["label"],
                "workers": run["workers"],
                "selected_ids": run["selected_ids"],
                "selected_titles": run["selected_titles"],
            }
            for run in (sequential_a, sequential_b, parallel)
        ],
        "comparisons": comparisons,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\nQUALIFICATION STABILITY CONTROL")
    print("=" * 80)
    print(f"Prompt: {PROMPT}")
    print(f"Qualification pool: {len(qualification_pool)} candidates")
    print()
    for run in (sequential_a, sequential_b, parallel):
        print(f"{run['label']:14} ({run['workers']} worker{'s' if run['workers'] != 1 else ''}): {run['selected_titles']}")
    print()
    for comparison in comparisons:
        print(f"{comparison['left']} vs {comparison['right']}")
        print(f"  same selected IDs:      {comparison['same_selected_ids']}")
        print(f"  same selected titles:   {comparison['same_selected_titles']}")
        print(f"  candidates with diffs:  {comparison['candidate_difference_count']}")
        print(f"  status changes:         {comparison['qualification_status_change_count']}")
        print(f"  diff fields:            {comparison['field_difference_counts']}")
    print()
    print(f"Saved details to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
