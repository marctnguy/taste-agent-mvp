from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import mvp.src.generative_v4.qualification_chain as qualification_module
import mvp.src.watchlist_personalization.reversible_history_watchlist_service as service_module


PROMPT = "I want something English from the 80s"
OUTPUT_PATH = Path("evaluation/performance/parallel_qualification_equivalence.json")


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


def _run_qualification(
    request: Any,
    qualification_pool: pd.DataFrame,
    route_ids: dict[str, set[str]],
    *,
    workers: int,
) -> dict[str, Any]:
    qualification_module.SEMANTIC_QUALIFICATION_MAX_WORKERS = workers
    qualified, records = qualification_module.qualify_candidates(
        request,
        qualification_pool,
        max_candidates=48,
    )
    selected = _selected_view(
        qualified,
        route_ids,
        request,
        requested_count=5,
    )
    return {
        "workers": workers,
        "qualified": qualified,
        "records": records,
        "selected_ids": selected["source_id"].astype(str).tolist()
        if not selected.empty
        else [],
        "selected_titles": selected["title"].astype(str).tolist()
        if not selected.empty
        else [],
    }


def main() -> None:
    service = service_module.load_reversible_history_watchlist_service()

    request = service_module.understand_request(
        PROMPT,
        client=service.tmdb_client,
    )
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

    sequential = _run_qualification(
        request,
        qualification_pool,
        route_ids,
        workers=1,
    )
    parallel = _run_qualification(
        request,
        qualification_pool,
        route_ids,
        workers=4,
    )

    sequential_records = _records_by_id(sequential["records"])
    parallel_records = _records_by_id(parallel["records"])

    all_ids = sorted(set(sequential_records) | set(parallel_records))
    record_differences: list[dict[str, Any]] = []
    fields = [
        "qualification_status",
        "supported_required_aspects",
        "unsupported_required_aspects",
        "supported_preferred_aspects",
        "unsupported_preferred_aspects",
        "violated_semantic_exclusions",
        "request_match",
        "caveat",
    ]

    for candidate_id in all_ids:
        left = sequential_records.get(candidate_id)
        right = parallel_records.get(candidate_id)
        if left is None or right is None:
            record_differences.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "candidate_missing_from_one_run",
                    "sequential_present": left is not None,
                    "parallel_present": right is not None,
                }
            )
            continue

        field_diffs = {
            field: {
                "sequential": left.get(field),
                "parallel": right.get(field),
            }
            for field in fields
            if left.get(field) != right.get(field)
        }
        if field_diffs:
            record_differences.append(
                {
                    "candidate_id": candidate_id,
                    "differences": field_diffs,
                }
            )

    payload = {
        "prompt": PROMPT,
        "qualification_pool_count": int(len(qualification_pool)),
        "sequential_workers": 1,
        "parallel_workers": 4,
        "same_selected_ids": sequential["selected_ids"] == parallel["selected_ids"],
        "same_selected_titles": sequential["selected_titles"]
        == parallel["selected_titles"],
        "sequential_selected_ids": sequential["selected_ids"],
        "parallel_selected_ids": parallel["selected_ids"],
        "sequential_selected_titles": sequential["selected_titles"],
        "parallel_selected_titles": parallel["selected_titles"],
        "qualification_record_difference_count": len(record_differences),
        "qualification_record_differences": record_differences,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\nPARALLEL QUALIFICATION EQUIVALENCE CHECK")
    print("=" * 72)
    print(f"Prompt: {PROMPT}")
    print(f"Qualification pool: {len(qualification_pool)} candidates")
    print(f"Sequential selected: {sequential['selected_titles']}")
    print(f"Parallel selected:   {parallel['selected_titles']}")
    print(f"Same selected IDs:   {payload['same_selected_ids']}")
    print(f"Same selected titles:{payload['same_selected_titles']}")
    print(
        "Qualification record differences: "
        f"{payload['qualification_record_difference_count']}"
    )
    print(f"Saved details to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
