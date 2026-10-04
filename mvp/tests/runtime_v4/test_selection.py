from __future__ import annotations

import pandas as pd

from mvp.src.generative_v4.intent_chain import RequestUnderstanding, understand_request
from mvp.src.generative_v4.selection_chain import select_candidates


def test_request_relevance_outweighs_higher_b3_when_candidates_are_qualified(runtime_v4_candidate_pool) -> None:
    request = understand_request("I want a film about performers and fame").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    qualified = runtime_v4_candidate_pool.copy()
    qualified.loc[:, "qualification_status"] = ["strong", "strong"]
    qualified.loc[:, "request_relevance"] = [0.15, 0.95]
    qualified.loc[:, "predicted_preference"] = [0.99, 0.55]
    qualified.loc[:, "b3_applicability_distance"] = [0.01, 0.75]

    result = select_candidates(request, qualified, recommendation_count=1)

    assert result.selected_frame.iloc[0]["source_id"] == "1002"
    assert result.selection_records[0].candidate_id == "1002"


def test_fewer_than_five_recommendations_is_valid(runtime_v4_candidate_pool) -> None:
    request = understand_request("I want a film about performers and fame").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    qualified = runtime_v4_candidate_pool.loc[[1]].copy()
    qualified.loc[:, "qualification_status"] = ["strong"]

    result = select_candidates(request, qualified, recommendation_count=5)

    assert len(result.selected_frame) == 1
    assert result.selection_report["selected_count"] == 1

