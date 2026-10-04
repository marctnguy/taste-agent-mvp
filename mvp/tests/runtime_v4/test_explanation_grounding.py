from __future__ import annotations

import pandas as pd

from mvp.src.generative.schemas import RecommendationResponse
from mvp.src.generative_v4.explanation_chain import explain_selection
from mvp.src.generative_v4.intent_chain import understand_request


def test_request_and_taste_evidence_remain_separate(monkeypatch, runtime_v4_candidate_pool) -> None:
    request = understand_request("I want a film about performers and fame").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    selected = runtime_v4_candidate_pool.loc[[1]].copy()
    selected.loc[:, "qualification_status"] = ["strong"]
    selected.loc[:, "supported_request_aspects"] = [["performers", "fame"]]
    selected.loc[:, "request_match"] = ["Supported request aspects: performers, fame"]
    selected.loc[:, "caveat"] = [None]

    monkeypatch.setattr(
        "mvp.src.generative_v4.explanation_chain._semantic_profile",
        lambda: pd.DataFrame({"dimension": ["melancholic"], "preference_association": [0.6], "evidence_count": [3], "evidence_confidence": [0.8]}),
    )
    monkeypatch.setattr(
        "mvp.src.generative_v4.explanation_chain.build_taste_evidence",
        lambda candidate, taste_profile: {
            "evidence_status": "supported",
            "positive_matches": [{"dimension": "melancholic"}],
            "possible_mismatches": [],
        },
    )
    monkeypatch.setattr(
        "mvp.src.generative_v4.explanation_chain.classify_semantic_vectors",
        lambda frame, cache_path=None: (
            pd.DataFrame({"source_id": frame["source_id"].astype(str), "melancholic": [1.0]}),
            {"semantic_classified": 1, "coverage": 1.0, "failures": []},
        ),
    )

    result = explain_selection(request, selected)

    assert result.response.recommendations[0].request_match == "Supported request aspects: performers, fame"
    assert result.response.recommendations[0].taste_signals == ["melancholic"]
    assert "Taste evidence:" in result.response.recommendations[0].why_it_may_fit
    assert result.semantic_report["semantic_classified"] == 1


def test_partial_candidates_preserve_caveats_in_explanations(monkeypatch, runtime_v4_candidate_pool) -> None:
    request = understand_request("performers and fame, preferably historical").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    selected = runtime_v4_candidate_pool.loc[[1]].copy()
    selected.loc[:, "qualification_status"] = ["partial"]
    selected.loc[:, "supported_request_aspects"] = [["performers", "fame"]]
    selected.loc[:, "unsupported_request_aspects"] = [["historical"]]
    selected.loc[:, "request_match"] = ["Supported request aspects: performers, fame"]
    selected.loc[:, "caveat"] = ["Unsupported aspects: historical"]

    monkeypatch.setattr(
        "mvp.src.generative_v4.explanation_chain._semantic_profile",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "mvp.src.generative_v4.explanation_chain.build_taste_evidence",
        lambda candidate, taste_profile: {
            "evidence_status": "unsupported",
            "positive_matches": [],
            "possible_mismatches": [],
        },
    )
    monkeypatch.setattr(
        "mvp.src.generative_v4.explanation_chain.classify_semantic_vectors",
        lambda frame, cache_path=None: (
            pd.DataFrame({"source_id": frame["source_id"].astype(str)}),
            {"semantic_classified": 1, "coverage": 1.0, "failures": []},
        ),
    )

    result = explain_selection(request, selected)

    assert result.response.recommendations[0].caveat == "Unsupported aspects: historical"
    assert "Caveat:" in result.response.recommendations[0].why_it_may_fit
    assert result.response.recommendations[0].request_match == "Supported request aspects: performers, fame"

