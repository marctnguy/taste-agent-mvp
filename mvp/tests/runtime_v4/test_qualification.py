from __future__ import annotations

import pandas as pd

from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.generative_v4.qualification_chain import qualify_candidates


def test_strong_qualification_for_stage_actress_request(runtime_v4_candidate_pool) -> None:
    request = understand_request("stage actress celebrity theatre").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    candidate = runtime_v4_candidate_pool.loc[[1]].copy()

    qualified, records = qualify_candidates(request, candidate, max_candidates=10)

    assert not qualified.empty
    assert records[0].qualification_status == "strong"
    assert set(records[0].supported_required_aspects) == {"performers", "fame"}


def test_unsupported_associative_leap_is_rejected(runtime_v4_candidate_pool) -> None:
    request = understand_request("something about performers, fame and show business, preferably period").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    candidate = runtime_v4_candidate_pool.loc[[0]].copy()

    qualified, records = qualify_candidates(request, candidate, max_candidates=10)

    assert qualified.empty
    assert records[0].qualification_status == "unsupported"
    assert "show_business" in records[0].unsupported_request_aspects or "performers" in records[0].unsupported_request_aspects


def test_partial_match_preserves_caveat(runtime_v4_candidate_pool) -> None:
    request = understand_request("performers and fame, preferably historical").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    candidate = runtime_v4_candidate_pool.loc[[1]].copy()
    candidate.loc[:, "tmdb_overview"] = "A celebrated stage actress struggles with celebrity in the theatre world."

    qualified, records = qualify_candidates(request, candidate, max_candidates=10)

    assert not qualified.empty
    assert records[0].qualification_status == "strong"
    assert records[0].caveat is None


def test_all_of_semantic_groups_require_every_concept_for_strong(monkeypatch) -> None:
    monkeypatch.setattr(
        "mvp.src.generative_v4.qualification_chain._qualify_with_llm",
        lambda *args, **kwargs: (None, 0, None, None, None),
    )
    request = understand_request("I want a film about performers and fame").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    candidate_frame = pd.DataFrame(
        [
            {
                "source_id": "a",
                "tmdb_id": 1,
                "title": "Complete Match",
                "year": 2001,
                "release_year": 2001,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A celebrated stage actress becomes a famous star.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["1"],
                "predicted_preference": 0.8,
                "request_relevance": 0.9,
            },
            {
                "source_id": "b",
                "tmdb_id": 2,
                "title": "Partial Match",
                "year": 2002,
                "release_year": 2002,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A stage actress works in the theatre world.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["2"],
                "predicted_preference": 0.7,
                "request_relevance": 0.85,
            },
        ]
    )

    qualified, records = qualify_candidates(request, candidate_frame, max_candidates=10)

    statuses = {record.candidate_id: record.qualification_status for record in records}
    assert statuses["a"] == "strong"
    assert statuses["b"] != "strong"
    assert not qualified.empty
