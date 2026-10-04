from __future__ import annotations

import pandas as pd

from mvp.src.generative_v4.intent_chain import RequestUnderstanding, understand_request
from mvp.src.generative_v4.selection_chain import select_candidates


def test_novelty_is_distinct_from_low_b3(monkeypatch) -> None:
    request = RequestUnderstanding(
        query="Show me something different from what I normally watch.",
        intent=understand_request("Show me something different from what I normally watch.").intent,
        request_mode="novelty",
        novelty_requested=True,
        reference_status="absent",
        request_relevance_mode="query_aware",
    )
    selected = pd.DataFrame(
        [
            {
                "source_id": "A",
                "tmdb_id": 1,
                "title": "Familiar Film",
                "year": 2001,
                "release_year": 2001,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A familiar film.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["1"],
                "predicted_preference": 0.9,
                "b3_applicability_distance": 0.95,
                "request_relevance": 0.5,
                "qualification_status": "strong",
                "emb_0000": 1.0,
                "emb_0001": 0.0,
            },
            {
                "source_id": "B",
                "tmdb_id": 2,
                "title": "Novel Film",
                "year": 2002,
                "release_year": 2002,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A more novel film.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["2"],
                "predicted_preference": 0.4,
                "b3_applicability_distance": 0.10,
                "request_relevance": 0.5,
                "qualification_status": "strong",
                "emb_0000": 0.0,
                "emb_0001": 1.0,
            },
        ]
    )
    monkeypatch.setattr(
        "mvp.src.generative_v4.selection_chain._build_history_embeddings",
        lambda: pd.DataFrame({"source_id": ["h1"], "emb_0000": [1.0], "emb_0001": [0.0]}),
    )

    result = select_candidates(request, selected, recommendation_count=1)

    assert result.selected_frame.iloc[0]["source_id"] == "B"
    assert result.selection_report["novelty_requested"] is True


def test_novelty_requires_explicit_novelty_language() -> None:
    request = understand_request("I want a strong request match even if it is only modestly compatible with my historical taste.")

    assert request.request_mode == "contextual"
    assert request.novelty_requested is False
    assert request.spec is not None
    assert request.spec.novelty_goal.enabled is False
    assert request.spec.personalization_instruction.priority == "current_request_primary"
