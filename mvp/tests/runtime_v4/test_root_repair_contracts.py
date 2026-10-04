from __future__ import annotations

import pandas as pd

from mvp.src.generative_v4.intent_chain import ReferenceResolution, understand_request
from mvp.src.generative_v4.qualification_chain import qualify_candidates
from mvp.src.generative_v4.runtime import _ensure_b3_predictions
from mvp.src.generative_v4.selection_chain import select_candidates


def test_semantic_requests_remain_contextual_and_preserve_concepts() -> None:
    request = understand_request("I want something comforting but not cheesy.")

    assert request.request_mode == "contextual"
    assert "comforting" in request.intent.requested_taste_dimensions
    assert "cheesy" in request.intent.exclusions
    assert "tone" not in request.intent.requested_taste_dimensions
    assert "mood" not in request.intent.requested_taste_dimensions


def test_surprise_requests_are_contextual_not_history_novelty() -> None:
    request = understand_request("I want to be mind-blown, recommend me something totally out there that surprises me.")

    assert request.request_mode == "contextual"
    assert request.novelty_requested is False
    assert request.intent.requested_taste_dimensions


def test_false_reference_spans_are_rejected_and_true_titles_survive() -> None:
    false_reference = understand_request("I want a European movie that feels like freedom and youthful.")
    true_reference = understand_request("Something like Aftersun.")

    assert false_reference.reference_title is None
    assert false_reference.reference_status == "absent"
    assert true_reference.reference_title == "aftersun"
    assert true_reference.request_mode == "reference"


def test_european_movies_map_to_country_constraints() -> None:
    request = understand_request("I want a European movie.")

    assert request.request_mode == "contextual"
    assert request.intent.requested_countries
    assert "FR" in request.intent.requested_countries or "GB" in request.intent.requested_countries


def test_aboutness_does_not_confuse_famous_with_about_fame(monkeypatch) -> None:
    monkeypatch.setattr(
        "mvp.src.generative_v4.qualification_chain._qualify_with_llm",
        lambda *args, **kwargs: (None, 0, None, None, None),
    )
    request = understand_request("I want something about fame and performers").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    candidate_frame = pd.DataFrame(
        [
            {
                "source_id": "1",
                "tmdb_id": 1,
                "title": "Behold a Pale Horse",
                "year": 1970,
                "release_year": 1970,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "Manuel Artiguez, a famous bandit during the Spanish civil war, has lived in French exile for 20 years.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["1"],
                "request_relevance": 0.9,
            }
        ]
    )

    qualified, records = qualify_candidates(request, candidate_frame, max_candidates=10)

    assert qualified.empty
    assert records[0].qualification_status == "unsupported"


def test_missing_b3_scores_are_not_faked_as_zero(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {"source_id": "1", "title": "Alpha"},
            {"source_id": "2", "title": "Beta"},
        ]
    )

    monkeypatch.setattr(
        "mvp.src.generative_v4.runtime.score_candidates",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"source_id": "1", "predicted_preference": 0.91},
                {"source_id": "2", "predicted_preference": 0.42},
            ]
        ),
    )

    scored, report = _ensure_b3_predictions(frame)
    assert report["b3_scored"] is True
    assert scored["predicted_preference"].notna().all()
    assert float(scored.loc[scored["source_id"] == "1", "predicted_preference"].iloc[0]) == 0.91

    missing = frame.copy()
    missing["predicted_preference"] = pd.NA
    result = select_candidates(
        understand_request("Recommend me something."),
        missing.assign(request_relevance=0.0, b3_applicability_distance=0.5, qualification_status="strong"),
        recommendation_count=1,
    )
    assert result.selection_records[0].b3_available is False
    assert result.selection_records[0].b3_predicted_preference is None
