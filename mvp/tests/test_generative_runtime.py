from __future__ import annotations

from pathlib import Path

import pandas as pd

from mvp.src.generative.context_builder import RecommendationRuntimeContext, hard_filter_candidates, parse_intent
from mvp.src.generative.schemas import (
    CandidateContextItem,
    CandidateSemanticEvidence,
    RecommendationItem,
    RecommendationResponse,
    RecommendationIntent,
)
from mvp.src.generative.validators import fallback_recommendations, validate_recommendation_response


def _candidate() -> CandidateContextItem:
    return CandidateContextItem(
        candidate_id="film-1",
        tmdb_id=1,
        title="Example Film",
        year=2001,
        genres=["Drama", "Comedy"],
        original_language="fr",
        production_countries=["FR"],
        predicted_preference=0.8,
        raw_rank=1,
        candidate_provenance=["top_rated"],
        candidate_source_ranks=["1"],
        semantic_evidence=[
            CandidateSemanticEvidence(
                dimension="melancholic",
                candidate_score=0.7,
                user_preference_association=0.08,
                evidence_count=10,
                evidence_confidence=0.8,
                direction="positive",
            )
        ],
    )


def test_parse_intent_examples() -> None:
    assert parse_intent("Recommend me something").intent_type == "GENERAL_DISCOVERY"
    assert parse_intent("I want something melancholic").intent_type == "MOOD_THEME"
    assert parse_intent("Something French").intent_type == "CONSTRAINT"
    assert parse_intent("Something from the 2000s").intent_type == "CONSTRAINT"
    assert parse_intent("Something completely different").intent_type == "NOVELTY"
    assert parse_intent("Why are you recommending this?").intent_type == "EXPLAIN"
    assert parse_intent("Recommend something, but no drama.").requested_genres == []
    assert parse_intent("Which film are you recommending because of my Goodreads history?").requested_genres == []
    soft = parse_intent("I want something weird but still emotionally engaging.")
    assert soft.intent_type == "MOOD_THEME"
    assert "bizarre" in soft.requested_taste_dimensions
    assert "emotional_intensity" in soft.requested_taste_dimensions


def test_hard_filter_and_fallback_when_no_candidates_match() -> None:
    candidate = _candidate()
    intent = RecommendationIntent(
        intent_type="CONSTRAINT",
        requested_languages=["ja"],
        requested_genres=[],
        requested_countries=[],
        requested_decades=[],
        exclusions=[],
        free_text_context="Japanese only",
    )

    context = RecommendationRuntimeContext(
        run_dir=Path("."),
        query="Japanese only",
        intent=intent,
        target_candidate_id=None,
        candidate_context_size=1,
        recommendation_count=5,
        candidate_frame=pd.DataFrame(),
        candidate_semantics=pd.DataFrame(),
        taste_profile=pd.DataFrame(),
        candidate_context=[candidate],
        positive_taste_profile=[],
        negative_taste_profile=[],
        hard_filters={"requested_genres": [], "requested_languages": ["ja"], "requested_countries": [], "requested_decades": [], "exclusions": []},
    )

    assert hard_filter_candidates(context) == []
    assert fallback_recommendations([candidate], intent, 5) == []


def test_validator_rejects_hallucinated_candidate_and_bad_evidence() -> None:
    candidate = _candidate()
    intent = RecommendationIntent(
        intent_type="GENERAL_DISCOVERY",
        requested_taste_dimensions=[],
        requested_genres=[],
        requested_languages=[],
        requested_countries=[],
        requested_decades=[],
        exclusions=[],
        free_text_context="",
    )
    response = RecommendationResponse(
        intent=intent,
        recommendations=[
            RecommendationItem(
                candidate_id="film-1",
                title="Example Film",
                year=2001,
                why_it_may_fit="Because of your books.",
                taste_signals=["made_up_dimension"],
                request_match="",
                caveat=None,
            )
        ],
        response_summary="Summary",
        methodology_note="Note",
    )

    validation = validate_recommendation_response(response, [candidate], intent, 5)
    assert not validation.passed
    assert any("Goodreads" in error for error in validation.errors)
    assert any("unsupported taste signal" in error for error in validation.errors)


def test_validator_rejects_unsupported_candidate_id() -> None:
    candidate = _candidate()
    intent = RecommendationIntent(
        intent_type="GENERAL_DISCOVERY",
        requested_taste_dimensions=[],
        requested_genres=[],
        requested_languages=[],
        requested_countries=[],
        requested_decades=[],
        exclusions=[],
        free_text_context="",
    )
    response = RecommendationResponse(
        intent=intent,
        recommendations=[
            RecommendationItem(
                candidate_id="unknown-film",
                title="Unknown",
                year=2001,
                why_it_may_fit="Grounded explanation.",
                taste_signals=[],
                request_match="",
                caveat=None,
            )
        ],
        response_summary="Summary",
        methodology_note="Note",
    )

    validation = validate_recommendation_response(response, [candidate], intent, 5)
    assert not validation.passed
    assert any("unsupported candidate_id" in error for error in validation.errors)


def test_validator_count_matches_eligible_candidates() -> None:
    candidate = _candidate()
    intent = RecommendationIntent(
        intent_type="GENERAL_DISCOVERY",
        requested_taste_dimensions=[],
        requested_genres=[],
        requested_languages=[],
        requested_countries=[],
        requested_decades=[],
        exclusions=[],
        free_text_context="",
    )
    response = RecommendationResponse(
        intent=intent,
        recommendations=[
            RecommendationItem(
                candidate_id="film-1",
                title="Example Film",
                year=2001,
                why_it_may_fit="Grounded explanation.",
                taste_signals=[],
                request_match="",
                caveat=None,
            )
        ],
        response_summary="Summary",
        methodology_note="Note",
    )
    validation = validate_recommendation_response(response, [candidate, candidate], intent, 5)
    assert not validation.passed
    assert any("expected 2 grounded recommendations" in error for error in validation.errors)


def test_explain_without_candidate_returns_clarification_state() -> None:
    from mvp.src.generative.recommendation_chain import run_recommendation_agent

    result = run_recommendation_agent("Why are you recommending this?", debug=True)
    assert result.runtime_metadata.generation_status == "needs_candidate_context"
    assert result.validation_passed is True
    assert result.response.recommendations == []
    assert "Which recommendation would you like me to explain?" in result.response.response_summary
