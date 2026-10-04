from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IntentType = Literal["GENERAL_DISCOVERY", "MOOD_THEME", "NOVELTY", "CONSTRAINT", "EXPLAIN"]


class RecommendationIntent(BaseModel):
    intent_type: IntentType
    requested_taste_dimensions: list[str] = Field(default_factory=list)
    requested_genres: list[str] = Field(default_factory=list)
    requested_languages: list[str] = Field(default_factory=list)
    requested_countries: list[str] = Field(default_factory=list)
    requested_decades: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    free_text_context: str = ""


class TasteSignal(BaseModel):
    dimension: str
    preference_association: float
    evidence_count: int
    evidence_confidence: float


class CandidateSemanticEvidence(BaseModel):
    dimension: str
    candidate_score: float
    user_preference_association: float
    evidence_count: int
    evidence_confidence: float
    direction: Literal["positive", "negative"]


class CandidateContextItem(BaseModel):
    candidate_id: str
    tmdb_id: int | None = None
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    original_language: str | None = None
    production_countries: list[str] = Field(default_factory=list)
    predicted_preference: float
    raw_rank: int
    candidate_provenance: list[str] = Field(default_factory=list)
    candidate_source_ranks: list[str] = Field(default_factory=list)
    semantic_evidence: list[CandidateSemanticEvidence] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    candidate_id: str
    title: str
    year: int | None = None
    why_it_may_fit: str
    taste_signals: list[str] = Field(default_factory=list)
    request_match: str | None = None
    caveat: str | None = None


class RecommendationResponse(BaseModel):
    intent: RecommendationIntent
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    response_summary: str
    methodology_note: str


class RuntimeMetadata(BaseModel):
    runtime_version: str = "runtime_v1"
    generation_model: str = "gpt-4o-mini"
    source_recommendation_run: str = "20261003T075635Z"
    run_id: str
    candidate_context_size: int
    recommendation_count: int
    intent_type: IntentType
    intent_prompt_version: str
    recommendation_prompt_version: str
    ranking_model: str = "B3"
    ranking_model_status: str = "exploratory_candidate_mvp"
    selection_layer: str = "langchain_contextual_selection"
    taste_model: str = "62_dim_interpretable_profile"
    goodreads_status: str = "descriptive_only"
    generation_status: Literal["generated", "fallback", "needs_candidate_context", "no_match"] = "generated"
    validation_passed: bool = False
    repair_attempted: bool = False
    fallback_used: bool = False
    hard_filters_applied: bool = False
    target_candidate_id: str | None = None
    llm_call_count: int = 0


class RecommendationRunResult(BaseModel):
    response: RecommendationResponse
    runtime_metadata: RuntimeMetadata
    validation_passed: bool
    validation_errors: list[str] = Field(default_factory=list)
    debug: dict[str, object] = Field(default_factory=dict)
