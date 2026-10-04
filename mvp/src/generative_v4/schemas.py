from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mvp.src.generative.schemas import (
    CandidateContextItem,
    CandidateSemanticEvidence,
    RecommendationIntent,
    RecommendationItem,
    RecommendationResponse,
)


RequestMode = Literal["generic", "contextual", "novelty", "reference", "clarification"]
InteractionMode = Literal["recommend", "explain_target", "explain_needs_target", "provenance", "profile_boundary"]
QualificationStatus = Literal["strong", "partial", "unsupported", "pending"]
ConstraintImportance = Literal["required", "preferred"]
SemanticSource = Literal["query", "reference", "novelty", "personalization", "structured_constraint"]


class StructuredConstraints(BaseModel):
    required_languages: list[str] = Field(default_factory=list)
    excluded_languages: list[str] = Field(default_factory=list)
    required_countries: list[str] = Field(default_factory=list)
    excluded_countries: list[str] = Field(default_factory=list)
    required_genres: list[str] = Field(default_factory=list)
    excluded_genres: list[str] = Field(default_factory=list)
    required_decades: list[str] = Field(default_factory=list)
    excluded_decades: list[str] = Field(default_factory=list)
    min_year: int | None = None
    max_year: int | None = None


class SemanticConcept(BaseModel):
    aspect_id: str
    concept: str
    importance: ConstraintImportance = "preferred"
    source_span: str = ""


class SemanticRequirementGroup(BaseModel):
    group_id: str
    mode: Literal["all_of", "any_of", "preferred"] = "all_of"
    concepts: list[SemanticConcept] = Field(default_factory=list)
    source_span: str = ""


class ReferenceSpec(BaseModel):
    title: str | None = None
    relation: str | None = None
    resolved_tmdb_id: int | None = None
    resolved_tmdb_title: str | None = None
    resolution_status: Literal["absent", "resolved", "unresolved"] = "absent"


class NoveltyGoal(BaseModel):
    enabled: bool = False
    priority: str | None = None


class PersonalizationInstruction(BaseModel):
    preference: str | None = None
    priority: str | None = None


class RequestSpec(BaseModel):
    intent_type: str = "GENERAL_DISCOVERY"
    mode: Literal["generic", "contextual"] = "generic"
    structured_constraints: StructuredConstraints = Field(default_factory=StructuredConstraints)
    semantic_requirement_groups: list[SemanticRequirementGroup] = Field(default_factory=list)
    semantic_requirements: list[SemanticConcept] = Field(default_factory=list)
    semantic_exclusions: list[SemanticConcept] = Field(default_factory=list)
    reference: ReferenceSpec = Field(default_factory=ReferenceSpec)
    novelty_goal: NoveltyGoal = Field(default_factory=NoveltyGoal)
    personalization_instruction: PersonalizationInstruction = Field(default_factory=PersonalizationInstruction)
    request_relevance_mode: Literal["broad", "query_aware"] = "broad"
    semantic_query_text: str = ""
    request_summary: str = ""
    llm_used: bool = False
    llm_call_count: int = 0
    runtime_input_tokens: int | None = None
    runtime_output_tokens: int | None = None
    runtime_total_tokens: int | None = None


class KeywordResolution(BaseModel):
    concept: str
    tmdb_keyword_id: int | None = None
    tmdb_keyword_name: str | None = None
    provenance: str = ""


class RetrievalPlan(BaseModel):
    request_mode: Literal["generic", "contextual"] = "generic"
    structured_constraints: StructuredConstraints = Field(default_factory=StructuredConstraints)
    semantic_query_text: str = ""
    semantic_concepts: list[SemanticConcept] = Field(default_factory=list)
    keyword_resolutions: list[KeywordResolution] = Field(default_factory=list)
    reference_seed: ReferenceSpec = Field(default_factory=ReferenceSpec)
    query_aware_expansion_required: bool = False
    supplemental_seed_sources: list[str] = Field(default_factory=list)


class RequestUnderstanding(BaseModel):
    query: str
    intent: RecommendationIntent
    spec: RequestSpec | None = None
    request_mode: RequestMode = "generic"
    interaction_mode: InteractionMode = "recommend"
    novelty_requested: bool = False
    reference_title: str | None = None
    reference_tmdb_id: int | None = None
    reference_status: Literal["absent", "resolved", "unresolved"] = "absent"
    reference_summary: str | None = None
    request_relevance_mode: Literal["broad", "query_aware"] = "broad"
    retrieval_plan: RetrievalPlan | None = None
    llm_used: bool = False
    llm_call_count: int = 0
    runtime_input_tokens: int | None = None
    runtime_output_tokens: int | None = None
    runtime_total_tokens: int | None = None


class RetrievedCandidate(CandidateContextItem):
    source_id: str | None = None
    tmdb_genres: list[str] = Field(default_factory=list)
    tmdb_original_language: str | None = None
    tmdb_production_countries: list[str] = Field(default_factory=list)
    tmdb_overview: str | None = None
    candidate_sources: list[str] = Field(default_factory=list)
    candidate_source_ranks: list[str] = Field(default_factory=list)
    document_text: str = ""
    document_hash: str = ""
    document_version: str = "film_doc_v2"
    request_relevance: float = 0.0
    request_relevance_rank: int = 0
    request_relevance_score: float = 0.0
    b3_applicability: float | None = None
    b3_applicability_distance: float | None = None
    novelty_score: float | None = None
    qualification_status: QualificationStatus = "pending"
    supported_required_aspects: list[str] = Field(default_factory=list)
    unsupported_required_aspects: list[str] = Field(default_factory=list)
    supported_preferred_aspects: list[str] = Field(default_factory=list)
    unsupported_preferred_aspects: list[str] = Field(default_factory=list)
    supported_request_aspects: list[str] = Field(default_factory=list)
    unsupported_request_aspects: list[str] = Field(default_factory=list)
    grounded_evidence: list[str] = Field(default_factory=list)
    qualification_reason: str = ""
    request_match: str | None = None
    caveat: str | None = None
    selection_rank: int | None = None
    selection_reason: str | None = None


class QualificationRecord(BaseModel):
    candidate_id: str
    qualification_status: QualificationStatus
    supported_required_aspects: list[str] = Field(default_factory=list)
    unsupported_required_aspects: list[str] = Field(default_factory=list)
    supported_request_aspects: list[str] = Field(default_factory=list)
    unsupported_request_aspects: list[str] = Field(default_factory=list)
    supported_preferred_aspects: list[str] = Field(default_factory=list)
    unsupported_preferred_aspects: list[str] = Field(default_factory=list)
    violated_semantic_exclusions: list[str] = Field(default_factory=list)
    grounded_evidence: list[str] = Field(default_factory=list)
    grounded_evidence_details: list[dict[str, Any]] = Field(default_factory=list)
    qualification_reason: str = ""
    request_match: str | None = None
    caveat: str | None = None
    llm_used: bool = False


class SelectionRecord(BaseModel):
    candidate_id: str
    selection_rank: int
    selection_reason: str = ""
    request_relevance: float = 0.0
    b3_predicted_preference: float | None = None
    b3_available: bool = False
    qualification_status: QualificationStatus = "pending"


class RuntimeV4Metadata(BaseModel):
    runtime_version: str = "runtime_v4"
    intent_prompt_version: str = "intent_v4"
    recommendation_prompt_version: str = "recommendation_v4"
    generation_model: str = "gpt-4o-mini"
    source_recommendation_run: str = "dynamic_tmdb_runtime_v4"
    run_id: str
    request_mode: RequestMode = "generic"
    interaction_mode: InteractionMode = "recommend"
    candidate_universe_count: int = 0
    post_constraint_candidate_count: int = 0
    contextual_retrieval_used: bool = False
    qualification_candidate_count: int = 0
    qualified_candidate_count: int = 0
    selected_count: int = 0
    semantic_classified_count: int = 0
    llm_call_count: int = 0
    intent_interpretation_llm_used: bool = False
    intent_interpretation_llm_call_count: int = 0
    semantic_qualification_llm_used: bool = False
    semantic_qualification_llm_call_count: int = 0
    selection_llm_used: bool = False
    selection_llm_call_count: int = 0
    explanation_llm_used: bool = False
    explanation_llm_call_count: int = 0
    runtime_llm_calls: int | None = None
    runtime_input_tokens: int | None = None
    runtime_output_tokens: int | None = None
    runtime_total_tokens: int | None = None
    judge_llm_calls: int | None = None
    repair_attempted: bool = False
    fallback_used: bool = False
    validation_passed: bool = False
    abstention_used: bool = False
    generation_status: Literal["generated", "abstained", "no_match", "fallback", "clarification", "validation_failed"] = "generated"
    hard_filters_applied: bool = False
    novelty_requested: bool = False
    reference_status: Literal["absent", "resolved", "unresolved"] = "absent"
    target_candidate_id: str | None = None
    request_relevance_mode: Literal["broad", "query_aware"] = "broad"


class RuntimeV4Result(BaseModel):
    response: RecommendationResponse
    runtime_metadata: RuntimeV4Metadata
    retrieval_diagnostics: dict[str, Any] = Field(default_factory=dict)
    qualification_records: list[QualificationRecord] = Field(default_factory=list)
    selection_records: list[SelectionRecord] = Field(default_factory=list)
    validation_passed: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


RuntimeV4RecommendationItem = RecommendationItem
RuntimeV4CandidateContextItem = RetrievedCandidate
RuntimeV4SemanticEvidence = CandidateSemanticEvidence
