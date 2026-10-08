from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from mvp.src.config import get_api_keys
from mvp.src.generative.runtime_version import GENERATION_MODEL
from mvp.src.generative_v4.schemas import (
    QualificationRecord,
    QualificationStatus,
    RetrievedCandidate,
    RequestUnderstanding,
    RequestSpec,
    SemanticConcept,
    SemanticRequirementGroup,
    StructuredConstraints,
)


DEFAULT_QUALIFICATION_MODEL = GENERATION_MODEL
SEMANTIC_QUALIFICATION_SHORTLIST_SIZE = 48
SEMANTIC_QUALIFICATION_BATCH_SIZE = 6
SEMANTIC_QUALIFICATION_MAX_WORKERS = 4

ASPECT_SYNONYMS: dict[str, list[str]] = {
    "comforting": ["comforting", "comfort", "cozy", "coziness", "warm", "gentle", "soothing"],
    "contemplative": ["contemplative", "meditative", "reflective", "thoughtful", "slow burn", "slow-burn"],
    "slow_burn": ["slow", "slow burn", "slow-burn", "slow paced", "slow pace"],
    "alternative": ["alternative", "offbeat", "unconventional", "non-mainstream"],
    "austere": ["austere", "visually austere", "minimal", "minimalist", "sparse", "ascetic"],
    "adolescence": ["adolescence", "adolescent", "teenage", "teenager", "coming of age", "coming-of-age"],
    "melancholic": ["melancholic", "melancholy", "sad", "wistful", "somber"],
    "intimate": ["intimate", "intimacy", "personal", "close"],
    "sentimental": ["sentimental", "sappy", "maudlin", "treacly"],
    "cheesy": ["cheesy", "corny", "cloying"],
    "performers": ["performer", "performers", "performance", "actor", "actress", "stage", "theatre", "theater"],
    "fame": ["fame", "famous", "celebrity", "celebrated", "star", "stars", "renown"],
    "show_business": ["show business", "entertainment", "cinema", "film industry", "behind the scenes", "industry"],
    "freedom": ["freedom", "liberation", "escape", "rebellion"],
    "youthful": ["youthful", "youth", "young", "coming of age", "coming-of-age"],
    "unsettling": ["unsettling", "anxiety", "anxious", "tense", "ominous"],
    "classic": ["classic", "canonical", "landmark", "old classic"],
    "quiet": ["quiet", "low-key", "subtle", "restrained"],
    "character_driven": ["character driven", "character-driven", "character", "persona", "intimate"],
    "mind_blown": ["mind blowing", "mind-blown", "surprising", "surprise", "out there", "experimental", "bizarre"],
    "performers": ["performer", "performers", "performance", "actor", "actress", "stage", "theatre", "theater"],
    "period_setting": ["period setting", "historical setting", "period piece", "set in the past", "set in a period"],
    "grief": ["grief", "mourning", "loss", "bereavement"],
    "filmmaking": ["filmmaking", "film making", "making films", "about filmmaking", "director", "directing", "cinema"],
    "politics": ["politics", "political", "campaign", "election", "government"],
    "quiet_domestic_routine": ["quiet domestic", "domestic routine", "domestic life", "household routine", "domestic"],
    "musicians": ["musician", "musicians", "music", "band", "songwriter", "singer"],
    "celebrity": ["celebrity", "celebrity culture", "celebrated"],
    "mind_blown": ["mind blown", "mind-blown", "mind blowing", "mind-blowing", "surprise", "surprising", "surprises me", "out there", "wild", "bizarre", "experimental"],
}

NEGATION_PATTERNS = (
    r"\b(?:not about|not actually about|no|without|excluding|must not be about)\s+([^.;!?]+)",
    r"\b(?:must not|do not want|don't want)\s+([^.;!?]+)",
)

_ABOUTNESS_MARKERS = (
    "about",
    "story",
    "stories",
    "focus",
    "focused",
    "centers on",
    "centred on",
    "explores",
    "portrait",
    "life of",
    "world of",
    "behind the scenes",
    "set in",
    "set against",
    "chronicle",
    "industry",
    "culture",
    "career",
)

_ABOUTNESS_REQUIRED_CONCEPTS = {"show_business", "filmmaking", "politics"}
_DIRECT_SUBJECT_MARKERS = (
    "stage",
    "theatre",
    "theater",
    "actor",
    "actress",
    "performer",
    "performance",
    "celebrity",
    "star",
    "music",
    "musician",
    "songwriter",
    "singer",
    "director",
    "cinema",
)
_TEMPORAL_CONCEPTS = {"classic", "historical", "period_setting", "old"}


class GroundedEvidence(BaseModel):
    aspect_id: str
    field: str
    evidence: str


class CandidateQualification(BaseModel):
    candidate_id: str
    qualification_status: QualificationStatus
    supported_required_aspects: list[str] = Field(default_factory=list)
    supported_preferred_aspects: list[str] = Field(default_factory=list)
    unsupported_required_aspects: list[str] = Field(default_factory=list)
    unsupported_preferred_aspects: list[str] = Field(default_factory=list)
    violated_semantic_exclusions: list[str] = Field(default_factory=list)
    grounded_evidence: list[GroundedEvidence] = Field(default_factory=list)
    reason: str = ""
    required_caveat: str | None = None


class QualificationBatch(BaseModel):
    items: list[CandidateQualification] = Field(default_factory=list)


@dataclass(frozen=True)
class QualificationOutput:
    candidate_id: str
    status: QualificationStatus
    supported_required_aspects: list[str]
    unsupported_required_aspects: list[str]
    supported_preferred_aspects: list[str]
    unsupported_preferred_aspects: list[str]
    violated_semantic_exclusions: list[str]
    grounded_evidence: list[str]
    grounded_evidence_details: list[dict[str, Any]]
    qualification_reason: str
    request_match: str | None
    caveat: str | None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).lower().split())


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _first_populated(*values: Any) -> Any:
    for value in values:
        if isinstance(value, list):
            cleaned = [item for item in value if not _is_missing_value(item) and str(item).strip()]
            if cleaned:
                return cleaned
            continue
        if not _is_missing_value(value) and str(value).strip():
            return value
    return None


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if not _is_missing_value(item) and str(item).strip()]
    if _is_missing_value(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.strip("[]")
    parts = [part.strip().strip("'\"") for part in text.split("|")] if "|" in text else [part.strip().strip("'\"") for part in text.split(",")]
    return [part for part in parts if part]


def _split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.strip("[]")
    parts = [part.strip().strip("'\"") for part in text.split("|")] if "|" in text else [part.strip().strip("'\"") for part in text.split(",")]
    return [part for part in parts if part]


def _canonicalize_constraint_values(values: list[str], kind: str) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        canonical = value
        if kind == "language":
            canonical = str(value).lower()
        elif kind == "country":
            canonical = str(value).upper()
        elif kind == "genre":
            canonical = str(value)
        elif kind == "decade":
            canonical = str(value)
        if canonical and canonical not in seen:
            seen.add(canonical)
            deduped.append(canonical)
    return deduped


def _canonical_concept(value: str) -> str | None:
    lower = _normalize_text(value)
    if not lower:
        return None
    if lower in {"want", "very", "high", "compatible", "must", "actually", "even", "match", "normally", "something", "film", "movie", "recommend", "recommendation", "tone", "mood", "feeling", "1", "2", "3"}:
        return None
    for canonical, aliases in ASPECT_SYNONYMS.items():
        if lower == canonical or lower in aliases:
            return canonical
    cleaned = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    if not cleaned or cleaned in {"want", "something", "film", "movie"}:
        return None
    return cleaned


def _semantic_value(concept: SemanticConcept | str | None) -> str | None:
    if concept is None:
        return None
    if isinstance(concept, SemanticConcept):
        positive, negative = _split_semantic_phrase(concept.concept or concept.aspect_id)
        return positive or negative
    positive, negative = _split_semantic_phrase(concept)
    return positive or negative


def _split_semantic_phrase(value: str | None) -> tuple[str | None, str | None]:
    normalized = _normalize_text(value).replace("_", " ")
    if not normalized:
        return None, None
    for prefix in ("not ", "no ", "without ", "avoid ", "exclude ", "excluding ", "must not ", "do not want ", "don't want "):
        if normalized.startswith(prefix):
            tail = normalized[len(prefix) :].strip()
            return None, _canonical_concept(tail)
    return _canonical_concept(normalized), None


def _candidate_year(candidate: RetrievedCandidate | pd.Series | dict[str, Any]) -> int | None:
    payload = candidate.model_dump() if isinstance(candidate, RetrievedCandidate) else candidate.to_dict() if isinstance(candidate, pd.Series) else candidate
    year = _first_populated(payload.get("year"), payload.get("release_year"))
    if year is None:
        return None
    try:
        return int(float(year))
    except Exception:
        return None


def _candidate_payload(candidate: RetrievedCandidate | pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, RetrievedCandidate):
        return candidate.model_dump()
    if isinstance(candidate, pd.Series):
        return candidate.to_dict()
    return dict(candidate)


def _candidate_field_text(candidate: RetrievedCandidate | pd.Series | dict[str, Any], field: str) -> str:
    payload = _candidate_payload(candidate)
    value = _first_populated(payload.get(field))
    if value is None:
        fallback_fields = {
            "overview": "tmdb_overview",
            "genres": "tmdb_genres",
            "original_language": "tmdb_original_language",
            "production_countries": "tmdb_production_countries",
        }
        fallback_field = fallback_fields.get(field)
        if fallback_field:
            value = _first_populated(payload.get(fallback_field))
    if value is None:
        return ""
    if isinstance(value, list):
        return _normalize_text(" ".join(str(item) for item in value if str(item).strip()))
    return _normalize_text(value)


def _candidate_grounding_text(candidate: RetrievedCandidate | pd.Series | dict[str, Any]) -> str:
    payload = _candidate_payload(candidate)
    parts = [
        _candidate_field_text(payload, "title"),
        _candidate_field_text(payload, "overview"),
        _candidate_field_text(payload, "document_text"),
        _candidate_field_text(payload, "candidate_sources"),
        _candidate_field_text(payload, "candidate_source_ranks"),
        _candidate_field_text(payload, "genres"),
        _candidate_field_text(payload, "original_language"),
        _candidate_field_text(payload, "production_countries"),
    ]
    return _normalize_text(" ".join(part for part in parts if part))


def _concept_match_state(candidate: RetrievedCandidate | pd.Series | dict[str, Any], concept: str) -> tuple[str, str | None]:
    payload = _candidate_payload(candidate)
    subject_text = _candidate_grounding_text(payload)
    overview_text = _candidate_field_text(payload, "overview") or _candidate_field_text(payload, "document_text")
    title_text = _candidate_field_text(payload, "title")
    source_text = _candidate_field_text(payload, "candidate_sources")
    candidate_text = subject_text
    synonyms = ASPECT_SYNONYMS.get(concept, [concept.replace("_", " ")])
    matched, synonym = _supported_by_synonym(candidate_text, concept)
    if not matched:
        return "unsupported", None

    if concept in _TEMPORAL_CONCEPTS:
        year = _candidate_year(candidate)
        current_year = datetime.now().year
        if year is not None and year > current_year - 15:
            return "ambiguous", synonym

    if concept in _ABOUTNESS_REQUIRED_CONCEPTS or concept in {"fame", "celebrity", "performers", "musicians"}:
        direct_markers = tuple(marker for marker in (*_ABOUTNESS_MARKERS, *_DIRECT_SUBJECT_MARKERS) if marker)
        if not any(marker in overview_text for marker in direct_markers) and not any(marker in source_text for marker in direct_markers):
            return "ambiguous", synonym
        return "supported", synonym

    overview_or_source_hit = any(alias in overview_text or alias in source_text for alias in synonyms)
    title_hit = any(alias in title_text for alias in synonyms)
    if overview_or_source_hit:
        return "supported", synonym
    if title_hit:
        return "ambiguous", synonym

    return "ambiguous", synonym


def _subject_matters_for_concept(candidate_text: str, concept: str, candidate: RetrievedCandidate | pd.Series | dict[str, Any]) -> bool:
    state, _ = _concept_match_state(candidate, concept)
    return state == "supported"


def _candidate_text(candidate: RetrievedCandidate | pd.Series | dict[str, Any]) -> str:
    if isinstance(candidate, RetrievedCandidate):
        payload = candidate.model_dump()
    elif isinstance(candidate, pd.Series):
        payload = candidate.to_dict()
    else:
        payload = candidate

    def _join(value: Any) -> str:
        if isinstance(value, list):
            return " ".join(str(item) for item in value if not _is_missing_value(item) and str(item).strip())
        if _is_missing_value(value):
            return ""
        return str(value)

    parts = [
        f"Title: {_normalize_text(payload.get('title'))}",
        f"Year: {_normalize_text(_first_populated(payload.get('year'), payload.get('release_year')))}",
        f"Genres: {_join(_first_populated(payload.get('genres'), payload.get('tmdb_genres')))}",
        f"Original language: {_normalize_text(_first_populated(payload.get('original_language'), payload.get('tmdb_original_language')))}",
        f"Production countries: {_join(_first_populated(payload.get('production_countries'), payload.get('tmdb_production_countries')))}",
        f"Overview: {_normalize_text(_first_populated(payload.get('overview'), payload.get('tmdb_overview')))}",
        f"Document: {_normalize_text(payload.get('document_text'))}",
        f"Candidate provenance: {_join(_first_populated(payload.get('candidate_provenance'), payload.get('candidate_sources')))}",
    ]
    return _normalize_text(" ".join(parts))


def _structured_constraint_satisfied(candidate: RetrievedCandidate | pd.Series | dict[str, Any], constraints: StructuredConstraints) -> tuple[bool, list[str], list[str]]:
    payload = candidate.model_dump() if isinstance(candidate, RetrievedCandidate) else candidate.to_dict() if isinstance(candidate, pd.Series) else dict(candidate)
    supported: list[str] = []
    unsupported: list[str] = []

    original_language_value = _first_populated(payload.get("original_language"), payload.get("tmdb_original_language"))
    genres_value = _first_populated(payload.get("genres"), payload.get("tmdb_genres"))
    countries_value = _first_populated(payload.get("production_countries"), payload.get("tmdb_production_countries"))
    year = _first_populated(payload.get("year"), payload.get("release_year"))
    original_language = str(original_language_value or "").lower()
    genres = {str(item).lower() for item in _normalize_list(genres_value)}
    countries = {str(item).upper() for item in _normalize_list(countries_value)}

    if constraints.required_languages:
        if original_language in {value.lower() for value in constraints.required_languages}:
            supported.extend([f"language:{original_language}"])
        else:
            unsupported.extend([f"language:{value}" for value in constraints.required_languages])

    if constraints.excluded_languages and original_language in {value.lower() for value in constraints.excluded_languages}:
        unsupported.extend([f"excluded_language:{value}" for value in constraints.excluded_languages])

    if constraints.required_countries:
        matches = countries.intersection({value.upper() for value in constraints.required_countries})
        if matches:
            supported.extend([f"country:{value}" for value in sorted(matches)])
        else:
            unsupported.extend([f"country:{value}" for value in constraints.required_countries])

    if constraints.excluded_countries and countries.intersection({value.upper() for value in constraints.excluded_countries}):
        unsupported.extend([f"excluded_country:{value}" for value in constraints.excluded_countries])

    if constraints.required_genres:
        if genres:
            matches = genres.intersection({value.lower() for value in constraints.required_genres})
            if matches:
                supported.extend([f"genre:{value}" for value in sorted(matches)])
            else:
                unsupported.extend([f"genre:{value}" for value in constraints.required_genres])
        else:
            unsupported.extend([f"genre:{value}" for value in constraints.required_genres])

    if constraints.excluded_genres:
        if not genres or genres.intersection({value.lower() for value in constraints.excluded_genres}):
            unsupported.extend([f"excluded_genre:{value}" for value in constraints.excluded_genres])

    if constraints.required_decades:
        decade = None
        if year is not None and str(year).strip():
            try:
                decade = f"{int(float(year)) // 10 * 10}s"
            except Exception:
                decade = None
        if decade and decade in constraints.required_decades:
            supported.append(f"decade:{decade}")
        else:
            unsupported.extend([f"decade:{value}" for value in constraints.required_decades])

    if constraints.excluded_decades:
        if year is not None:
            try:
                decade = f"{int(float(year)) // 10 * 10}s"
            except Exception:
                decade = None
            if decade and decade in constraints.excluded_decades:
                unsupported.extend([f"excluded_decade:{value}" for value in constraints.excluded_decades])

    if constraints.min_year is not None:
        try:
            if year is not None and int(float(year)) >= int(constraints.min_year):
                supported.append(f"min_year:{constraints.min_year}")
            else:
                unsupported.append(f"min_year:{constraints.min_year}")
        except Exception:
            unsupported.append(f"min_year:{constraints.min_year}")
    if constraints.max_year is not None:
        try:
            if year is not None and int(float(year)) <= int(constraints.max_year):
                supported.append(f"max_year:{constraints.max_year}")
            else:
                unsupported.append(f"max_year:{constraints.max_year}")
        except Exception:
            unsupported.append(f"max_year:{constraints.max_year}")

    return not unsupported, list(dict.fromkeys(supported)), list(dict.fromkeys(unsupported))


def _supported_by_synonym(candidate_text: str, concept: str) -> tuple[bool, str | None]:
    synonyms = ASPECT_SYNONYMS.get(concept, [concept.replace("_", " ")])
    for synonym in synonyms:
        if synonym in candidate_text:
            return True, synonym
    return False, None


def _extract_concepts_from_query(query: str) -> tuple[list[str], list[str]]:
    lower = _normalize_text(query)
    positive: list[str] = []
    negative: list[str] = []

    for canonical, aliases in ASPECT_SYNONYMS.items():
        if any(alias in lower for alias in aliases):
            positive.append(canonical)

    for pattern in NEGATION_PATTERNS:
        for match in re.finditer(pattern, lower):
            tail = match.group(1)
            tail = re.split(r"\b(?:but|and|or|so|while|though|yet)\b", tail, maxsplit=1)[0]
            parts = re.split(r",|\band/or\b|\band\b|\bor\b|/", tail)
            for part in parts:
                canonical = _canonical_concept(part)
                if canonical:
                    negative.append(canonical)
    return list(dict.fromkeys(positive)), list(dict.fromkeys(negative))


def _request_spec(request: RequestUnderstanding) -> RequestSpec:
    if request.spec is not None:
        return request.spec
    structured = StructuredConstraints(
        required_languages=[value.lower() for value in request.intent.requested_languages],
        required_countries=[value.upper() for value in request.intent.requested_countries],
        required_genres=[value for value in request.intent.requested_genres],
        required_decades=[value for value in request.intent.requested_decades],
    )
    semantic_requirements = []
    for value in request.intent.requested_taste_dimensions:
        canonical = _semantic_value(value)
        if canonical:
            semantic_requirements.append(
                SemanticConcept(aspect_id=canonical, concept=canonical, importance="required", source_span=request.query)
            )
    semantic_exclusions = []
    for value in request.intent.exclusions:
        canonical = _semantic_value(value)
        if canonical:
            semantic_exclusions.append(
                SemanticConcept(aspect_id=canonical, concept=canonical, importance="required", source_span=request.query)
            )
    return RequestSpec(
        intent_type=request.intent.intent_type,
        mode="contextual" if any([structured.required_languages, structured.required_countries, structured.required_genres, structured.required_decades, semantic_requirements, semantic_exclusions]) else "generic",
        structured_constraints=structured,
        semantic_requirements=semantic_requirements,
        semantic_exclusions=semantic_exclusions,
        reference=ReferenceSpec(
            title=request.reference_title,
            resolved_tmdb_id=request.reference_tmdb_id,
            resolved_tmdb_title=request.reference_summary,
            resolution_status=request.reference_status,
        ),
        novelty_goal=NoveltyGoal(enabled=request.novelty_requested),
        personalization_instruction=PersonalizationInstruction(),
        request_relevance_mode=request.request_relevance_mode,
        semantic_query_text=request.query,
        request_summary=request.query,
    )


def _fallback_qualification_output(request: RequestUnderstanding, candidate: RetrievedCandidate | pd.Series | dict[str, Any]) -> QualificationOutput:
    spec = _request_spec(request)
    candidate_id = str(_normalize_candidate_payload(candidate).get("candidate_id") or _normalize_candidate_payload(candidate).get("source_id") or "")
    candidate_text = _candidate_text(candidate)
    structured_ok, structured_supported, structured_unsupported = _structured_constraint_satisfied(candidate, spec.structured_constraints)
    groups = _semantic_requirement_groups(spec)
    required_aspects = [_semantic_value(concept) for group in groups if group.mode in {"all_of", "any_of"} for concept in group.concepts]
    preferred_aspects = [_semantic_value(concept) for group in groups if group.mode == "preferred" for concept in group.concepts]
    excluded_concepts = [_semantic_value(concept) for concept in spec.semantic_exclusions]

    supported_required: list[str] = list(structured_supported)
    supported_preferred: list[str] = []
    unsupported_required: list[str] = list(structured_unsupported)
    unsupported_preferred: list[str] = []
    violated_exclusions: list[str] = []
    evidence: list[str] = []
    evidence_details: list[dict[str, Any]] = []

    def _append_evidence(concept_id: str, synonym: str | None, field: str) -> None:
        evidence.append(f"{concept_id} via {synonym or concept_id}")
        evidence_details.append({"aspect_id": concept_id, "field": field, "evidence": synonym or concept_id})

    for group in groups:
        group_supported = False
        group_missing: list[str] = []
        for concept in group.concepts:
            concept_id = _semantic_value(concept)
            if not concept_id:
                continue
            match_state, synonym = _concept_match_state(candidate, concept_id)
            if match_state == "supported":
                group_supported = True
                if group.mode == "preferred":
                    supported_preferred.append(concept_id)
                else:
                    supported_required.append(concept_id)
                _append_evidence(concept_id, synonym, "overview")
            elif match_state == "ambiguous":
                group_missing.append(concept_id)
                if group.mode == "preferred":
                    unsupported_preferred.append(concept_id)
                else:
                    unsupported_required.append(concept_id)
            else:
                group_missing.append(concept_id)
                if group.mode == "preferred":
                    unsupported_preferred.append(concept_id)
                else:
                    unsupported_required.append(concept_id)
        if group.mode == "all_of" and group_missing:
            unsupported_required.extend(group_missing)
        elif group.mode == "any_of" and not group_supported:
            unsupported_required.extend(group_missing or [_semantic_value(concept) for concept in group.concepts if _semantic_value(concept)])
        elif group.mode == "preferred":
            unsupported_preferred.extend(group_missing)

    for concept in preferred_aspects:
        if not concept:
            continue
        match_state, synonym = _concept_match_state(candidate, concept)
        if match_state == "supported":
            supported_preferred.append(concept)
            _append_evidence(concept, synonym, "overview")
        else:
            unsupported_preferred.append(concept)

    for concept in excluded_concepts:
        if not concept:
            continue
        match_state, synonym = _concept_match_state(candidate, concept)
        if match_state == "supported":
            violated_exclusions.append(concept)
            evidence_details.append({"aspect_id": concept, "field": "overview", "evidence": synonym or concept})

    candidate_sources_value = candidate.model_dump().get("candidate_sources") if isinstance(candidate, RetrievedCandidate) else candidate.get("candidate_sources") if isinstance(candidate, dict) else candidate.to_dict().get("candidate_sources")
    if spec.reference.resolution_status == "resolved" and any("reference_recommendations" in str(value) for value in _split_values(candidate_sources_value)):
        supported_preferred.append("reference_similarity")
        evidence.append("reference_recommendations provenance")
        evidence_details.append({"aspect_id": "reference_similarity", "field": "candidate_sources", "evidence": "reference_recommendations"})

    if spec.novelty_goal.enabled:
        supported_preferred.append("novelty_requested")

    required_group_unsatisfied = False
    for group in groups:
        if group.mode not in {"all_of", "any_of"}:
            continue
        group_concepts = [_semantic_value(concept) for concept in group.concepts if _semantic_value(concept)]
        if not group_concepts:
            continue
        if group.mode == "all_of" and not set(group_concepts).issubset(set(supported_required)):
            required_group_unsatisfied = True
            break
        if group.mode == "any_of" and not any(concept in supported_required for concept in group_concepts):
            required_group_unsatisfied = True
            break
    required_failure = bool(structured_unsupported or required_group_unsatisfied)

    if violated_exclusions:
        status: QualificationStatus = "unsupported"
    elif structured_unsupported:
        status = "unsupported"
    elif required_failure and required_aspects:
        status = "unsupported"
    elif unsupported_preferred and not supported_preferred and not required_aspects:
        status = "strong"
    elif required_aspects or preferred_aspects or spec.semantic_exclusions:
        status = "strong" if not required_failure and not violated_exclusions else "unsupported"
    elif not structured_unsupported and not violated_exclusions:
        status = "strong"
    else:
        status = "partial"

    if status == "unsupported":
        reason = "Grounded evidence was insufficient to support the request, or the candidate violated an explicit semantic exclusion."
        caveat = (
            f"Semantic exclusions violated: {', '.join(violated_exclusions[:5])}."
            if violated_exclusions
            else (
                f"Unsupported structured constraints: {', '.join((structured_unsupported or unsupported_required)[:5])}."
                if (structured_unsupported or unsupported_required)
                else "No grounded request match could be established."
            )
        )
        request_match = None
    elif status == "partial":
        reason = "Candidate grounds part of the request, but not all requested semantics are directly supported."
        caveat = (
            f"Unsupported aspects: {', '.join((unsupported_required or unsupported_preferred)[:5])}"
            if (unsupported_required or unsupported_preferred)
            else "Partial grounding only."
        )
        request_match = f"Supported request aspects: {', '.join((supported_required or supported_preferred)[:5])}" if (supported_required or supported_preferred) else None
    else:
        reason = "Candidate has grounded evidence that directly supports the request."
        request_match = f"Supported request aspects: {', '.join((supported_required or supported_preferred)[:5])}" if (supported_required or supported_preferred) else "Grounded request match."
        caveat = None

    return QualificationOutput(
        candidate_id=candidate_id,
        status=status,
        supported_required_aspects=list(dict.fromkeys(supported_required)),
        unsupported_required_aspects=list(dict.fromkeys(unsupported_required)),
        supported_preferred_aspects=list(dict.fromkeys(supported_preferred)),
        unsupported_preferred_aspects=list(dict.fromkeys(unsupported_preferred)),
        violated_semantic_exclusions=list(dict.fromkeys(violated_exclusions)),
        grounded_evidence=evidence[:5],
        grounded_evidence_details=evidence_details[:5],
        qualification_reason=reason,
        request_match=request_match,
        caveat=caveat,
    )


def _normalize_candidate_payload(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    payload = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    def _clean(value: Any) -> Any:
        if isinstance(value, list):
            cleaned = []
            for item in value:
                normalized = _clean(item)
                if normalized is not None:
                    cleaned.append(normalized)
            return cleaned
        try:
            if value is None or pd.isna(value):
                return None
        except Exception:
            pass
        return value

    payload = {key: _clean(value) for key, value in payload.items()}
    for key in ("genres", "tmdb_genres", "production_countries", "tmdb_production_countries", "candidate_provenance", "candidate_sources", "candidate_source_ranks"):
        if key in payload:
            payload[key] = _split_values(payload[key])
    if not payload.get("genres"):
        payload["genres"] = _split_values(payload.get("tmdb_genres"))
    if not payload.get("original_language"):
        payload["original_language"] = payload.get("tmdb_original_language")
    if not payload.get("production_countries"):
        payload["production_countries"] = _split_values(payload.get("tmdb_production_countries"))
    if "original_language" not in payload and "tmdb_original_language" in payload:
        payload["original_language"] = payload["tmdb_original_language"]
    if "overview" not in payload and "tmdb_overview" in payload:
        payload["overview"] = payload["tmdb_overview"]
    if "year" not in payload and "release_year" in payload:
        payload["year"] = payload["release_year"]
    if not payload.get("candidate_provenance") and payload.get("candidate_sources"):
        payload["candidate_provenance"] = list(payload.get("candidate_sources") or [])
    if "source_id" not in payload and payload.get("candidate_id") is not None:
        payload["source_id"] = str(payload["candidate_id"])
    if "candidate_id" not in payload and payload.get("source_id") is not None:
        payload["candidate_id"] = str(payload["source_id"])
    if "raw_rank" not in payload and payload.get("rank") is not None:
        payload["raw_rank"] = int(payload["rank"])
    if "raw_rank" not in payload:
        payload["raw_rank"] = 0
    return payload


def _build_candidate_document(candidate: dict[str, Any]) -> str:
    parts = [
        f"Title: {candidate.get('title') or ''}",
        f"Year: {_first_populated(candidate.get('year'), candidate.get('release_year')) or ''}",
        f"Structured facts: genres={_first_populated(candidate.get('genres'), candidate.get('tmdb_genres')) or []}; language={_first_populated(candidate.get('original_language'), candidate.get('tmdb_original_language')) or ''}; countries={_first_populated(candidate.get('production_countries'), candidate.get('tmdb_production_countries')) or []}.",
        f"Subject / content evidence: {_first_populated(candidate.get('overview'), candidate.get('tmdb_overview')) or ''}",
        f"Candidate provenance: {_first_populated(candidate.get('candidate_provenance'), candidate.get('candidate_sources')) or []}",
    ]
    return _normalize_text(" ".join(str(part) for part in parts if str(part).strip()))


def _candidate_payload_for_llm(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id") or candidate.get("source_id"),
        "title": candidate.get("title"),
        "structured_facts": {
            "year": _first_populated(candidate.get("year"), candidate.get("release_year")),
            "genres": _normalize_list(_first_populated(candidate.get("genres"), candidate.get("tmdb_genres"))),
            "original_language": _first_populated(candidate.get("original_language"), candidate.get("tmdb_original_language")),
            "production_countries": _normalize_list(_first_populated(candidate.get("production_countries"), candidate.get("tmdb_production_countries"))),
        },
        "subject_evidence": _first_populated(candidate.get("overview"), candidate.get("tmdb_overview")) or "",
        "candidate_provenance": _first_populated(candidate.get("candidate_provenance"), candidate.get("candidate_sources")) or [],
        "document_text": _build_candidate_document(candidate),
    }


def _request_payload_for_llm(request: RequestUnderstanding) -> dict[str, Any]:
    spec = _request_spec(request)
    return {
        "query": request.query,
        "intent_type": spec.intent_type,
        "mode": spec.mode,
        "structured_constraints": spec.structured_constraints.model_dump(),
        "semantic_requirement_groups": [group.model_dump() for group in spec.semantic_requirement_groups],
        "semantic_requirements": [concept.model_dump() for concept in spec.semantic_requirements],
        "semantic_exclusions": [concept.model_dump() for concept in spec.semantic_exclusions],
        "reference": spec.reference.model_dump(),
        "novelty_goal": spec.novelty_goal.model_dump(),
        "personalization_instruction": spec.personalization_instruction.model_dump(),
        "request_relevance_mode": spec.request_relevance_mode,
        "semantic_query_text": spec.semantic_query_text,
    }


def _semantic_requirement_groups(spec: RequestSpec) -> list[SemanticRequirementGroup]:
    if spec.semantic_requirement_groups:
        return spec.semantic_requirement_groups
    required = [concept for concept in spec.semantic_requirements if concept.importance == "required"]
    preferred = [concept for concept in spec.semantic_requirements if concept.importance != "required"]
    groups: list[SemanticRequirementGroup] = []
    if required:
        groups.append(
            SemanticRequirementGroup(
                group_id="required_semantics",
                mode="all_of",
                concepts=required,
                source_span=required[0].source_span or "",
            )
        )
    if preferred:
        groups.append(
            SemanticRequirementGroup(
                group_id="preferred_semantics",
                mode="preferred",
                concepts=preferred,
                source_span=preferred[0].source_span or "",
            )
        )
    return groups


def _qualify_with_llm(request: RequestUnderstanding, candidate_rows: list[dict[str, Any]]) -> tuple[list[QualificationOutput] | None, int, int | None, int | None, int | None]:
    keys = get_api_keys()
    if not keys.openai_api_key or not candidate_rows:
        return None, 0, None, None, None

    llm = ChatOpenAI(model=DEFAULT_QUALIFICATION_MODEL, temperature=0, api_key=keys.openai_api_key)
    structured = llm.with_structured_output(QualificationBatch, include_raw=True)
    request_payload = _request_payload_for_llm(request)
    prompt = (
        "You are conservatively qualifying film candidates against a structured semantic request.\n"
        "Use only the supplied evidence.\n"
        "Guidance:\n"
        "- Structured constraints are hard constraints.\n"
        "- Semantic exclusions are admission blockers.\n"
        "- Do not infer subject matter from fame of cast/crew alone.\n"
        "- Be conservative: if grounded evidence does not establish the requested meaning, mark it unsupported.\n"
        "- A resolved reference may be used only from the supplied reference metadata.\n"
        "- Return candidate-level structured judgments only.\n"
    )
    try:
        result = structured.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        "Request payload:\n"
                        f"{request_payload}\n\n"
                        "Candidate payloads:\n"
                        f"{candidate_rows}\n"
                    )
                ),
            ]
        )
    except Exception:
        return None, 1, None, None, None
    parsed = result.get("parsed") if isinstance(result, dict) else result
    raw = result.get("raw") if isinstance(result, dict) else None
    usage_metadata = getattr(raw, "usage_metadata", None) if raw is not None else None
    input_tokens = usage_metadata.get("input_tokens") if isinstance(usage_metadata, dict) else None
    output_tokens = usage_metadata.get("output_tokens") if isinstance(usage_metadata, dict) else None
    total_tokens = usage_metadata.get("total_tokens") if isinstance(usage_metadata, dict) else None
    if parsed is None:
        return None, 1, input_tokens, output_tokens, total_tokens
    items = [
        QualificationOutput(
            candidate_id=str(item.candidate_id),
            status=item.qualification_status,
            supported_required_aspects=list(item.supported_required_aspects),
            unsupported_required_aspects=list(item.unsupported_required_aspects),
            supported_preferred_aspects=list(item.supported_preferred_aspects),
            unsupported_preferred_aspects=list(item.unsupported_preferred_aspects),
            violated_semantic_exclusions=list(item.violated_semantic_exclusions),
            grounded_evidence=[f"{evidence.aspect_id}:{evidence.field}:{evidence.evidence}" for evidence in item.grounded_evidence],
            grounded_evidence_details=[evidence.model_dump() for evidence in item.grounded_evidence],
            qualification_reason=item.reason,
            request_match=f"Supported request aspects: {', '.join(item.supported_required_aspects[:5])}" if item.supported_required_aspects else (
                f"Supported request aspects: {', '.join(item.supported_preferred_aspects[:5])}" if item.supported_preferred_aspects else None
            ),
            caveat=item.required_caveat,
        )
        for item in parsed.items
    ]
    return items, 1, input_tokens, output_tokens, total_tokens


def _record_from_output(candidate_id: str, output: QualificationOutput, llm_used: bool) -> QualificationRecord:
    supported = list(dict.fromkeys(output.supported_required_aspects + output.supported_preferred_aspects))
    unsupported = list(dict.fromkeys(output.unsupported_required_aspects + output.unsupported_preferred_aspects))
    return QualificationRecord(
        candidate_id=candidate_id,
        qualification_status=output.status,
        supported_required_aspects=output.supported_required_aspects,
        unsupported_required_aspects=output.unsupported_required_aspects,
        supported_request_aspects=supported,
        unsupported_request_aspects=unsupported,
        supported_preferred_aspects=output.supported_preferred_aspects,
        unsupported_preferred_aspects=output.unsupported_preferred_aspects,
        violated_semantic_exclusions=output.violated_semantic_exclusions,
        grounded_evidence=output.grounded_evidence,
        grounded_evidence_details=output.grounded_evidence_details,
        qualification_reason=output.qualification_reason,
        request_match=output.request_match,
        caveat=output.caveat,
        llm_used=llm_used,
    )


def qualify_candidates(
    request: RequestUnderstanding,
    candidate_frame: pd.DataFrame,
    *,
    max_candidates: int = SEMANTIC_QUALIFICATION_SHORTLIST_SIZE,
) -> tuple[pd.DataFrame, list[QualificationRecord]]:
    frame = candidate_frame.copy().reset_index(drop=True)
    if frame.empty:
        return frame, []

    if "predicted_preference" not in frame.columns:
        frame["predicted_preference"] = np.nan
    frame["predicted_preference"] = pd.to_numeric(frame["predicted_preference"], errors="coerce")
    if "request_relevance" in frame.columns:
        frame = frame.sort_values(
            ["request_relevance", "predicted_preference", "source_id"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    else:
        frame = frame.sort_values(["predicted_preference", "source_id"], ascending=[False, True], na_position="last").reset_index(drop=True)

    spec = _request_spec(request)

    records: list[QualificationRecord] = []
    qualified_rows = []
    total_llm_call_count = 0
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    candidate_subset = frame.head(max_candidates).copy().reset_index(drop=True)
    if candidate_subset.empty:
        qualified_frame = base_frame = frame.iloc[0:0].copy()
        qualified_frame.attrs["llm_usage"] = {"llm_call_count": 0, "runtime_input_tokens": None, "runtime_output_tokens": None, "runtime_total_tokens": None}
        return qualified_frame, []

    base_frame = candidate_subset.copy()

    # Each qualification batch is independent: same request, disjoint candidate rows,
    # no shared ranking state. Running these calls concurrently preserves the exact
    # candidate set, prompt, model, batch size, and downstream ordering while removing
    # avoidable network wait time. Keep concurrency deliberately bounded to reduce
    # rate-limit risk.
    batch_jobs: list[tuple[int, pd.DataFrame, list[dict[str, Any]], list[str]]] = []
    for batch_index, start in enumerate(
        range(0, len(candidate_subset), SEMANTIC_QUALIFICATION_BATCH_SIZE)
    ):
        batch = candidate_subset.iloc[
            start : start + SEMANTIC_QUALIFICATION_BATCH_SIZE
        ].copy().reset_index(drop=True)
        candidate_payloads = [
            _candidate_payload_for_llm(_normalize_candidate_payload(row))
            for _, row in batch.iterrows()
        ]
        candidate_ids = [
            str(payload.get("candidate_id") or payload.get("source_id") or "")
            for payload in candidate_payloads
        ]
        duplicate_ids = sorted(
            {candidate_id for candidate_id in candidate_ids if candidate_ids.count(candidate_id) > 1}
        )
        if duplicate_ids:
            raise RuntimeError(
                f"Qualification batch contains duplicate candidate IDs: {duplicate_ids}"
            )
        batch_jobs.append((batch_index, batch, candidate_payloads, candidate_ids))

    def _run_batch(job):
        batch_index, batch, candidate_payloads, candidate_ids = job
        llm_outputs, llm_call_count, input_tokens, output_tokens, batch_total_tokens = (
            _qualify_with_llm(request, candidate_payloads)
        )
        return (
            batch_index,
            batch,
            candidate_ids,
            llm_outputs,
            llm_call_count,
            input_tokens,
            output_tokens,
            batch_total_tokens,
        )

    max_workers = min(SEMANTIC_QUALIFICATION_MAX_WORKERS, len(batch_jobs))
    if max_workers <= 1:
        batch_results = [_run_batch(job) for job in batch_jobs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map preserves input order; sorting again below is defensive and
            # makes the ordering guarantee explicit for downstream record construction.
            batch_results = list(executor.map(_run_batch, batch_jobs))

    batch_results.sort(key=lambda item: item[0])

    for (
        _,
        batch,
        candidate_ids,
        llm_outputs,
        llm_call_count,
        input_tokens,
        output_tokens,
        batch_total_tokens,
    ) in batch_results:
        total_llm_call_count += int(llm_call_count or 0)
        if total_input_tokens is None:
            total_input_tokens = input_tokens
        elif input_tokens is not None:
            total_input_tokens += int(input_tokens)
        if total_output_tokens is None:
            total_output_tokens = output_tokens
        elif output_tokens is not None:
            total_output_tokens += int(output_tokens)
        if total_tokens is None:
            total_tokens = batch_total_tokens
        elif batch_total_tokens is not None:
            total_tokens += int(batch_total_tokens)

        llm_used = llm_outputs is not None
        output_lookup: dict[str, QualificationOutput] = {}
        if llm_outputs is not None:
            parsed_ids = [str(item.candidate_id) for item in llm_outputs]
            parsed_lookup = {str(item.candidate_id): item for item in llm_outputs}
            duplicate_parsed_ids = sorted(
                {candidate_id for candidate_id in parsed_ids if parsed_ids.count(candidate_id) > 1}
            )
            missing_ids = sorted(set(candidate_ids).difference(parsed_lookup))
            extra_ids = sorted(set(parsed_lookup).difference(candidate_ids))
            if duplicate_parsed_ids or missing_ids or extra_ids:
                raise RuntimeError(
                    "Qualification LLM returned mismatched candidate IDs: "
                    f"duplicates={duplicate_parsed_ids}, missing={missing_ids}, extra={extra_ids}"
                )
            output_lookup = parsed_lookup

        for _, row in batch.iterrows():
            candidate_payload = _normalize_candidate_payload(row)
            if candidate_payload.get("predicted_preference") is None:
                candidate_payload["predicted_preference"] = 0.0
            if candidate_payload.get("raw_rank") is None:
                candidate_payload["raw_rank"] = 0
            if candidate_payload.get("request_relevance_score") is None:
                candidate_payload["request_relevance_score"] = float(
                    candidate_payload.get("request_relevance") or 0.0
                )
            if candidate_payload.get("request_relevance_rank") is None:
                candidate_payload["request_relevance_rank"] = 0
            candidate = RetrievedCandidate.model_validate(candidate_payload)
            output = output_lookup.get(candidate.candidate_id) if output_lookup else None
            if output is None:
                output = _fallback_qualification_output(request, candidate)
            structured_ok, structured_supported, structured_unsupported = (
                _structured_constraint_satisfied(candidate, spec.structured_constraints)
            )
            if (
                not structured_ok or output.violated_semantic_exclusions
            ) and output.status != "unsupported":
                violated = list(dict.fromkeys(output.violated_semantic_exclusions))
                if not structured_ok:
                    violated.extend(structured_unsupported)
                output = QualificationOutput(
                    candidate_id=output.candidate_id,
                    status="unsupported",
                    supported_required_aspects=list(
                        dict.fromkeys(
                            [*output.supported_required_aspects, *structured_supported]
                        )
                    ),
                    unsupported_required_aspects=list(
                        dict.fromkeys(
                            [*output.unsupported_required_aspects, *structured_unsupported]
                        )
                    ),
                    supported_preferred_aspects=output.supported_preferred_aspects,
                    unsupported_preferred_aspects=output.unsupported_preferred_aspects,
                    violated_semantic_exclusions=list(dict.fromkeys(violated)),
                    grounded_evidence=output.grounded_evidence,
                    grounded_evidence_details=output.grounded_evidence_details,
                    qualification_reason="Candidate violates a structured hard constraint.",
                    request_match=None,
                    caveat=(
                        f"Unsupported structured constraints: {', '.join((structured_unsupported or structured_supported)[:5])}"
                        if (structured_supported or structured_unsupported)
                        else "Candidate violates a structured hard constraint."
                    ),
                )
            record = _record_from_output(
                candidate.candidate_id, output, llm_used=llm_used
            )
            records.append(record)
            qualified_row = dict(candidate_payload)
            qualified_row.update(record.model_dump())
            qualified_row["candidate_id"] = candidate.candidate_id
            qualified_row["source_id"] = candidate.candidate_id
            if record.qualification_status != "unsupported":
                qualified_rows.append(qualified_row)

    qualified_frame = pd.DataFrame(qualified_rows)
    if qualified_frame.empty:
        qualified_frame = base_frame.iloc[0:0].copy()
    llm_usage = {
        "llm_call_count": int(total_llm_call_count),
        "runtime_input_tokens": total_input_tokens,
        "runtime_output_tokens": total_output_tokens,
        "runtime_total_tokens": total_tokens,
    }
    if not qualified_frame.empty and "request_relevance" in qualified_frame.columns:
        qualified_frame = qualified_frame.sort_values(
            ["qualification_status", "request_relevance", "predicted_preference", "source_id"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    qualified_frame.attrs["llm_usage"] = llm_usage
    return qualified_frame, records
