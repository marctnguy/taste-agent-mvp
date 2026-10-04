from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from mvp.src.config import get_api_keys
from mvp.src.generative.context_builder import parse_intent as legacy_parse_intent
from mvp.src.generative.runtime_version import GENERATION_MODEL
from mvp.src.generative_v4.schemas import (
    KeywordResolution,
    NoveltyGoal,
    PersonalizationInstruction,
    ReferenceSpec,
    RequestSpec,
    RequestUnderstanding,
    RetrievalPlan,
    SemanticConcept,
    SemanticRequirementGroup,
    StructuredConstraints,
)
from mvp.src.prepare import TMDBClient


REQUEST_MODEL = GENERATION_MODEL

_REFERENCE_PATTERNS = (
    r"\blike\s+([^,.;!?]+)",
    r"\bsimilar to\s+([^,.;!?]+)",
    r"\bbased on\s+([^,.;!?]+)",
    r"\bafter watching\s+([^,.;!?]+)",
    r"\bi just watched\s+([^,.;!?]+)",
    r"\bi watched\s+([^,.;!?]+)",
    r"\bwatching\s+([^,.;!?]+)",
)

_LANGUAGE_ALIASES = {
    "french": "fr",
    "english": "en",
    "japanese": "ja",
    "italian": "it",
    "spanish": "es",
    "german": "de",
    "korean": "ko",
    "portuguese": "pt",
    "hindi": "hi",
}

_COUNTRY_ALIASES = {
    "france": "FR",
    "japan": "JP",
    "united states": "US",
    "usa": "US",
    "u.s.": "US",
    "uk": "GB",
    "united kingdom": "GB",
    "britain": "GB",
    "spain": "ES",
    "italy": "IT",
    "germany": "DE",
    "korea": "KR",
}

_GENRE_ALIASES = {
    "comedy": "Comedy",
    "drama": "Drama",
    "thriller": "Thriller",
    "documentary": "Documentary",
    "crime": "Crime",
    "mystery": "Mystery",
    "action": "Action",
    "romance": "Romance",
    "history": "History",
    "horror": "Horror",
    "science fiction": "Science Fiction",
    "sci fi": "Science Fiction",
    "animation": "Animation",
}

_SEMANTIC_ALIASES: dict[str, list[str]] = {
    "comforting": ["comforting", "comfort", "cozy", "coziness", "warm", "gentle", "soothing"],
    "performers": ["performer", "performers", "performance", "actor", "actress", "stage", "theatre", "theater"],
    "fame": ["fame", "famous", "star", "stars", "renown"],
    "show_business": ["show business", "entertainment", "cinema", "film industry", "behind the scenes", "industry"],
    "period_setting": ["period setting", "historical setting", "period piece", "set in the past", "set in a period"],
    "melancholic": ["melancholic", "melancholy", "sad", "wistful", "somber"],
    "intimate": ["intimate", "intimacy", "personal", "close"],
    "grief": ["grief", "mourning", "loss", "bereavement"],
    "filmmaking": ["filmmaking", "film making", "making films", "about filmmaking", "director", "directing", "cinema"],
    "politics": ["politics", "political", "campaign", "election", "government"],
    "quiet_domestic_routine": ["quiet domestic", "domestic routine", "domestic life", "household routine", "domestic"],
    "unsettling": ["unsettling", "anxiety", "anxious", "tense", "ominous"],
    "mind_blown": ["mind blown", "mind-blown", "mind blowing", "mind-blowing", "surprise", "surprising", "surprises me", "out there", "wild", "bizarre", "experimental"],
    "musicians": ["musician", "musicians", "music", "band", "songwriter", "singer"],
    "celebrity": ["celebrity", "celebrity culture", "celebrated"],
    "historical": ["historical", "history", "period", "old", "past"],
}

_FILLER_CONCEPTS = {
    "want",
    "very",
    "high",
    "compatible",
    "must",
    "actually",
    "even",
    "match",
    "normally",
    "something",
    "film",
    "movie",
    "recommend",
    "recommendation",
    "tone",
    "mood",
    "feeling",
    "1",
    "2",
    "3",
}

_EXPLICIT_NOVELTY_PHRASES = (
    "different from what i normally watch",
    "different from what i usually watch",
    "something different from what i normally watch",
    "something different from what i usually watch",
    "outside my usual",
    "outside my normal",
    "surprise me",
    "something novel",
    "more novel than usual",
    "less familiar than usual",
    "not something i'd normally watch",
    "not something i would normally watch",
)

_VALID_INTENT_TYPES = {"GENERAL_DISCOVERY", "MOOD_THEME", "NOVELTY", "CONSTRAINT", "EXPLAIN"}
_EXPLANATION_PHRASES = (
    "why are you recommending this",
    "why are you recommending that",
    "why did you recommend this",
    "why did this one rank highly",
    "why this film",
    "why this one",
    "why does this fit",
    "tell me why this film",
    "tell me why this one",
    "explain this recommendation",
    "explain why this",
    "explain why you picked this",
)
_PROVENANCE_PHRASES = (
    "goodreads",
    "book history",
    "book ratings",
    "reading history",
    "reading taste",
    "because of my books",
    "because of my goodreads history",
    "influence my recommendations",
    "influence these recommendations",
)
_PROFILE_BOUNDARY_PHRASES = (
    "personality",
    "identity",
    "what kind of person",
    "what does this say about me",
    "what do these recommendations say about me",
    "what do these recommendations say about my personality",
    "sexual orientation",
    "political beliefs",
    "religion",
    "ethnicity",
    "mental health",
    "diagnosis",
)

_EUROPEAN_PRODUCTION_COUNTRIES = {
    "AL",
    "AD",
    "AT",
    "BE",
    "BG",
    "BA",
    "BY",
    "CH",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GR",
    "HR",
    "HU",
    "IE",
    "IS",
    "IT",
    "LT",
    "LU",
    "LV",
    "MC",
    "ME",
    "MK",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "RS",
    "SE",
    "SI",
    "SK",
    "SM",
    "UA",
}

_REGION_TO_COUNTRIES = {
    "europe": sorted(_EUROPEAN_PRODUCTION_COUNTRIES),
    "european": sorted(_EUROPEAN_PRODUCTION_COUNTRIES),
}

_REFERENCE_REJECTION_PHRASES = (
    "what you know about my taste",
    "what you know about my preferences",
    "based on what you know about my taste",
    "based on what you know about my preferences",
    "my taste",
    "my preferences",
    "my preference",
    "your taste",
    "your preferences",
    "freedom",
    "summer",
    "grief",
    "nostalgia",
)

_REFERENCE_REJECTION_TOKENS = {
    "taste",
    "preferences",
    "preference",
    "freedom",
    "summer",
    "grief",
    "nostalgia",
    "personality",
    "identity",
}

_NEGATION_PREFIXES = (
    "not ",
    "no ",
    "without ",
    "avoid ",
    "exclude ",
    "excluding ",
    "must not ",
    "do not want ",
    "don't want ",
)


@dataclass(frozen=True)
class ReferenceResolution:
    title: str | None
    tmdb_id: int | None
    status: str
    summary: str | None


@dataclass(frozen=True)
class RequestInterpretation:
    spec: RequestSpec
    request: RequestUnderstanding
    llm_used: bool
    llm_call_count: int
    runtime_input_tokens: int | None
    runtime_output_tokens: int | None
    runtime_total_tokens: int | None


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().split())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _canonical_semantic_value(value: str | None) -> str | None:
    if value is None:
        return None
    canonical = _canonical_concept(value)
    return canonical or None


def _split_semantic_phrase(value: str | None) -> tuple[str | None, str | None]:
    normalized = _normalize_text(value or "").replace("_", " ")
    if not normalized:
        return None, None
    for prefix in _NEGATION_PREFIXES:
        if normalized.startswith(prefix):
            tail = normalized[len(prefix) :].strip()
            return None, _canonical_semantic_value(tail)
    return _canonical_semantic_value(normalized), None


def _canonicalize_semantic_concepts(concepts: list[SemanticConcept], *, source_span: str) -> list[SemanticConcept]:
    canonicalized: list[SemanticConcept] = []
    seen: set[str] = set()
    for concept in concepts:
        canonical, negated = _split_semantic_phrase(concept.concept or concept.aspect_id)
        if negated:
            canonical = None
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        canonicalized.append(
            SemanticConcept(
                aspect_id=canonical,
                concept=canonical,
                importance=concept.importance,
                source_span=concept.source_span or source_span,
            )
    )
    return canonicalized


def _canonicalize_semantic_groups_with_negations(groups: list[SemanticRequirementGroup], *, query: str) -> tuple[list[SemanticRequirementGroup], list[SemanticConcept]]:
    canonical_groups: list[SemanticRequirementGroup] = []
    extracted_exclusions: list[SemanticConcept] = []
    seen_exclusions: set[str] = set()
    for group in groups:
        concepts: list[SemanticConcept] = []
        for concept in group.concepts:
            canonical, negated = _split_semantic_phrase(concept.concept or concept.aspect_id)
            if negated:
                if negated not in seen_exclusions:
                    seen_exclusions.add(negated)
                    extracted_exclusions.append(
                        SemanticConcept(
                            aspect_id=negated,
                            concept=negated,
                            importance="required",
                            source_span=concept.source_span or group.source_span or query,
                        )
                    )
                continue
            if not canonical:
                continue
            concepts.append(
                SemanticConcept(
                    aspect_id=canonical,
                    concept=canonical,
                    importance=concept.importance,
                    source_span=concept.source_span or group.source_span or query,
                )
            )
        if not concepts:
            continue
        canonical_groups.append(
            SemanticRequirementGroup(
                group_id=group.group_id,
                mode=group.mode,
                concepts=concepts,
                source_span=group.source_span or query,
            )
        )
    return canonical_groups, extracted_exclusions


def _canonicalize_semantic_groups(groups: list[SemanticRequirementGroup], *, query: str) -> list[SemanticRequirementGroup]:
    canonical_groups: list[SemanticRequirementGroup] = []
    for group in groups:
        concepts = _canonicalize_semantic_concepts(group.concepts, source_span=group.source_span or query)
        if not concepts:
            continue
        canonical_groups.append(
            SemanticRequirementGroup(
                group_id=group.group_id,
                mode=group.mode,
                concepts=concepts,
                source_span=group.source_span or query,
            )
        )
    return canonical_groups


def _has_meaningful_semantic_intent(spec: RequestSpec) -> bool:
    if spec.semantic_requirement_groups or spec.semantic_requirements or spec.semantic_exclusions:
        return True
    constraints = spec.structured_constraints
    return bool(
        constraints.required_languages
        or constraints.excluded_languages
        or constraints.required_countries
        or constraints.excluded_countries
        or constraints.required_genres
        or constraints.excluded_genres
        or constraints.required_decades
        or constraints.excluded_decades
        or constraints.min_year is not None
        or constraints.max_year is not None
    )


def _is_plausible_reference_span(span: str) -> bool:
    normalized = _normalize_text(span)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _REFERENCE_REJECTION_PHRASES):
        return False
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return False
    if len(tokens) > 6:
        return False
    if len(tokens) == 1 and tokens[0] in _REFERENCE_REJECTION_TOKENS:
        return False
    if any(token in _REFERENCE_REJECTION_TOKENS for token in tokens) and any(
        phrase in normalized for phrase in ("what you know", "my taste", "my preferences", "your taste", "your preferences")
    ):
        return False
    return True


def _canonical_language(value: str) -> str | None:
    lower = value.lower()
    if lower in _LANGUAGE_ALIASES.values():
        return lower
    return _LANGUAGE_ALIASES.get(lower)


def _canonical_country(value: str) -> str | None:
    upper = value.upper()
    if upper in _COUNTRY_ALIASES.values():
        return upper
    return _COUNTRY_ALIASES.get(value.lower())


def _canonical_genre(value: str) -> str | None:
    if value in _GENRE_ALIASES.values():
        return value
    return _GENRE_ALIASES.get(value.lower())


def _canonical_decade(value: str) -> str | None:
    match = re.search(r"(19|20)\d0s", value)
    if match:
        return match.group(0)
    if re.fullmatch(r"(19|20)\d0s", value.strip()):
        return value.strip()
    return None


def _canonical_concept(value: str) -> str | None:
    lower = _normalize_text(value)
    if not lower:
        return None
    if lower in _FILLER_CONCEPTS:
        return None
    if lower in _SEMANTIC_ALIASES:
        return lower
    for canonical, aliases in _SEMANTIC_ALIASES.items():
        if lower == canonical or lower in aliases:
            return canonical
    cleaned = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    if cleaned in _FILLER_CONCEPTS:
        return None
    return cleaned or None


def _normalize_title_concept(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^a-z0-9]+", "_", _normalize_text(value)).strip("_")
    return cleaned or None


def _contains_explicit_novelty_request(query: str) -> bool:
    lower = _normalize_text(query)
    return any(phrase in lower for phrase in _EXPLICIT_NOVELTY_PHRASES)


def _is_explanation_request(query: str) -> bool:
    lower = _normalize_text(query)
    return any(phrase in lower for phrase in _EXPLANATION_PHRASES)


def _is_provenance_request(query: str) -> bool:
    lower = _normalize_text(query)
    return any(phrase in lower for phrase in _PROVENANCE_PHRASES)


def _is_profile_boundary_request(query: str) -> bool:
    lower = _normalize_text(query)
    return any(phrase in lower for phrase in _PROFILE_BOUNDARY_PHRASES)


def _determine_interaction_mode(query: str, candidate_id: str | None = None) -> str:
    if _is_explanation_request(query):
        return "explain_target" if candidate_id is not None else "explain_needs_target"
    if _is_provenance_request(query):
        return "provenance"
    if _is_profile_boundary_request(query):
        return "profile_boundary"
    return "recommend"


def _reference_concept_tokens(reference_title: str | None) -> set[str]:
    normalized = _normalize_title_concept(reference_title)
    if not normalized:
        return set()
    tokens = {normalized}
    tokens.update(part for part in normalized.split("_") if len(part) > 2)
    return tokens


def _filter_reference_concepts(
    concepts: list[SemanticConcept],
    reference_title: str | None,
) -> list[SemanticConcept]:
    reference_tokens = _reference_concept_tokens(reference_title)
    if not reference_tokens:
        return concepts
    filtered: list[SemanticConcept] = []
    for concept in concepts:
        concept_key = _normalize_title_concept(concept.concept or concept.aspect_id)
        if concept_key and any(token and (concept_key == token or concept_key.startswith(token)) for token in reference_tokens):
            continue
        filtered.append(concept)
    return filtered


def _build_semantic_requirement_groups(
    semantic_requirements: list[SemanticConcept],
    semantic_exclusions: list[SemanticConcept],
    reference_title: str | None,
) -> list[SemanticRequirementGroup]:
    groups: list[SemanticRequirementGroup] = []
    required = [concept for concept in semantic_requirements if concept.importance == "required"]
    preferred = [concept for concept in semantic_requirements if concept.importance != "required"]
    required = _filter_reference_concepts(required, reference_title)
    preferred = _filter_reference_concepts(preferred, reference_title)
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


def _canonical_intent_type(value: str, query: str) -> str:
    if value in _VALID_INTENT_TYPES:
        return value
    legacy_intent = legacy_parse_intent(query).intent_type
    if legacy_intent in _VALID_INTENT_TYPES:
        return legacy_intent
    return "GENERAL_DISCOVERY"


def _clean_reference_text(text: str) -> str:
    cleaned = text.strip().strip("\"'“”‘’")
    cleaned = re.sub(r"\b(?:that|which)\s+tmdb\s+cannot\s+resolve\b.*$", "", cleaned).strip()
    cleaned = re.split(r"\b(?:and|but|so|because|since|with|for)\b", cleaned, maxsplit=1)[0].strip()
    cleaned = cleaned.rstrip(",;:.!?")
    return cleaned


def _extract_reference_title(query: str) -> str | None:
    text = _normalize_text(query)
    for pattern in _REFERENCE_PATTERNS:
        match = re.search(pattern, text, flags=re.I)
        if match:
            candidate = _clean_reference_text(match.group(1))
            if candidate and _is_plausible_reference_span(candidate):
                return candidate
    return None


def _resolve_reference_title(title: str, client: TMDBClient | None = None) -> ReferenceResolution:
    if not title:
        return ReferenceResolution(None, None, "absent", None)
    keys = get_api_keys()
    client = client or (TMDBClient(api_key=keys.tmdb_api_key) if keys.tmdb_api_key else None)
    if client is None:
        return ReferenceResolution(title, None, "unresolved", None)
    try:
        match = client.search_movie(title)
    except Exception:
        match = None
    if not match or match.get("id") is None:
        return ReferenceResolution(title, None, "unresolved", None)
    tmdb_id = int(match["id"])
    summary = str(match.get("title") or match.get("original_title") or title)
    return ReferenceResolution(title, tmdb_id, "resolved", summary)


def _extract_usage(payload: Any) -> tuple[int | None, int | None, int | None]:
    if payload is None:
        return None, None, None
    usage = None
    if isinstance(payload, dict):
        usage = payload.get("usage_metadata") or payload.get("response_metadata", {}).get("token_usage") or payload.get("response_metadata")
    else:
        usage = getattr(payload, "usage_metadata", None) or getattr(payload, "response_metadata", None)
    if not isinstance(usage, dict):
        return None, None, None
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("promptTokenCount")
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("candidates_token_count")
    total_tokens = usage.get("total_tokens") or usage.get("totalTokenCount")
    try:
        input_tokens = int(input_tokens) if input_tokens is not None else None
    except Exception:
        input_tokens = None
    try:
        output_tokens = int(output_tokens) if output_tokens is not None else None
    except Exception:
        output_tokens = None
    try:
        total_tokens = int(total_tokens) if total_tokens is not None else None
    except Exception:
        total_tokens = None
    return input_tokens, output_tokens, total_tokens


def _llm_request_spec(query: str) -> tuple[RequestSpec | None, int, int | None, int | None, int | None]:
    keys = get_api_keys()
    if not getattr(keys, "openai_api_key", None):
        return None, 0, None, None, None

    llm = ChatOpenAI(model=REQUEST_MODEL, temperature=0, api_key=keys.openai_api_key)
    structured = llm.with_structured_output(RequestSpec, include_raw=True)
    prompt = (
        "Interpret the user's film request into a structured semantic request.\n"
        "Return only the schema fields. Do not recommend films. Do not retrieve data.\n"
        "Important:\n"
        "- Separate structured metadata constraints from semantic requirements and exclusions.\n"
        "- Do not turn filler words into film concepts.\n"
        "- If a reference title is present, capture it.\n"
        "- If the request is about novelty or current-vs-historical taste, mark that explicitly.\n"
        "- Preserve request priority instructions such as current request vs historical taste.\n"
    )
    try:
        result = structured.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
        parsed = result.get("parsed") if isinstance(result, dict) else result
        raw = result.get("raw") if isinstance(result, dict) else None
        input_tokens, output_tokens, total_tokens = _extract_usage(raw)
        return parsed, 1, input_tokens, output_tokens, total_tokens
    except Exception:
        return None, 0, None, None, None


def _extract_negated_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    for match in re.finditer(r"\b(?:not about|not actually about|no|without|excluding|must not(?: actually)? be about|must not mention|avoid|exclude|excluding)\s+([^.;!?]+)", text):
        tail = match.group(1)
        tail = re.sub(r"^(?:actually\s+)?(?:be\s+)?about\s+", "", tail).strip()
        tail = re.sub(r"^(?:actually\s+)?(?:be\s+)?", "", tail).strip()
        tail = re.split(r"\b(?:but|so|while|though|yet)\b", tail, maxsplit=1)[0]
        parts = re.split(r",|\band/or\b|\band\b|\bor\b|/", tail)
        for part in parts:
            canonical = _canonical_concept(part)
            if canonical:
                concepts.append(canonical)
    for match in re.finditer(r"\bnot\s+([^.;!?]+)", text):
        tail = match.group(1).strip()
        if tail.startswith(("only ", "just ")):
            continue
        tail = re.split(r"\b(?:but|so|while|though|yet)\b", tail, maxsplit=1)[0]
        parts = re.split(r",|\band/or\b|\band\b|\bor\b|/", tail)
        for part in parts:
            canonical = _canonical_concept(part)
            if canonical:
                concepts.append(canonical)
    return _dedupe(concepts)


def _extract_preferred_concepts(text: str) -> list[str]:
    lower = _normalize_text(text)
    concepts: list[str] = []
    for match in re.finditer(r"\bpreferably(?:\s+with\s+(?:a|an|the))?\s+([^,.;!?]+)", lower):
        tail = match.group(1)
        parts = re.split(r",|\band\b|\bor\b|/", tail)
        for part in parts:
            canonical = _canonical_concept(part)
            if canonical:
                concepts.append(canonical)
    for match in re.finditer(r"\b(?:still|a little|somewhat|slightly|but still)\s+([^,.;!?]+)", lower):
        tail = match.group(1)
        parts = re.split(r",|\band\b|\bor\b|/", tail)
        for part in parts:
            canonical = _canonical_concept(part)
            if canonical:
                concepts.append(canonical)
    return _dedupe(concepts)


def _extract_positive_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    lower = _normalize_text(text)
    for canonical, aliases in _SEMANTIC_ALIASES.items():
        if canonical == "historical" and any(phrase in lower for phrase in ("historical taste", "historical compatibility", "my historical taste")):
            continue
        if any(alias in lower for alias in aliases):
            concepts.append(canonical)
    return _dedupe(concepts)


def _novelty_goal_from_query(query: str, parsed: NoveltyGoal | None = None) -> NoveltyGoal:
    explicit = _contains_explicit_novelty_request(query)
    if not explicit:
        return NoveltyGoal(enabled=False, priority=None)
    return NoveltyGoal(enabled=True, priority="history_distance")


def _extract_structured_constraints(text: str) -> StructuredConstraints:
    lower = _normalize_text(text)
    constraints = StructuredConstraints()
    for alias, language in _LANGUAGE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            constraints.required_languages.append(language)
    for alias, country in _COUNTRY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            constraints.required_countries.append(country)
    for region, countries in _REGION_TO_COUNTRIES.items():
        if re.search(rf"\b{re.escape(region)}\b", lower):
            constraints.required_countries.extend(countries)
    for alias, genre in _GENRE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            if re.search(rf"\b(?:no|not|without|exclude|excluding|do not want|don't want|must not want)\s+{re.escape(alias)}\b", lower):
                constraints.excluded_genres.append(genre)
            else:
                constraints.required_genres.append(genre)
    for decade in ("2020s", "2010s", "2000s", "1990s", "1980s", "1970s", "1960s", "1950s"):
        if decade in lower:
            if re.search(rf"\b(?:no|not|without|exclude|excluding)\s+{re.escape(decade)}\b", lower):
                constraints.excluded_decades.append(decade)
            else:
                constraints.required_decades.append(decade)
    year_match = re.search(r"\b(19|20)\d{2}\b", lower)
    if year_match:
        year = int(year_match.group(0))
        if any(marker in lower for marker in ("after", "since", "from")):
            constraints.min_year = year
        if any(marker in lower for marker in ("before", "until", "through", "up to")):
            constraints.max_year = year
    constraints.required_languages = _dedupe([value.lower() for value in constraints.required_languages])
    constraints.required_countries = _dedupe([value.upper() for value in constraints.required_countries])
    constraints.required_genres = _dedupe([value for value in constraints.required_genres])
    constraints.required_decades = _dedupe([value for value in constraints.required_decades])
    constraints.excluded_genres = _dedupe([value for value in constraints.excluded_genres])
    constraints.excluded_languages = _dedupe([value.lower() for value in constraints.excluded_languages])
    constraints.excluded_countries = _dedupe([value.upper() for value in constraints.excluded_countries])
    constraints.excluded_decades = _dedupe([value for value in constraints.excluded_decades])
    return constraints


def _extract_personalization_instruction(text: str) -> PersonalizationInstruction:
    lower = _normalize_text(text)
    if "high-b3" in lower or "high b3" in lower:
        return PersonalizationInstruction(preference="high B3 compatibility", priority="historical_taste_secondary")
    if "modestly compatible with my historical taste" in lower:
        return PersonalizationInstruction(preference="modest historical taste compatibility", priority="current_request_primary")
    if "historical taste" in lower and "request match" in lower:
        return PersonalizationInstruction(preference="request match first, historical taste second", priority="current_request_primary")
    return PersonalizationInstruction()


def _deterministic_request_spec(query: str) -> RequestSpec:
    lower = _normalize_text(query)
    structured_constraints = _extract_structured_constraints(query)
    preferred_concepts = set(_extract_preferred_concepts(query))
    semantic_exclusions = _canonicalize_semantic_concepts(
        [
            SemanticConcept(aspect_id=concept, concept=concept, importance="required", source_span=query)
            for concept in _extract_negated_concepts(query)
        ],
        source_span=query,
    )
    semantic_requirements = [
        SemanticConcept(
            aspect_id=concept,
            concept=concept,
            importance="preferred" if concept in preferred_concepts else "required",
            source_span=query,
        )
        for concept in _extract_positive_concepts(query)
        if concept not in {item.concept or item.aspect_id for item in semantic_exclusions}
    ]
    semantic_requirements = _canonicalize_semantic_concepts(semantic_requirements, source_span=query)
    novelty_goal = _novelty_goal_from_query(query)
    reference_title = _extract_reference_title(query)
    request_mode = "generic"
    if novelty_goal.enabled:
        request_mode = "novelty"
    elif reference_title:
        request_mode = "reference"
    elif semantic_requirements or semantic_exclusions or _has_meaningful_semantic_intent(
        RequestSpec(
            structured_constraints=structured_constraints,
            semantic_requirement_groups=[],
            semantic_requirements=semantic_requirements,
            semantic_exclusions=semantic_exclusions,
        )
    ) or _extract_personalization_instruction(query).preference:
        request_mode = "contextual"
    if _is_explanation_request(query):
        request_mode = "clarification"

    request_relevance_mode = "query_aware" if request_mode != "generic" else "broad"
    intent_type = _canonical_intent_type(legacy_parse_intent(query).intent_type, query)
    spec = RequestSpec(
        intent_type=intent_type,
        mode="contextual" if request_mode != "generic" else "generic",
        structured_constraints=structured_constraints,
        semantic_requirement_groups=_build_semantic_requirement_groups(semantic_requirements, semantic_exclusions, reference_title),
        semantic_requirements=semantic_requirements,
        semantic_exclusions=semantic_exclusions,
        reference=ReferenceSpec(title=reference_title, relation="similar_to" if reference_title else None),
        novelty_goal=novelty_goal,
        personalization_instruction=_extract_personalization_instruction(query),
        request_relevance_mode=request_relevance_mode,
        semantic_query_text=query.strip(),
        request_summary=query.strip(),
    )
    return spec


def _normalize_request_spec(spec: RequestSpec, query: str, client: TMDBClient | None = None) -> RequestSpec:
    fallback = _deterministic_request_spec(query)
    structured = spec.structured_constraints.model_copy()
    fallback_structured = fallback.structured_constraints
    structured.required_languages = _dedupe(
        [
            _canonical_language(lang) or str(lang).lower()
            for lang in (structured.required_languages or fallback_structured.required_languages)
            if _canonical_language(lang)
        ]
    )
    structured.excluded_languages = _dedupe(
        [
            _canonical_language(lang) or str(lang).lower()
            for lang in (structured.excluded_languages or fallback_structured.excluded_languages)
            if _canonical_language(lang)
        ]
    )
    structured.required_countries = _dedupe(
        [
            _canonical_country(country) or str(country).upper()
            for country in (structured.required_countries or fallback_structured.required_countries)
            if _canonical_country(country)
        ]
    )
    structured.excluded_countries = _dedupe(
        [
            _canonical_country(country) or str(country).upper()
            for country in (structured.excluded_countries or fallback_structured.excluded_countries)
            if _canonical_country(country)
        ]
    )
    structured.required_genres = _dedupe(
        [
            _canonical_genre(genre) or str(genre)
            for genre in (structured.required_genres or fallback_structured.required_genres)
            if _canonical_genre(genre)
        ]
    )
    structured.excluded_genres = _dedupe(
        [
            _canonical_genre(genre) or str(genre)
            for genre in (structured.excluded_genres or fallback_structured.excluded_genres)
            if _canonical_genre(genre)
        ]
    )
    structured.required_decades = _dedupe(
        [
            _canonical_decade(decade) or str(decade)
            for decade in (structured.required_decades or fallback_structured.required_decades)
            if _canonical_decade(decade)
        ]
    )
    structured.excluded_decades = _dedupe(
        [
            _canonical_decade(decade) or str(decade)
            for decade in (structured.excluded_decades or fallback_structured.excluded_decades)
            if _canonical_decade(decade)
        ]
    )
    if structured.min_year is not None and structured.max_year is not None and structured.min_year > structured.max_year:
        structured.min_year, structured.max_year = structured.max_year, structured.min_year

    semantic_requirements_source = spec.semantic_requirements or fallback.semantic_requirements
    semantic_requirements = _canonicalize_semantic_concepts(semantic_requirements_source, source_span=query)
    semantic_exclusions_source = spec.semantic_exclusions or fallback.semantic_exclusions
    semantic_exclusions = _canonicalize_semantic_concepts(semantic_exclusions_source, source_span=query)
    semantic_requirement_groups_source = spec.semantic_requirement_groups or fallback.semantic_requirement_groups
    semantic_requirement_groups, group_exclusions = _canonicalize_semantic_groups_with_negations(semantic_requirement_groups_source, query=query)
    semantic_exclusions = _dedupe(
        [
            *_dedupe([concept.concept for concept in semantic_exclusions if concept.concept]),
            *[concept.concept for concept in group_exclusions if concept.concept],
        ]
    )
    semantic_exclusions = [
        SemanticConcept(aspect_id=concept, concept=concept, importance="required", source_span=query)
        for concept in semantic_exclusions
    ]

    reference = spec.reference.model_copy()
    fallback_reference = fallback.reference
    if not reference.title and fallback_reference.title:
        reference = fallback_reference.model_copy()
    if reference.title and reference.resolution_status != "resolved":
        resolved = _resolve_reference_title(reference.title, client=client)
        reference = ReferenceSpec(
            title=resolved.title or reference.title,
            relation=reference.relation or "similar_to",
            resolved_tmdb_id=resolved.tmdb_id,
            resolved_tmdb_title=resolved.summary,
            resolution_status=resolved.status,
            )

    novelty_goal = spec.novelty_goal.model_copy()
    if not novelty_goal.enabled and fallback.novelty_goal.enabled:
        novelty_goal = fallback.novelty_goal.model_copy()
    novelty_goal = _novelty_goal_from_query(query, novelty_goal)

    semantic_requirements = _filter_reference_concepts(semantic_requirements, reference.title)
    semantic_requirement_groups = _canonicalize_semantic_groups(
        semantic_requirement_groups or _build_semantic_requirement_groups(semantic_requirements, semantic_exclusions, reference.title),
        query=query,
    )

    request_mode = spec.mode
    if novelty_goal.enabled:
        request_mode = "contextual"
    elif reference.resolution_status == "resolved":
        request_mode = "contextual"
    elif semantic_requirement_groups or semantic_requirements or semantic_exclusions or structured.required_languages or structured.required_countries or structured.required_genres or structured.required_decades:
        request_mode = "contextual"
    elif spec.request_relevance_mode == "query_aware":
        request_mode = "contextual"

    request_relevance_mode = "query_aware" if request_mode != "generic" else "broad"
    if request_mode == "generic" and not semantic_requirement_groups and not semantic_requirements and not semantic_exclusions and not any(
        [
            structured.required_languages,
            structured.required_countries,
            structured.required_genres,
            structured.required_decades,
            structured.min_year is not None,
            structured.max_year is not None,
        ]
    ):
        request_relevance_mode = "broad"

    return RequestSpec(
        intent_type=_canonical_intent_type(spec.intent_type, query),
        mode=request_mode,
        structured_constraints=structured,
        semantic_requirement_groups=semantic_requirement_groups,
        semantic_requirements=semantic_requirements,
        semantic_exclusions=semantic_exclusions,
        reference=reference,
        novelty_goal=novelty_goal,
        personalization_instruction=spec.personalization_instruction if (spec.personalization_instruction.preference or spec.personalization_instruction.priority) else fallback.personalization_instruction,
        request_relevance_mode=request_relevance_mode,
        semantic_query_text=spec.semantic_query_text or query.strip(),
        request_summary=spec.request_summary or query.strip(),
    )


def _legacy_intent_from_spec(spec: RequestSpec) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    taste_dimensions = [concept.concept or concept.aspect_id for concept in spec.semantic_requirements if (concept.concept or concept.aspect_id)]
    for group in spec.semantic_requirement_groups:
        taste_dimensions.extend([concept.concept or concept.aspect_id for concept in group.concepts if (concept.concept or concept.aspect_id)])
    genres = list(spec.structured_constraints.required_genres)
    languages = list(spec.structured_constraints.required_languages)
    countries = list(spec.structured_constraints.required_countries)
    decades = list(spec.structured_constraints.required_decades)
    exclusions = [concept.concept for concept in spec.semantic_exclusions if concept.concept]
    return _dedupe(taste_dimensions), _dedupe(genres), _dedupe(languages), _dedupe(countries), _dedupe(decades), _dedupe(exclusions)


def _build_intent(spec: RequestSpec) -> Any:
    taste_dimensions, genres, languages, countries, decades, exclusions = _legacy_intent_from_spec(spec)
    from mvp.src.generative.schemas import RecommendationIntent

    return RecommendationIntent(
        intent_type=spec.intent_type,  # type: ignore[arg-type]
        requested_taste_dimensions=taste_dimensions,
        requested_genres=genres,
        requested_languages=languages,
        requested_countries=countries,
        requested_decades=decades,
        exclusions=exclusions,
        free_text_context=spec.request_summary,
    )


def _build_retrieval_plan(spec: RequestSpec) -> RetrievalPlan:
    semantic_concepts = [*spec.semantic_requirements, *spec.semantic_exclusions]
    for group in spec.semantic_requirement_groups:
        semantic_concepts.extend(group.concepts)
    keyword_resolutions = [
        KeywordResolution(concept=concept.concept, tmdb_keyword_name=concept.concept.replace("_", " "), provenance="semantic_concept")
        for concept in semantic_concepts
    ]
    supplemental_seed_sources: list[str] = []
    if spec.reference.title:
        supplemental_seed_sources.append("reference_title")
    if spec.reference.resolution_status == "resolved":
        supplemental_seed_sources.append("reference_recommendations")
    if spec.structured_constraints.required_genres:
        supplemental_seed_sources.extend([f"genre:{genre}" for genre in spec.structured_constraints.required_genres])
    if spec.structured_constraints.required_languages:
        supplemental_seed_sources.extend([f"language:{language}" for language in spec.structured_constraints.required_languages])
    if spec.structured_constraints.required_decades:
        supplemental_seed_sources.extend([f"decade:{decade}" for decade in spec.structured_constraints.required_decades])
    if spec.structured_constraints.required_countries:
        supplemental_seed_sources.extend([f"country:{country}" for country in spec.structured_constraints.required_countries])
    if semantic_concepts:
        supplemental_seed_sources.extend([f"semantic:{concept.concept}" for concept in semantic_concepts])
    return RetrievalPlan(
        request_mode=spec.mode,
        structured_constraints=spec.structured_constraints,
        semantic_query_text=spec.semantic_query_text,
        semantic_concepts=semantic_concepts,
        keyword_resolutions=keyword_resolutions,
        reference_seed=spec.reference,
        query_aware_expansion_required=spec.mode != "generic" or bool(semantic_concepts) or bool(spec.reference.title),
        supplemental_seed_sources=_dedupe(supplemental_seed_sources),
    )


def understand_request(query: str, candidate_id: str | None = None, *, client: TMDBClient | None = None) -> RequestUnderstanding:
    parsed_spec, llm_call_count, input_tokens, output_tokens, total_tokens = _llm_request_spec(query)
    if parsed_spec is None:
        parsed_spec = _deterministic_request_spec(query)
        llm_used = False
    else:
        llm_used = True
    spec = _normalize_request_spec(parsed_spec, query, client=client)
    plan = _build_retrieval_plan(spec)

    reference_title = spec.reference.title
    reference_status = spec.reference.resolution_status
    reference_tmdb_id = spec.reference.resolved_tmdb_id
    reference_summary = spec.reference.resolved_tmdb_title

    if reference_status == "absent" and reference_title:
        resolution = _resolve_reference_title(reference_title, client=client)
        reference_status = resolution.status
        reference_tmdb_id = resolution.tmdb_id
        reference_summary = resolution.summary
        spec = spec.model_copy(
            update={
                "reference": ReferenceSpec(
                    title=reference_title,
                    relation=spec.reference.relation,
                    resolved_tmdb_id=reference_tmdb_id,
                    resolved_tmdb_title=reference_summary,
                    resolution_status=reference_status,
                )
            }
        )
        plan = _build_retrieval_plan(spec)

    novelty_requested = spec.novelty_goal.enabled
    request_mode = "generic"
    if novelty_requested:
        request_mode = "novelty"
    elif spec.reference.title:
        request_mode = "reference"
    elif spec.mode == "contextual":
        request_mode = "contextual"
    if request_mode == "generic" and (
        spec.semantic_requirement_groups
        or spec.semantic_requirements
        or spec.semantic_exclusions
        or _has_meaningful_semantic_intent(spec)
        or spec.personalization_instruction.preference
    ):
        request_mode = "contextual"
    if _is_explanation_request(query):
        request_mode = "clarification"
    interaction_mode = _determine_interaction_mode(query, candidate_id)
    request = RequestUnderstanding(
        query=query,
        intent=_build_intent(spec),
        spec=spec,
        request_mode=request_mode,
        interaction_mode=interaction_mode,
        novelty_requested=novelty_requested,
        reference_title=reference_title,
        reference_tmdb_id=reference_tmdb_id,
        reference_status=reference_status,
        reference_summary=reference_summary,
        request_relevance_mode=spec.request_relevance_mode,
        retrieval_plan=plan,
        llm_used=llm_used,
        llm_call_count=llm_call_count,
        runtime_input_tokens=input_tokens,
        runtime_output_tokens=output_tokens,
        runtime_total_tokens=total_tokens,
    )

    if request.request_mode == "clarification" and candidate_id is None:
        request = request.model_copy(update={"request_mode": "clarification"})
    if request.interaction_mode == "explain_target" and candidate_id is None:
        request = request.model_copy(update={"interaction_mode": "explain_needs_target"})
    return request
