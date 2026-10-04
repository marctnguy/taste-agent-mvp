from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mvp.src.generative.schemas import CandidateContextItem, RecommendationIntent, RecommendationResponse


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


_GOODREADS_RANKING_PATTERNS = (
    r"because of your books",
    r"because your books",
    r"reading taste",
    r"reading profile",
    r"goodreads.*rank",
    r"goodreads.*score",
    r"books.*increased",
)


def _candidate_lookup(candidate_context: list[CandidateContextItem]) -> dict[str, CandidateContextItem]:
    return {candidate.candidate_id: candidate for candidate in candidate_context}


def _candidate_dimensions(candidate: CandidateContextItem) -> set[str]:
    return {item.dimension for item in candidate.semantic_evidence}


def _candidate_hard_filter_passes(candidate: CandidateContextItem, intent: RecommendationIntent) -> bool:
    if intent.requested_languages and candidate.original_language:
        if candidate.original_language.lower() not in {value.lower() for value in intent.requested_languages}:
            return False
    if intent.requested_countries:
        countries = {country.lower() for country in candidate.production_countries}
        if not countries.intersection({value.lower() for value in intent.requested_countries}):
            return False
    if intent.requested_decades and candidate.year is not None:
        decade = f"{candidate.year // 10 * 10}s"
        if decade not in intent.requested_decades:
            return False
    if intent.requested_genres:
        genres = {genre.lower() for genre in candidate.genres}
        if not genres.intersection({value.lower() for value in intent.requested_genres}):
            return False
    return True


def _candidate_request_passes(candidate: CandidateContextItem, intent: RecommendationIntent) -> bool:
    if not _candidate_hard_filter_passes(candidate, intent):
        return False
    if intent.exclusions:
        haystack = " ".join(
            [
                candidate.title.lower(),
                " ".join(candidate.genres).lower(),
                (candidate.original_language or "").lower(),
                " ".join(candidate.production_countries).lower(),
                " ".join(candidate.candidate_provenance).lower(),
            ]
        )
        if any(str(exclusion).lower() in haystack for exclusion in intent.exclusions):
            return False
    return True


def eligible_candidate_count(candidate_context: list[CandidateContextItem], intent: RecommendationIntent) -> int:
    return sum(1 for candidate in candidate_context if _candidate_request_passes(candidate, intent))


def validate_recommendation_response(
    response: RecommendationResponse,
    candidate_context: list[CandidateContextItem],
    intent: RecommendationIntent,
    recommendation_count: int,
    target_candidate_id: str | None = None,
) -> ValidationResult:
    errors: list[str] = []
    lookup = _candidate_lookup(candidate_context)
    seen: set[str] = set()
    expected_count = eligible_candidate_count(candidate_context, intent)

    if intent.intent_type == "EXPLAIN" and target_candidate_id:
        if len(response.recommendations) != 1:
            errors.append("EXPLAIN responses must contain exactly one grounded recommendation.")
    else:
        expected_count = min(recommendation_count, expected_count)
        if len(response.recommendations) != expected_count:
            errors.append(
                f"Returned {len(response.recommendations)} recommendations; expected {expected_count} grounded recommendations for the eligible candidate set."
            )

    for index, item in enumerate(response.recommendations):
        if item.candidate_id not in lookup:
            errors.append(f"Recommendation {index} uses unsupported candidate_id {item.candidate_id!r}.")
            continue

        candidate = lookup[item.candidate_id]
        if item.title != candidate.title:
            errors.append(f"Recommendation {item.candidate_id!r} title mismatch.")

        if candidate.year is not None and item.year is not None and int(item.year) != int(candidate.year):
            errors.append(f"Recommendation {item.candidate_id!r} year mismatch.")

        if item.candidate_id in seen:
            errors.append(f"Duplicate candidate_id {item.candidate_id!r} returned.")
        seen.add(item.candidate_id)

        if not _candidate_hard_filter_passes(candidate, intent):
            errors.append(f"Recommendation {item.candidate_id!r} violates a hard filter.")

        candidate_dims = _candidate_dimensions(candidate)
        for signal in item.taste_signals:
            signal_dimension = signal.split(":", 1)[0].strip()
            if signal_dimension not in candidate_dims:
                errors.append(f"Recommendation {item.candidate_id!r} uses unsupported taste signal {signal_dimension!r}.")

        if any(re.search(pattern, item.why_it_may_fit, flags=re.I) for pattern in _GOODREADS_RANKING_PATTERNS):
            errors.append(f"Recommendation {item.candidate_id!r} appears to use Goodreads as ranking evidence.")
        if item.request_match and any(re.search(pattern, item.request_match, flags=re.I) for pattern in _GOODREADS_RANKING_PATTERNS):
            errors.append(f"Recommendation {item.candidate_id!r} request_match appears to use Goodreads as ranking evidence.")
        if item.caveat and any(re.search(pattern, item.caveat, flags=re.I) for pattern in _GOODREADS_RANKING_PATTERNS):
            errors.append(f"Recommendation {item.candidate_id!r} caveat appears to use Goodreads as ranking evidence.")

    if intent.intent_type == "EXPLAIN" and target_candidate_id and response.recommendations:
        if response.recommendations[0].candidate_id != str(target_candidate_id):
            errors.append(
                f"EXPLAIN response targeted {target_candidate_id!r} but returned {response.recommendations[0].candidate_id!r}."
            )

    if response.recommendations == [] and intent.intent_type != "EXPLAIN" and candidate_context:
        if expected_count > 0:
            errors.append("Structured response returned no recommendations despite eligible candidates being available.")

    return ValidationResult(passed=not errors, errors=errors)


def fallback_recommendations(
    candidate_context: list[CandidateContextItem],
    intent: RecommendationIntent,
    recommendation_count: int,
) -> list[dict[str, Any]]:
    eligible = [candidate for candidate in candidate_context if _candidate_hard_filter_passes(candidate, intent)]
    selected = sorted(eligible, key=lambda candidate: (candidate.raw_rank, -candidate.predicted_preference, candidate.candidate_id))[:recommendation_count]
    fallback_items = []
    for candidate in selected:
        taste_signals = [evidence.dimension for evidence in candidate.semantic_evidence[:2]]
        caveat = None if taste_signals else "Grounded semantic evidence was limited, so this fallback is based on B3 relevance and metadata only."
        fallback_items.append(
            {
                "candidate_id": candidate.candidate_id,
                "title": candidate.title,
                "year": candidate.year,
                "why_it_may_fit": "B3 places this title near the top of the supplied shortlist and the grounded evidence supports it.",
                "taste_signals": taste_signals,
                "request_match": "Grounded fallback selection.",
                "caveat": caveat,
            }
        )
    return fallback_items
