from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from mvp.src.generative_v4.schemas import RuntimeV4Result
from mvp.src.semantics import SEMANTIC_COLUMNS


_GOODREADS_PATTERNS = (
    r"because of your books",
    r"because your books",
    r"reading taste",
    r"reading profile",
    r"goodreads.*rank",
    r"goodreads.*score",
    r"books.*increased",
)

_SENSITIVE_PATTERNS = (
    r"your personality",
    r"your identity",
    r"says about you",
    r"psychology",
    r"type of person",
    r"you are an",
    r"you are a",
    r"because you are",
)

_CERTAINTY_PATTERNS = (
    r"probability",
    r"certain",
    r"certainty",
    r"guaranteed",
    r"likely to like",
    r"likelihood",
)


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def _candidate_lookup(result: RuntimeV4Result) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for candidate in result.debug.get("candidate_context", []):
        if isinstance(candidate, dict):
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            if candidate_id:
                lookup[candidate_id] = candidate
    return lookup


def _request_spec(result: RuntimeV4Result) -> dict[str, Any]:
    request = result.debug.get("request_spec")
    return request if isinstance(request, dict) else {}


def _structured_constraints(result: RuntimeV4Result) -> dict[str, Any]:
    spec = _request_spec(result)
    constraints = spec.get("structured_constraints", {})
    if not isinstance(constraints, dict):
        return {}
    return {
        "genres": constraints.get("required_genres", []) or [],
        "languages": constraints.get("required_languages", []) or [],
        "countries": constraints.get("required_countries", []) or [],
        "decades": constraints.get("required_decades", []) or [],
        "min_year": constraints.get("min_year"),
        "max_year": constraints.get("max_year"),
    }


def _semantic_exclusions(result: RuntimeV4Result) -> list[str]:
    spec = _request_spec(result)
    exclusions = spec.get("semantic_exclusions", [])
    if not isinstance(exclusions, list):
        return []
    return [str(item.get("concept", "")) for item in exclusions if isinstance(item, dict) and item.get("concept")]


def _semantic_requirement_groups(result: RuntimeV4Result) -> list[dict[str, Any]]:
    spec = _request_spec(result)
    groups = spec.get("semantic_requirement_groups", [])
    if not isinstance(groups, list):
        return []
    return [group for group in groups if isinstance(group, dict)]


def _candidate_satisfies_constraints(candidate: dict[str, Any], constraints: dict[str, Any]) -> bool:
    genres = {str(item).lower() for item in candidate.get("genres", []) if item is not None}
    languages = str(candidate.get("original_language", "")).lower()
    countries = {str(item).lower() for item in candidate.get("production_countries", []) if item is not None}
    year = candidate.get("year")

    for genre in constraints.get("genres", []):
        if str(genre).lower() not in genres:
            return False
    for language in constraints.get("languages", []):
        if languages != str(language).lower():
            return False
    for country in constraints.get("countries", []):
        if str(country).lower() not in countries:
            return False
    if constraints.get("decades"):
        if year is None:
            return False
        candidate_decade = f"{int(year) // 10 * 10}s"
        allowed_decades = {str(decade) for decade in constraints.get("decades", [])}
        if candidate_decade not in allowed_decades:
            return False
    min_year = constraints.get("min_year")
    if min_year is not None and year is not None and int(year) < int(min_year):
        return False
    max_year = constraints.get("max_year")
    if max_year is not None and year is not None and int(year) > int(max_year):
        return False
    return True


def _selection_lookup(result: RuntimeV4Result) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for candidate in result.selection_records:
        lookup[str(candidate.candidate_id)] = candidate.model_dump()
    return lookup


def _aggregate_text(result: RuntimeV4Result) -> str:
    chunks = [
        str(result.response.response_summary or ""),
        str(result.response.methodology_note or ""),
    ]
    for rec in result.response.recommendations:
        chunks.extend(
            [
                str(rec.why_it_may_fit or ""),
                str(rec.request_match or ""),
                str(rec.caveat or ""),
                " ".join(str(item) for item in rec.taste_signals or []),
            ]
        )
    return " ".join(chunks).lower()


def _interaction_mode(result: RuntimeV4Result) -> str:
    return str(getattr(result.runtime_metadata, "interaction_mode", "recommend")).strip()


def validate_runtime_v4_result(result: RuntimeV4Result) -> ValidationResult:
    errors: list[str] = []
    candidate_lookup = _candidate_lookup(result)
    selection_lookup = _selection_lookup(result)
    recommendation_ids = [str(rec.candidate_id) for rec in result.response.recommendations]
    constraints = _structured_constraints(result)
    semantic_exclusions = set(_semantic_exclusions(result))
    semantic_groups = _semantic_requirement_groups(result)

    if len(recommendation_ids) != len(set(recommendation_ids)):
        errors.append("Duplicate candidates were returned.")

    if not recommendation_ids:
        if result.runtime_metadata.generation_status not in {"no_match", "abstained", "clarification"}:
            errors.append("Empty recommendation list is only valid for abstention states.")
    for rec in result.response.recommendations:
        candidate = candidate_lookup.get(str(rec.candidate_id))
        if candidate is None:
            errors.append(f"Recommendation {rec.candidate_id!r} is not in the retrieved candidate context.")
            continue
        if str(candidate.get("qualification_status")) == "unsupported":
            errors.append(f"Recommendation {rec.candidate_id!r} should not have been selected after unsupported qualification.")
        if candidate.get("violated_semantic_exclusions"):
            errors.append(f"Recommendation {rec.candidate_id!r} violated a semantic exclusion during qualification.")
        if constraints and not _candidate_satisfies_constraints(candidate, constraints):
            errors.append(f"Recommendation {rec.candidate_id!r} violates a structured hard constraint.")
        if semantic_exclusions:
            haystack = " ".join(
                [
                    str(candidate.get("title", "")).lower(),
                    " ".join(str(item) for item in candidate.get("genres", [])).lower(),
                    str(candidate.get("original_language", "")).lower(),
                    " ".join(str(item) for item in candidate.get("production_countries", [])).lower(),
                    str(candidate.get("overview", "")).lower(),
                    " ".join(str(item) for item in candidate.get("candidate_provenance", [])).lower(),
                ]
            )
            if any(exclusion in haystack for exclusion in semantic_exclusions):
                errors.append(f"Recommendation {rec.candidate_id!r} violated a semantic exclusion from the request.")
        if rec.request_match and not candidate.get("supported_request_aspects") and str(candidate.get("qualification_status")) != "strong":
            errors.append(f"Recommendation {rec.candidate_id!r} contains a request match claim without grounded support.")
        if str(candidate.get("qualification_status")) == "partial" and not rec.caveat:
            errors.append(f"Recommendation {rec.candidate_id!r} is partial but the caveat was dropped.")
        if str(candidate.get("qualification_status")) == "strong" and semantic_groups:
            supported = {str(item) for item in candidate.get("supported_request_aspects", []) if item is not None}
            for group in semantic_groups:
                mode = str(group.get("mode", "all_of"))
                concepts = [str(item.get("concept") or item.get("aspect_id") or "") for item in group.get("concepts", []) if isinstance(item, dict)]
                concepts = [concept for concept in concepts if concept]
                if not concepts:
                    continue
                if mode == "all_of" and not set(concepts).issubset(supported):
                    errors.append(f"Recommendation {rec.candidate_id!r} was marked strong without satisfying an all_of semantic group.")
                if mode == "any_of" and not any(concept in supported for concept in concepts):
                    errors.append(f"Recommendation {rec.candidate_id!r} was marked strong without satisfying an any_of semantic group.")
        if rec.taste_signals:
            evidence_dims = {
                str(item.get("dimension"))
                for item in candidate.get("semantic_evidence", [])
                if isinstance(item, dict) and item.get("dimension")
            }
            for signal in rec.taste_signals:
                if signal not in SEMANTIC_COLUMNS:
                    errors.append(f"Recommendation {rec.candidate_id!r} uses unsupported taste signal {signal!r}.")
                elif evidence_dims and signal not in evidence_dims:
                    errors.append(f"Recommendation {rec.candidate_id!r} uses an ungrounded taste signal {signal!r}.")

        selection_record = selection_lookup.get(str(rec.candidate_id))
        if selection_record is None:
            errors.append(f"Recommendation {rec.candidate_id!r} is missing a selection record.")
        elif str(selection_record.get("qualification_status")) == "unsupported":
            errors.append(f"Recommendation {rec.candidate_id!r} was selected despite unsupported qualification.")

    text = _aggregate_text(result)
    if any(re.search(pattern, text, flags=re.I) for pattern in _GOODREADS_PATTERNS):
        errors.append("Goodreads appears to have been used as ranking evidence.")
    if any(re.search(pattern, text, flags=re.I) for pattern in _SENSITIVE_PATTERNS):
        errors.append("Sensitive inference was introduced.")
    if any(re.search(pattern, text, flags=re.I) for pattern in _CERTAINTY_PATTERNS):
        errors.append("B3 was described with probability or certainty language.")

    interaction_mode = _interaction_mode(result)
    target_candidate_id = str(getattr(result.runtime_metadata, "target_candidate_id", "") or "").strip()
    if interaction_mode == "explain_target":
        if not target_candidate_id:
            errors.append("EXPLAIN_TARGET was selected without a preserved target candidate.")
        if len(recommendation_ids) != 1:
            errors.append("EXPLAIN_TARGET must return exactly one grounded explanation.")
        elif recommendation_ids[0] != target_candidate_id:
            errors.append("EXPLAIN_TARGET must explain the supplied target candidate only.")
        candidate_ids = {
            str(candidate.get("candidate_id", ""))
            for candidate in result.debug.get("candidate_context", [])
            if isinstance(candidate, dict)
        }
        if target_candidate_id and target_candidate_id not in candidate_ids:
            errors.append("EXPLAIN_TARGET target candidate was not preserved in candidate context.")
    elif interaction_mode == "explain_needs_target":
        if recommendation_ids:
            errors.append("EXPLAIN_NEEDS_TARGET must not generate a recommendation slate.")
        if result.runtime_metadata.generation_status != "clarification":
            errors.append("EXPLAIN_NEEDS_TARGET must return a clarification state.")
    elif interaction_mode == "provenance":
        if recommendation_ids:
            errors.append("PROVENANCE must not generate a recommendation slate.")
        if "goodreads" not in text or not any(token in text for token in ("descriptive only", "not used", "does not rank", "does not select", "not rank")):
            errors.append("PROVENANCE must truthfully state that Goodreads is descriptive only and not used for ranking or selection.")
    elif interaction_mode == "profile_boundary":
        if recommendation_ids:
            errors.append("PROFILE_BOUNDARY must not generate a recommendation slate.")
        if not any(token in text for token in ("taste pattern", "cultural preference", "do not infer", "don't infer", "cannot infer", "not infer")):
            errors.append("PROFILE_BOUNDARY must explain the boundary around personality or identity inference.")

    return ValidationResult(passed=not errors, errors=errors)
