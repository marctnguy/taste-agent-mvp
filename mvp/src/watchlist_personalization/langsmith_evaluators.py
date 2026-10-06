from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult, run_evaluator
from pydantic import BaseModel, Field

from mvp.src.generative.runtime_version import GENERATION_MODEL


DEFAULT_JUDGE_MODEL = GENERATION_MODEL
WATCHLIST_SEMANTIC_FIT_RUBRIC_VERSION = "watchlist_semantic_fit_v1"


class WatchlistSemanticFitJudgement(BaseModel):
    semantic_request_fit: int = Field(ge=1, le=5)
    comment: str = ""


_LLM_SCORE_CACHE: dict[tuple[str, str], WatchlistSemanticFitJudgement] = {}


def _score_result(key: str, passed: bool, comment: str = "") -> EvaluationResult:
    return EvaluationResult(key=key, score=1.0 if passed else 0.0, comment=comment or None)


def _run_payload(run: Any) -> dict[str, Any]:
    outputs = getattr(run, "outputs", {}) or {}
    if isinstance(outputs, dict) and "output" in outputs and isinstance(outputs["output"], dict):
        return outputs["output"]
    if isinstance(outputs, dict):
        return outputs
    return {}


def _example_payload(example: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(example, dict):
        inputs = example.get("inputs", {}) or {}
        outputs = example.get("outputs", {}) or {}
        metadata = example.get("metadata", {}) or {}
        return inputs, outputs, metadata
    inputs = getattr(example, "inputs", {}) or {}
    outputs = getattr(example, "outputs", {}) or {}
    metadata = getattr(example, "metadata", {}) or {}
    return inputs, outputs, metadata


def _response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response", {})
    return response if isinstance(response, dict) else {}


def _runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime_metadata", {})
    return runtime if isinstance(runtime, dict) else {}


def _debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
    debug = payload.get("debug", {})
    return debug if isinstance(debug, dict) else {}


def _recommendations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    recs = _response_payload(payload).get("recommendations", [])
    if isinstance(recs, list):
        return [item for item in recs if isinstance(item, dict)]
    return []


def _selection_report(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("selection_report", {})
    return report if isinstance(report, dict) else {}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


@run_evaluator
def duplicate_candidates(run: Any, example: Any) -> EvaluationResult:
    rec_ids = [str(rec.get("candidate_id", "")) for rec in _recommendations(_run_payload(run))]
    passed = len(rec_ids) == len(set(rec_ids))
    return _score_result("duplicate_candidates", passed)


@run_evaluator
def watched_reference_leakage(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    recs = _recommendations(payload)
    debug = _debug_payload(payload)
    excluded_ids = {str(item) for item in debug.get("consumed_source_ids", []) if str(item).strip()}
    if not excluded_ids:
        excluded_ids = {str(item) for item in debug.get("consumed_source", []) if str(item).strip()}
    excluded_tmdb_ids = {str(item) for item in debug.get("consumed_tmdb_ids", []) if str(item).strip()}
    reference_title = _normalize_text(debug.get("reference_title"))
    passed = True
    for rec in recs:
        candidate_id = str(rec.get("candidate_id", ""))
        candidate_tmdb_id = str(rec.get("tmdb_id", ""))
        if candidate_id and candidate_id in excluded_ids:
            passed = False
            break
        if candidate_tmdb_id and candidate_tmdb_id in excluded_tmdb_ids:
            passed = False
            break
        if reference_title and _normalize_text(rec.get("title")) == reference_title:
            passed = False
            break
    return _score_result("watched_reference_leakage", passed, comment=f"excluded_ids={len(excluded_ids)}")


def _candidate_satisfies_constraints(candidate: dict[str, Any], constraints: dict[str, Any]) -> bool:
    genres = {str(item).lower() for item in candidate.get("genres", []) if item is not None}
    language = str(candidate.get("original_language", "")).lower()
    countries = {str(item).lower() for item in candidate.get("production_countries", []) if item is not None}
    year = candidate.get("year")

    def _matches_any(values: list[str], universe: set[str]) -> bool:
        return not values or bool(universe.intersection({str(value).lower() for value in values}))

    for group in constraints.get("semantic_requirement_groups", []) or []:
        if not isinstance(group, dict):
            continue
        mode = str(group.get("mode", "all_of"))
        concepts = [item for item in group.get("concepts", []) or [] if isinstance(item, dict)]
        group_terms = {
            _normalize_text(concept.get("concept") or concept.get("aspect_id"))
            for concept in concepts
            if _normalize_text(concept.get("concept") or concept.get("aspect_id"))
        }
        if mode == "preferred":
            continue
        any_required = True
        if not group_terms:
            return False
        if mode == "any_of" and not any(term in " ".join([candidate.get("title", ""), candidate.get("overview", ""), " ".join(sorted(genres)), language, " ".join(sorted(countries))]).lower() for term in group_terms):
            return False
        if mode == "all_of":
            joined = " ".join([candidate.get("title", ""), candidate.get("overview", ""), " ".join(sorted(genres)), language, " ".join(sorted(countries))]).lower()
            if not all(term in joined for term in group_terms):
                return False

    if constraints.get("required_genres") and not _matches_any(constraints.get("required_genres", []), genres):
        return False
    if constraints.get("required_languages") and not _matches_any(constraints.get("required_languages", []), {language}):
        return False
    if constraints.get("required_countries") and not _matches_any(constraints.get("required_countries", []), countries):
        return False
    if constraints.get("required_decades"):
        if year is None:
            return False
        try:
            candidate_decade = f"{int(float(year)) // 10 * 10}s"
        except Exception:
            return False
        if candidate_decade not in {str(decade).lower() for decade in constraints.get("required_decades", [])}:
            return False

    for genre in constraints.get("excluded_genres", []) or []:
        if str(genre).lower() in genres:
            return False
    for constraint_language in constraints.get("excluded_languages", []) or []:
        if language == str(constraint_language).lower():
            return False
    for country in constraints.get("excluded_countries", []) or []:
        if str(country).lower() in countries:
            return False
    for decade in constraints.get("excluded_decades", []) or []:
        if year is None:
            continue
        try:
            expected = f"{int(float(year)) // 10 * 10}s"
        except Exception:
            continue
        if expected == str(decade):
            return False
    return True


@run_evaluator
def structured_constraint_compliance(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    debug = _debug_payload(payload)
    recs = _recommendations(payload)
    spec = debug.get("controlled_request_spec", {})
    if not isinstance(spec, dict):
        return EvaluationResult(key="structured_constraint_compliance", score=None, comment="N/A - no structured request spec.")
    structured_constraints = spec.get("structured_constraints", {}) if isinstance(spec, dict) else {}
    semantic_groups = spec.get("semantic_requirement_groups", []) if isinstance(spec, dict) else []
    if not any(
        [
            structured_constraints.get("required_genres"),
            structured_constraints.get("required_languages"),
            structured_constraints.get("required_countries"),
            structured_constraints.get("required_decades"),
            structured_constraints.get("excluded_genres"),
            structured_constraints.get("excluded_languages"),
            structured_constraints.get("excluded_countries"),
            structured_constraints.get("excluded_decades"),
            structured_constraints.get("min_year") is not None,
            structured_constraints.get("max_year") is not None,
            semantic_groups,
        ]
    ):
        return EvaluationResult(key="structured_constraint_compliance", score=None, comment="N/A - no structured constraints in frozen request.")
    passed = True
    for rec in recs:
        candidate_constraints = dict(structured_constraints)
        candidate_constraints["semantic_requirement_groups"] = semantic_groups
        if not _candidate_satisfies_constraints(rec, candidate_constraints):
            passed = False
            break
        min_year = structured_constraints.get("min_year")
        max_year = structured_constraints.get("max_year")
        year = rec.get("year")
        if min_year is not None and year is not None and int(float(year)) < int(min_year):
            passed = False
            break
        if max_year is not None and year is not None and int(float(year)) > int(max_year):
            passed = False
            break
    return _score_result("structured_constraint_compliance", passed)


@run_evaluator
def output_validity(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    recommendations = _recommendations(payload)
    runtime = _runtime_payload(payload)
    valid_structure = isinstance(payload.get("response"), dict) and isinstance(payload.get("runtime_metadata"), dict)
    selected_count = _selection_report(payload).get("selected_count")
    passed = (
        valid_structure
        and isinstance(recommendations, list)
        and all(isinstance(rec.get("candidate_id"), (str, int)) for rec in recommendations)
        and all(rec.get("title") is not None for rec in recommendations)
        and (selected_count is None or int(selected_count) == len(recommendations))
        and runtime.get("variant") in {"A", "B"}
    )
    return _score_result("output_validity", passed)


@run_evaluator
def requested_slate_completion(run: Any, example: Any) -> EvaluationResult:
    payload = _run_payload(run)
    report = _selection_report(payload)
    runtime = _runtime_payload(payload)
    selected_count = int(report.get("selected_count", 0) or 0)
    eligible_count = int(report.get("eligible_count", 0) or 0)
    requested_count = int(runtime.get("requested_count", report.get("slate_size", runtime.get("recommendation_count", 5))) or 5)
    expected = min(requested_count, eligible_count)
    passed = requested_count > 0 and selected_count == expected
    return _score_result("requested_slate_completion", passed, comment=f"requested={requested_count}, expected={expected}, actual={selected_count}")


def _judge_payload(run: Any, example: Any) -> dict[str, Any]:
    payload = _run_payload(run)
    inputs, outputs, _ = _example_payload(example)
    recs = [
        {
            "candidate_id": rec.get("candidate_id"),
            "title": rec.get("title"),
            "year": rec.get("year"),
            "genres": rec.get("genres", []),
            "original_language": rec.get("original_language"),
            "production_countries": rec.get("production_countries", []),
            "request_match": rec.get("request_match"),
            "caveat": rec.get("caveat"),
        }
        for rec in _recommendations(payload)
    ]
    return {
        "query": inputs.get("prompt") or inputs.get("query", ""),
        "test_purpose": outputs.get("test_purpose", ""),
        "recommendations": recs,
        "runtime_metadata": _runtime_payload(payload),
        "selection_report": _selection_report(payload),
    }


def _score_with_llm(run: Any, example: Any) -> WatchlistSemanticFitJudgement:
    cache_key = (str(getattr(run, "id", "run")), str(getattr(example, "id", "example")))
    if cache_key in _LLM_SCORE_CACHE:
        return _LLM_SCORE_CACHE[cache_key]

    from mvp.src.config import get_api_keys

    payload = _judge_payload(run, example)
    prompt = (
        "You are judging whether a recommendation slate fits a frozen movie request.\n"
        f"Rubric version: {WATCHLIST_SEMANTIC_FIT_RUBRIC_VERSION}\n"
        "Score semantic request fit from 1 to 5.\n"
        "1 = clearly poor fit, 3 = mixed/adequate, 5 = strong fit.\n\n"
        f"Request: {payload['query']}\n"
        f"Selection report: {payload['selection_report']}\n"
        f"Recommendations: {payload['recommendations']}\n"
        "Focus on request semantics and hard-constraint compliance only.\n"
        "Do not score personal appeal or user taste.\n"
    )

    openai_api_key = get_api_keys().openai_api_key
    if not openai_api_key:
        judged = WatchlistSemanticFitJudgement(
            semantic_request_fit=3,
            comment=f"Disabled offline; rubric={WATCHLIST_SEMANTIC_FIT_RUBRIC_VERSION}",
        )
    else:
        llm = ChatOpenAI(model=DEFAULT_JUDGE_MODEL, temperature=0, api_key=openai_api_key)
        judged = llm.with_structured_output(WatchlistSemanticFitJudgement).invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content="Return the structured semantic-fit score now."),
            ]
        )
        judged = judged.model_copy(update={"comment": f"{judged.comment or ''} rubric={WATCHLIST_SEMANTIC_FIT_RUBRIC_VERSION}".strip()})
    _LLM_SCORE_CACHE[cache_key] = judged
    return judged


@run_evaluator
def semantic_request_fit(run: Any, example: Any) -> EvaluationResult:
    judged = _score_with_llm(run, example)
    return EvaluationResult(key="semantic_request_fit", score=float(judged.semantic_request_fit), comment=judged.comment or None)
