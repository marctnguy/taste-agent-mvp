from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.run_helpers import trace

from mvp.src.config import get_api_keys, get_langsmith_config, load_runtime_env
from mvp.src.generative.context_builder import (
    DEFAULT_CANDIDATE_CONTEXT_SIZE,
    DEFAULT_RECOMMENDATION_COUNT,
    RecommendationRuntimeContext,
    build_recommendation_runtime_context,
    parse_intent,
    hard_filter_candidates,
)
from mvp.src.generative.contextual_retrieval import build_runtime_v3_recommendation_context
from mvp.src.generative.prompts import (
    INTENT_PROMPT_VERSION,
    RECOMMENDATION_PROMPT_TEMPLATE,
    RECOMMENDATION_PROMPT_VERSION,
    REPAIR_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from mvp.src.generative.runtime_version import (
    GENERATION_MODEL,
    RUNTIME_VERSION,
    SOURCE_RECOMMENDATION_RUN,
)
from mvp.src.generative.schemas import RecommendationItem, RecommendationResponse, RecommendationRunResult, RuntimeMetadata
from mvp.src.generative.validators import fallback_recommendations, validate_recommendation_response


DEFAULT_GENERATIVE_MODEL = GENERATION_MODEL
DEFAULT_GENERATIVE_RUN_DIR = Path("mvp/artifacts/recommendation_runs/20261003T075635Z")


def _langsmith_project() -> str:
    config = get_langsmith_config()
    return config.project or "taste-agent-capstone"


def _candidate_payload(candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "tmdb_id": candidate.tmdb_id,
        "title": candidate.title,
        "year": candidate.year,
        "genres": candidate.genres,
        "original_language": candidate.original_language,
        "production_countries": candidate.production_countries,
        "predicted_preference": candidate.predicted_preference,
        "raw_rank": candidate.raw_rank,
        "candidate_provenance": candidate.candidate_provenance,
        "candidate_source_ranks": candidate.candidate_source_ranks,
        "semantic_evidence": [evidence.model_dump() for evidence in candidate.semantic_evidence],
    }


def _taste_payload(context: RecommendationRuntimeContext) -> dict[str, Any]:
    return {
        "positive_taste_profile": [signal.model_dump() for signal in context.positive_taste_profile],
        "negative_taste_profile": [signal.model_dump() for signal in context.negative_taste_profile],
    }


def _trace_tags(
    intent_type: str,
    target_candidate_id: str | None = None,
    *,
    runtime_version: str = RUNTIME_VERSION,
    intent_prompt_version: str = INTENT_PROMPT_VERSION,
    recommendation_prompt_version: str = RECOMMENDATION_PROMPT_VERSION,
) -> list[str]:
    tags = [
        runtime_version,
        intent_prompt_version,
        recommendation_prompt_version,
        f"intent:{intent_type}",
        "ranking_model:B3",
        "ranking_model_status:exploratory_candidate_mvp",
        "selection_layer:langchain_contextual_selection",
        "taste_model:62_dim_interpretable_profile",
        "goodreads_status:descriptive_only",
    ]
    if target_candidate_id:
        tags.append(f"target_candidate_id:{target_candidate_id}")
    return tags


def _trace_metadata(
    context: RecommendationRuntimeContext,
    *,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    source_recommendation_run: str,
    generation_status: str,
    validation_passed: bool,
    repair_attempted: bool,
    fallback_used: bool,
    hard_filters_applied: bool,
    llm_call_count: int,
) -> dict[str, Any]:
    return {
        "runtime_version": runtime_version,
        "intent_prompt_version": intent_prompt_version,
        "recommendation_prompt_version": recommendation_prompt_version,
        "generation_model": DEFAULT_GENERATIVE_MODEL,
        "candidate_context_size": context.candidate_context_size,
        "recommendation_count": context.recommendation_count,
        "intent_type": context.intent.intent_type,
        "ranking_model": "B3",
        "ranking_model_status": "exploratory_candidate_mvp",
        "selection_layer": "langchain_contextual_selection",
        "taste_model": "62_dim_interpretable_profile",
        "goodreads_status": "descriptive_only",
        "generation_status": generation_status,
        "validation_passed": validation_passed,
        "repair_attempted": repair_attempted,
        "fallback_used": fallback_used,
        "hard_filters_applied": hard_filters_applied,
        "source_recommendation_run": source_recommendation_run,
        "target_candidate_id": context.target_candidate_id,
        "llm_call_count": llm_call_count,
    }


def _build_messages(context: RecommendationRuntimeContext, eligible_candidates: list[Any]) -> list[Any]:
    candidate_json = json.dumps([_candidate_payload(candidate) for candidate in eligible_candidates], ensure_ascii=False)
    taste_json = json.dumps(_taste_payload(context), ensure_ascii=False)
    hard_filters_json = json.dumps(context.hard_filters, ensure_ascii=False)
    intent_json = context.intent.model_dump_json()
    user_prompt = RECOMMENDATION_PROMPT_TEMPLATE.format(
        query=context.query,
        intent_json=intent_json,
        hard_filters_json=hard_filters_json,
        candidate_json=candidate_json,
        taste_profile_json=taste_json,
        recommendation_count=context.recommendation_count,
    )
    if context.intent.intent_type == "EXPLAIN" and context.target_candidate_id:
        user_prompt += (
            "\n\nEXPLAIN target:\n"
            f"candidate_id={context.target_candidate_id}\n"
            "Return one grounded explanation for this exact candidate only."
        )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]


def _format_response(
    response: RecommendationResponse,
    context: RecommendationRuntimeContext,
    *,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    source_recommendation_run: str,
    generation_status: str,
    validation_passed: bool,
    repair_attempted: bool,
    fallback_used: bool,
    hard_filters_applied: bool,
    llm_call_count: int,
) -> RecommendationRunResult:
    runtime_metadata = RuntimeMetadata(
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        candidate_context_size=context.candidate_context_size,
        recommendation_count=context.recommendation_count,
        intent_type=context.intent.intent_type,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        generation_status=generation_status,  # type: ignore[arg-type]
        validation_passed=validation_passed,
        repair_attempted=repair_attempted,
        fallback_used=fallback_used,
        hard_filters_applied=hard_filters_applied,
        target_candidate_id=context.target_candidate_id,
        llm_call_count=llm_call_count,
    )
    return RecommendationRunResult(
        response=response,
        runtime_metadata=runtime_metadata,
        validation_passed=validation_passed,
        validation_errors=[],
        debug={},
    )


def _needs_candidate_context(
    context: RecommendationRuntimeContext,
    *,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    source_recommendation_run: str,
) -> RecommendationRunResult:
    response = RecommendationResponse(
        intent=context.intent,
        recommendations=[],
        response_summary="Which recommendation would you like me to explain?",
        methodology_note="EXPLAIN requests require a resolvable candidate_id or prior recommendation context.",
    )
    return _format_response(
        response,
        context,
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
        generation_status="needs_candidate_context",
        validation_passed=True,
        repair_attempted=False,
        fallback_used=False,
        hard_filters_applied=False,
        llm_call_count=0,
    )


def _no_match_response(
    context: RecommendationRuntimeContext,
    *,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    source_recommendation_run: str,
) -> RecommendationRunResult:
    response = RecommendationResponse(
        intent=context.intent,
        recommendations=[],
        response_summary="No supplied candidate satisfied the hard constraints, so no grounded recommendations could be returned.",
        methodology_note="Hard constraints were applied deterministically before any generative selection step.",
    )
    return _format_response(
        response,
        context,
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
        generation_status="no_match",
        validation_passed=True,
        repair_attempted=False,
        fallback_used=True,
        hard_filters_applied=True,
        llm_call_count=0,
    )


def _fallback_response(
    context: RecommendationRuntimeContext,
    *,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    source_recommendation_run: str,
    generation_status: str = "fallback",
) -> RecommendationRunResult:
    fallback_items = fallback_recommendations(context.candidate_context, context.intent, context.recommendation_count)
    recommendations = [RecommendationItem(**item) for item in fallback_items]
    response = RecommendationResponse(
        intent=context.intent,
        recommendations=recommendations,
        response_summary="A grounded fallback selection was used because the generative path was unavailable or failed validation.",
        methodology_note="Selections were drawn from the frozen B3 shortlist, filtered deterministically for hard constraints, and explained only with grounded metadata and eligible semantic evidence.",
    )
    return _format_response(
        response,
        context,
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
        generation_status=generation_status,
        validation_passed=False,
        repair_attempted=False,
        fallback_used=True,
        hard_filters_applied=bool(context.intent.requested_genres or context.intent.requested_languages or context.intent.requested_countries or context.intent.requested_decades or context.intent.exclusions),
        llm_call_count=0,
    )


def _generate_with_llm(context: RecommendationRuntimeContext, eligible_candidates: list[Any]) -> tuple[RecommendationResponse | None, int]:
    keys = get_api_keys()
    if not keys.openai_api_key:
        return None, 0

    llm = ChatOpenAI(model=DEFAULT_GENERATIVE_MODEL, temperature=0, api_key=keys.openai_api_key)
    structured = llm.with_structured_output(RecommendationResponse)
    messages = _build_messages(context, eligible_candidates)
    response = structured.invoke(messages)
    return response, 1


def _repair_with_llm(
    context: RecommendationRuntimeContext,
    eligible_candidates: list[Any],
    validation_errors: list[str],
) -> tuple[RecommendationResponse | None, int]:
    keys = get_api_keys()
    if not keys.openai_api_key:
        return None, 0
    llm = ChatOpenAI(model=DEFAULT_GENERATIVE_MODEL, temperature=0, api_key=keys.openai_api_key)
    structured = llm.with_structured_output(RecommendationResponse)
    messages = _build_messages(context, eligible_candidates)
    messages.append(HumanMessage(content=REPAIR_PROMPT_TEMPLATE.format(validation_errors="\n".join(validation_errors))))
    response = structured.invoke(messages)
    return response, 1


def run_recommendation_agent(
    query: str,
    count: int = DEFAULT_RECOMMENDATION_COUNT,
    candidate_context_size: int = DEFAULT_CANDIDATE_CONTEXT_SIZE,
    run_dir: str | Path = DEFAULT_GENERATIVE_RUN_DIR,
    candidate_id: str | None = None,
    debug: bool = False,
    runtime_version: str = RUNTIME_VERSION,
    intent_prompt_version: str = INTENT_PROMPT_VERSION,
    recommendation_prompt_version: str = RECOMMENDATION_PROMPT_VERSION,
    source_recommendation_run: str = SOURCE_RECOMMENDATION_RUN,
) -> RecommendationRunResult:
    load_runtime_env()
    parsed_intent = parse_intent(query)
    with trace(
        "taste_agent_request",
        run_type="chain",
        inputs={"query": query, "candidate_id": candidate_id, "count": count},
        metadata={"runtime_version": runtime_version, "source_recommendation_run": source_recommendation_run},
        tags=_trace_tags(
            parsed_intent.intent_type,
            candidate_id,
            runtime_version=runtime_version,
            intent_prompt_version=intent_prompt_version,
            recommendation_prompt_version=recommendation_prompt_version,
        ),
        project_name=_langsmith_project(),
    ) as root_run:
        with trace("interpret_intent", run_type="chain", parent=root_run, inputs={"query": query, "candidate_id": candidate_id}, project_name=_langsmith_project()) as interpret_run:
            if runtime_version == "runtime_v3":
                context, retrieval_diagnostics = build_runtime_v3_recommendation_context(
                    query=query,
                    target_candidate_id=candidate_id,
                    candidate_context_size=candidate_context_size,
                    recommendation_count=count,
                    run_dir=run_dir,
                )
            else:
                context = build_recommendation_runtime_context(
                    query=query,
                    target_candidate_id=candidate_id,
                    candidate_context_size=candidate_context_size,
                    recommendation_count=count,
                    run_dir=run_dir,
                )
                retrieval_diagnostics = {}
            interpret_run.end(outputs={"intent": context.intent.model_dump(), "target_candidate_id": context.target_candidate_id})
        if retrieval_diagnostics:
            root_run.metadata.update({f"retrieval_{key}": value for key, value in asdict(retrieval_diagnostics).items()})

        hard_filter_candidates_count = 0
        with trace(
            "build_candidate_context",
            run_type="chain",
            parent=root_run,
            inputs={"candidate_context_size": context.candidate_context_size, "run_dir": str(context.run_dir)},
            project_name=_langsmith_project(),
        ) as build_run:
            hard_filter_candidates_count = len(context.candidate_context)
            build_run.end(outputs={"candidate_count": hard_filter_candidates_count, "target_candidate_id": context.target_candidate_id})

        with trace(
            "apply_hard_filters",
            run_type="chain",
            parent=root_run,
            inputs={"intent": context.intent.model_dump(), "candidate_count": hard_filter_candidates_count},
            project_name=_langsmith_project(),
        ) as filter_run:
            eligible_candidates = hard_filter_candidates(context)
            filter_run.end(outputs={"eligible_candidate_ids": [candidate.candidate_id for candidate in eligible_candidates], "eligible_candidate_count": len(eligible_candidates)})

        if context.intent.intent_type == "EXPLAIN" and not context.target_candidate_id:
            result = _needs_candidate_context(
                context,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
            )
            if debug:
                result.debug = {
                    "intent": context.intent.model_dump(),
                    "candidate_context": [candidate.model_dump() for candidate in eligible_candidates],
                    "candidate_ids_supplied": [candidate.candidate_id for candidate in eligible_candidates],
                    "candidate_ranks": {candidate.candidate_id: candidate.raw_rank for candidate in eligible_candidates},
                    "validation": {"passed": True, "errors": []},
                    "needs_candidate_context": True,
                }
            root_run.metadata.update(
                _trace_metadata(
                    context,
                    runtime_version=runtime_version,
                    intent_prompt_version=intent_prompt_version,
                    recommendation_prompt_version=recommendation_prompt_version,
                    source_recommendation_run=source_recommendation_run,
                    generation_status=result.runtime_metadata.generation_status,
                    validation_passed=result.validation_passed,
                    repair_attempted=result.runtime_metadata.repair_attempted,
                    fallback_used=result.runtime_metadata.fallback_used,
                    hard_filters_applied=result.runtime_metadata.hard_filters_applied,
                    llm_call_count=result.runtime_metadata.llm_call_count,
                )
            )
            root_run.end(outputs=result.model_dump())
            return result

        if context.intent.intent_type == "EXPLAIN" and not eligible_candidates:
            result = _needs_candidate_context(
                context,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
            )
            if debug:
                result.debug = {
                    "intent": context.intent.model_dump(),
                    "candidate_context": [candidate.model_dump() for candidate in eligible_candidates],
                    "candidate_ids_supplied": [candidate.candidate_id for candidate in eligible_candidates],
                    "candidate_ranks": {candidate.candidate_id: candidate.raw_rank for candidate in eligible_candidates},
                    "validation": {"passed": True, "errors": []},
                    "needs_candidate_context": True,
                }
            root_run.metadata.update(
                _trace_metadata(
                    context,
                    runtime_version=runtime_version,
                    intent_prompt_version=intent_prompt_version,
                    recommendation_prompt_version=recommendation_prompt_version,
                    source_recommendation_run=source_recommendation_run,
                    generation_status=result.runtime_metadata.generation_status,
                    validation_passed=result.validation_passed,
                    repair_attempted=result.runtime_metadata.repair_attempted,
                    fallback_used=result.runtime_metadata.fallback_used,
                    hard_filters_applied=result.runtime_metadata.hard_filters_applied,
                    llm_call_count=result.runtime_metadata.llm_call_count,
                )
            )
            root_run.end(outputs=result.model_dump())
            return result

        if not eligible_candidates:
            result = _no_match_response(
                context,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
            )
            if debug:
                result.debug = {
                    "intent": context.intent.model_dump(),
                    "candidate_context": [candidate.model_dump() for candidate in eligible_candidates],
                    "candidate_ids_supplied": [candidate.candidate_id for candidate in eligible_candidates],
                    "candidate_ranks": {candidate.candidate_id: candidate.raw_rank for candidate in eligible_candidates},
                    "validation": {"passed": True, "errors": []},
                    "no_match": True,
                }
            root_run.metadata.update(
                _trace_metadata(
                    context,
                    runtime_version=runtime_version,
                    intent_prompt_version=intent_prompt_version,
                    recommendation_prompt_version=recommendation_prompt_version,
                    source_recommendation_run=source_recommendation_run,
                    generation_status=result.runtime_metadata.generation_status,
                    validation_passed=result.validation_passed,
                    repair_attempted=result.runtime_metadata.repair_attempted,
                    fallback_used=result.runtime_metadata.fallback_used,
                    hard_filters_applied=result.runtime_metadata.hard_filters_applied,
                    llm_call_count=result.runtime_metadata.llm_call_count,
                )
            )
            root_run.end(outputs=result.model_dump())
            return result

        with trace(
            "generate_recommendations",
            run_type="llm",
            parent=root_run,
            inputs={"candidate_ids": [candidate.candidate_id for candidate in eligible_candidates], "count": count},
            project_name=_langsmith_project(),
        ) as generate_run:
            initial_response, call_count = _generate_with_llm(context, eligible_candidates)
            generate_run.end(outputs={"llm_call_count": call_count, "candidate_ids": [candidate.candidate_id for candidate in eligible_candidates]})

        if initial_response is None:
            result = _fallback_response(
                context,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
            )
            if debug:
                result.debug = {
                    "intent": context.intent.model_dump(),
                    "candidate_context": [candidate.model_dump() for candidate in eligible_candidates],
                    "candidate_ids_supplied": [candidate.candidate_id for candidate in eligible_candidates],
                    "candidate_ranks": {candidate.candidate_id: candidate.raw_rank for candidate in eligible_candidates},
                    "validation": {"passed": False, "errors": []},
                    "fallback_used": True,
                }
            root_run.metadata.update(
                _trace_metadata(
                    context,
                    runtime_version=runtime_version,
                    intent_prompt_version=intent_prompt_version,
                    recommendation_prompt_version=recommendation_prompt_version,
                    source_recommendation_run=source_recommendation_run,
                    generation_status=result.runtime_metadata.generation_status,
                    validation_passed=result.validation_passed,
                    repair_attempted=result.runtime_metadata.repair_attempted,
                    fallback_used=result.runtime_metadata.fallback_used,
                    hard_filters_applied=result.runtime_metadata.hard_filters_applied,
                    llm_call_count=result.runtime_metadata.llm_call_count,
                )
            )
            root_run.end(outputs=result.model_dump())
            return result

        with trace(
            "validate_response",
            run_type="chain",
            parent=root_run,
            inputs={"recommendation_count": count, "intent_type": context.intent.intent_type},
            project_name=_langsmith_project(),
        ) as validate_run:
            validation = validate_recommendation_response(initial_response, eligible_candidates, context.intent, count, target_candidate_id=context.target_candidate_id)
            validate_run.end(outputs={"passed": validation.passed, "errors": validation.errors})

        if validation.passed:
            response = initial_response.model_copy(update={"intent": context.intent})
            result = _format_response(
                response,
                context,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
                generation_status="generated",
                validation_passed=True,
                repair_attempted=False,
                fallback_used=False,
                hard_filters_applied=bool(context.intent.requested_genres or context.intent.requested_languages or context.intent.requested_countries or context.intent.requested_decades or context.intent.exclusions),
                llm_call_count=call_count,
            )
            result.validation_errors = []
            if debug:
                result.debug = {
                    "intent": context.intent.model_dump(),
                    "candidate_context": [candidate.model_dump() for candidate in eligible_candidates],
                    "candidate_ids_supplied": [candidate.candidate_id for candidate in eligible_candidates],
                    "candidate_ranks": {candidate.candidate_id: candidate.raw_rank for candidate in eligible_candidates},
                    "validation": validation.__dict__,
                }
            root_run.metadata.update(
                _trace_metadata(
                    context,
                    runtime_version=runtime_version,
                    intent_prompt_version=intent_prompt_version,
                    recommendation_prompt_version=recommendation_prompt_version,
                    source_recommendation_run=source_recommendation_run,
                    generation_status=result.runtime_metadata.generation_status,
                    validation_passed=result.validation_passed,
                    repair_attempted=result.runtime_metadata.repair_attempted,
                    fallback_used=result.runtime_metadata.fallback_used,
                    hard_filters_applied=result.runtime_metadata.hard_filters_applied,
                    llm_call_count=result.runtime_metadata.llm_call_count,
                )
            )
            root_run.end(outputs=result.model_dump())
            return result

        with trace(
            "repair_response",
            run_type="llm",
            parent=root_run,
            inputs={"validation_errors": validation.errors},
            project_name=_langsmith_project(),
        ) as repair_run:
            repaired_response, repair_calls = _repair_with_llm(context, eligible_candidates, validation.errors)
            repair_run.end(outputs={"llm_call_count": repair_calls})

        if repaired_response is not None:
            call_count += repair_calls
            repaired_validation = validate_recommendation_response(
                repaired_response,
                eligible_candidates,
                context.intent,
                count,
                target_candidate_id=context.target_candidate_id,
            )
            if repaired_validation.passed:
                response = repaired_response.model_copy(update={"intent": context.intent})
                result = _format_response(
                    response,
                    context,
                    runtime_version=runtime_version,
                    intent_prompt_version=intent_prompt_version,
                    recommendation_prompt_version=recommendation_prompt_version,
                    source_recommendation_run=source_recommendation_run,
                    generation_status="generated",
                    validation_passed=True,
                    repair_attempted=True,
                    fallback_used=False,
                    hard_filters_applied=bool(context.intent.requested_genres or context.intent.requested_languages or context.intent.requested_countries or context.intent.requested_decades or context.intent.exclusions),
                    llm_call_count=call_count,
                )
                result.validation_errors = []
                if debug:
                    result.debug = {
                        "intent": context.intent.model_dump(),
                        "candidate_context": [candidate.model_dump() for candidate in eligible_candidates],
                        "candidate_ids_supplied": [candidate.candidate_id for candidate in eligible_candidates],
                        "candidate_ranks": {candidate.candidate_id: candidate.raw_rank for candidate in eligible_candidates},
                        "validation": repaired_validation.__dict__,
                    }
                root_run.metadata.update(
                    _trace_metadata(
                        context,
                        runtime_version=runtime_version,
                        intent_prompt_version=intent_prompt_version,
                        recommendation_prompt_version=recommendation_prompt_version,
                        source_recommendation_run=source_recommendation_run,
                        generation_status=result.runtime_metadata.generation_status,
                        validation_passed=result.validation_passed,
                        repair_attempted=result.runtime_metadata.repair_attempted,
                        fallback_used=result.runtime_metadata.fallback_used,
                        hard_filters_applied=result.runtime_metadata.hard_filters_applied,
                        llm_call_count=result.runtime_metadata.llm_call_count,
                    )
                )
                root_run.end(outputs=result.model_dump())
                return result

        result = _fallback_response(
            context,
            runtime_version=runtime_version,
            intent_prompt_version=intent_prompt_version,
            recommendation_prompt_version=recommendation_prompt_version,
            source_recommendation_run=source_recommendation_run,
        )
        result.runtime_metadata.repair_attempted = True
        result.runtime_metadata.llm_call_count = call_count
        result.validation_errors = validation.errors
        if debug:
            result.debug = {
                "intent": context.intent.model_dump(),
                "candidate_context": [candidate.model_dump() for candidate in eligible_candidates],
                "candidate_ids_supplied": [candidate.candidate_id for candidate in eligible_candidates],
                "candidate_ranks": {candidate.candidate_id: candidate.raw_rank for candidate in eligible_candidates},
                "validation": validation.__dict__,
                "repair_attempted": True,
            }
        root_run.metadata.update(
            _trace_metadata(
                context,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
                generation_status=result.runtime_metadata.generation_status,
                validation_passed=result.validation_passed,
                repair_attempted=result.runtime_metadata.repair_attempted,
                fallback_used=result.runtime_metadata.fallback_used,
                hard_filters_applied=result.runtime_metadata.hard_filters_applied,
                llm_call_count=result.runtime_metadata.llm_call_count,
            )
        )
        root_run.end(outputs=result.model_dump())
        return result
