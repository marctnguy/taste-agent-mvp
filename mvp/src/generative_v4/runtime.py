from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langsmith.run_helpers import trace

from mvp.src.config import get_langsmith_config, load_runtime_env
from mvp.src.generative.schemas import RecommendationResponse
from mvp.src.generative_v4.explanation_chain import explain_selection
from mvp.src.generative_v4.explanation_chain import _semantic_profile
from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.generative_v4.qualification_chain import qualify_candidates
from mvp.src.generative_v4.schemas import RuntimeV4Metadata, RuntimeV4Result, SelectionRecord
from mvp.src.generative_v4.selection_chain import select_candidates
from mvp.src.generative_v4.validators import validate_runtime_v4_result
from mvp.src.mvp_deployment import score_candidates
from mvp.src.retrieval.catalog_retrieval import discover_catalog_for_request


DEFAULT_V4_CANDIDATE_CONTEXT_SIZE = 50
DEFAULT_V4_RECOMMENDATION_COUNT = 5
FROZEN_RECOMMENDATION_RUN_DIR = Path("mvp/artifacts/recommendation_runs/20261003T075635Z")
FROZEN_CANDIDATE_POOL_PATH = FROZEN_RECOMMENDATION_RUN_DIR / "candidate_pool.csv"


def _langsmith_project() -> str:
    config = get_langsmith_config()
    return config.project or "taste-agent-capstone"


def _trace_tags(request_mode: str, novelty_requested: bool, reference_status: str, target_candidate_id: str | None = None) -> list[str]:
    tags = [
        "runtime_v4",
        f"request_mode:{request_mode}",
        f"novelty_requested:{str(novelty_requested).lower()}",
        f"reference_status:{reference_status}",
    ]
    if target_candidate_id:
        tags.append(f"target_candidate_id:{target_candidate_id}")
    return tags


def _usage_value(value: int | None) -> int | None:
    return int(value) if value is not None else None


def _sum_usage(*values: int | None) -> int | None:
    total = 0
    seen = False
    for value in values:
        if value is None:
            continue
        seen = True
        total += int(value)
    return total if seen else None


def _sum_tokens(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values if value is not None)


def _hard_filters_applied(request) -> bool:
    spec = getattr(request, "spec", None)
    if spec is not None:
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
            or spec.semantic_exclusions
        )
    intent = getattr(request, "intent", None)
    return bool(
        getattr(intent, "requested_genres", [])
        or getattr(intent, "requested_languages", [])
        or getattr(intent, "requested_countries", [])
        or getattr(intent, "requested_decades", [])
        or getattr(intent, "exclusions", [])
    )


def _load_frozen_candidate_pool(candidate_pool: pd.DataFrame | None = None) -> pd.DataFrame:
    if candidate_pool is not None:
        frame = candidate_pool.copy()
    elif FROZEN_CANDIDATE_POOL_PATH.exists():
        frame = pd.read_csv(FROZEN_CANDIDATE_POOL_PATH)
    else:
        return pd.DataFrame()
    if "source_id" in frame.columns:
        frame["source_id"] = frame["source_id"].astype(str)
    if "candidate_id" not in frame.columns and "source_id" in frame.columns:
        frame["candidate_id"] = frame["source_id"].astype(str)
    elif "candidate_id" in frame.columns:
        frame["candidate_id"] = frame["candidate_id"].astype(str)
    return frame


def _ensure_b3_predictions(frame: pd.DataFrame, model_dir: str | Path = "mvp/artifacts/models/b3_mvp") -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"b3_predictions_available": 0, "b3_predictions_missing": 0, "b3_scored": False}
    scored = frame.copy()
    if "predicted_preference" in scored.columns:
        existing = pd.to_numeric(scored["predicted_preference"], errors="coerce")
        if existing.notna().any():
            scored["predicted_preference"] = existing
            scored["b3_available"] = existing.notna()
            return scored, {
                "b3_predictions_available": int(existing.notna().sum()),
                "b3_predictions_missing": int(existing.isna().sum()),
                "b3_scored": False,
            }
    try:
        predictions = score_candidates(frame, model_dir=model_dir, top_k=max(len(frame), 1))
    except Exception:
        scored["predicted_preference"] = np.nan
        scored["b3_available"] = False
        return scored, {"b3_predictions_available": 0, "b3_predictions_missing": int(len(scored)), "b3_scored": False}

    prediction_map = predictions.drop_duplicates(subset=["source_id"]).set_index("source_id")["predicted_preference"]
    scored["predicted_preference"] = scored["source_id"].astype(str).map(prediction_map)
    scored["b3_available"] = scored["predicted_preference"].notna()
    return scored, {
        "b3_predictions_available": int(scored["b3_available"].sum()),
        "b3_predictions_missing": int((~scored["b3_available"]).sum()),
        "b3_scored": True,
    }


def _resolve_target_candidate(candidate_id: str | None, candidate_pool: pd.DataFrame | None = None) -> pd.DataFrame:
    if candidate_id is None:
        return pd.DataFrame()
    frame = _load_frozen_candidate_pool(candidate_pool)
    if frame.empty:
        return pd.DataFrame()
    candidate_key = str(candidate_id).strip()
    if "source_id" in frame.columns:
        match = frame.loc[frame["source_id"].astype(str) == candidate_key].copy()
        if not match.empty:
            match.loc[:, "source_id"] = match["source_id"].astype(str)
            match.loc[:, "candidate_id"] = match["source_id"].astype(str)
            return match.reset_index(drop=True)
    if "candidate_id" in frame.columns:
        match = frame.loc[frame["candidate_id"].astype(str) == candidate_key].copy()
        if not match.empty:
            match.loc[:, "source_id"] = match.get("source_id", match["candidate_id"]).astype(str)
            match.loc[:, "candidate_id"] = match["candidate_id"].astype(str)
            return match.reset_index(drop=True)
    return pd.DataFrame()


def _interaction_result(
    *,
    request,
    interaction_mode: str,
    generation_status: str,
    response_summary: str,
    methodology_note: str,
    source_recommendation_run: str,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    target_candidate_id: str | None = None,
    candidate_context: list[dict[str, Any]] | None = None,
    selection_records: list[Any] | None = None,
    qualification_records: list[Any] | None = None,
    llm_call_count: int = 0,
    runtime_llm_calls: int | None = None,
    runtime_input_tokens: int | None = None,
    runtime_output_tokens: int | None = None,
    runtime_total_tokens: int | None = None,
    selected_count: int = 0,
    semantic_classified_count: int = 0,
    candidate_universe_count: int = 0,
    post_constraint_candidate_count: int = 0,
    qualification_candidate_count: int = 0,
    qualified_candidate_count: int = 0,
    contextual_retrieval_used: bool = False,
    abstention_used: bool = True,
) -> RuntimeV4Result:
    response = RecommendationResponse(
        intent=request.intent,
        recommendations=[],
        response_summary=response_summary,
        methodology_note=methodology_note,
    )
    metadata = RuntimeV4Metadata(
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        request_mode=request.request_mode,
        interaction_mode=interaction_mode,
        candidate_universe_count=candidate_universe_count,
        post_constraint_candidate_count=post_constraint_candidate_count,
        contextual_retrieval_used=contextual_retrieval_used,
        qualification_candidate_count=qualification_candidate_count,
        qualified_candidate_count=qualified_candidate_count,
        selected_count=selected_count,
        semantic_classified_count=semantic_classified_count,
        llm_call_count=llm_call_count,
        intent_interpretation_llm_used=request.llm_used,
        intent_interpretation_llm_call_count=request.llm_call_count,
        semantic_qualification_llm_used=False,
        semantic_qualification_llm_call_count=0,
        selection_llm_used=False,
        selection_llm_call_count=0,
        explanation_llm_used=False,
        explanation_llm_call_count=0,
        runtime_llm_calls=runtime_llm_calls,
        runtime_input_tokens=runtime_input_tokens,
        runtime_output_tokens=runtime_output_tokens,
        runtime_total_tokens=runtime_total_tokens,
        validation_passed=True,
        generation_status=generation_status,
        hard_filters_applied=_hard_filters_applied(request),
        novelty_requested=request.novelty_requested,
        reference_status=request.reference_status,
        target_candidate_id=target_candidate_id,
        request_relevance_mode=request.request_relevance_mode,
        abstention_used=abstention_used,
    )
    debug_payload = {
        "request": request.model_dump(),
        "request_spec": request.spec.model_dump() if request.spec is not None else None,
        "retrieval_plan": request.retrieval_plan.model_dump() if request.retrieval_plan is not None else None,
        "candidate_context": candidate_context or [],
        "qualification_records": [record.model_dump() if hasattr(record, "model_dump") else record for record in (qualification_records or [])],
        "selection_records": [record.model_dump() if hasattr(record, "model_dump") else record for record in (selection_records or [])],
        "runtime_llm_calls": runtime_llm_calls,
        "runtime_input_tokens": runtime_input_tokens,
        "runtime_output_tokens": runtime_output_tokens,
        "runtime_total_tokens": runtime_total_tokens,
        "intent_interpretation_llm_used": request.llm_used,
        "intent_interpretation_llm_call_count": request.llm_call_count,
        "semantic_qualification_llm_used": False,
        "semantic_qualification_llm_call_count": 0,
        "selection_llm_used": False,
        "selection_llm_call_count": 0,
        "explanation_llm_used": False,
        "explanation_llm_call_count": 0,
        "abstention_used": abstention_used,
        "interaction_mode": interaction_mode,
    }
    result = RuntimeV4Result(
        response=response,
        runtime_metadata=metadata,
        retrieval_diagnostics={},
        qualification_records=qualification_records or [],
        selection_records=selection_records or [],
        validation_passed=True,
        validation_errors=[],
        debug=debug_payload,
    )
    return result


def _empty_result(
    *,
    request,
    generation_status: str,
    message: str,
    source_recommendation_run: str,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    llm_call_count: int = 0,
    runtime_llm_calls: int | None = None,
    runtime_input_tokens: int | None = None,
    runtime_output_tokens: int | None = None,
    runtime_total_tokens: int | None = None,
) -> RuntimeV4Result:
    response = RecommendationResponse(
        intent=request.intent,
        recommendations=[],
        response_summary=message,
        methodology_note="Request relevance must be grounded before selection. Abstention is valid when no candidate survives qualification.",
    )
    metadata = RuntimeV4Metadata(
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
        run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        request_mode=request.request_mode,
        interaction_mode=getattr(request, "interaction_mode", "recommend"),
        candidate_universe_count=0,
        post_constraint_candidate_count=0,
        contextual_retrieval_used=request.request_mode != "generic",
        qualification_candidate_count=0,
        qualified_candidate_count=0,
        selected_count=0,
        semantic_classified_count=0,
        llm_call_count=llm_call_count,
        intent_interpretation_llm_used=request.llm_used,
        intent_interpretation_llm_call_count=request.llm_call_count,
        semantic_qualification_llm_used=False,
        semantic_qualification_llm_call_count=0,
        selection_llm_used=False,
        selection_llm_call_count=0,
        explanation_llm_used=False,
        explanation_llm_call_count=0,
        runtime_llm_calls=runtime_llm_calls,
        runtime_input_tokens=runtime_input_tokens,
        runtime_output_tokens=runtime_output_tokens,
        runtime_total_tokens=runtime_total_tokens,
        validation_passed=True,
        generation_status=generation_status,
        hard_filters_applied=_hard_filters_applied(request),
        novelty_requested=request.novelty_requested,
        reference_status=request.reference_status,
        target_candidate_id=None,
        request_relevance_mode=request.request_relevance_mode,
        abstention_used=True,
    )
    return RuntimeV4Result(
        response=response,
        runtime_metadata=metadata,
        retrieval_diagnostics={},
        qualification_records=[],
        selection_records=[],
        validation_passed=True,
        validation_errors=[],
        debug={"request": request.model_dump(), "abstained": True, "abstention_used": True},
    )


def run_runtime_v4(
    query: str,
    count: int = DEFAULT_V4_RECOMMENDATION_COUNT,
    candidate_context_size: int = DEFAULT_V4_CANDIDATE_CONTEXT_SIZE,
    candidate_pool: pd.DataFrame | None = None,
    candidate_id: str | None = None,
    watched_ids: list[str] | None = None,
    debug: bool = False,
    runtime_version: str = "runtime_v4",
    intent_prompt_version: str = "intent_v4",
    recommendation_prompt_version: str = "recommendation_v4",
    source_recommendation_run: str = "dynamic_tmdb_runtime_v4",
) -> RuntimeV4Result:
    load_runtime_env()

    with trace(
        "taste_agent_request_v4",
        run_type="chain",
        inputs={"query": query, "candidate_id": candidate_id, "count": count},
        metadata={"runtime_version": runtime_version, "source_recommendation_run": source_recommendation_run},
        tags=_trace_tags("pending", False, "absent", candidate_id),
        project_name=_langsmith_project(),
    ) as root_run:
        with trace(
            "intent_interpretation",
            run_type="chain",
            parent=root_run,
            inputs={"query": query},
            project_name=_langsmith_project(),
        ) as intent_run:
            request = understand_request(query, candidate_id)
            intent_run.end(
                outputs={
                    "request": request.model_dump(),
                    "request_spec": request.spec.model_dump() if request.spec is not None else None,
                    "retrieval_plan": request.retrieval_plan.model_dump() if request.retrieval_plan is not None else None,
                    "llm_used": request.llm_used,
                    "llm_call_count": request.llm_call_count,
                    "runtime_input_tokens": request.runtime_input_tokens,
                    "runtime_output_tokens": request.runtime_output_tokens,
                    "runtime_total_tokens": request.runtime_total_tokens,
                }
            )

        with trace(
            "retrieval_plan",
            run_type="chain",
            parent=root_run,
            inputs={"request_mode": request.request_mode, "query": query},
            project_name=_langsmith_project(),
        ) as retrieval_plan_run:
            retrieval_plan_run.end(
                outputs={
                    "retrieval_plan": request.retrieval_plan.model_dump() if request.retrieval_plan is not None else None,
                    "query_aware_expansion_required": bool(request.retrieval_plan.query_aware_expansion_required) if request.retrieval_plan is not None else False,
                }
            )

        with trace(
            "interaction_routing",
            run_type="chain",
            parent=root_run,
            inputs={"query": query, "candidate_id": candidate_id, "interaction_mode": request.interaction_mode},
            project_name=_langsmith_project(),
        ) as routing_run:
            routing_run.end(
                outputs={
                    "interaction_mode": request.interaction_mode,
                    "request_mode": request.request_mode,
                    "target_candidate_id": candidate_id,
                }
            )

        if request.interaction_mode != "recommend":
            if request.interaction_mode == "explain_target":
                target_frame = _resolve_target_candidate(candidate_id, candidate_pool)
                if target_frame.empty:
                    result = _interaction_result(
                        request=request,
                        interaction_mode="explain_needs_target",
                        generation_status="clarification",
                        response_summary="I can explain the recommendation, but I need the exact target candidate first.",
                        methodology_note="EXPLAIN_TARGET requires a resolvable target candidate_id. No new recommendation slate was generated.",
                        source_recommendation_run=source_recommendation_run,
                        runtime_version=runtime_version,
                        intent_prompt_version=intent_prompt_version,
                        recommendation_prompt_version=recommendation_prompt_version,
                        candidate_context=[],
                        target_candidate_id=None,
                        llm_call_count=request.llm_call_count,
                        runtime_llm_calls=request.llm_call_count,
                        runtime_input_tokens=request.runtime_input_tokens,
                        runtime_output_tokens=request.runtime_output_tokens,
                        runtime_total_tokens=request.runtime_total_tokens,
                        abstention_used=True,
                        )
                else:
                    target_frame, target_b3_report = _ensure_b3_predictions(target_frame)
                    with trace(
                        "target_resolution",
                        run_type="chain",
                        parent=root_run,
                        inputs={"candidate_id": candidate_id},
                        project_name=_langsmith_project(),
                    ) as target_run:
                        target_run.end(
                            outputs={
                                "target_candidate_id": str(target_frame.iloc[0].get("source_id", candidate_id)),
                                "target_candidate_title": str(target_frame.iloc[0].get("title", "")),
                                "target_candidate_found": True,
                                "b3_prediction_report": target_b3_report,
                            }
                        )

                    with trace(
                        "target_explanation",
                        run_type="chain",
                        parent=root_run,
                        inputs={"candidate_id": candidate_id},
                        project_name=_langsmith_project(),
                    ) as explanation_run:
                        explanation = explain_selection(request, target_frame)
                        explanation_run.end(outputs=explanation.semantic_report)

                    response = explanation.response.model_copy()
                    if response.recommendations:
                        recommendation = response.recommendations[0].model_copy()
                        if any(term in query.lower() for term in ("award", "awards", "masterpiece", "acclaim", "reputation")):
                            caveat = "Awards, acclaim, or masterpiece status are not grounded in the current film data, so I can't verify that claim."
                            if recommendation.caveat:
                                recommendation = recommendation.model_copy(update={"caveat": f"{recommendation.caveat} {caveat}"})
                            else:
                                recommendation = recommendation.model_copy(update={"caveat": caveat})
                        response = response.model_copy(
                            update={
                                "recommendations": [recommendation],
                                "response_summary": "Explaining the supplied target candidate with grounded evidence only.",
                                "methodology_note": "No new recommendation slate was generated; the answer stays on the supplied target candidate and uses only grounded evidence.",
                            }
                        )

                    candidate_context_records = explanation.selected_frame.to_dict(orient="records")
                    for record in candidate_context_records:
                        source_id = record.get("source_id") or record.get("candidate_id")
                        if source_id is not None:
                            record["source_id"] = str(source_id)
                            record["candidate_id"] = str(source_id)
                    selected_records = [
                        SelectionRecord(
                            candidate_id=str(row.get("source_id", candidate_id)),
                            selection_rank=int(index + 1),
                            selection_reason="Targeted explanation used the supplied candidate only.",
                            request_relevance=0.0,
                            b3_predicted_preference=float(row.get("predicted_preference")) if pd.notna(row.get("predicted_preference", np.nan)) else None,
                            b3_available=bool(pd.notna(row.get("predicted_preference", np.nan))),
                            qualification_status=str(row.get("qualification_status", "strong")),
                        )
                        for index, row in explanation.selected_frame.iterrows()
                    ]
                    if candidate_context_records and selected_records:
                        candidate_context_records[0].setdefault("selection_rank", 1)
                        candidate_context_records[0].setdefault("selection_reason", "Targeted explanation used the supplied candidate only.")
                        candidate_context_records[0].setdefault("qualification_status", "strong")
                        candidate_context_records[0].setdefault("supported_request_aspects", candidate_context_records[0].get("supported_request_aspects", []))
                    result = RuntimeV4Result(
                        response=response,
                        runtime_metadata=RuntimeV4Metadata(
                            runtime_version=runtime_version,
                            intent_prompt_version=intent_prompt_version,
                            recommendation_prompt_version=recommendation_prompt_version,
                            source_recommendation_run=source_recommendation_run,
                            run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                            request_mode=request.request_mode,
                            interaction_mode=request.interaction_mode,
                            candidate_universe_count=int(len(candidate_context_records)),
                            post_constraint_candidate_count=int(len(candidate_context_records)),
                            contextual_retrieval_used=False,
                            qualification_candidate_count=int(len(candidate_context_records)),
                            qualified_candidate_count=int(len(candidate_context_records)),
                            selected_count=int(len(selected_records)),
                            semantic_classified_count=int(explanation.semantic_report.get("semantic_classified", 0) or 0),
                            llm_call_count=request.llm_call_count,
                            intent_interpretation_llm_used=request.llm_used,
                            intent_interpretation_llm_call_count=request.llm_call_count,
                            semantic_qualification_llm_used=False,
                            semantic_qualification_llm_call_count=0,
                            selection_llm_used=False,
                            selection_llm_call_count=0,
                            explanation_llm_used=False,
                            explanation_llm_call_count=0,
                            runtime_llm_calls=request.llm_call_count,
                            runtime_input_tokens=request.runtime_input_tokens,
                            runtime_output_tokens=request.runtime_output_tokens,
                            runtime_total_tokens=request.runtime_total_tokens,
                            repair_attempted=False,
                            fallback_used=False,
                            validation_passed=True,
                            generation_status="generated" if selected_records else "clarification",
                            hard_filters_applied=_hard_filters_applied(request),
                            novelty_requested=request.novelty_requested,
                            reference_status=request.reference_status,
                            target_candidate_id=str(candidate_id) if candidate_id is not None else None,
                            request_relevance_mode=request.request_relevance_mode,
                            abstention_used=not bool(selected_records),
                        ),
                        retrieval_diagnostics={
                            "interaction_mode": request.interaction_mode,
                            "target_candidate_id": str(candidate_id) if candidate_id is not None else None,
                            "target_candidate_found": bool(candidate_context_records),
                        },
                        qualification_records=[],
                        selection_records=selected_records,
                        validation_passed=True,
                        validation_errors=[],
                        debug={
                            "request": request.model_dump(),
                            "request_spec": request.spec.model_dump() if request.spec is not None else None,
                            "retrieval_plan": request.retrieval_plan.model_dump() if request.retrieval_plan is not None else None,
                                "retrieval_diagnostics": {
                                    "interaction_mode": request.interaction_mode,
                                    "target_candidate_id": str(candidate_id) if candidate_id is not None else None,
                                    "target_candidate_found": bool(candidate_context_records),
                                },
                                "candidate_context": candidate_context_records,
                                "qualification_records": [],
                                "selection_records": [record.model_dump() for record in selected_records],
                                "runtime_llm_calls": request.llm_call_count,
                                "runtime_input_tokens": request.runtime_input_tokens,
                                "runtime_output_tokens": request.runtime_output_tokens,
                                "runtime_total_tokens": request.runtime_total_tokens,
                                "intent_interpretation_llm_used": request.llm_used,
                                "intent_interpretation_llm_call_count": request.llm_call_count,
                            "semantic_qualification_llm_used": False,
                            "semantic_qualification_llm_call_count": 0,
                            "selection_llm_used": False,
                            "selection_llm_call_count": 0,
                            "explanation_llm_used": False,
                            "explanation_llm_call_count": 0,
                            "abstention_used": not bool(selected_records),
                            "interaction_mode": request.interaction_mode,
                        },
                    )
                    with trace(
                        "explanation_validation",
                        run_type="chain",
                        parent=root_run,
                        inputs={"selected_count": int(len(selected_records))},
                        project_name=_langsmith_project(),
                    ) as validation_run:
                        validation = validate_runtime_v4_result(result)
                        validation_run.end(outputs={"passed": validation.passed, "errors": validation.errors})
                    result.validation_passed = validation.passed
                    result.validation_errors = validation.errors
                    result.runtime_metadata.validation_passed = validation.passed
                    if not validation.passed:
                        result.runtime_metadata.generation_status = "abstained" if len(result.response.recommendations) == 0 else "fallback"
                    with trace(
                        "final_response",
                        run_type="chain",
                        parent=root_run,
                        inputs={"recommendation_count": int(len(result.response.recommendations))},
                        project_name=_langsmith_project(),
                    ) as final_response_run:
                        final_response_run.end(
                            outputs={
                                "response_summary": result.response.response_summary,
                                "recommendation_count": int(len(result.response.recommendations)),
                                "generation_status": result.runtime_metadata.generation_status,
                            }
                        )
                    root_run.metadata.update(
                        {
                            "runtime_version": runtime_version,
                            "intent_prompt_version": intent_prompt_version,
                            "recommendation_prompt_version": recommendation_prompt_version,
                            "generation_status": result.runtime_metadata.generation_status,
                            "validation_passed": result.validation_passed,
                            "repair_attempted": result.runtime_metadata.repair_attempted,
                            "fallback_used": result.runtime_metadata.fallback_used,
                            "hard_filters_applied": result.runtime_metadata.hard_filters_applied,
                            "llm_call_count": result.runtime_metadata.llm_call_count,
                            "runtime_llm_calls": result.runtime_metadata.runtime_llm_calls,
                            "runtime_input_tokens": result.runtime_metadata.runtime_input_tokens,
                            "runtime_output_tokens": result.runtime_metadata.runtime_output_tokens,
                            "runtime_total_tokens": result.runtime_metadata.runtime_total_tokens,
                            "request_mode": request.request_mode,
                            "interaction_mode": request.interaction_mode,
                            "intent_interpretation_llm_used": request.llm_used,
                            "intent_interpretation_llm_call_count": request.llm_call_count,
                            "semantic_qualification_llm_used": False,
                            "semantic_qualification_llm_call_count": 0,
                            "selection_llm_used": False,
                            "selection_llm_call_count": 0,
                            "explanation_llm_used": False,
                            "explanation_llm_call_count": 0,
                            "abstention_used": not bool(selected_records),
                        }
                    )
                    root_run.end(outputs=result.model_dump())
                    return result
            clarification_mode = request.interaction_mode in {"explain_needs_target", "provenance", "profile_boundary"}
            if clarification_mode:
                message_map = {
                    "explain_needs_target": "Which recommendation do you mean? I need a target candidate before I can explain it.",
                    "provenance": "None. Goodreads is descriptive only and does not select films.",
                    "profile_boundary": "I can describe grounded taste patterns, but I do not infer personality or identity from viewing history.",
                }
                method_map = {
                    "explain_needs_target": "EXPLAIN requests without a target candidate must ask for clarification instead of generating a new slate.",
                    "provenance": "Goodreads remains descriptive metadata only and does not affect film selection.",
                    "profile_boundary": "Taste Agent models cultural preference patterns only and does not infer sensitive identity traits.",
                }
                result = _interaction_result(
                    request=request,
                    interaction_mode=request.interaction_mode,
                    generation_status="clarification",
                    response_summary=message_map.get(request.interaction_mode, "Clarification required."),
                    methodology_note=method_map.get(request.interaction_mode, "No recommendation slate was generated."),
                    source_recommendation_run=source_recommendation_run,
                    runtime_version=runtime_version,
                    intent_prompt_version=intent_prompt_version,
                    recommendation_prompt_version=recommendation_prompt_version,
                    candidate_context=[],
                    target_candidate_id=str(candidate_id) if candidate_id is not None else None,
                    llm_call_count=request.llm_call_count,
                    runtime_llm_calls=request.llm_call_count,
                    runtime_input_tokens=request.runtime_input_tokens,
                    runtime_output_tokens=request.runtime_output_tokens,
                    runtime_total_tokens=request.runtime_total_tokens,
                    abstention_used=True,
                )
                with trace(
                    "final_response",
                    run_type="chain",
                    parent=root_run,
                    inputs={"recommendation_count": 0},
                    project_name=_langsmith_project(),
                ) as final_response_run:
                    final_response_run.end(
                        outputs={
                            "response_summary": result.response.response_summary,
                            "recommendation_count": 0,
                            "generation_status": result.runtime_metadata.generation_status,
                        }
                    )
                root_run.metadata.update(
                    {
                        "runtime_version": runtime_version,
                        "intent_prompt_version": intent_prompt_version,
                        "recommendation_prompt_version": recommendation_prompt_version,
                        "generation_status": result.runtime_metadata.generation_status,
                        "validation_passed": result.validation_passed,
                        "repair_attempted": result.runtime_metadata.repair_attempted,
                        "fallback_used": result.runtime_metadata.fallback_used,
                        "hard_filters_applied": result.runtime_metadata.hard_filters_applied,
                        "llm_call_count": result.runtime_metadata.llm_call_count,
                        "runtime_llm_calls": result.runtime_metadata.runtime_llm_calls,
                        "runtime_input_tokens": result.runtime_metadata.runtime_input_tokens,
                        "runtime_output_tokens": result.runtime_metadata.runtime_output_tokens,
                        "runtime_total_tokens": result.runtime_metadata.runtime_total_tokens,
                        "request_mode": request.request_mode,
                        "interaction_mode": request.interaction_mode,
                        "intent_interpretation_llm_used": request.llm_used,
                        "intent_interpretation_llm_call_count": request.llm_call_count,
                        "semantic_qualification_llm_used": False,
                        "semantic_qualification_llm_call_count": 0,
                        "selection_llm_used": False,
                        "selection_llm_call_count": 0,
                        "explanation_llm_used": False,
                        "explanation_llm_call_count": 0,
                        "abstention_used": True,
                    }
                )
                root_run.end(outputs=result.model_dump())
                return result

        with trace("catalog_retrieval", run_type="chain", parent=root_run, inputs={"query": query}, project_name=_langsmith_project()) as retrieval_run:
            with trace(
                "semantic_retrieval",
                run_type="chain",
                parent=retrieval_run,
                inputs={"candidate_context_size": candidate_context_size, "request_mode": request.request_mode, "retrieval_plan": request.retrieval_plan.model_dump() if request.retrieval_plan is not None else None},
                project_name=_langsmith_project(),
            ) as semantic_run:
                retrieval = discover_catalog_for_request(
                    request,
                    candidate_limit=max(candidate_context_size * 6, 120),
                    candidate_context_size=candidate_context_size,
                    candidate_pool=candidate_pool,
                    watched_ids=watched_ids,
                )
                candidate_context_frame, b3_prediction_report = _ensure_b3_predictions(retrieval.catalog_frame)
                semantic_run.end(
                    outputs={
                        "candidate_universe_count": retrieval.retrieval_diagnostics.candidate_universe_count,
                        "post_constraint_candidate_count": retrieval.retrieval_diagnostics.post_constraint_candidate_count,
                        "request_relevance_method": retrieval.retrieval_diagnostics.request_relevance_method,
                        "request_relevance_top_source_ids": candidate_context_frame.head(min(5, len(candidate_context_frame)))["source_id"].astype(str).tolist()
                        if not candidate_context_frame.empty
                        else [],
                        "b3_prediction_report": b3_prediction_report,
                    }
                )
            retrieval_run.end(outputs={"retrieval_diagnostics": retrieval.retrieval_diagnostics.__dict__, "augmentation_report": retrieval.augmentation_report})

        with trace(
            "hard_constraints",
            run_type="chain",
            parent=root_run,
            inputs={"candidate_count": int(len(candidate_context_frame))},
            project_name=_langsmith_project(),
        ) as hard_filter_run:
            hard_filter_run.end(
                outputs={
                    "candidate_ids": candidate_context_frame["source_id"].astype(str).tolist() if "source_id" in candidate_context_frame.columns else [],
                    "candidate_count": int(len(candidate_context_frame)),
                }
            )

        with trace(
            "semantic_qualification",
            run_type="chain",
            parent=root_run,
            inputs={"candidate_count": int(len(candidate_context_frame)), "max_candidates": min(candidate_context_size, 20)},
            project_name=_langsmith_project(),
        ) as qualification_run:
            qualified_frame, qualification_records = qualify_candidates(
                request,
                candidate_context_frame,
                max_candidates=min(candidate_context_size, 20),
            )
            qualification_run.end(
                outputs={
                    "qualified_candidate_count": int(len(qualified_frame)),
                    "qualified_candidate_ids": qualified_frame.get("source_id", pd.Series(dtype=str)).astype(str).tolist() if not qualified_frame.empty else [],
                    "llm_usage": qualified_frame.attrs.get("llm_usage", {}),
                }
            )

        qualification_usage = qualified_frame.attrs.get("llm_usage", {}) if hasattr(qualified_frame, "attrs") else {}
        runtime_llm_calls = _sum_usage(request.llm_call_count, int(qualification_usage.get("llm_call_count", 0) or 0))
        runtime_input_tokens = _sum_tokens(request.runtime_input_tokens, qualification_usage.get("runtime_input_tokens"))
        runtime_output_tokens = _sum_tokens(request.runtime_output_tokens, qualification_usage.get("runtime_output_tokens"))
        runtime_total_tokens = _sum_tokens(request.runtime_total_tokens, qualification_usage.get("runtime_total_tokens"))

        if qualified_frame.empty:
            result = _empty_result(
                request=request,
                generation_status="no_match",
                message="No grounded recommendation survived request qualification, so the system is abstaining.",
                source_recommendation_run=source_recommendation_run,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                runtime_llm_calls=runtime_llm_calls,
                runtime_input_tokens=runtime_input_tokens,
                runtime_output_tokens=runtime_output_tokens,
                runtime_total_tokens=runtime_total_tokens,
            )
            result.runtime_metadata.candidate_universe_count = retrieval.retrieval_diagnostics.candidate_universe_count
            result.runtime_metadata.post_constraint_candidate_count = retrieval.retrieval_diagnostics.post_constraint_candidate_count
            result.runtime_metadata.contextual_retrieval_used = retrieval.retrieval_diagnostics.contextual_retrieval_used
            result.runtime_metadata.qualification_candidate_count = int(len(candidate_context_frame))
            result.runtime_metadata.request_relevance_mode = request.request_relevance_mode
            result.runtime_metadata.runtime_llm_calls = runtime_llm_calls
            result.runtime_metadata.runtime_input_tokens = runtime_input_tokens
            result.runtime_metadata.runtime_output_tokens = runtime_output_tokens
            result.runtime_metadata.runtime_total_tokens = runtime_total_tokens
            result.runtime_metadata.llm_call_count = runtime_llm_calls or 0
            result.debug.update(
                {
                    "request": request.model_dump(),
                    "retrieval_diagnostics": asdict(retrieval.retrieval_diagnostics),
                    "retrieval_plan": request.retrieval_plan.model_dump() if request.retrieval_plan is not None else None,
                    "candidate_context": candidate_context_frame.to_dict(orient="records"),
                    "qualification_records": [record.model_dump() for record in qualification_records],
                    "selection_records": [],
                }
            )
            root_run.metadata.update(
                {
                    "runtime_version": runtime_version,
                    "intent_prompt_version": intent_prompt_version,
                    "recommendation_prompt_version": recommendation_prompt_version,
                    "generation_status": result.runtime_metadata.generation_status,
                    "validation_passed": result.validation_passed,
                    "repair_attempted": result.runtime_metadata.repair_attempted,
                    "fallback_used": result.runtime_metadata.fallback_used,
                    "hard_filters_applied": result.runtime_metadata.hard_filters_applied,
                    "llm_call_count": result.runtime_metadata.llm_call_count,
                    "runtime_llm_calls": result.runtime_metadata.runtime_llm_calls,
                    "runtime_input_tokens": result.runtime_metadata.runtime_input_tokens,
                    "runtime_output_tokens": result.runtime_metadata.runtime_output_tokens,
                    "runtime_total_tokens": result.runtime_metadata.runtime_total_tokens,
                    "request_mode": request.request_mode,
                    "interaction_mode": request.interaction_mode,
                    "intent_interpretation_llm_used": request.llm_used,
                    "intent_interpretation_llm_call_count": request.llm_call_count,
                    "semantic_qualification_llm_used": False,
                    "semantic_qualification_llm_call_count": 0,
                    "selection_llm_used": False,
                    "selection_llm_call_count": 0,
                    "explanation_llm_used": False,
                    "explanation_llm_call_count": 0,
                    "abstention_used": True,
                }
            )
            root_run.end(outputs=result.model_dump())
            return result

        with trace(
            "b3_personalization",
            run_type="chain",
            parent=root_run,
            inputs={"qualified_candidate_count": int(len(qualified_frame))},
            project_name=_langsmith_project(),
        ) as b3_run:
            b3_run.end(
                outputs={
                    "qualified_candidate_count": int(len(qualified_frame)),
                    "candidate_ids": qualified_frame["source_id"].astype(str).tolist(),
                }
            )

        with trace(
            "novelty_diagnostic",
            run_type="chain",
            parent=root_run,
            inputs={"request_mode": request.request_mode, "novelty_requested": request.novelty_requested},
            project_name=_langsmith_project(),
        ) as novelty_run:
            novelty_run.end(
                outputs={
                    "applicable": bool(request.novelty_requested or request.request_mode == "novelty"),
                    "novelty_requested": request.novelty_requested,
                    "request_mode": request.request_mode,
                }
            )

        with trace(
            "taste_evidence",
            run_type="chain",
            parent=root_run,
            inputs={"selected_candidate_count": int(len(qualified_frame))},
            project_name=_langsmith_project(),
        ) as taste_run:
            taste_profile = _semantic_profile()
            taste_run.end(
                outputs={
                    "taste_profile_rows": int(len(taste_profile)),
                    "taste_profile_dimensions": taste_profile["dimension"].astype(str).tolist()[:10] if not taste_profile.empty and "dimension" in taste_profile.columns else [],
                }
            )

        with trace(
            "selection",
            run_type="chain",
            parent=root_run,
            inputs={"request_mode": request.request_mode, "recommendation_count": count},
            project_name=_langsmith_project(),
        ) as selection_run:
            selection = select_candidates(request, qualified_frame, recommendation_count=count)
            selection_run.end(outputs=selection.selection_report)

        with trace(
            "selection_validation",
            run_type="chain",
            parent=root_run,
            inputs={"selected_candidate_count": int(len(selection.selected_frame))},
            project_name=_langsmith_project(),
        ) as selection_validation_run:
            selected_ids = selection.selected_frame["source_id"].astype(str).tolist() if "source_id" in selection.selected_frame.columns else []
            record_ids = [record.candidate_id for record in selection.selection_records]
            selection_validation_run.end(
                outputs={
                    "selected_count": int(len(selection.selected_frame)),
                    "selection_record_count": int(len(selection.selection_records)),
                    "selected_ids": selected_ids,
                    "selection_record_ids": record_ids,
                    "passed": selected_ids == record_ids,
                }
            )

        with trace(
            "explanation",
            run_type="chain",
            parent=root_run,
            inputs={"selected_candidate_count": int(len(selection.selected_frame))},
            project_name=_langsmith_project(),
        ) as explanation_run:
            explanation = explain_selection(request, selection.selected_frame)
            explanation_run.end(outputs=explanation.semantic_report)

        candidate_context_records = qualified_frame.to_dict(orient="records")
        for record in candidate_context_records:
            source_id = record.get("source_id") or record.get("candidate_id")
            if source_id is not None:
                record["source_id"] = str(source_id)
                record["candidate_id"] = str(source_id)
        selected_records_by_id = {str(row["source_id"]): row.to_dict() for _, row in explanation.selected_frame.iterrows()}
        for record in candidate_context_records:
            source_id = str(record.get("source_id", ""))
            if source_id in selected_records_by_id:
                selected_row = selected_records_by_id[source_id]
                record.update(
                    {
                        "semantic_evidence": selected_row.get("semantic_evidence", []),
                        "selection_rank": selected_row.get("selection_rank"),
                        "selection_reason": selected_row.get("selection_reason"),
                    }
                )

        result = RuntimeV4Result(
            response=explanation.response,
            runtime_metadata=RuntimeV4Metadata(
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
                run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                request_mode=request.request_mode,
                candidate_universe_count=retrieval.retrieval_diagnostics.candidate_universe_count,
                post_constraint_candidate_count=retrieval.retrieval_diagnostics.post_constraint_candidate_count,
                contextual_retrieval_used=retrieval.retrieval_diagnostics.contextual_retrieval_used,
                qualification_candidate_count=int(len(candidate_context_frame)),
                qualified_candidate_count=int(len(qualified_frame)),
                selected_count=int(len(selection.selected_frame)),
                semantic_classified_count=int(explanation.semantic_report.get("semantic_classified", 0) or 0),
                            llm_call_count=request.llm_call_count,
                intent_interpretation_llm_used=request.llm_used,
                intent_interpretation_llm_call_count=request.llm_call_count,
                semantic_qualification_llm_used=bool(qualification_usage.get("llm_call_count", 0)),
                semantic_qualification_llm_call_count=int(qualification_usage.get("llm_call_count", 0) or 0),
                selection_llm_used=False,
                selection_llm_call_count=0,
                explanation_llm_used=False,
                explanation_llm_call_count=0,
                runtime_llm_calls=runtime_llm_calls,
                runtime_input_tokens=runtime_input_tokens,
                runtime_output_tokens=runtime_output_tokens,
                runtime_total_tokens=runtime_total_tokens,
                repair_attempted=False,
                fallback_used=False,
                validation_passed=True,
                generation_status="generated" if len(selection.selected_frame) else "no_match",
                hard_filters_applied=_hard_filters_applied(request),
                novelty_requested=request.novelty_requested,
                reference_status=request.reference_status,
                target_candidate_id=candidate_id,
                request_relevance_mode=request.request_relevance_mode,
                abstention_used=bool(len(selection.selected_frame) == 0),
            ),
            retrieval_diagnostics=asdict(retrieval.retrieval_diagnostics),
            qualification_records=qualification_records,
            selection_records=selection.selection_records,
            validation_passed=True,
            validation_errors=[],
            debug={
                "request": request.model_dump(),
                "request_spec": request.spec.model_dump() if request.spec is not None else None,
                "retrieval_plan": request.retrieval_plan.model_dump() if request.retrieval_plan is not None else None,
                "retrieval_diagnostics": asdict(retrieval.retrieval_diagnostics),
                "augmentation_report": retrieval.augmentation_report,
                "candidate_context": candidate_context_records,
                "qualification_records": [record.model_dump() for record in qualification_records],
                "selection_records": [record.model_dump() for record in selection.selection_records],
                            "runtime_llm_calls": request.llm_call_count,
                            "runtime_input_tokens": request.runtime_input_tokens,
                            "runtime_output_tokens": request.runtime_output_tokens,
                            "runtime_total_tokens": request.runtime_total_tokens,
                "intent_interpretation_llm_used": request.llm_used,
                "intent_interpretation_llm_call_count": request.llm_call_count,
                "semantic_qualification_llm_used": bool(qualification_usage.get("llm_call_count", 0)),
                "semantic_qualification_llm_call_count": int(qualification_usage.get("llm_call_count", 0) or 0),
                "selection_llm_used": False,
                "selection_llm_call_count": 0,
                "explanation_llm_used": False,
                "explanation_llm_call_count": 0,
                "abstention_used": bool(len(selection.selected_frame) == 0),
            },
        )

        with trace(
            "explanation_validation",
            run_type="chain",
            parent=root_run,
            inputs={"selected_count": int(len(selection.selected_frame))},
            project_name=_langsmith_project(),
        ) as validation_run:
            validation = validate_runtime_v4_result(result)
            validation_run.end(outputs={"passed": validation.passed, "errors": validation.errors})
        result.validation_passed = validation.passed
        result.validation_errors = validation.errors
        result.runtime_metadata.validation_passed = validation.passed
        if not validation.passed:
            result.runtime_metadata.generation_status = "validation_failed"

        with trace(
            "final_response",
            run_type="chain",
            parent=root_run,
            inputs={"recommendation_count": int(len(result.response.recommendations))},
            project_name=_langsmith_project(),
        ) as final_response_run:
            final_response_run.end(
                outputs={
                    "response_summary": result.response.response_summary,
                    "recommendation_count": int(len(result.response.recommendations)),
                    "generation_status": result.runtime_metadata.generation_status,
                }
            )

        root_run.metadata.update(
            {
                "runtime_version": runtime_version,
                "intent_prompt_version": intent_prompt_version,
                "recommendation_prompt_version": recommendation_prompt_version,
                "generation_status": result.runtime_metadata.generation_status,
                "validation_passed": result.validation_passed,
                "repair_attempted": result.runtime_metadata.repair_attempted,
                "fallback_used": result.runtime_metadata.fallback_used,
                "hard_filters_applied": result.runtime_metadata.hard_filters_applied,
                "llm_call_count": result.runtime_metadata.llm_call_count,
                "runtime_llm_calls": result.runtime_metadata.runtime_llm_calls,
                "runtime_input_tokens": result.runtime_metadata.runtime_input_tokens,
                "runtime_output_tokens": result.runtime_metadata.runtime_output_tokens,
                "runtime_total_tokens": result.runtime_metadata.runtime_total_tokens,
                "request_mode": request.request_mode,
                "interaction_mode": request.interaction_mode,
                "intent_interpretation_llm_used": request.llm_used,
                "intent_interpretation_llm_call_count": request.llm_call_count,
                "semantic_qualification_llm_used": bool(qualification_usage.get("llm_call_count", 0)),
                "semantic_qualification_llm_call_count": int(qualification_usage.get("llm_call_count", 0) or 0),
                "selection_llm_used": False,
                "selection_llm_call_count": 0,
                "explanation_llm_used": False,
                "explanation_llm_call_count": 0,
                "abstention_used": bool(len(selection.selected_frame) == 0),
            }
        )
        root_run.end(outputs=result.model_dump())
        return result
