from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mvp.src.candidates import load_consumption_history
from mvp.src.generative.schemas import RecommendationItem, RecommendationResponse
from mvp.src.generative_v4.intent_chain import RequestUnderstanding
from mvp.src.generative_v4.schemas import SelectionRecord
from mvp.src.mvp_deployment import build_taste_evidence
from mvp.src.prepare import classify_semantic_vectors
from mvp.src.semantics import SemanticVectorStore
from mvp.src.taste_profile import build_taste_profile


SELECTION_SEMANTIC_CACHE_PATH = Path("mvp/artifacts/generative/runtime_v4_selected_semantic_vectors.csv")


@dataclass(frozen=True)
class ExplanationResult:
    response: RecommendationResponse
    selected_frame: pd.DataFrame
    semantic_report: dict[str, Any]


def _semantic_profile() -> pd.DataFrame:
    history = load_consumption_history()
    semantic_vectors = SemanticVectorStore("mvp/artifacts/semantic_vectors").load()
    if history.empty or semantic_vectors.empty:
        return pd.DataFrame()
    try:
        return build_taste_profile(history, semantic_vectors)
    except Exception:
        return pd.DataFrame()


def _selected_semantic_frame(selected_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if selected_frame.empty:
        return pd.DataFrame(), {"semantic_classified": 0, "semantic_total": 0, "coverage": 0.0, "failures": []}
    frame = selected_frame.copy()
    for column in ["canonical_id", "source_id"]:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    if "canonical_id" not in frame.columns:
        frame["canonical_id"] = frame["source_id"].astype(str)
    try:
        semantic_vectors, report = classify_semantic_vectors(frame, cache_path=SELECTION_SEMANTIC_CACHE_PATH)
    except Exception:
        return pd.DataFrame(), {"semantic_classified": 0, "semantic_total": int(len(frame)), "coverage": 0.0, "failures": ["semantic_classification_unavailable"]}
    semantic_vectors = semantic_vectors.rename(columns={"canonical_id": "source_id"})
    semantic_vectors["source_id"] = semantic_vectors["source_id"].astype(str)
    return semantic_vectors, report


def _build_recommendation_item(
    row: pd.Series,
    request: RequestUnderstanding,
    taste_profile: pd.DataFrame,
) -> RecommendationItem:
    candidate_series = row.copy()
    evidence = build_taste_evidence(candidate_series, taste_profile) if not taste_profile.empty else {"evidence_status": "unavailable", "positive_matches": [], "possible_mismatches": []}
    row["semantic_evidence"] = list(evidence.get("positive_matches", [])) + list(evidence.get("possible_mismatches", []))
    taste_signals = [str(item.get("dimension")) for item in evidence.get("positive_matches", []) if item.get("dimension")]
    if not taste_signals and row.get("semantic_evidence"):
        taste_signals = [str(item.get("dimension")) for item in row.get("semantic_evidence", [])[:3] if isinstance(item, dict)]

    supported = [str(value) for value in row.get("supported_request_aspects", []) if value]
    unsupported = [str(value) for value in row.get("unsupported_request_aspects", []) if value]
    qualification_status = str(row.get("qualification_status", "pending"))
    request_match = row.get("request_match")
    caveat = row.get("caveat")
    if not request_match and request.request_mode == "generic":
        request_match = "Generic discovery match."
    if not request_match and supported:
        request_match = f"Supported request aspects: {', '.join(supported[:5])}"
    if qualification_status == "partial" and not caveat and unsupported:
        caveat = f"Unsupported aspects preserved as caveats: {', '.join(unsupported[:5])}."
    if qualification_status == "unsupported":
        caveat = caveat or "Unsupported candidates are not eligible for selection."
    if not taste_signals and caveat is None:
        caveat = "Grounded semantic evidence was limited for this candidate."

    why_parts = []
    if request_match:
        why_parts.append(str(request_match))
    if request.request_mode == "reference" and (request.reference_title or request.reference_tmdb_id):
        if "reference_recommendations" in " ".join(str(item) for item in row.get("candidate_provenance", [])).lower():
            why_parts.append("Reference context was used to ground this selection.")
    if request.request_mode == "novelty" or request.novelty_requested:
        if pd.notna(row.get("novelty_score")):
            why_parts.append("Novelty diagnostics favored this candidate over closer history matches.")
    if taste_signals:
        why_parts.append(f"Taste evidence: {', '.join(taste_signals[:5])}.")
    if caveat and qualification_status == "partial":
        why_parts.append(f"Caveat: {caveat}")
    why_it_may_fit = " ".join(why_parts) if why_parts else "Grounded request fit and taste evidence were limited, so the explanation is intentionally cautious."

    return RecommendationItem(
        candidate_id=str(row["source_id"]),
        title=str(row.get("title") or ""),
        year=int(row["year"]) if pd.notna(row.get("year")) else None,
        why_it_may_fit=why_it_may_fit,
        taste_signals=taste_signals[:5],
        request_match=request_match,
        caveat=caveat,
    )


def explain_selection(
    request: RequestUnderstanding,
    selected_frame: pd.DataFrame,
) -> ExplanationResult:
    frame = selected_frame.copy().reset_index(drop=True)
    if frame.empty:
        response = RecommendationResponse(
            intent=request.intent,
            recommendations=[],
            response_summary="No grounded recommendation survived request qualification, so the system is abstaining.",
            methodology_note="Request relevance had to be grounded before any selection; no candidate met that bar.",
        )
        return ExplanationResult(response=response, selected_frame=frame, semantic_report={"semantic_classified": 0, "coverage": 0.0})

    taste_profile = _semantic_profile()
    semantic_vectors, semantic_report = _selected_semantic_frame(frame)
    b3_active = bool(pd.to_numeric(frame.get("predicted_preference", pd.Series(dtype=float)), errors="coerce").notna().any())
    if not semantic_vectors.empty:
        frame = frame.merge(semantic_vectors, on="source_id", how="left", suffixes=("", "_semantic"))
    items = []
    semantic_evidence_rows = []
    for _, row in frame.iterrows():
        mutable_row = row.copy()
        item = _build_recommendation_item(mutable_row, request, taste_profile)
        items.append(item)
        semantic_evidence_rows.append(mutable_row.get("semantic_evidence", []))
    if len(semantic_evidence_rows) == len(frame):
        frame = frame.copy()
        frame["semantic_evidence"] = semantic_evidence_rows
    response_summary = f"Returned {len(items)} grounded recommendation{'s' if len(items) != 1 else ''} from the qualified catalog."
    if b3_active:
        methodology_note = (
            "Request relevance was checked first, then frozen B3 compatibility was used to order only the qualified candidates, "
            "and the explanation only verbalizes grounded evidence already attached to the selected films."
        )
    else:
        methodology_note = (
            "Request relevance was checked first, then the qualified candidates were ordered with grounded tie-breakers because frozen B3 scores were unavailable, "
            "and the explanation only verbalizes grounded evidence already attached to the selected films."
        )
    response = RecommendationResponse(
        intent=request.intent,
        recommendations=items,
        response_summary=response_summary,
        methodology_note=methodology_note,
    )
    return ExplanationResult(response=response, selected_frame=frame, semantic_report=semantic_report)
