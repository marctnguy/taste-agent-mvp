from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mvp.src.candidates import load_condition_a_enriched, load_consumption_history
from mvp.src.generative_v4.intent_chain import RequestUnderstanding
from mvp.src.generative_v4.schemas import SelectionRecord
from mvp.src.retrieval.catalog_retrieval import build_canonical_film_documents, embed_documents


@dataclass(frozen=True)
class SelectionResult:
    selected_frame: pd.DataFrame
    selection_records: list[SelectionRecord]
    selection_report: dict[str, Any]


def _status_rank(value: str | None) -> int:
    if value == "strong":
        return 3
    if value == "partial":
        return 2
    if value == "pending":
        return 1
    return 0


def _candidate_embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def _build_history_embeddings() -> pd.DataFrame:
    history = load_consumption_history()
    enriched = load_condition_a_enriched()
    if "source_id" not in history.columns or "source_id" not in enriched.columns:
        return pd.DataFrame()
    merged = history.merge(enriched, on="source_id", how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    documents = build_canonical_film_documents(merged)
    embeddings, _, _ = embed_documents(documents)
    embeddings["source_id"] = embeddings["source_id"].astype(str)
    return embeddings


def _novelty_scores(frame: pd.DataFrame) -> pd.Series:
    emb_cols = _candidate_embedding_columns(frame)
    if not emb_cols:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    history_embeddings = _build_history_embeddings()
    if history_embeddings.empty:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    history_cols = _candidate_embedding_columns(history_embeddings)
    if not history_cols:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    candidate_matrix = frame.loc[:, emb_cols].to_numpy(dtype=float)
    history_matrix = history_embeddings.loc[:, history_cols].to_numpy(dtype=float)
    candidate_matrix = candidate_matrix / np.clip(np.linalg.norm(candidate_matrix, axis=1, keepdims=True), 1e-12, None)
    history_matrix = history_matrix / np.clip(np.linalg.norm(history_matrix, axis=1, keepdims=True), 1e-12, None)
    similarities = candidate_matrix @ history_matrix.T
    nearest = similarities.max(axis=1)
    return pd.Series(1.0 - nearest, index=frame.index, dtype=float)


def select_candidates(
    request: RequestUnderstanding,
    qualified_frame: pd.DataFrame,
    *,
    recommendation_count: int = 5,
) -> SelectionResult:
    frame = qualified_frame.copy().reset_index(drop=True)
    if frame.empty:
        return SelectionResult(frame, [], {"selected_count": 0, "abstained": True})

    if "request_relevance" not in frame.columns:
        frame["request_relevance"] = 0.0
    if "b3_applicability_distance" not in frame.columns:
        frame["b3_applicability_distance"] = np.nan
    if "predicted_preference" not in frame.columns:
        frame["predicted_preference"] = np.nan
    frame["predicted_preference"] = pd.to_numeric(frame["predicted_preference"], errors="coerce")
    if "qualification_status" not in frame.columns:
        frame["qualification_status"] = "pending"
    if "source_id" not in frame.columns and "candidate_id" in frame.columns:
        frame["source_id"] = frame["candidate_id"].astype(str)
    if "source_id" not in frame.columns:
        frame["source_id"] = frame.index.astype(str)

    frame["qualification_rank"] = frame["qualification_status"].map(_status_rank) if "qualification_status" in frame.columns else 0
    b3_available = bool(pd.to_numeric(frame["predicted_preference"], errors="coerce").notna().any())

    if request.request_mode == "novelty" or request.novelty_requested:
        frame["novelty_score"] = _novelty_scores(frame)
        frame = frame.sort_values(
            ["novelty_score", "request_relevance", "predicted_preference", "source_id"],
            ascending=[False, False, False, True],
            na_position="last",
        )
        selection_reason = "Novelty-ranked selection using nearest-neighbour distance from the consumed-film embedding distribution."
    elif request.request_mode == "generic":
        frame = frame.sort_values(
            ["predicted_preference", "request_relevance", "b3_applicability_distance", "source_id"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        selection_reason = (
            "Generic discovery ranked by frozen B3 compatibility with request relevance as a tie-breaker."
            if b3_available
            else "Generic discovery ranked by request relevance with stable tie-breakers because B3 was unavailable for this slate."
        )
    else:
        frame = frame.sort_values(
            ["qualification_rank", "request_relevance", "predicted_preference", "b3_applicability_distance", "source_id"],
            ascending=[False, False, False, True, True],
            na_position="last",
        )
        selection_reason = (
            "Contextual selection prioritized request-fit first, then frozen B3 compatibility among qualified candidates."
            if b3_available
            else "Contextual selection prioritized request-fit first with stable tie-breakers because B3 was unavailable for this slate."
        )

    selected = frame.head(recommendation_count).copy().reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["selection_reason"] = selection_reason
    if "novelty_score" not in selected.columns:
        selected["novelty_score"] = np.nan

    records = [
        SelectionRecord(
            candidate_id=str(row["source_id"]),
            selection_rank=int(index + 1),
            selection_reason=selection_reason,
            request_relevance=float(row.get("request_relevance", 0.0)) if pd.notna(row.get("request_relevance", np.nan)) else 0.0,
            b3_predicted_preference=float(row.get("predicted_preference")) if pd.notna(row.get("predicted_preference", np.nan)) else None,
            b3_available=bool(pd.notna(row.get("predicted_preference", np.nan))),
            qualification_status=str(row.get("qualification_status", "pending")),
        )
        for index, row in selected.iterrows()
    ]
    report = {
        "selected_count": int(len(selected)),
        "abstained": len(selected) == 0,
        "novelty_requested": bool(request.novelty_requested),
        "request_mode": request.request_mode,
        "b3_available_count": int(pd.to_numeric(frame["predicted_preference"], errors="coerce").notna().sum()),
        "b3_available_selected_count": int(pd.to_numeric(selected["predicted_preference"], errors="coerce").notna().sum()),
    }
    total_candidates = int(len(frame))
    report["b3_available_coverage"] = float(report["b3_available_count"] / total_candidates) if total_candidates else 0.0
    if "novelty_score" in frame.columns:
        numeric_novelty = pd.to_numeric(frame["novelty_score"], errors="coerce")
        if numeric_novelty.notna().any():
            report["novelty_diagnostic"] = {
                "min": float(numeric_novelty.min()),
                "mean": float(numeric_novelty.mean()),
                "max": float(numeric_novelty.max()),
            }
    return SelectionResult(selected, records, report)
