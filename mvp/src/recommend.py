from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mvp.src.mvp_deployment import build_taste_evidence, score_candidates as score_candidates_v1


def filter_watched_candidates(
    candidates: pd.DataFrame,
    watched_ids: Iterable[str] | None = None,
    id_column: str = "source_id",
) -> pd.DataFrame:
    frame = candidates.copy()
    if watched_ids is None or id_column not in frame.columns:
        return frame
    watched = {str(value) for value in watched_ids}
    return frame[~frame[id_column].astype(str).isin(watched)].copy()


def score_candidates(
    candidates: pd.DataFrame | str | Path,
    model_dir: str | Path | None = None,
    watched_ids: Iterable[str] | None = None,
    top_k: int = 10,
    candidate_semantic_vectors: pd.DataFrame | str | Path | None = None,
    taste_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return score_candidates_v1(
        candidates,
        model_dir=model_dir or "mvp/artifacts/models/b3_mvp",
        watched_ids=watched_ids,
        top_k=top_k,
        candidate_semantic_vectors=candidate_semantic_vectors,
        taste_profile=taste_profile,
    )


def rank_candidates(
    candidates: pd.DataFrame | str | Path,
    model_dir: str | Path | None = None,
    watched_ids: Iterable[str] | None = None,
    top_n: int = 10,
    candidate_semantic_vectors: pd.DataFrame | str | Path | None = None,
    taste_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return score_candidates(
        candidates,
        model_dir=model_dir,
        watched_ids=watched_ids,
        top_k=top_n,
        candidate_semantic_vectors=candidate_semantic_vectors,
        taste_profile=taste_profile,
    )


def explain_candidate(
    candidate_row: pd.Series,
    taste_profile: pd.DataFrame | None = None,
    top_n: int = 3,
) -> str:
    evidence = build_taste_evidence(candidate_row, taste_profile)
    if evidence["evidence_status"] == "unavailable":
        return "Grounded explanation unavailable until a taste profile is built."
    positive = evidence.get("positive_matches", [])[:top_n]
    if not positive:
        return "This title matches the current profile, but no strong semantic evidence was available."
    reasons = [str(item["dimension"]).replace("_", " ") for item in positive]
    return "Grounded reasons: " + ", ".join(reasons)
