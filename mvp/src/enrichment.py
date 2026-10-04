from __future__ import annotations

import numpy as np
import pandas as pd

from mvp.src.data import add_preference_weight_column, infer_work_id_column
from mvp.src.semantics import SEMANTIC_COLUMNS, coerce_semantic_frame, ensure_semantic_columns


def merge_observations_with_vectors(
    observations: pd.DataFrame,
    semantic_vectors: pd.DataFrame,
    id_column: str | None = None,
) -> pd.DataFrame:
    obs = observations.copy()
    vectors = ensure_semantic_columns(semantic_vectors)
    key = id_column or infer_work_id_column(obs)
    if key not in obs.columns or key not in vectors.columns:
        raise KeyError(f"Unable to merge observations and vectors on '{key}'.")
    return obs.merge(vectors, on=key, how="inner")


def build_cross_media_preference_signal(
    history: pd.DataFrame,
    semantic_vectors: pd.DataFrame,
    id_column: str | None = None,
    target_col: str = "preference_weight",
) -> pd.DataFrame:
    """
    Build a source-specific semantic signal without pretending books and films are the same observations.

    The output is a per-dimension weighted mean that can be used later as an auxiliary
    preference signal, but it is intentionally not mixed into the film model by default.
    """

    merged = merge_observations_with_vectors(history, semantic_vectors, id_column=id_column)
    if target_col not in merged.columns:
        merged = add_preference_weight_column(merged)
    weights = pd.to_numeric(merged[target_col], errors="coerce").fillna(0.0).to_numpy()
    features = merged[SEMANTIC_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weight_sum = float(np.abs(weights).sum())
    if weight_sum == 0:
        return pd.DataFrame([{"dimension": dim, "weighted_mean": 0.0} for dim in SEMANTIC_COLUMNS])
    values = (features.to_numpy(dtype=float).T @ weights) / weight_sum
    return pd.DataFrame({"dimension": SEMANTIC_COLUMNS, "weighted_mean": values})
