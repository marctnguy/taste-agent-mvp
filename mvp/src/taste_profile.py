from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from mvp.src.data import infer_work_id_column
from mvp.src.semantics import SEMANTIC_COLUMNS, coerce_semantic_frame, ensure_semantic_columns


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _evidence_confidence(x: np.ndarray) -> float:
    """
    Round 1 heuristic:
    70% evidence volume + 30% variation.

    Evidence volume grows with observation count; variation is derived from
    the spread of the dimension values within the observed set.
    """

    n = len(x)
    if n == 0:
        return 0.0
    evidence_volume = float(min(1.0, n / 30.0))
    variation = float(min(1.0, np.std(x) / 0.5))
    return float(max(0.0, min(1.0, 0.7 * evidence_volume + 0.3 * variation)))


def build_taste_profile(
    observations: pd.DataFrame,
    semantic_vectors: pd.DataFrame,
    id_column: str | None = None,
    target_col: str = "preference_weight",
    dimensions: Sequence[str] | None = None,
) -> pd.DataFrame:
    obs = observations.copy()
    vectors = ensure_semantic_columns(semantic_vectors)
    key = id_column or infer_work_id_column(obs)
    if key not in obs.columns or key not in vectors.columns:
        raise KeyError(f"Unable to join observations and semantic vectors on '{key}'.")
    obs = obs.copy()
    vectors = vectors.copy()
    obs[key] = obs[key].astype(str)
    vectors[key] = vectors[key].astype(str)
    merged = obs.merge(vectors, on=key, how="inner")
    if target_col not in merged.columns:
        raise KeyError(f"Missing target column: {target_col}")

    target = pd.to_numeric(merged[target_col], errors="coerce").to_numpy(dtype=float)
    rows = []
    selected_dimensions = list(dimensions) if dimensions is not None else list(SEMANTIC_COLUMNS)
    for dimension in selected_dimensions:
        feature = pd.to_numeric(merged[dimension], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(feature) & np.isfinite(target)
        n = int(mask.sum())
        x = feature[mask]
        y = target[mask]
        evidence_count = int(np.sum(x >= 0.25)) if n else 0
        rows.append(
            {
                "dimension": dimension,
                "prevalence": float(np.mean(x)) if n else np.nan,
                "pearson_preference_association": _safe_corr(x, y),
                "evidence_count": evidence_count,
                "evidence_confidence": _evidence_confidence(x),
            }
        )
    return pd.DataFrame(rows)


def top_associations(profile: pd.DataFrame, kind: str = "positive", n: int = 10) -> pd.DataFrame:
    frame = profile.copy()
    frame = frame.dropna(subset=["pearson_preference_association"])
    if kind == "positive":
        frame = frame.sort_values(
            ["pearson_preference_association", "evidence_confidence", "evidence_count"],
            ascending=[False, False, False],
        )
    else:
        frame = frame.sort_values(
            ["pearson_preference_association", "evidence_confidence", "evidence_count"],
            ascending=[True, False, False],
        )
    return frame.head(n).reset_index(drop=True)
