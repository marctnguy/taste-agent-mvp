from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mvp.src.data import add_decade_column, parse_multi_value_column
from mvp.src.semantics import SEMANTIC_COLUMNS


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return _safe_corr(_rankdata(y_true), _rankdata(y_pred))


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = len(y_true) if k is None else min(k, len(y_true))
    order = np.argsort(y_score)[::-1][:k]
    ideal = np.argsort(y_true)[::-1][:k]

    def _dcg(indices: np.ndarray) -> float:
        gains = np.maximum(y_true[indices] - np.min(y_true), 0.0)
        discounts = np.log2(np.arange(2, len(indices) + 2))
        return float(np.sum(gains / discounts))

    actual_dcg = _dcg(order)
    ideal_dcg = _dcg(ideal)
    return float(actual_dcg / ideal_dcg) if ideal_dcg > 0 else float("nan")


def evaluate_regression(y_true: Sequence[float], y_pred: Sequence[float], ndcg_k: int | None = 10) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) == 0:
        return {"n": 0.0, "mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "spearman": float("nan"), "ndcg_at_k": float("nan")}
    metrics = {
        "n": float(len(y_true_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "pearson": _safe_corr(y_true_arr, y_pred_arr),
        "spearman": _spearman_corr(y_true_arr, y_pred_arr),
        "ndcg_at_k": _ndcg_at_k(y_true_arr, y_pred_arr, k=ndcg_k),
    }
    return metrics


@dataclass
class ConventionalMetadataFeatureBuilder:
    genre_column_candidates: tuple[str, ...] = ("tmdb_genres", "genres", "genre")
    language_column_candidates: tuple[str, ...] = ("tmdb_original_language", "language", "original_language")
    country_column_candidates: tuple[str, ...] = ("tmdb_production_countries", "country", "production_country")
    year_column_candidates: tuple[str, ...] = ("release_year", "year")
    decade_column: str = "release_decade"
    genre_values: list[str] = field(default_factory=list)
    language_values: list[str] = field(default_factory=list)
    country_values: list[str] = field(default_factory=list)
    decade_values: list[str] = field(default_factory=list)
    year_mean: float = 0.0
    year_std: float = 1.0

    def _find_column(self, df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _collect_tokens(self, df: pd.DataFrame, column: str | None) -> list[str]:
        if column is None:
            return []
        tokens: set[str] = set()
        for values in parse_multi_value_column(df[column]):
            tokens.update(token.lower() for token in values)
        return sorted(tokens)

    def fit(self, df: pd.DataFrame) -> "ConventionalMetadataFeatureBuilder":
        frame = df.copy()
        year_col = self._find_column(frame, self.year_column_candidates)
        if year_col is not None:
            years = pd.to_numeric(frame[year_col], errors="coerce")
            self.year_mean = float(years.mean()) if years.notna().any() else 0.0
            self.year_std = float(years.std(ddof=0)) if years.notna().any() else 1.0
            if not np.isfinite(self.year_std) or self.year_std == 0:
                self.year_std = 1.0
        self.genre_values = self._collect_tokens(frame, self._find_column(frame, self.genre_column_candidates))
        self.language_values = self._collect_tokens(frame, self._find_column(frame, self.language_column_candidates))
        self.country_values = self._collect_tokens(frame, self._find_column(frame, self.country_column_candidates))
        if self.decade_column in frame.columns:
            self.decade_values = sorted({str(value) for value in frame[self.decade_column].dropna().astype(str).tolist()})
        elif year_col is not None:
            decades = pd.to_numeric(frame[year_col], errors="coerce")
            decade_labels = (np.floor(decades / 10) * 10).dropna().astype(int).astype(str) + "s"
            self.decade_values = sorted(set(decade_labels.tolist()))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        if self.decade_column not in frame.columns:
            frame = add_decade_column(
                frame,
                year_col=self._find_column(frame, self.year_column_candidates),
                output_col=self.decade_column,
            )

        features: dict[str, pd.Series] = {}

        year_col = self._find_column(frame, self.year_column_candidates)
        if year_col is not None:
            years = pd.to_numeric(frame[year_col], errors="coerce")
            features["release_year_z"] = ((years - self.year_mean) / self.year_std).fillna(0.0)
        else:
            features["release_year_z"] = pd.Series(0.0, index=frame.index)

        def add_one_hot(prefix: str, column: str | None, known_values: list[str]) -> None:
            if column is None or not known_values:
                return
            parsed = parse_multi_value_column(frame[column])
            for value in known_values:
                features[f"{prefix}__{value}"] = pd.Series(
                    [1.0 if value in {token.lower() for token in values} else 0.0 for values in parsed],
                    index=frame.index,
                )

        add_one_hot("genre", self._find_column(frame, self.genre_column_candidates), self.genre_values)
        add_one_hot("language", self._find_column(frame, self.language_column_candidates), self.language_values)
        add_one_hot("country", self._find_column(frame, self.country_column_candidates), self.country_values)

        if self.decade_values:
            decade_series = frame[self.decade_column].astype(str).fillna("")
            for value in self.decade_values:
                features[f"decade__{value}"] = (decade_series == value).astype(float)

        return pd.DataFrame(features, index=frame.index).fillna(0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class ModelBundle:
    name: str
    model: RidgeCV
    feature_space: str
    feature_builder: ConventionalMetadataFeatureBuilder | None = None
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = "preference_weight"
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_ridge_model(
    train_df: pd.DataFrame,
    target_col: str = "preference_weight",
    feature_builder: ConventionalMetadataFeatureBuilder | None = None,
    feature_columns: Sequence[str] | None = None,
    model_name: str = "ridge_model",
) -> ModelBundle:
    if target_col not in train_df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    if feature_builder is not None:
        X = feature_builder.transform(train_df)
        feature_space = "metadata"
        feature_columns = list(X.columns)
    else:
        if feature_columns is None:
            feature_columns = [column for column in SEMANTIC_COLUMNS if column in train_df.columns]
        X = train_df.reindex(columns=list(feature_columns)).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feature_space = "semantic"

    y = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(y)
    X = X.loc[valid_mask]
    y = y[valid_mask]

    alphas = np.logspace(-3, 3, 25)
    model = RidgeCV(alphas=alphas, fit_intercept=True)
    model.fit(X.to_numpy(dtype=float), y)

    return ModelBundle(
        name=model_name,
        model=model,
        feature_space=feature_space,
        feature_builder=feature_builder,
        feature_columns=list(feature_columns),
        target_column=target_col,
        metadata={"alphas": alphas.tolist()},
    )


def train_semantic_model(
    train_df: pd.DataFrame,
    target_col: str = "preference_weight",
    semantic_columns: Sequence[str] = SEMANTIC_COLUMNS,
    model_name: str = "semantic_ridge",
) -> ModelBundle:
    return fit_ridge_model(
        train_df,
        target_col=target_col,
        feature_columns=semantic_columns,
        feature_builder=None,
        model_name=model_name,
    )


def _build_feature_matrix(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    if bundle.feature_space == "metadata":
        if bundle.feature_builder is None:
            raise ValueError("Metadata model bundle is missing its feature builder.")
        return bundle.feature_builder.transform(df).reindex(columns=bundle.feature_columns, fill_value=0.0)
    return df.reindex(columns=bundle.feature_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def predict_with_bundle(bundle: ModelBundle, df: pd.DataFrame) -> np.ndarray:
    matrix = _build_feature_matrix(bundle, df)
    return bundle.model.predict(matrix.to_numpy(dtype=float))


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(Path(path))


def train_and_evaluate_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "preference_weight",
) -> dict[str, dict[str, Any]]:
    baseline_builder = ConventionalMetadataFeatureBuilder().fit(train_df)
    baseline_bundle = fit_ridge_model(
        train_df,
        target_col=target_col,
        feature_builder=baseline_builder,
        model_name="baseline_metadata",
    )
    semantic_bundle = train_semantic_model(train_df, target_col=target_col, model_name="semantic_ridge")

    results = {}
    for name, bundle in {"baseline": baseline_bundle, "semantic": semantic_bundle}.items():
        predictions = predict_with_bundle(bundle, test_df)
        results[name] = {
            "bundle": bundle,
            "predictions": predictions,
            "metrics": evaluate_regression(test_df[target_col].to_numpy(dtype=float), predictions),
        }
    return results
