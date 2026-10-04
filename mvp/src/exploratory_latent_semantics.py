from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mvp.src.config import get_api_keys
from mvp.src.data import load_table, normalize_column_names
from mvp.src.models import ConventionalMetadataFeatureBuilder, evaluate_regression, predict_with_bundle
from mvp.src.semantics import SEMANTIC_COLUMNS
from mvp.src.prepare import _frozen_split_bins


EXPLORATORY_DIR = Path("mvp/artifacts/experiments/exploratory_latent_semantics")
EMBEDDING_CACHE_DIR = EXPLORATORY_DIR / "embedding_cache"
DOCUMENTS_PATH = EXPLORATORY_DIR / "canonical_film_documents.csv"
MANIFEST_PATH = EXPLORATORY_DIR / "embedding_manifest.csv"
EMBEDDINGS_PATH = EMBEDDING_CACHE_DIR / "film_embeddings.csv"
CV_RESULTS_PATH = EXPLORATORY_DIR / "cv_results.json"
FOLD_SELECTIONS_PATH = EXPLORATORY_DIR / "fold_selections.json"
DIAGNOSTICS_PATH = EXPLORATORY_DIR / "representation_diagnostics.json"
CONFIG_PATH = EXPLORATORY_DIR / "config.json"
README_PATH = EXPLORATORY_DIR / "README.md"
MODEL_PATH = EXPLORATORY_DIR / "final_selected_model.joblib"

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "32"))
PCA_CANDIDATES = (16, 32, 64, 128)
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
NDCG_K = 10
OUTER_SPLITS = 5
INNER_SPLITS = 5
RANDOM_STATE = 42
DOCUMENT_TEMPLATE_VERSION = "film_doc_v1"


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _openai_post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def load_film_population() -> pd.DataFrame:
    split = pd.read_csv("mvp/data/processed/primary_holdout_split.csv")
    condition_a = pd.read_csv("mvp/data/processed/condition_a_enriched.csv")
    merged = split.merge(condition_a, on="source_id", how="inner", validate="one_to_one")
    train = merged.loc[merged["split"] == "train"].copy().reset_index(drop=True)
    if len(train) != 393:
        raise RuntimeError(f"Expected 393 training films, found {len(train)}.")
    return train


def load_frozen_semantics() -> pd.DataFrame:
    frame = pd.read_csv("mvp/artifacts/semantic_vectors/semantic_vectors.csv")
    return frame.rename(columns={"canonical_id": "source_id"})


def build_canonical_film_documents(films: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = films.sort_values("source_id").reset_index(drop=True)
    for _, row in ordered.iterrows():
        lines = [f"Title: {_normalize_text(row.get('title'))}"]
        fields_used = ["title"]
        if not _is_missing(row.get("release_year")):
            lines.append(f"Release year: {int(float(row.get('release_year')))}")
            fields_used.append("release_year")
        if not _is_missing(row.get("tmdb_genres")):
            lines.append(f"Genres: {_normalize_text(row.get('tmdb_genres')).replace('|', ' | ')}")
            fields_used.append("tmdb_genres")
        if not _is_missing(row.get("tmdb_original_language")):
            lines.append(f"Original language: {_normalize_text(row.get('tmdb_original_language'))}")
            fields_used.append("tmdb_original_language")
        if not _is_missing(row.get("tmdb_production_countries")):
            lines.append(
                f"Production countries: {_normalize_text(row.get('tmdb_production_countries')).replace('|', ' | ')}"
            )
            fields_used.append("tmdb_production_countries")
        if not _is_missing(row.get("tmdb_overview")):
            lines.append(f"Overview: {_normalize_text(row.get('tmdb_overview'))}")
            fields_used.append("tmdb_overview")
        document = "\n".join(lines)
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "title": row.get("title"),
                "document_text": document,
                "document_hash": _sha256(document),
                "document_length": len(document),
                "fields_used": "|".join(fields_used),
                "template_version": DOCUMENT_TEMPLATE_VERSION,
            }
        )
    frame = pd.DataFrame(rows)
    DOCUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DOCUMENTS_PATH, index=False)
    return frame


def _embed_texts(texts: list[str], api_key: str, model: str) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    response = _openai_post_json("https://api.openai.com/v1/embeddings", api_key, payload, timeout=180)
    data = sorted(response.get("data", []), key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in data]


def _embed_texts_with_fallback(texts: list[str], api_key: str, model: str) -> list[list[float]]:
    try:
        return _embed_texts(texts, api_key, model)
    except Exception:
        if len(texts) <= 1:
            raise
        midpoint = len(texts) // 2
        left = _embed_texts_with_fallback(texts[:midpoint], api_key, model)
        right = _embed_texts_with_fallback(texts[midpoint:], api_key, model)
        return left + right


def _load_embedding_cache() -> pd.DataFrame:
    if not EMBEDDINGS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(EMBEDDINGS_PATH)
    if "source_id" in frame.columns:
        frame["source_id"] = frame["source_id"].astype(str)
    return frame


def _embedding_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def _semantic_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in SEMANTIC_COLUMNS if column in frame.columns]


def _write_embedding_artifacts(embedding_frame: pd.DataFrame, manifest_frame: pd.DataFrame) -> None:
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embedding_frame.to_csv(EMBEDDINGS_PATH, index=False, float_format="%.8f")
    manifest_frame.to_csv(MANIFEST_PATH, index=False)


def _build_stratified_splits(frame: pd.DataFrame, *, n_splits: int, random_state: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.Series]:
    strata = _strata(frame)
    min_class = int(strata.value_counts().min())
    if min_class < 2:
        raise RuntimeError("Stratified CV requires at least two samples in every rating bin.")
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        raise RuntimeError("Stratified CV could not be built with the available class balance.")
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(frame, strata)), strata


def _ensure_aligned_rows(frame: pd.DataFrame, source_ids: Sequence[str], *, label: str) -> pd.DataFrame:
    indexed = frame.copy()
    if "source_id" not in indexed.columns:
        raise KeyError(f"{label} is missing source_id.")
    indexed["source_id"] = indexed["source_id"].astype(str)
    lookup = indexed.set_index("source_id")
    missing = [source_id for source_id in source_ids if source_id not in lookup.index]
    if missing:
        raise RuntimeError(f"{label} is missing {len(missing)} required source_id values.")
    return lookup.loc[list(source_ids)].reset_index()


def _score_range(predictions: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(predictions)),
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
    }


def _fit_ridge_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> RidgeCV:
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv_splits, fit_intercept=True)
    model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
    return model


def _fit_metadata_fold(
    fold_index: int,
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[FoldResult, dict[str, Any]]:
    builder = ConventionalMetadataFeatureBuilder().fit(train_fold)
    X_train = builder.transform(train_fold).reset_index(drop=True)
    X_val = builder.transform(val_fold).reset_index(drop=True)
    ridge = _fit_ridge_cv(X_train, train_fold["preference_weight"], cv_splits)
    preds = ridge.predict(X_val.to_numpy(dtype=float))
    metrics = _score_predictions(val_fold["preference_weight"], preds)
    feature_columns = X_train.columns.tolist()
    result = FoldResult(
        fold=fold_index,
        architecture="a1",
        alpha=float(ridge.alpha_),
        feature_count=int(len(feature_columns)),
        pca_dim=None,
        explained_variance=None,
        metrics=metrics,
        prediction_range=_score_range(preds),
        feature_columns=feature_columns,
    )
    selection = {
        "architecture": "a1",
        "fold": fold_index,
        "alpha": float(ridge.alpha_),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "train_source_ids": train_fold["source_id"].astype(str).tolist(),
        "val_source_ids": val_fold["source_id"].astype(str).tolist(),
        "metrics": metrics,
        "prediction_range": _score_range(preds),
    }
    return result, selection


def _fit_semantic_fold(
    fold_index: int,
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    semantic_train: pd.DataFrame,
    semantic_val: pd.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[FoldResult, dict[str, Any]]:
    feature_columns = _semantic_feature_columns(semantic_train)
    X_train = semantic_train.loc[:, feature_columns].reset_index(drop=True)
    X_val = semantic_val.loc[:, feature_columns].reset_index(drop=True)
    ridge = _fit_ridge_cv(X_train, train_fold["preference_weight"], cv_splits)
    preds = ridge.predict(X_val.to_numpy(dtype=float))
    metrics = _score_predictions(val_fold["preference_weight"], preds)
    result = FoldResult(
        fold=fold_index,
        architecture="b",
        alpha=float(ridge.alpha_),
        feature_count=int(len(feature_columns)),
        pca_dim=None,
        explained_variance=None,
        metrics=metrics,
        prediction_range=_score_range(preds),
        feature_columns=feature_columns,
    )
    selection = {
        "architecture": "b",
        "fold": fold_index,
        "alpha": float(ridge.alpha_),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "train_source_ids": train_fold["source_id"].astype(str).tolist(),
        "val_source_ids": val_fold["source_id"].astype(str).tolist(),
        "metrics": metrics,
        "prediction_range": _score_range(preds),
    }
    return result, selection


def _evaluate_latent_fold(
    fold_index: int,
    architecture: str,
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    embedding_train: pd.DataFrame,
    embedding_val: pd.DataFrame,
) -> tuple[FoldResult, dict[str, Any]]:
    if architecture not in {"b2", "b3"}:
        raise ValueError(f"Unsupported latent architecture: {architecture}")

    emb_cols = _embedding_feature_columns(embedding_train)
    emb_train = embedding_train.loc[:, emb_cols].to_numpy(dtype=float)
    emb_val = embedding_val.loc[:, emb_cols].to_numpy(dtype=float)
    inner_splits, _ = _build_stratified_splits(train_fold, n_splits=INNER_SPLITS, random_state=RANDOM_STATE + fold_index)
    candidate_dims = _valid_pca_candidates(emb_train.shape[1], len(train_fold))
    candidate_rows: list[dict[str, Any]] = []

    for dim in candidate_dims:
        fold_metrics: list[dict[str, float]] = []
        explained_variance: list[float] = []
        for inner_train_idx, inner_val_idx in inner_splits:
            inner_train = train_fold.iloc[inner_train_idx].reset_index(drop=True)
            inner_val = train_fold.iloc[inner_val_idx].reset_index(drop=True)
            inner_emb_train = emb_train[inner_train_idx]
            inner_emb_val = emb_train[inner_val_idx]
            scaler = StandardScaler().fit(inner_emb_train)
            pca = PCA(
                n_components=dim,
                whiten=True,
                random_state=RANDOM_STATE + fold_index,
                svd_solver="randomized",
            )
            pca.fit(scaler.transform(inner_emb_train))
            inner_latent_train = pca.transform(scaler.transform(inner_emb_train))
            inner_latent_val = pca.transform(scaler.transform(inner_emb_val))
            latent_columns = [f"latent_pca_{index + 1:03d}" for index in range(dim)]

            if architecture == "b2":
                X_inner_train = pd.DataFrame(inner_latent_train, columns=latent_columns)
                X_inner_val = pd.DataFrame(inner_latent_val, columns=latent_columns)
            else:
                inner_builder = ConventionalMetadataFeatureBuilder().fit(inner_train)
                X_inner_train = pd.concat(
                    [
                        inner_builder.transform(inner_train).reset_index(drop=True),
                        pd.DataFrame(inner_latent_train, columns=latent_columns),
                    ],
                    axis=1,
                )
                X_inner_val = pd.concat(
                    [
                        inner_builder.transform(inner_val).reset_index(drop=True),
                        pd.DataFrame(inner_latent_val, columns=latent_columns),
                    ],
                    axis=1,
                )

            inner_ridge = _fit_ridge_cv(X_inner_train, inner_train["preference_weight"], _build_stratified_splits(inner_train, n_splits=INNER_SPLITS, random_state=RANDOM_STATE + fold_index)[0])
            inner_preds = inner_ridge.predict(X_inner_val.to_numpy(dtype=float))
            fold_metrics.append(_score_predictions(inner_val["preference_weight"], inner_preds))
            explained_variance.append(float(pca.explained_variance_ratio_.sum()))

        candidate_rows.append(
            {
                "pca_dim": int(dim),
                "mean_mae": float(np.mean([row["mae"] for row in fold_metrics])),
                "mean_rmse": float(np.mean([row["rmse"] for row in fold_metrics])),
                "mean_pearson": float(np.mean([row["pearson"] for row in fold_metrics])),
                "mean_spearman": float(np.mean([row["spearman"] for row in fold_metrics])),
                "mean_ndcg_at_k": float(np.mean([row["ndcg_at_k"] for row in fold_metrics])),
                "mean_explained_variance": float(np.mean(explained_variance)),
            }
        )

    selected_row = sorted(
        candidate_rows,
        key=lambda row: (row["mean_spearman"], row["mean_ndcg_at_k"], -row["mean_mae"]),
        reverse=True,
    )[0]
    selected_dim = int(selected_row["pca_dim"])

    scaler = StandardScaler().fit(emb_train)
    pca = PCA(
        n_components=selected_dim,
        whiten=True,
        random_state=RANDOM_STATE + fold_index,
        svd_solver="randomized",
    )
    pca.fit(scaler.transform(emb_train))
    latent_train = pca.transform(scaler.transform(emb_train))
    latent_val = pca.transform(scaler.transform(emb_val))
    latent_columns = [f"latent_pca_{index + 1:03d}" for index in range(selected_dim)]

    if architecture == "b2":
        X_train = pd.DataFrame(latent_train, columns=latent_columns)
        X_val = pd.DataFrame(latent_val, columns=latent_columns)
        redundancy = None
        metadata_feature_count = 0
    else:
        metadata_builder = ConventionalMetadataFeatureBuilder().fit(train_fold)
        metadata_train = metadata_builder.transform(train_fold).reset_index(drop=True)
        metadata_val = metadata_builder.transform(val_fold).reset_index(drop=True)
        X_train = pd.concat([metadata_train, pd.DataFrame(latent_train, columns=latent_columns)], axis=1)
        X_val = pd.concat([metadata_val, pd.DataFrame(latent_val, columns=latent_columns)], axis=1)
        metadata_cols = metadata_train.columns.tolist()
        latent_corrs: list[float] = []
        for latent_index in range(selected_dim):
            latent_series = pd.Series(latent_train[:, latent_index])
            per_feature_corrs = []
            for column in metadata_cols:
                meta_series = pd.Series(metadata_train[column].to_numpy(dtype=float))
                if meta_series.nunique(dropna=False) < 2 or latent_series.nunique(dropna=False) < 2:
                    continue
                corr = latent_series.corr(meta_series)
                if pd.notna(corr):
                    per_feature_corrs.append(abs(float(corr)))
            if per_feature_corrs:
                latent_corrs.append(max(per_feature_corrs))
        redundancy = {
            "mean_abs_max_corr": float(np.mean(latent_corrs)) if latent_corrs else float("nan"),
            "max_abs_corr": float(np.max(latent_corrs)) if latent_corrs else float("nan"),
            "selected_metadata_features": int(len(metadata_cols)),
        }
        metadata_feature_count = int(len(metadata_cols))

    ridge = _fit_ridge_cv(X_train, train_fold["preference_weight"], _build_stratified_splits(train_fold, n_splits=INNER_SPLITS, random_state=RANDOM_STATE + fold_index)[0])
    preds = ridge.predict(X_val.to_numpy(dtype=float))
    metrics = _score_predictions(val_fold["preference_weight"], preds)
    result = FoldResult(
        fold=fold_index,
        architecture=architecture,
        alpha=float(ridge.alpha_),
        feature_count=int(X_train.shape[1]),
        pca_dim=selected_dim,
        explained_variance=float(pca.explained_variance_ratio_.sum()),
        metrics=metrics,
        prediction_range=_score_range(preds),
        redundancy=redundancy,
        feature_columns=X_train.columns.tolist(),
        metadata_feature_count=metadata_feature_count,
    )
    selection = {
        "architecture": architecture,
        "fold": fold_index,
        "alpha": float(ridge.alpha_),
        "pca_dim": selected_dim,
        "feature_count": int(X_train.shape[1]),
        "metadata_feature_count": metadata_feature_count,
        "latent_feature_count": int(selected_dim),
        "feature_columns": X_train.columns.tolist(),
        "train_source_ids": train_fold["source_id"].astype(str).tolist(),
        "val_source_ids": val_fold["source_id"].astype(str).tolist(),
        "candidate_dim_scores": candidate_rows,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "metrics": metrics,
        "prediction_range": _score_range(preds),
        "redundancy": redundancy,
    }
    return result, selection


def _fit_fold(
    fold_index: int,
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    semantic_train: pd.DataFrame,
    semantic_val: pd.DataFrame,
    embedding_train: pd.DataFrame,
    embedding_val: pd.DataFrame,
) -> tuple[list[FoldResult], list[dict[str, Any]]]:
    cv_splits, _ = _build_stratified_splits(train_fold, n_splits=INNER_SPLITS, random_state=RANDOM_STATE + fold_index)
    fold_results: list[FoldResult] = []
    fold_selections: list[dict[str, Any]] = []

    metadata_result, metadata_selection = _fit_metadata_fold(fold_index, train_fold, val_fold, cv_splits)
    fold_results.append(metadata_result)
    fold_selections.append(metadata_selection)

    semantic_result, semantic_selection = _fit_semantic_fold(fold_index, train_fold, val_fold, semantic_train, semantic_val, cv_splits)
    fold_results.append(semantic_result)
    fold_selections.append(semantic_selection)

    for architecture in ("b2", "b3"):
        latent_result, latent_selection = _evaluate_latent_fold(
            fold_index,
            architecture,
            train_fold,
            val_fold,
            embedding_train,
            embedding_val,
        )
        fold_results.append(latent_result)
        fold_selections.append(latent_selection)

    return fold_results, fold_selections


def generate_film_embeddings(documents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cached = _load_embedding_cache()
    if not cached.empty and {"source_id", "document_hash", "embedding_model"}.issubset(cached.columns):
        current = documents.loc[:, ["source_id", "document_hash"]].copy()
        cached_keys = cached.loc[:, ["source_id", "document_hash", "embedding_model"]].copy()
        if (
            set(cached_keys["source_id"].astype(str)) == set(current["source_id"].astype(str))
            and set(cached_keys["document_hash"].astype(str)) == set(current["document_hash"].astype(str))
            and cached_keys["embedding_model"].nunique() == 1
            and cached_keys["embedding_model"].iloc[0] == OPENAI_EMBEDDING_MODEL
        ):
            manifest = cached.loc[:, ["source_id", "title", "document_hash", "embedding_model", "embedding_dim", "status"]].copy()
            manifest["error"] = ""
            _write_embedding_artifacts(cached, manifest)
            report = {
                "provider": "openai",
                "model": OPENAI_EMBEDDING_MODEL,
                "embedding_dim": int(cached["embedding_dim"].iloc[0]),
                "coverage": 1.0,
                "successful": int((cached["status"] == "success").sum()),
                "failed": int((cached["status"] != "success").sum()),
            }
            return cached, manifest, report

    keys = get_api_keys()
    if not keys.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for latent film embeddings.")

    vector_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    batch_size = max(1, EMBEDDING_BATCH_SIZE)
    embedding_dim = None
    for start in range(0, len(documents), batch_size):
        batch = documents.iloc[start : start + batch_size].reset_index(drop=True)
        try:
            embeddings = _embed_texts_with_fallback(batch["document_text"].tolist(), keys.openai_api_key, OPENAI_EMBEDDING_MODEL)
        except Exception as exc:
            for _, row in batch.iterrows():
                manifest_rows.append(
                    {
                        "source_id": row["source_id"],
                        "title": row["title"],
                        "document_hash": row["document_hash"],
                        "embedding_model": OPENAI_EMBEDDING_MODEL,
                        "embedding_dim": pd.NA,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            continue

        if embeddings and embedding_dim is None:
            embedding_dim = len(embeddings[0])
        for row, vector in zip(batch.to_dict(orient="records"), embeddings, strict=True):
            vector_row = {
                "source_id": row["source_id"],
                "title": row["title"],
                "document_hash": row["document_hash"],
                "embedding_model": OPENAI_EMBEDDING_MODEL,
                "embedding_dim": len(vector),
                "status": "success",
            }
            vector_row.update({f"emb_{index:04d}": float(value) for index, value in enumerate(vector)})
            vector_rows.append(vector_row)
            manifest_rows.append(
                {
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "document_hash": row["document_hash"],
                    "embedding_model": OPENAI_EMBEDDING_MODEL,
                    "embedding_dim": len(vector),
                    "status": "success",
                    "error": "",
                }
            )
        partial = pd.DataFrame(vector_rows)
        EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial.to_csv(EMBEDDINGS_PATH, index=False, float_format="%.8f")

    embedding_frame = pd.DataFrame(vector_rows)
    manifest_frame = pd.DataFrame(manifest_rows)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embedding_frame.to_csv(EMBEDDINGS_PATH, index=False, float_format="%.8f")
    manifest_frame.to_csv(MANIFEST_PATH, index=False)
    report = {
        "provider": "openai",
        "model": OPENAI_EMBEDDING_MODEL,
        "embedding_dim": int(embedding_dim or 0),
        "coverage": float(len(embedding_frame) / len(documents)) if len(documents) else 0.0,
        "successful": int(len(embedding_frame)),
        "failed": int((manifest_frame["status"] != "success").sum()) if not manifest_frame.empty else 0,
    }
    return embedding_frame, manifest_frame, report


def _strata(frame: pd.DataFrame) -> pd.Series:
    return _frozen_split_bins(frame["rating"])


def _build_metadata_matrix(frame: pd.DataFrame) -> tuple[ConventionalMetadataFeatureBuilder, pd.DataFrame]:
    builder = ConventionalMetadataFeatureBuilder().fit(frame)
    matrix = builder.transform(frame).reset_index(drop=True)
    return builder, matrix


def _assemble_feature_frame(
    frame: pd.DataFrame,
    architecture: str,
    metadata_builder: ConventionalMetadataFeatureBuilder | None = None,
    embedding_frame: pd.DataFrame | None = None,
    semantic_frame: pd.DataFrame | None = None,
    pca_model: PCA | None = None,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    parts = []
    if architecture in {"a1", "b3"}:
        if metadata_builder is None:
            raise ValueError("Metadata builder required.")
        parts.append(metadata_builder.transform(frame).reset_index(drop=True))
    if architecture == "b":
        if semantic_frame is None:
            raise ValueError("Semantic frame required.")
        parts.append(semantic_frame.loc[:, list(semantic_frame.columns)].reset_index(drop=True))
    elif architecture in {"b2", "b3"}:
        if embedding_frame is None or pca_model is None or scaler is None:
            raise ValueError("Embedding PCA components required.")
        emb = embedding_frame.loc[:, [column for column in embedding_frame.columns if column.startswith("emb_")]].to_numpy(dtype=float)
        emb = scaler.transform(emb)
        emb = pca_model.transform(emb)
        emb_columns = [f"latent_pca_{index+1:03d}" for index in range(emb.shape[1])]
        parts.append(pd.DataFrame(emb, columns=emb_columns, index=frame.index).reset_index(drop=True))
    return pd.concat(parts, axis=1)


def _valid_pca_candidates(max_components: int, n_samples: int) -> list[int]:
    valid = [dim for dim in PCA_CANDIDATES if dim <= max_components and dim < n_samples]
    return valid or [min(max_components, n_samples - 1)]


def _score_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return evaluate_regression(y_true.to_numpy(dtype=float), y_pred, ndcg_k=NDCG_K)


def _ndcg_audit() -> dict[str, Any]:
    from mvp.src.models import _ndcg_at_k

    sample = np.array([-1.0, -0.25, 0.0, 0.6, 1.0], dtype=float)
    shifted = sample - sample.min()
    audit = {
        "historical_metric_behavior": "gains = max(y_true - min(y_true), 0), so negative preference_weight values are shifted upward by the fold minimum and then ranked non-negatively",
        "historical_unchanged": True,
        "diagnostic_variant": "not required because the historical implementation already performs a non-negative shift",
        "example_historical_ndcg": float(_ndcg_at_k(sample, sample, k=5)),
        "example_shifted_ndcg": float(_ndcg_at_k(shifted, shifted, k=5)),
    }
    return audit


@dataclass
class FoldResult:
    fold: int
    architecture: str
    alpha: float
    feature_count: int
    pca_dim: int | None
    explained_variance: float | None
    metrics: dict[str, float]
    prediction_range: dict[str, float]
    feature_columns: list[str] | None = None
    metadata_feature_count: int | None = None
    redundancy: dict[str, float] | None = None


def _aggregate_fold_results(folds: list[FoldResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for architecture in sorted({fold.architecture for fold in folds}):
        subset = [fold for fold in folds if fold.architecture == architecture]
        out[architecture] = {
            metric: {
                "mean": float(np.mean([fold.metrics[metric] for fold in subset])),
                "std": float(np.std([fold.metrics[metric] for fold in subset], ddof=0)),
            }
            for metric in ["mae", "rmse", "pearson", "spearman", "ndcg_at_k"]
        }
    return out


def run_exploratory_latent_semantics_checkpoint() -> dict[str, Any]:
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    training_films = load_film_population()
    semantic_frame = load_frozen_semantics()
    documents = build_canonical_film_documents(training_films)
    embeddings, manifest, embedding_report = generate_film_embeddings(documents)

    if len(embeddings) != len(documents):
        raise RuntimeError(f"Embedding coverage incomplete: {len(embeddings)}/{len(documents)}")
    training_ids = training_films["source_id"].astype(str).tolist()
    embedding_lookup = _ensure_aligned_rows(embeddings, training_ids, label="embedding cache").set_index("source_id")
    semantic_lookup = _ensure_aligned_rows(semantic_frame, training_ids, label="semantic vectors").set_index("source_id")
    if len(embedding_lookup) != len(training_films):
        raise RuntimeError("Embedding cache does not cover the full 393-film training population.")
    if len(semantic_lookup) != len(training_films):
        raise RuntimeError("Semantic vectors do not cover the full 393-film training population.")

    outer_splits, outer_strata = _build_stratified_splits(training_films, n_splits=OUTER_SPLITS, random_state=RANDOM_STATE)

    fold_results: list[FoldResult] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_index, (train_idx, val_idx) in enumerate(outer_splits, start=1):
        fold_train = training_films.iloc[train_idx].reset_index(drop=True)
        fold_val = training_films.iloc[val_idx].reset_index(drop=True)
        train_ids = fold_train["source_id"].astype(str).tolist()
        val_ids = fold_val["source_id"].astype(str).tolist()
        embedding_train = embedding_lookup.loc[train_ids].reset_index()
        embedding_val = embedding_lookup.loc[val_ids].reset_index()
        semantic_train = semantic_lookup.loc[train_ids].reset_index()
        semantic_val = semantic_lookup.loc[val_ids].reset_index()

        fold_models, fold_selections = _fit_fold(
            fold_index,
            fold_train,
            fold_val,
            semantic_train,
            semantic_val,
            embedding_train,
            embedding_val,
        )
        fold_results.extend(fold_models)
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_n": int(len(fold_train)),
                "val_n": int(len(fold_val)),
                "train_source_ids": train_ids,
                "val_source_ids": val_ids,
                "train_rating_bins": _strata(fold_train).value_counts().sort_index().to_dict(),
                "validation_rating_bins": _strata(fold_val).value_counts().sort_index().to_dict(),
                "architectures": {item["architecture"]: item for item in fold_selections},
            }
        )

    cv_summary = _aggregate_fold_results(fold_results)
    comparison = {
        "b2_vs_b": {
            metric: float(cv_summary["b2"][metric]["mean"] - cv_summary["b"][metric]["mean"])
            for metric in ["mae", "rmse", "pearson", "spearman", "ndcg_at_k"]
        },
        "b3_vs_a1": {
            metric: float(cv_summary["b3"][metric]["mean"] - cv_summary["a1"][metric]["mean"])
            for metric in ["mae", "rmse", "pearson", "spearman", "ndcg_at_k"]
        },
    }
    diagnostics = {
        "embedding_provider": "openai",
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimensionality": int(embedding_report["embedding_dim"]),
        "embedding_coverage": float(embedding_report["coverage"]),
        "embedding_successful": int(embedding_report["successful"]),
        "embedding_failed": int(embedding_report["failed"]),
        "semantic_coverage": float(len(semantic_lookup) / len(training_films)),
        "semantic_successful": int(len(semantic_lookup)),
        "semantic_failed": int(len(training_films) - len(semantic_lookup)),
        "document_template_version": DOCUMENT_TEMPLATE_VERSION,
        "document_fields": ["title", "release_year", "tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"],
        "document_length": {
            "min": int(documents["document_length"].min()),
            "mean": float(documents["document_length"].mean()),
            "max": int(documents["document_length"].max()),
        },
        "document_field_coverage": {
            field: int(documents["fields_used"].str.contains(field).sum())
            for field in ["release_year", "tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"]
        },
        "ndcg_audit": _ndcg_audit(),
        "outer_split_strata": outer_strata.value_counts().sort_index().to_dict(),
        "selected_pca_dims": {
            "b2": [fold["architectures"]["b2"]["pca_dim"] for fold in fold_summaries],
            "b3": [fold["architectures"]["b3"]["pca_dim"] for fold in fold_summaries],
        },
        "redundancy_b3": [fold["architectures"]["b3"]["redundancy"] for fold in fold_summaries],
        "holdout_used": False,
    }
    config = {
        "experiment_name": "exploratory_latent_semantics",
        "status": "training_only_checkpoint",
        "train_population": int(len(training_films)),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "random_state": RANDOM_STATE,
        "embedding_provider": "openai",
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimensionality": int(embedding_report["embedding_dim"]),
        "document_template_version": DOCUMENT_TEMPLATE_VERSION,
        "document_fields": diagnostics["document_fields"],
        "pca_candidates": list(PCA_CANDIDATES),
        "ridge_alphas": RIDGE_ALPHAS.tolist(),
        "metric_primary": "spearman",
        "metric_tiebreaker": "ndcg_at_k",
        "holdout_used": False,
        "train_source_ids": training_ids,
        "train_split_path": "mvp/data/processed/primary_holdout_split.csv",
    }
    cv_results = {
        "summary": cv_summary,
        "comparisons": comparison,
        "embedding_report": embedding_report,
        "fold_results": [
            {
                "fold": item.fold,
                "architecture": item.architecture,
                "alpha": item.alpha,
                "feature_count": item.feature_count,
                "pca_dim": item.pca_dim,
                "explained_variance": item.explained_variance,
                "metrics": item.metrics,
                "prediction_range": item.prediction_range,
                "feature_columns": item.feature_columns,
                "metadata_feature_count": item.metadata_feature_count,
                "redundancy": item.redundancy,
            }
            for item in fold_results
        ],
    }
    CV_RESULTS_PATH.write_text(json.dumps(cv_results, indent=2))
    FOLD_SELECTIONS_PATH.write_text(json.dumps(fold_summaries, indent=2))
    DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics, indent=2))
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    README_PATH.write_text(
        "\n".join(
            [
                "# Exploratory Latent Semantics",
                "",
                "Training-only nested CV for A1, B, B2, and B3.",
                "No 99-film holdout labels were used.",
                f"Embedding model: {OPENAI_EMBEDDING_MODEL} ({embedding_report['embedding_dim']} dims).",
            ]
        )
    )

    best_architecture = max(
        ["b2", "b3"],
        key=lambda name: (
            cv_summary[name]["spearman"]["mean"],
            cv_summary[name]["ndcg_at_k"]["mean"],
        ),
    )
    joblib.dump(
        {
            "best_architecture": best_architecture,
            "note": "No holdout-trained model was produced; this bundle only records the exploratory selection.",
        },
        MODEL_PATH,
    )

    checkpoint = {
        "canonical_film_document_schema": {
            "fields": diagnostics["document_fields"],
            "template_version": DOCUMENT_TEMPLATE_VERSION,
            "document_file": str(DOCUMENTS_PATH),
        },
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimensionality": int(embedding_report["embedding_dim"]),
        "embedding_coverage": float(embedding_report["coverage"]),
        "semantic_coverage": float(diagnostics["semantic_coverage"]),
        "ndcg_audit": diagnostics["ndcg_audit"],
        "nested_cv_procedure": {
            "outer_cv": f"{OUTER_SPLITS}-fold stratified CV on the 393 training films",
            "inner_cv": f"{INNER_SPLITS}-fold stratified CV on each outer-train fold",
            "alpha_selection": "RidgeCV on training-only folds",
            "pca_selection": "inner CV on outer-train, then final RidgeCV on outer-train with the selected PCA dimensionality",
            "holdout_used": False,
        },
        "b2_selected_pca_dims_per_outer_fold": diagnostics["selected_pca_dims"]["b2"],
        "b3_selected_pca_dims_per_outer_fold": diagnostics["selected_pca_dims"]["b3"],
        "outer_cv_results": cv_summary,
        "comparison": comparison,
        "representation_diagnostics": diagnostics,
        "artifact_paths": {
            "config": str(CONFIG_PATH),
            "canonical_film_documents": str(DOCUMENTS_PATH),
            "embedding_manifest": str(MANIFEST_PATH),
            "embedding_cache": str(EMBEDDINGS_PATH),
            "cv_results": str(CV_RESULTS_PATH),
            "fold_selections": str(FOLD_SELECTIONS_PATH),
            "representation_diagnostics": str(DIAGNOSTICS_PATH),
            "readme": str(README_PATH),
            "model_bundle": str(MODEL_PATH),
        },
        "warnings": [
            "B3 redundancy diagnostic is a simple correlation summary, not a causal test.",
            "The exploratory study is training-only; no 99-film holdout evaluation was run.",
        ],
        "holdout_confirmation": "The 99-film holdout was never used.",
    }
    return checkpoint
