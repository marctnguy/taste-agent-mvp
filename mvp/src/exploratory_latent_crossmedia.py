from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mvp.src.config import get_api_keys
from mvp.src.data import load_table, normalize_column_names
from mvp.src.models import ConventionalMetadataFeatureBuilder, evaluate_regression
from mvp.src.prepare import _frozen_split_bins


EXPERIMENT_DIR = Path("mvp/artifacts/experiments/exploratory_latent_crossmedia")
BOOK_EMBEDDING_CACHE_DIR = EXPERIMENT_DIR / "embedding_cache"
BOOK_DOCUMENTS_PATH = EXPERIMENT_DIR / "canonical_book_documents.csv"
BOOK_MANIFEST_PATH = EXPERIMENT_DIR / "book_embedding_manifest.csv"
BOOK_EMBEDDINGS_PATH = BOOK_EMBEDDING_CACHE_DIR / "book_embeddings.csv"
CROSSMEDIA_FEATURES_PATH = EXPERIMENT_DIR / "crossmedia_features.csv"
CV_RESULTS_PATH = EXPERIMENT_DIR / "cv_results.json"
FOLD_SELECTIONS_PATH = EXPERIMENT_DIR / "fold_selections.json"
DIAGNOSTICS_PATH = EXPERIMENT_DIR / "crossmedia_diagnostics.json"
COEFFICIENTS_PATH = EXPERIMENT_DIR / "crossmedia_coefficients.csv"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
README_PATH = EXPERIMENT_DIR / "README.md"

BASELINE_LATENT_DIR = Path("mvp/artifacts/experiments/exploratory_latent_semantics")
BASELINE_CV_RESULTS_PATH = BASELINE_LATENT_DIR / "cv_results.json"
BASELINE_FOLD_SELECTIONS_PATH = BASELINE_LATENT_DIR / "fold_selections.json"
FILM_EMBEDDINGS_PATH = BASELINE_LATENT_DIR / "embedding_cache" / "film_embeddings.csv"

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "32"))
PCA_CANDIDATES = (16, 32, 64, 128)
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
OUTER_SPLITS = 5
INNER_SPLITS = 5
RANDOM_STATE = 42
BOOK_DOC_TEMPLATE_VERSION = "book_doc_v1"
EXPERIMENT_STATUS = "exploratory_training_only"
NDCG_K = 10


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


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _vector_columns(frame: pd.DataFrame, prefix: str = "emb_") -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _score_range(predictions: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(predictions)),
        "min": float(np.min(predictions)),
        "max": float(np.max(predictions)),
    }


def _strata(frame: pd.DataFrame) -> pd.Series:
    return _frozen_split_bins(frame["rating"])


def _build_stratified_splits(frame: pd.DataFrame, *, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = _strata(frame)
    min_class = int(strata.value_counts().min())
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        raise RuntimeError("Stratified CV could not be built with the available class balance.")
    splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(frame, strata))


def _fit_ridge_cv(X: pd.DataFrame, y: pd.Series, *, random_state: int) -> RidgeCV:
    strata = _strata(pd.DataFrame({"rating": y}))
    cv_splits = _build_stratified_splits(pd.DataFrame({"rating": y}), n_splits=INNER_SPLITS, random_state=random_state)
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv_splits, fit_intercept=True)
    model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
    return model


def load_film_population() -> pd.DataFrame:
    split = pd.read_csv("mvp/data/processed/primary_holdout_split.csv")
    condition_a = pd.read_csv("mvp/data/processed/condition_a_enriched.csv")
    merged = split.merge(condition_a, on="source_id", how="inner", validate="one_to_one")
    train = merged.loc[merged["split"] == "train"].copy().reset_index(drop=True)
    if len(train) != 393:
        raise RuntimeError(f"Expected 393 training films, found {len(train)}.")
    train["source_id"] = train["source_id"].astype(str)
    return train


def load_goodreads_books() -> pd.DataFrame:
    raw = normalize_column_names(load_table("mvp/data/raw/taste-agent-combined-ingestion.csv"))
    books = raw.loc[(raw["media_type"] == "book") & raw["rating"].notna()].copy().reset_index(drop=True)
    if "source_id" not in books.columns and "canonical_id" in books.columns:
        books["source_id"] = books["canonical_id"].astype(str)
    books["source_id"] = books["source_id"].astype(str)
    if not BOOK_EMBEDDINGS_PATH.exists() and not Path("mvp/artifacts/experiments/condition_c_goodreads_crossmedia/goodreads_metadata_cache.csv").exists():
        return books

    cache_path = Path("mvp/artifacts/experiments/condition_c_goodreads_crossmedia/goodreads_metadata_cache.csv")
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        cache["source_id"] = cache["source_id"].astype(str)
        cache = cache.drop_duplicates(subset=["source_id"]).copy()
        cols = [
            "source_id",
            "google_books_title",
            "google_books_authors",
            "google_books_description",
            "google_books_categories",
            "google_books_published_year",
        ]
        books = books.merge(cache.loc[:, [col for col in cols if col in cache.columns]], on="source_id", how="left")
    return books.reset_index(drop=True)


def build_book_documents(books: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = books.sort_values("source_id").reset_index(drop=True)
    for _, row in ordered.iterrows():
        title_value = row.get("google_books_title")
        if _is_missing(title_value):
            title_value = row.get("title")
        author_value = row.get("google_books_authors")
        if _is_missing(author_value):
            author_value = row.get("creator")
        title = _normalize_text(title_value)
        author = _normalize_text(author_value)
        pub_year = row.get("google_books_published_year")
        if _is_missing(pub_year):
            pub_year = row.get("year")
        categories = _normalize_text(row.get("google_books_categories"))
        description = _normalize_text(row.get("google_books_description"))
        lines = [f"Title: {title}" if title else "Title: "]
        fields_used = ["title"]
        if author:
            lines.append(f"Author: {author}")
            fields_used.append("author")
        if not _is_missing(pub_year):
            lines.append(f"Publication year: {int(float(pub_year))}")
            fields_used.append("publication_year")
        if categories:
            lines.append(f"Categories: {categories.replace('|', ' | ')}")
            fields_used.append("categories")
        if description:
            lines.append(f"Description: {description}")
            fields_used.append("description")
        document_text = "\n".join(lines)
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "title": row.get("title"),
                "author": author or pd.NA,
                "publication_year": int(float(pub_year)) if not _is_missing(pub_year) else pd.NA,
                "categories": categories or pd.NA,
                "description": description or pd.NA,
                "document_text": document_text,
                "document_hash": _sha256(document_text),
                "document_length": len(document_text),
                "fields_used": "|".join(fields_used),
                "template_version": BOOK_DOC_TEMPLATE_VERSION,
            }
        )
    documents = pd.DataFrame(rows)
    BOOK_DOCUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    documents.to_csv(BOOK_DOCUMENTS_PATH, index=False)
    return documents


def _load_embedding_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "source_id" in frame.columns:
        frame["source_id"] = frame["source_id"].astype(str)
    return frame


def _write_embedding_cache(embeddings: pd.DataFrame, manifest: pd.DataFrame) -> None:
    BOOK_EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(BOOK_EMBEDDINGS_PATH, index=False, float_format="%.8f")
    manifest.to_csv(BOOK_MANIFEST_PATH, index=False)


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


def generate_book_embeddings(documents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cached = _load_embedding_cache(BOOK_EMBEDDINGS_PATH)
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
            _write_embedding_cache(cached, manifest)
            return cached, manifest, {
                "provider": "openai",
                "model": OPENAI_EMBEDDING_MODEL,
                "embedding_dim": int(cached["embedding_dim"].iloc[0]),
                "coverage": 1.0,
                "successful": int((cached["status"] == "success").sum()),
                "failed": int((cached["status"] != "success").sum()),
            }

    keys = get_api_keys()
    if not keys.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for latent Goodreads embeddings.")

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
        _write_embedding_cache(pd.DataFrame(vector_rows), pd.DataFrame(manifest_rows))

    embedding_frame = pd.DataFrame(vector_rows)
    manifest_frame = pd.DataFrame(manifest_rows)
    _write_embedding_cache(embedding_frame, manifest_frame)
    return embedding_frame, manifest_frame, {
        "provider": "openai",
        "model": OPENAI_EMBEDDING_MODEL,
        "embedding_dim": int(embedding_dim or 0),
        "coverage": float(len(embedding_frame) / len(documents)) if len(documents) else 0.0,
        "successful": int(len(embedding_frame)),
        "failed": int((manifest_frame["status"] != "success").sum()) if not manifest_frame.empty else 0,
    }


def load_film_embeddings() -> pd.DataFrame:
    if not FILM_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "Missing frozen film embeddings from the exploratory latent semantics run."
        )
    frame = pd.read_csv(FILM_EMBEDDINGS_PATH)
    frame["source_id"] = frame["source_id"].astype(str)
    return frame


def load_baseline_b3() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not BASELINE_CV_RESULTS_PATH.exists() or not BASELINE_FOLD_SELECTIONS_PATH.exists():
        raise FileNotFoundError("Missing frozen B3 latent-semantics artifacts.")
    cv_results = json.loads(BASELINE_CV_RESULTS_PATH.read_text())
    fold_selection_rows = json.loads(BASELINE_FOLD_SELECTIONS_PATH.read_text())
    baseline_folds = [row for row in cv_results.get("fold_results", []) if row.get("architecture") == "b3"]
    baseline_folds = sorted(baseline_folds, key=lambda row: row["fold"])
    fold_map = {row["fold"]: row for row in fold_selection_rows}
    b3_summaries = []
    for row in baseline_folds:
        fold = int(row["fold"])
        fold_selection = fold_map[fold]["architectures"]["b3"]
        b3_summaries.append(
            {
                "fold": fold,
                "architecture": "b3",
                "alpha": fold_selection["alpha"],
                "pca_dim": fold_selection["pca_dim"],
                "feature_count": fold_selection["feature_count"],
                "feature_columns": fold_selection["feature_columns"],
                "train_source_ids": fold_selection["train_source_ids"],
                "val_source_ids": fold_selection["val_source_ids"],
                "metrics": row["metrics"],
                "prediction_range": row["prediction_range"],
                "explained_variance": row["explained_variance"],
            }
        )
    return cv_results, fold_selection_rows, {str(item["fold"]): item for item in b3_summaries}


def _build_feature_frame(
    frame: pd.DataFrame,
    metadata_builder: ConventionalMetadataFeatureBuilder,
    latent_train: np.ndarray,
    crossmedia_lookup: pd.DataFrame,
    added_columns: list[str],
) -> pd.DataFrame:
    metadata = metadata_builder.transform(frame).reset_index(drop=True)
    latent_cols = [f"latent_pca_{index + 1:03d}" for index in range(latent_train.shape[1])]
    latent = pd.DataFrame(latent_train, columns=latent_cols)
    crossmedia = crossmedia_lookup.loc[frame["source_id"].astype(str).tolist(), added_columns].reset_index(drop=True)
    return pd.concat([metadata, latent, crossmedia], axis=1)


def _build_redundancy_summary(feature_train: pd.DataFrame, crossmedia_cols: list[str]) -> dict[str, Any]:
    other_cols = [column for column in feature_train.columns if column not in crossmedia_cols]
    per_feature_max: dict[str, float] = {}
    for cross_col in crossmedia_cols:
        series = feature_train[cross_col].to_numpy(dtype=float)
        corrs = []
        for other_col in other_cols:
            other = feature_train[other_col].to_numpy(dtype=float)
            corr = _safe_corr(series, other)
            if np.isfinite(corr):
                corrs.append(abs(float(corr)))
        per_feature_max[cross_col] = float(max(corrs)) if corrs else float("nan")
    values = list(per_feature_max.values())
    return {
        "per_feature_max_abs_corr": per_feature_max,
        "mean_abs_max_corr": float(np.mean(values)) if values else float("nan"),
        "max_abs_corr": float(np.max(values)) if values else float("nan"),
        "reference_feature_count": int(len(other_cols)),
    }


def _evaluate_fold(
    fold_index: int,
    architecture: str,
    train_fold: pd.DataFrame,
    val_fold: pd.DataFrame,
    film_train_embeddings: np.ndarray,
    film_val_embeddings: np.ndarray,
    crossmedia_lookup: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if architecture not in {"x1", "x2"}:
        raise ValueError(f"Unsupported architecture: {architecture}")

    added_columns = ["book_alignment_latent_global"] if architecture == "x1" else [
        "book_similarity_positive",
        "book_similarity_negative",
    ]
    y_train = train_fold["preference_weight"].copy()
    y_val = val_fold["preference_weight"].copy()
    inner_splits = _build_stratified_splits(train_fold, n_splits=INNER_SPLITS, random_state=RANDOM_STATE + fold_index)
    candidate_rows: list[dict[str, Any]] = []
    pca_candidates = [dim for dim in PCA_CANDIDATES if dim < len(train_fold) and dim < film_train_embeddings.shape[1]]
    if not pca_candidates:
        pca_candidates = [min(film_train_embeddings.shape[1], len(train_fold) - 1)]

    for dim in pca_candidates:
        inner_metrics = []
        inner_explained = []
        for inner_train_idx, inner_val_idx in inner_splits:
            inner_train = train_fold.iloc[inner_train_idx].reset_index(drop=True)
            inner_val = train_fold.iloc[inner_val_idx].reset_index(drop=True)
            inner_metadata_builder = ConventionalMetadataFeatureBuilder().fit(inner_train)
            inner_train_emb = film_train_embeddings[inner_train_idx]
            inner_val_emb = film_train_embeddings[inner_val_idx]
            scaler = StandardScaler().fit(inner_train_emb)
            pca = PCA(n_components=dim, whiten=True, random_state=RANDOM_STATE + fold_index, svd_solver="randomized")
            pca.fit(scaler.transform(inner_train_emb))
            latent_train = pca.transform(scaler.transform(inner_train_emb))
            latent_val = pca.transform(scaler.transform(inner_val_emb))
            X_inner_train = _build_feature_frame(inner_train, inner_metadata_builder, latent_train, crossmedia_lookup, added_columns)
            X_inner_val = _build_feature_frame(inner_val, inner_metadata_builder, latent_val, crossmedia_lookup, added_columns)
            ridge = _fit_ridge_cv(X_inner_train, inner_train["preference_weight"], random_state=RANDOM_STATE + fold_index)
            preds = ridge.predict(X_inner_val.to_numpy(dtype=float))
            inner_metrics.append(evaluate_regression(inner_val["preference_weight"], preds, ndcg_k=NDCG_K))
            inner_explained.append(float(pca.explained_variance_ratio_.sum()))
        candidate_rows.append(
            {
                "pca_dim": int(dim),
                "mean_mae": float(np.mean([row["mae"] for row in inner_metrics])),
                "mean_rmse": float(np.mean([row["rmse"] for row in inner_metrics])),
                "mean_pearson": float(np.mean([row["pearson"] for row in inner_metrics])),
                "mean_spearman": float(np.mean([row["spearman"] for row in inner_metrics])),
                "mean_ndcg_at_k": float(np.mean([row["ndcg_at_k"] for row in inner_metrics])),
                "mean_explained_variance": float(np.mean(inner_explained)),
            }
        )

    selected_row = sorted(
        candidate_rows,
        key=lambda row: (row["mean_spearman"], row["mean_ndcg_at_k"], -row["mean_mae"]),
        reverse=True,
    )[0]
    selected_dim = int(selected_row["pca_dim"])

    scaler = StandardScaler().fit(film_train_embeddings)
    pca = PCA(
        n_components=selected_dim,
        whiten=True,
        random_state=RANDOM_STATE + fold_index,
        svd_solver="randomized",
    )
    pca.fit(scaler.transform(film_train_embeddings))
    latent_train = pca.transform(scaler.transform(film_train_embeddings))
    latent_val = pca.transform(scaler.transform(film_val_embeddings))

    outer_metadata_builder = ConventionalMetadataFeatureBuilder().fit(train_fold)
    X_train = _build_feature_frame(train_fold, outer_metadata_builder, latent_train, crossmedia_lookup, added_columns)
    X_val = _build_feature_frame(val_fold, outer_metadata_builder, latent_val, crossmedia_lookup, added_columns)
    ridge = _fit_ridge_cv(X_train, y_train, random_state=RANDOM_STATE + fold_index)
    preds = ridge.predict(X_val.to_numpy(dtype=float))
    metrics = evaluate_regression(y_val, preds, ndcg_k=NDCG_K)
    coef = pd.Series(ridge.coef_, index=X_train.columns)
    coefficient_records = []
    for column in added_columns:
        coefficient_records.append(
            {
                "feature": column,
                "coefficient": float(coef[column]),
                "sign": "positive" if coef[column] > 0 else "negative" if coef[column] < 0 else "zero",
            }
        )
    fold_result = {
        "fold": fold_index,
        "architecture": architecture,
        "alpha": float(ridge.alpha_),
        "pca_dim": selected_dim,
        "feature_count": int(X_train.shape[1]),
        "metadata_feature_count": int(len([c for c in X_train.columns if c.startswith("release_year_z") or "__" in c or c.startswith("decade__")])),
        "latent_feature_count": int(selected_dim),
        "crossmedia_feature_count": int(len(added_columns)),
        "metrics": metrics,
        "prediction_range": _score_range(preds),
        "feature_columns": X_train.columns.tolist(),
        "crossmedia_coefficients": coefficient_records,
        "candidate_dim_scores": candidate_rows,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    selection = {
        "fold": fold_index,
        "architecture": architecture,
        "train_source_ids": train_fold["source_id"].astype(str).tolist(),
        "val_source_ids": val_fold["source_id"].astype(str).tolist(),
        "selected_pca_dim": selected_dim,
        "selected_alpha": float(ridge.alpha_),
        "feature_count": int(X_train.shape[1]),
        "feature_columns": X_train.columns.tolist(),
        "crossmedia_feature_columns": added_columns,
        "crossmedia_coefficients": coefficient_records,
        "candidate_dim_scores": candidate_rows,
        "metrics": metrics,
        "prediction_range": _score_range(preds),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "redundancy": _build_redundancy_summary(X_train, added_columns),
    }
    diagnostics = {
        "feature_training_frame": X_train,
        "crossmedia_series": X_train.loc[:, added_columns],
        "redundancy": selection["redundancy"],
        "selected_dim": selected_dim,
    }
    return fold_result, selection, diagnostics


def _aggregate_fold_results(folds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for architecture in sorted({fold["architecture"] for fold in folds}):
        subset = [fold for fold in folds if fold["architecture"] == architecture]
        summary[architecture] = {
            metric: {
                "mean": float(np.mean([fold["metrics"][metric] for fold in subset])),
                "std": float(np.std([fold["metrics"][metric] for fold in subset], ddof=0)),
            }
            for metric in ["mae", "rmse", "pearson", "spearman", "ndcg_at_k"]
        }
    return summary


def _compare_metrics(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]) -> dict[str, float]:
    return {metric: float(a[metric]["mean"] - b[metric]["mean"]) for metric in ["mae", "rmse", "pearson", "spearman", "ndcg_at_k"]}


def _feature_distribution(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {key: float("nan") for key in ["min", "p25", "median", "mean", "p75", "max", "std"]}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "std": float(np.std(values, ddof=0)),
    }


def _compute_book_centroids(book_embeddings: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    normalized = _l2_normalize_rows(book_embeddings)
    total_weight = float(np.sum(np.abs(weights)))
    if total_weight == 0.0:
        raise RuntimeError("Goodreads weights sum to zero; cannot build a centroid.")

    def _weighted_centroid(mask: np.ndarray, use_abs_weights: bool) -> np.ndarray:
        subset = normalized[mask]
        subset_weights = np.abs(weights[mask]) if use_abs_weights else weights[mask]
        if len(subset) == 0 or float(np.sum(np.abs(subset_weights))) == 0.0:
            return np.zeros(normalized.shape[1], dtype=float)
        centroid = np.average(subset, axis=0, weights=np.abs(subset_weights))
        return _l2_normalize_vector(centroid)

    global_centroid = np.average(normalized, axis=0, weights=weights)
    global_centroid = _l2_normalize_vector(global_centroid)
    positive_centroid = _weighted_centroid(weights > 0, use_abs_weights=True)
    negative_centroid = _weighted_centroid(weights < 0, use_abs_weights=True)
    return {
        "global": global_centroid,
        "positive": positive_centroid,
        "negative": negative_centroid,
        "normalized_book_embeddings": normalized,
    }


def _load_previous_outer_folds() -> list[dict[str, Any]]:
    if not BASELINE_FOLD_SELECTIONS_PATH.exists():
        raise FileNotFoundError("Missing frozen outer fold selections from the prior latent semantics experiment.")
    return json.loads(BASELINE_FOLD_SELECTIONS_PATH.read_text())


def run_exploratory_latent_crossmedia_checkpoint() -> dict[str, Any]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    BOOK_EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    training_films = load_film_population()
    b3_baseline_cv, b3_baseline_folds, b3_baseline_by_fold = load_baseline_b3()
    outer_folds = _load_previous_outer_folds()
    training_ids = set(training_films["source_id"].astype(str))
    for fold in outer_folds:
        if set(fold["train_source_ids"]) | set(fold["val_source_ids"]) != training_ids:
            raise RuntimeError("Frozen outer folds do not match the 393-film training population.")

    books = load_goodreads_books()
    book_documents = build_book_documents(books)
    book_embeddings, book_manifest, embedding_report = generate_book_embeddings(book_documents)
    if len(book_embeddings) != len(book_documents):
        raise RuntimeError(f"Book embedding coverage incomplete: {len(book_embeddings)}/{len(book_documents)}")

    film_embeddings = load_film_embeddings()
    film_embeddings["source_id"] = film_embeddings["source_id"].astype(str)
    film_lookup = film_embeddings.set_index("source_id")
    missing_film_ids = sorted(training_ids.difference(set(film_lookup.index.astype(str))))
    if missing_film_ids:
        raise RuntimeError(f"Frozen film embeddings are missing {len(missing_film_ids)} training films.")
    film_lookup = film_lookup.loc[training_films["source_id"].astype(str)].copy()

    book_doc_ids = book_documents["source_id"].astype(str).tolist()
    book_embeddings = book_embeddings.set_index("source_id").loc[book_doc_ids].reset_index()
    book_matrix = book_embeddings.loc[:, _vector_columns(book_embeddings)].to_numpy(dtype=float)
    film_matrix = film_lookup.loc[:, _vector_columns(film_lookup)].to_numpy(dtype=float)
    book_weights = books.set_index("source_id").loc[book_doc_ids]["preference_weight"].to_numpy(dtype=float)
    if len(book_matrix) != len(book_weights):
        raise RuntimeError("Book embeddings do not align with Goodreads weights.")

    centroids = _compute_book_centroids(book_matrix, book_weights)
    normalized_film_matrix = _l2_normalize_rows(film_matrix)
    x1_alignment = normalized_film_matrix @ centroids["global"]
    x2_positive = normalized_film_matrix @ centroids["positive"]
    x2_negative = normalized_film_matrix @ centroids["negative"]
    x2_margin = x2_positive - x2_negative

    crossmedia_features = pd.DataFrame(
        {
            "source_id": training_films["source_id"].astype(str),
            "title": training_films["title"],
            "book_alignment_latent_global": x1_alignment,
            "book_similarity_positive": x2_positive,
            "book_similarity_negative": x2_negative,
            "book_crossmedia_margin": x2_margin,
        }
    )
    CROSSMEDIA_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    crossmedia_features.to_csv(CROSSMEDIA_FEATURES_PATH, index=False)
    crossmedia_lookup = crossmedia_features.set_index("source_id")

    outer_results: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for fold_row in outer_folds:
        fold_index = int(fold_row["fold"])
        train_ids = [str(source_id) for source_id in fold_row["train_source_ids"]]
        val_ids = [str(source_id) for source_id in fold_row["val_source_ids"]]
        train_fold = training_films.set_index("source_id").loc[train_ids].reset_index()
        val_fold = training_films.set_index("source_id").loc[val_ids].reset_index()

        train_emb = normalized_film_matrix[[training_films["source_id"].astype(str).tolist().index(source_id) for source_id in train_ids]]
        val_emb = normalized_film_matrix[[training_films["source_id"].astype(str).tolist().index(source_id) for source_id in val_ids]]

        for architecture in ("x1", "x2"):
            fold_result, selection, diagnostics = _evaluate_fold(
                fold_index,
                architecture,
                train_fold,
                val_fold,
                film_train_embeddings=train_emb,
                film_val_embeddings=val_emb,
                crossmedia_lookup=crossmedia_lookup,
            )
            outer_results.append(fold_result)
            fold_summaries.append(selection)
            diagnostics_rows.append(
                {
                    "fold": fold_index,
                    "architecture": architecture,
                    "redundancy": selection["redundancy"],
                    "book_alignment_distribution_train": _feature_distribution(crossmedia_lookup.loc[train_ids, "book_alignment_latent_global"] if architecture == "x1" else crossmedia_lookup.loc[train_ids, "book_similarity_positive"]),
                    "book_alignment_distribution_val": _feature_distribution(crossmedia_lookup.loc[val_ids, "book_alignment_latent_global"] if architecture == "x1" else crossmedia_lookup.loc[val_ids, "book_similarity_positive"]),
                }
            )
            for coef in fold_result["crossmedia_coefficients"]:
                coefficient_rows.append(
                    {
                        "fold": fold_index,
                        "architecture": architecture,
                        "feature": coef["feature"],
                        "coefficient": coef["coefficient"],
                        "sign": coef["sign"],
                        "alpha": fold_result["alpha"],
                        "selected_pca_dim": fold_result["pca_dim"],
                        "feature_count": fold_result["feature_count"],
                    }
                )

    summary = _aggregate_fold_results(outer_results)
    comparisons = {
        "x1_vs_b3": _compare_metrics(summary["x1"], b3_baseline_cv["summary"]["b3"]),
        "x2_vs_b3": _compare_metrics(summary["x2"], b3_baseline_cv["summary"]["b3"]),
        "x2_vs_x1": _compare_metrics(summary["x2"], summary["x1"]),
    }

    x1_coeffs = [row["coefficient"] for row in coefficient_rows if row["architecture"] == "x1"]
    x2_positive_coeffs = [row["coefficient"] for row in coefficient_rows if row["architecture"] == "x2" and row["feature"] == "book_similarity_positive"]
    x2_negative_coeffs = [row["coefficient"] for row in coefficient_rows if row["architecture"] == "x2" and row["feature"] == "book_similarity_negative"]

    global_distribution = _feature_distribution(crossmedia_features["book_alignment_latent_global"])
    positive_distribution = _feature_distribution(crossmedia_features["book_similarity_positive"])
    negative_distribution = _feature_distribution(crossmedia_features["book_similarity_negative"])
    margin_distribution = _feature_distribution(crossmedia_features["book_crossmedia_margin"])

    corr_matrix = {
        "x1_vs_x2_positive": float(_safe_corr(crossmedia_features["book_alignment_latent_global"].to_numpy(dtype=float), crossmedia_features["book_similarity_positive"].to_numpy(dtype=float))),
        "x1_vs_x2_negative": float(_safe_corr(crossmedia_features["book_alignment_latent_global"].to_numpy(dtype=float), crossmedia_features["book_similarity_negative"].to_numpy(dtype=float))),
        "x1_vs_margin": float(_safe_corr(crossmedia_features["book_alignment_latent_global"].to_numpy(dtype=float), crossmedia_features["book_crossmedia_margin"].to_numpy(dtype=float))),
        "positive_vs_negative": float(_safe_corr(crossmedia_features["book_similarity_positive"].to_numpy(dtype=float), crossmedia_features["book_similarity_negative"].to_numpy(dtype=float))),
        "positive_vs_margin": float(_safe_corr(crossmedia_features["book_similarity_positive"].to_numpy(dtype=float), crossmedia_features["book_crossmedia_margin"].to_numpy(dtype=float))),
        "negative_vs_margin": float(_safe_corr(crossmedia_features["book_similarity_negative"].to_numpy(dtype=float), crossmedia_features["book_crossmedia_margin"].to_numpy(dtype=float))),
    }

    positive_count = int((books["preference_weight"] > 0).sum())
    negative_count = int((books["preference_weight"] < 0).sum())
    neutral_count = int((books["preference_weight"] == 0).sum())

    diagnostics = {
        "experiment_status": EXPERIMENT_STATUS,
        "holdout_used": False,
        "same_outer_folds_as_previous_latent_experiment": True,
        "canonical_book_document_schema": {
            "fields": ["title", "author", "publication_year", "categories", "description"],
            "template_version": BOOK_DOC_TEMPLATE_VERSION,
        },
        "book_document_coverage": {
            "title": int(book_documents["title"].notna().sum()),
            "author": int(book_documents["author"].notna().sum()),
            "publication_year": int(book_documents["publication_year"].notna().sum()),
            "categories": int(book_documents["categories"].notna().sum()),
            "description": int(book_documents["description"].notna().sum()),
        },
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimensionality": 1536,
        "book_embedding_coverage": float(embedding_report["coverage"]),
        "book_embedding_successful": int(embedding_report["successful"]),
        "book_embedding_failed": int(embedding_report["failed"]),
        "shared_embedding_space": "OpenAI text-embedding-3-small, 1536 dimensions, with L2 normalization applied before centroid construction and cosine similarity.",
        "book_counts": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "total": int(len(books)),
        },
        "leakage_audit": {
            "goodreads_input_scope": "51 rated Goodreads books only; no unread, unfinished, or missing-rating rows.",
            "film_target_usage": "No film targets were used in book document construction or centroid fitting.",
            "fixed_profile_across_folds": True,
            "book_centroid_formula": "book_taste_global = sum(w_i * E_i) / sum(abs(w_i)), with E_i and the centroid both L2-normalized for cosine similarity.",
            "positive_negative_formula": "positive centroid uses only preference_weight > 0; negative centroid uses only preference_weight < 0; neutral books are excluded from both.",
            "book_doc_exclusion": ["rating", "preference_weight", "user_review", "notes", "film_targets"],
        },
        "feature_distributions": {
            "x1_global_alignment": global_distribution,
            "x2_positive_similarity": positive_distribution,
            "x2_negative_similarity": negative_distribution,
            "x2_margin": margin_distribution,
        },
        "feature_correlations": corr_matrix,
        "redundancy_summary": {
            "x1": [row["redundancy"] for row in fold_summaries if row["architecture"] == "x1"],
            "x2": [row["redundancy"] for row in fold_summaries if row["architecture"] == "x2"],
        },
        "film_embedding_reuse": {
            "source": str(FILM_EMBEDDINGS_PATH),
            "regenerated": False,
            "coverage": 1.0,
        },
        "baseline_b3_source": {
            "cv_results_path": str(BASELINE_CV_RESULTS_PATH),
            "fold_selections_path": str(BASELINE_FOLD_SELECTIONS_PATH),
        },
    }

    config = {
        "experiment_name": "exploratory_latent_crossmedia",
        "status": EXPERIMENT_STATUS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "random_state": RANDOM_STATE,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimensionality": 1536,
        "book_document_template_version": BOOK_DOC_TEMPLATE_VERSION,
        "crossmedia_features": [
            "book_alignment_latent_global",
            "book_similarity_positive",
            "book_similarity_negative",
            "book_crossmedia_margin",
        ],
        "model_features": {
            "b3": "frozen baseline from exploratory_latent_semantics",
            "x1": ["b3", "book_alignment_latent_global"],
            "x2": ["b3", "book_similarity_positive", "book_similarity_negative"],
        },
        "pca_candidates": list(PCA_CANDIDATES),
        "ridge_alphas": RIDGE_ALPHAS.tolist(),
        "selection_metric": "spearman",
        "tie_breaker": "ndcg_at_k",
        "holdout_used": False,
        "train_population": int(len(training_films)),
        "book_population": int(len(books)),
        "baseline_b3_path": str(BASELINE_CV_RESULTS_PATH),
        "film_embedding_path": str(FILM_EMBEDDINGS_PATH),
        "crossmedia_features_path": str(CROSSMEDIA_FEATURES_PATH),
    }

    coefficients_frame = pd.DataFrame(coefficient_rows)
    coefficient_summary: list[dict[str, Any]] = []
    for architecture in ("x1", "x2"):
        subset = coefficients_frame.loc[coefficients_frame["architecture"] == architecture].copy()
        for feature in subset["feature"].unique():
            values = subset.loc[subset["feature"] == feature, "coefficient"].to_numpy(dtype=float)
            mean_value = float(np.mean(values))
            coefficient_summary.append(
                {
                    "architecture": architecture,
                    "feature": feature,
                    "folds": int(len(values)),
                    "mean": mean_value,
                    "std": float(np.std(values, ddof=0)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "sign_consistency": float(np.mean(np.sign(values) == np.sign(mean_value))) if len(values) else float("nan"),
                    "effectively_shrunk_to_zero": bool(abs(mean_value) < 0.05),
                }
            )
    coefficients_frame.to_csv(COEFFICIENTS_PATH, index=False)

    b3_summary = b3_baseline_cv["summary"]["b3"]
    fold_results_for_json = {
        "b3": b3_baseline_folds,
        "x1": [fold for fold in outer_results if fold["architecture"] == "x1"],
        "x2": [fold for fold in outer_results if fold["architecture"] == "x2"],
    }
    cv_results = {
        "summary": {
            "b3": b3_summary,
            "x1": summary["x1"],
            "x2": summary["x2"],
        },
        "comparisons": comparisons,
        "baseline_b3_reference": {
            "cv_results_path": str(BASELINE_CV_RESULTS_PATH),
            "fold_selections_path": str(BASELINE_FOLD_SELECTIONS_PATH),
        },
        "fold_results": fold_results_for_json,
    }

    CV_RESULTS_PATH.write_text(json.dumps(cv_results, indent=2))
    FOLD_SELECTIONS_PATH.write_text(json.dumps({"b3": b3_baseline_folds, "x1": [fold for fold in fold_summaries if fold["architecture"] == "x1"], "x2": [fold for fold in fold_summaries if fold["architecture"] == "x2"]}, indent=2))
    DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics, indent=2))
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    README_PATH.write_text(
        "\n".join(
            [
                "# Exploratory Latent Crossmedia",
                "",
                "Training-only comparison of B3, X1, and X2 on the frozen 393-film population.",
                "The 99-film holdout was never accessed.",
                f"Embedding model: {OPENAI_EMBEDDING_MODEL} (1536 dims).",
            ]
        )
    )

    checkpoint = {
        "canonical_book_document_schema": diagnostics["canonical_book_document_schema"],
        "book_document_coverage": diagnostics["book_document_coverage"],
        "book_embedding_coverage": diagnostics["book_embedding_coverage"],
        "shared_embedding_space": diagnostics["shared_embedding_space"],
        "book_counts": diagnostics["book_counts"],
        "leakage_audit": diagnostics["leakage_audit"],
        "x1_formula": "B3 + book_alignment_latent_global -> Ridge",
        "x2_formula": "B3 + book_similarity_positive + book_similarity_negative -> Ridge (margin kept for diagnostics only)",
        "outer_cv_results": cv_results["summary"],
        "comparisons": comparisons,
        "feature_distributions": diagnostics["feature_distributions"],
        "coefficients": coefficient_summary,
        "redundancy_summary": diagnostics["redundancy_summary"],
        "warnings": [
            "The crossmedia branch is exploratory training-only; no holdout evaluation was run.",
            "B3 was reused from the frozen latent-semantics run and not modified.",
            "The diagnostic margin feature is intentionally excluded from the X2 model to avoid deterministic collinearity.",
        ],
        "holdout_confirmation": "The 99-film holdout was never accessed.",
        "artifact_paths": {
            "config": str(CONFIG_PATH),
            "canonical_book_documents": str(BOOK_DOCUMENTS_PATH),
            "book_embedding_manifest": str(BOOK_MANIFEST_PATH),
            "crossmedia_features": str(CROSSMEDIA_FEATURES_PATH),
            "cv_results": str(CV_RESULTS_PATH),
            "fold_selections": str(FOLD_SELECTIONS_PATH),
            "crossmedia_diagnostics": str(DIAGNOSTICS_PATH),
            "crossmedia_coefficients": str(COEFFICIENTS_PATH),
            "readme": str(README_PATH),
        },
        "no_posthoc_architecture_changes": True,
    }
    return checkpoint
