from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mvp.src.config import get_api_keys
from mvp.src.data import load_table, normalize_column_names
from mvp.src.models import ConventionalMetadataFeatureBuilder, evaluate_regression
from mvp.src.semantics import SEMANTIC_COLUMNS, SemanticVectorStore
from mvp.src.taste_profile import build_taste_profile


MODEL_DIR = Path("mvp/artifacts/models/b3_mvp")
MODEL_PATH = MODEL_DIR / "model.joblib"
PCA_PATH = MODEL_DIR / "pca.joblib"
METADATA_PREPROCESSOR_PATH = MODEL_DIR / "metadata_preprocessor.joblib"
CONFIG_PATH = MODEL_DIR / "config.json"
FEATURE_SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
TRAINING_MANIFEST_PATH = MODEL_DIR / "training_manifest.json"
CV_RESULTS_PATH = MODEL_DIR / "cv_results.json"
RECOMMENDATION_CACHE_DIR = MODEL_DIR / "candidate_embedding_cache"
CANDIDATE_EMBEDDING_CACHE_PATH = RECOMMENDATION_CACHE_DIR / "candidate_embeddings.csv"

TRAINING_SPLIT_PATH = Path("mvp/data/processed/primary_holdout_split.csv")
TRAINING_ENRICHED_PATH = Path("mvp/data/processed/condition_a_enriched.csv")
TRAINING_EMBEDDINGS_PATH = Path("mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv")
TRAINING_FOLDS_PATH = Path("mvp/artifacts/experiments/exploratory_latent_semantics/fold_selections.json")

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONALITY = 1536
DOCUMENT_TEMPLATE_VERSION = "film_doc_v1"
MODEL_STATUS = "exploratory_candidate_mvp"
ARCHITECTURE = "B3"
TARGET_COLUMN = "preference_weight"
PCA_CANDIDATES = (16, 32, 64, 128)
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
RANDOM_STATE = 42
NDCG_K = 10


@dataclass
class B3MVPBundle:
    model_status: str
    architecture: str
    target_column: str
    ridge: Ridge
    scaler: StandardScaler
    selected_pca_dim: int
    selected_alpha: float
    embedding_model: str
    embedding_dimensionality: int
    metadata_feature_columns: list[str]
    latent_feature_columns: list[str]
    feature_columns: list[str]
    training_population: int
    trained_source_ids: list[str]
    source_experiment: str
    fit_timestamp: str
    holdout_used: bool = False
    notes: str = "Frozen deployable B3 MVP"


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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_columns(frame: pd.DataFrame, prefix: str = "emb_") -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def _filter_watched_candidates(
    candidates: pd.DataFrame,
    watched_ids: Iterable[str] | None = None,
    id_column: str = "source_id",
) -> pd.DataFrame:
    frame = candidates.copy()
    if watched_ids is None or id_column not in frame.columns:
        return frame
    watched = {str(value) for value in watched_ids}
    return frame[~frame[id_column].astype(str).isin(watched)].copy()


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _build_strata(frame: pd.DataFrame) -> pd.Series:
    ratings = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return pd.cut(
        ratings,
        bins=[-1.0, 2.0, 3.0, 4.0, 5.1],
        labels=["low", "mid", "high", "top"],
        include_lowest=True,
    ).astype(str)


def _build_folds() -> list[dict[str, Any]]:
    if TRAINING_FOLDS_PATH.exists():
        return json.loads(TRAINING_FOLDS_PATH.read_text())
    training = load_training_population()
    strata = _build_strata(training)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(training, strata), start=1):
        folds.append(
            {
                "fold": fold_index,
                "train_source_ids": training.iloc[train_idx]["source_id"].astype(str).tolist(),
                "val_source_ids": training.iloc[val_idx]["source_id"].astype(str).tolist(),
            }
        )
    return folds


def load_training_population() -> pd.DataFrame:
    split = pd.read_csv(TRAINING_SPLIT_PATH)
    enriched = pd.read_csv(TRAINING_ENRICHED_PATH)
    merged = split.merge(enriched, on="source_id", how="inner", validate="one_to_one")
    train = merged.loc[merged["split"] == "train"].copy().reset_index(drop=True)
    if len(train) != 393:
        raise RuntimeError(f"Expected 393 training films, found {len(train)}.")
    train["source_id"] = train["source_id"].astype(str)
    return train


def load_training_embeddings() -> pd.DataFrame:
    frame = pd.read_csv(TRAINING_EMBEDDINGS_PATH)
    frame["source_id"] = frame["source_id"].astype(str)
    return frame


def load_model_bundle(model_dir: str | Path = MODEL_DIR) -> tuple[B3MVPBundle, PCA, ConventionalMetadataFeatureBuilder]:
    model_dir = Path(model_dir)
    bundle = joblib.load(model_dir / "model.joblib")
    pca = joblib.load(model_dir / "pca.joblib")
    metadata_preprocessor = joblib.load(model_dir / "metadata_preprocessor.joblib")
    if not isinstance(bundle, B3MVPBundle):
        raise TypeError("model.joblib does not contain a B3MVPBundle.")
    return bundle, pca, metadata_preprocessor


def build_film_document(row: pd.Series) -> str:
    lines = [f"Title: {_normalize_text(row.get('title'))}"]
    if not _is_missing(row.get("release_year")):
        lines.append(f"Release year: {int(float(row.get('release_year')))}")
    if not _is_missing(row.get("tmdb_genres")):
        lines.append(f"Genres: {_normalize_text(row.get('tmdb_genres')).replace('|', ' | ')}")
    if not _is_missing(row.get("tmdb_original_language")):
        lines.append(f"Original language: {_normalize_text(row.get('tmdb_original_language'))}")
    if not _is_missing(row.get("tmdb_production_countries")):
        lines.append(
            f"Production countries: {_normalize_text(row.get('tmdb_production_countries')).replace('|', ' | ')}"
        )
    if not _is_missing(row.get("tmdb_overview")):
        lines.append(f"Overview: {_normalize_text(row.get('tmdb_overview'))}")
    return "\n".join(lines)


def build_film_documents(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = frame.sort_values("source_id").reset_index(drop=True)
    for _, row in ordered.iterrows():
        document = build_film_document(row)
        rows.append(
            {
                "source_id": str(row.get("source_id", "")),
                "title": row.get("title"),
                "document_text": document,
                "document_hash": _sha256(document),
                "document_length": len(document),
                "template_version": DOCUMENT_TEMPLATE_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _load_embedding_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "source_id" in frame.columns:
        frame["source_id"] = frame["source_id"].astype(str)
    return frame


def _write_embedding_cache(embeddings: pd.DataFrame, manifest: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.to_csv(cache_path, index=False, float_format="%.8f")
    manifest.to_csv(cache_path.with_name("candidate_embedding_manifest.csv"), index=False)


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


def load_or_generate_candidate_embeddings(
    documents: pd.DataFrame,
    cache_path: str | Path = CANDIDATE_EMBEDDING_CACHE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cache_path = Path(cache_path)
    cached = _load_embedding_cache(cache_path)
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
            _write_embedding_cache(cached, manifest, cache_path)
            return cached, manifest, {
                "provider": "openai",
                "model": OPENAI_EMBEDDING_MODEL,
                "embedding_dim": int(cached["embedding_dim"].iloc[0]),
                "coverage": 1.0,
                "successful": int((cached["status"] == "success").sum()),
                "failed": int((cached["status"] != "success").sum()),
                "cache_status": "hit",
                "cache_hits": int(len(documents)),
                "cache_misses": 0,
            }

    keys = get_api_keys()
    if not keys.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate candidate embeddings.")

    vector_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    batch_size = 32
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
        _write_embedding_cache(pd.DataFrame(vector_rows), pd.DataFrame(manifest_rows), cache_path)

    embeddings = pd.DataFrame(vector_rows)
    manifest = pd.DataFrame(manifest_rows)
    _write_embedding_cache(embeddings, manifest, cache_path)
    return embeddings, manifest, {
        "provider": "openai",
        "model": OPENAI_EMBEDDING_MODEL,
        "embedding_dim": int(embedding_dim or 0),
        "coverage": float(len(embeddings) / len(documents)) if len(documents) else 0.0,
        "successful": int(len(embeddings)),
        "failed": int((manifest["status"] != "success").sum()) if not manifest.empty else 0,
        "cache_status": "miss",
        "cache_hits": 0,
        "cache_misses": int(len(documents)),
    }


def _build_feature_frame(
    frame: pd.DataFrame,
    metadata_builder: ConventionalMetadataFeatureBuilder,
    latent_matrix: np.ndarray,
    model_columns: list[str],
) -> pd.DataFrame:
    metadata = metadata_builder.transform(frame).reset_index(drop=True)
    latent_cols = [f"latent_pca_{index + 1:03d}" for index in range(latent_matrix.shape[1])]
    latent = pd.DataFrame(latent_matrix, columns=latent_cols)
    features = pd.concat([metadata, latent], axis=1)
    return features.reindex(columns=model_columns, fill_value=0.0)


def _metadata_columns(metadata_builder: ConventionalMetadataFeatureBuilder, frame: pd.DataFrame) -> list[str]:
    return metadata_builder.transform(frame).reset_index(drop=True).columns.tolist()


def _fit_fixed_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
    return model


def select_b3_deployment_hyperparameters(
    training_films: pd.DataFrame,
    film_embeddings: pd.DataFrame,
    folds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if folds is None:
        folds = _build_folds()
    film_embeddings = film_embeddings.copy()
    film_embeddings["source_id"] = film_embeddings["source_id"].astype(str)
    film_embedding_lookup = film_embeddings.set_index("source_id")
    emb_cols = _vector_columns(film_embeddings)
    if len(emb_cols) != OPENAI_EMBEDDING_DIMENSIONALITY:
        raise RuntimeError(f"Expected {OPENAI_EMBEDDING_DIMENSIONALITY} embedding dimensions, found {len(emb_cols)}.")

    training_lookup = training_films.set_index("source_id")
    candidate_rows: list[dict[str, Any]] = []
    for pca_dim in PCA_CANDIDATES:
        if pca_dim >= len(training_films) or pca_dim >= len(emb_cols):
            continue
        for alpha in RIDGE_ALPHAS:
            fold_metrics = []
            fold_explained = []
            for fold in folds:
                train_ids = [str(value) for value in fold["train_source_ids"]]
                val_ids = [str(value) for value in fold["val_source_ids"]]
                fold_train = training_lookup.loc[train_ids].reset_index()
                fold_val = training_lookup.loc[val_ids].reset_index()
                fold_train_emb = film_embedding_lookup.loc[train_ids, emb_cols].to_numpy(dtype=float)
                fold_val_emb = film_embedding_lookup.loc[val_ids, emb_cols].to_numpy(dtype=float)
                metadata_builder = ConventionalMetadataFeatureBuilder().fit(fold_train)
                metadata_columns = _metadata_columns(metadata_builder, fold_train)
                scaler = StandardScaler().fit(fold_train_emb)
                pca = PCA(n_components=pca_dim, whiten=True, random_state=RANDOM_STATE, svd_solver="randomized")
                pca.fit(scaler.transform(fold_train_emb))
                latent_train = pca.transform(scaler.transform(fold_train_emb))
                latent_val = pca.transform(scaler.transform(fold_val_emb))
                latent_columns = [f"latent_pca_{index + 1:03d}" for index in range(pca_dim)]
                X_train = _build_feature_frame(
                    fold_train,
                    metadata_builder,
                    latent_train,
                    metadata_columns + latent_columns,
                )
                X_val = _build_feature_frame(
                    fold_val,
                    metadata_builder,
                    latent_val,
                    metadata_columns + latent_columns,
                )
                ridge = _fit_fixed_alpha(X_train, fold_train[TARGET_COLUMN], float(alpha))
                predictions = ridge.predict(X_val.to_numpy(dtype=float))
                metrics = evaluate_regression(fold_val[TARGET_COLUMN].to_numpy(dtype=float), predictions, ndcg_k=NDCG_K)
                fold_metrics.append(metrics)
                fold_explained.append(float(pca.explained_variance_ratio_.sum()))

            candidate_rows.append(
                {
                    "pca_dim": int(pca_dim),
                    "alpha": float(alpha),
                    "mean_mae": float(np.mean([row["mae"] for row in fold_metrics])),
                    "mean_rmse": float(np.mean([row["rmse"] for row in fold_metrics])),
                    "mean_pearson": float(np.mean([row["pearson"] for row in fold_metrics])),
                    "mean_spearman": float(np.mean([row["spearman"] for row in fold_metrics])),
                    "mean_ndcg_at_k": float(np.mean([row["ndcg_at_k"] for row in fold_metrics])),
                    "mean_explained_variance": float(np.mean(fold_explained)),
                }
            )

    selected = sorted(
        candidate_rows,
        key=lambda row: (row["mean_spearman"], row["mean_ndcg_at_k"], -row["mean_mae"]),
        reverse=True,
    )[0]
    return {
        "selected_pca_dim": int(selected["pca_dim"]),
        "selected_alpha": float(selected["alpha"]),
        "candidate_rows": candidate_rows,
    }


def fit_mvp_b3_model(
    model_dir: str | Path = MODEL_DIR,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    recommendation_cache_dir = model_dir / "candidate_embedding_cache"
    recommendation_cache_dir.mkdir(parents=True, exist_ok=True)

    training_films = load_training_population()
    film_embeddings = load_training_embeddings()
    folds = _build_folds()
    selection = select_b3_deployment_hyperparameters(training_films, film_embeddings, folds=folds)
    selected_pca_dim = selection["selected_pca_dim"]
    selected_alpha = selection["selected_alpha"]

    emb_cols = _vector_columns(film_embeddings)
    film_embeddings = film_embeddings.set_index("source_id").loc[training_films["source_id"].astype(str)].reset_index()
    training_embedding_matrix = film_embeddings.loc[:, emb_cols].to_numpy(dtype=float)

    metadata_builder = ConventionalMetadataFeatureBuilder().fit(training_films)
    metadata_columns = _metadata_columns(metadata_builder, training_films)
    scaler = StandardScaler().fit(training_embedding_matrix)
    pca = PCA(
        n_components=selected_pca_dim,
        whiten=True,
        random_state=RANDOM_STATE,
        svd_solver="randomized",
    )
    pca.fit(scaler.transform(training_embedding_matrix))
    latent_train = pca.transform(scaler.transform(training_embedding_matrix))

    latent_columns = [f"latent_pca_{index + 1:03d}" for index in range(selected_pca_dim)]
    metadata_features = metadata_builder.transform(training_films).reset_index(drop=True)
    X_train = pd.concat([metadata_features, pd.DataFrame(latent_train, columns=latent_columns)], axis=1)
    feature_columns = X_train.columns.tolist()
    ridge = _fit_fixed_alpha(X_train, training_films[TARGET_COLUMN], selected_alpha)

    bundle = B3MVPBundle(
        model_status=MODEL_STATUS,
        architecture=ARCHITECTURE,
        target_column=TARGET_COLUMN,
        ridge=ridge,
        scaler=scaler,
        selected_pca_dim=selected_pca_dim,
        selected_alpha=selected_alpha,
        embedding_model=OPENAI_EMBEDDING_MODEL,
        embedding_dimensionality=OPENAI_EMBEDDING_DIMENSIONALITY,
        metadata_feature_columns=metadata_columns,
        latent_feature_columns=latent_columns,
        feature_columns=feature_columns,
        training_population=int(len(training_films)),
        trained_source_ids=training_films["source_id"].astype(str).tolist(),
        source_experiment="exploratory_latent_semantics",
        fit_timestamp=datetime.now(timezone.utc).isoformat(),
        holdout_used=False,
    )

    joblib.dump(bundle, model_dir / "model.joblib")
    joblib.dump(pca, model_dir / "pca.joblib")
    joblib.dump(metadata_builder, model_dir / "metadata_preprocessor.joblib")

    feature_schema = {
        "model_status": MODEL_STATUS,
        "architecture": ARCHITECTURE,
        "target": TARGET_COLUMN,
        "training_population": int(len(training_films)),
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimensionality": OPENAI_EMBEDDING_DIMENSIONALITY,
        "selected_pca_dim": selected_pca_dim,
        "selected_alpha": selected_alpha,
        "metadata_feature_families": {
            "genre": metadata_builder.genre_values,
            "language": metadata_builder.language_values,
            "country": metadata_builder.country_values,
            "decade": metadata_builder.decade_values,
            "year": "release_year_z",
        },
        "metadata_feature_columns": metadata_features.columns.tolist(),
        "latent_feature_columns": latent_columns,
        "feature_columns": feature_columns,
        "candidate_schema": {
            "required": ["title"],
            "recommended": ["source_id", "canonical_id", "tmdb_id", "release_year", "tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"],
            "optional": ["embedding_model", "embedding_dim", "semantic_vector_62"],
            "document_template_version": DOCUMENT_TEMPLATE_VERSION,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "embedding_dimensionality": OPENAI_EMBEDDING_DIMENSIONALITY,
        },
    }

    config = {
        "model_status": MODEL_STATUS,
        "architecture": ARCHITECTURE,
        "training_population": int(len(training_films)),
        "target": TARGET_COLUMN,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "original_embedding_dimensionality": OPENAI_EMBEDDING_DIMENSIONALITY,
        "selected_pca_dimensionality": selected_pca_dim,
        "selected_ridge_alpha": selected_alpha,
        "metadata_feature_families": ["genre", "language", "country", "year", "decade"],
        "source_experiment": "exploratory_latent_semantics",
        "fit_timestamp": bundle.fit_timestamp,
        "holdout_used": False,
        "holdout_confirmation": "The frozen 99-film holdout was never used.",
    }

    training_manifest = {
        "model_status": MODEL_STATUS,
        "architecture": ARCHITECTURE,
        "source_experiment": "exploratory_latent_semantics",
        "training_population": int(len(training_films)),
        "training_source_ids": training_films["source_id"].astype(str).tolist(),
        "selected_pca_dim": selected_pca_dim,
        "selected_alpha": selected_alpha,
        "cross_validation": {
            "fold_count": len(folds),
            "selection_metric": "spearman",
            "tie_breaker": "ndcg_at_k",
            "candidate_rows": selection["candidate_rows"],
        },
        "holdout_used": False,
        "holdout_confirmation": "The frozen 99-film holdout was never used.",
    }

    cv_results = {
        "selection": selection,
        "fold_count": len(folds),
        "training_population": int(len(training_films)),
    }

    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    (model_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2))
    (model_dir / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2))
    (model_dir / "cv_results.json").write_text(json.dumps(cv_results, indent=2))

    return {
        "model_dir": str(model_dir),
        "model_status": MODEL_STATUS,
        "architecture": ARCHITECTURE,
        "selected_pca_dim": selected_pca_dim,
        "selected_alpha": selected_alpha,
        "training_population": int(len(training_films)),
        "feature_columns": feature_columns,
        "config_path": str(model_dir / "config.json"),
        "feature_schema_path": str(model_dir / "feature_schema.json"),
        "training_manifest_path": str(model_dir / "training_manifest.json"),
        "cv_results_path": str(model_dir / "cv_results.json"),
        "model_path": str(model_dir / "model.joblib"),
        "pca_path": str(model_dir / "pca.joblib"),
        "metadata_preprocessor_path": str(model_dir / "metadata_preprocessor.joblib"),
    }


def _load_candidate_frame(candidates: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(candidates, pd.DataFrame):
        frame = candidates.copy()
    else:
        frame = normalize_column_names(load_table(candidates))
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()

    def _coalesce(target: str, sources: Sequence[str]) -> None:
        if target in frame.columns:
            return
        for source in sources:
            if source in frame.columns:
                frame[target] = frame[source]
                return

    if "source_id" not in frame.columns and "canonical_id" in frame.columns:
        frame["source_id"] = frame["canonical_id"].astype(str)
    if "source_id" not in frame.columns and "tmdb_id" in frame.columns:
        frame["source_id"] = frame["tmdb_id"].astype(str)
    _coalesce("title", ("title_x", "title_y", "worktitle", "name"))
    if "title" not in frame.columns:
        raise KeyError("Candidate file must include a title column.")
    if "source_id" not in frame.columns:
        raise KeyError("Candidate file must include source_id, canonical_id, or tmdb_id.")
    _coalesce("release_year", ("release_year_x", "release_year_y", "year_x", "year_y"))
    if "release_year" not in frame.columns and "year" in frame.columns:
        frame["release_year"] = frame["year"]
    if "year" not in frame.columns and "release_year" in frame.columns:
        frame["year"] = frame["release_year"]
    frame["source_id"] = frame["source_id"].astype(str)
    return frame


def _candidate_embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def validate_candidate_contract(frame: pd.DataFrame, required_schema: Sequence[str] | None = None) -> None:
    required_schema = required_schema or ("source_id", "title")
    missing = [column for column in required_schema if column not in frame.columns]
    if missing:
        raise KeyError(f"Candidate frame is missing required columns: {missing}")


def score_candidates(
    candidates: pd.DataFrame | str | Path,
    model_dir: str | Path = MODEL_DIR,
    watched_ids: Iterable[str] | None = None,
    top_k: int = 10,
    candidate_semantic_vectors: pd.DataFrame | str | Path | None = None,
    taste_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = _load_candidate_frame(candidates)
    validate_candidate_contract(frame)
    bundle, pca, metadata_builder = load_model_bundle(model_dir)

    if watched_ids is not None:
        frame = _filter_watched_candidates(frame, watched_ids=watched_ids, id_column="source_id")

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "tmdb_id",
                "title",
                "release_year",
                "candidate_sources",
                "predicted_preference",
                "rank",
                "model_status",
                "architecture",
                "taste_evidence",
            ]
        )

    if candidate_semantic_vectors is not None:
        vectors = candidate_semantic_vectors if isinstance(candidate_semantic_vectors, pd.DataFrame) else SemanticVectorStore(candidate_semantic_vectors).load()
        if "canonical_id" in vectors.columns and "source_id" not in vectors.columns:
            vectors = vectors.rename(columns={"canonical_id": "source_id"})
        if "source_id" in vectors.columns:
            vectors["source_id"] = vectors["source_id"].astype(str)
            frame = frame.merge(vectors, on="source_id", how="left", suffixes=("", "_semantic"))

    emb_cols = _candidate_embedding_columns(frame)
    if len(emb_cols) == OPENAI_EMBEDDING_DIMENSIONALITY:
        embeddings = frame.loc[:, emb_cols].to_numpy(dtype=float)
    else:
        documents = build_film_documents(frame)
        embeddings, _, _ = load_or_generate_candidate_embeddings(documents)
        embeddings = embeddings.set_index("source_id").loc[frame["source_id"].astype(str)].reset_index()
        emb_cols = _vector_columns(embeddings)
        embeddings = embeddings.loc[:, emb_cols].to_numpy(dtype=float)

    if embeddings.shape[1] != OPENAI_EMBEDDING_DIMENSIONALITY:
        raise ValueError(
            f"Candidate embeddings must have {OPENAI_EMBEDDING_DIMENSIONALITY} dimensions before PCA."
        )

    metadata = metadata_builder.transform(frame).reset_index(drop=True)
    scaled = bundle.scaler.transform(embeddings)
    latent = pca.transform(scaled)
    latent_columns = [f"latent_pca_{index + 1:03d}" for index in range(latent.shape[1])]
    features = pd.concat([metadata, pd.DataFrame(latent, columns=latent_columns)], axis=1)
    features = features.reindex(columns=bundle.feature_columns, fill_value=0.0)

    predictions = bundle.ridge.predict(features.to_numpy(dtype=float))
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("Prediction produced non-finite values.")

    ranked = frame.copy()
    ranked["predicted_preference"] = predictions
    ranked = ranked.sort_values(["predicted_preference", "source_id"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked = ranked.head(top_k).copy()
    ranked["model_status"] = bundle.model_status
    ranked["architecture"] = bundle.architecture
    if taste_profile is not None:
        ranked["taste_evidence"] = ranked.apply(lambda row: build_taste_evidence(row, taste_profile), axis=1)
    else:
        ranked["taste_evidence"] = ranked.apply(lambda row: build_taste_evidence(row, None), axis=1)

    return ranked.loc[
        :,
        [
            "source_id",
            *(["tmdb_id"] if "tmdb_id" in ranked.columns else []),
            "title",
            *(["year"] if "year" in ranked.columns else []),
            *(["release_year"] if "release_year" in ranked.columns else []),
            *(["candidate_sources"] if "candidate_sources" in ranked.columns else []),
            "predicted_preference",
            "rank",
            "model_status",
            "architecture",
            "taste_evidence",
        ]
        + [column for column in ["tmdb_genres", "tmdb_original_language", "tmdb_production_countries"] if column in ranked.columns],
    ]


def build_taste_evidence(
    candidate_row: pd.Series,
    taste_profile: pd.DataFrame | None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if taste_profile is None or taste_profile.empty:
        return {"evidence_status": "unavailable", "positive_matches": [], "possible_mismatches": []}

    profile = taste_profile.copy()
    profile = profile.dropna(subset=["pearson_preference_association"])
    if "dimension" not in profile.columns:
        raise ValueError("Taste profile is missing the canonical dimension column.")
    profile_dimensions = set(profile["dimension"].astype(str).tolist())
    expected_dimensions = set(SEMANTIC_COLUMNS)
    if profile_dimensions != expected_dimensions:
        missing = sorted(expected_dimensions - profile_dimensions)
        extra = sorted(profile_dimensions - expected_dimensions)
        raise ValueError(
            "Taste profile taxonomy mismatch: "
            f"missing={missing}, extra={extra}"
        )

    candidate_vector = {}
    for dimension in SEMANTIC_COLUMNS:
        # Some candidate exports may be sparse or partially classified; treat missing
        # semantic coordinates as neutral evidence rather than aborting the run.
        candidate_vector[dimension] = pd.to_numeric(candidate_row.get(dimension, 0.0), errors="coerce")
    if not candidate_vector:
        return {"evidence_status": "unavailable", "positive_matches": [], "possible_mismatches": []}
    positive_matches = []
    possible_mismatches = []
    for _, row in profile.iterrows():
        dimension = str(row["dimension"])
        candidate_score = candidate_vector.get(dimension)
        if candidate_score is None or pd.isna(candidate_score):
            continue
        evidence_count = int(row["evidence_count"])
        evidence_confidence = float(row["evidence_confidence"])
        association = float(row["pearson_preference_association"])
        if evidence_count < 3 or evidence_confidence < 0.4:
            continue
        if association > 0 and candidate_score >= threshold:
            positive_matches.append(
                {
                    "dimension": dimension,
                    "candidate_score": float(candidate_score),
                    "user_preference_association": association,
                    "evidence_count": evidence_count,
                    "evidence_confidence": evidence_confidence,
                }
            )
        elif association < 0 and candidate_score >= threshold:
            possible_mismatches.append(
                {
                    "dimension": dimension,
                    "candidate_score": float(candidate_score),
                    "user_preference_association": association,
                    "evidence_count": evidence_count,
                    "evidence_confidence": evidence_confidence,
                }
            )

    positive_matches = sorted(positive_matches, key=lambda row: (row["user_preference_association"], row["evidence_confidence"]), reverse=True)[:5]
    possible_mismatches = sorted(possible_mismatches, key=lambda row: (row["user_preference_association"], row["evidence_confidence"]))[:5]
    return {
        "evidence_status": "descriptive_not_causal",
        "positive_matches": positive_matches,
        "possible_mismatches": possible_mismatches,
    }


def build_crossmedia_profile_comparison(
    film_profile: pd.DataFrame,
    book_profile: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if book_profile is None or book_profile.empty:
        return {
            "agreements": [],
            "divergences": [],
            "film_evidence": film_profile.to_dict(orient="records"),
            "book_evidence": [],
            "ranking_effect": False,
        }

    film = film_profile.copy().set_index("dimension")
    book = book_profile.copy().set_index("dimension")
    common = film.index.intersection(book.index)
    agreements = []
    divergences = []
    for dimension in common:
        film_assoc = float(film.loc[dimension, "pearson_preference_association"])
        book_assoc = float(book.loc[dimension, "pearson_preference_association"])
        item = {
            "dimension": dimension,
            "film_association": film_assoc,
            "book_association": book_assoc,
        }
        if np.sign(film_assoc) == np.sign(book_assoc):
            agreements.append(item)
        else:
            divergences.append(item)
    return {
        "agreements": agreements,
        "divergences": divergences,
        "film_evidence": film_profile.to_dict(orient="records"),
        "book_evidence": book_profile.to_dict(orient="records"),
        "ranking_effect": False,
    }
