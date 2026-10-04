from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mvp.src.config import get_api_keys
from mvp.src.data import (
    GOODREADS_SEMANTIC_VECTOR_DIR,
    load_table,
    normalize_column_names,
)
from mvp.src.models import ConventionalMetadataFeatureBuilder, evaluate_regression, fit_ridge_model, predict_with_bundle
from mvp.src.semantics import SEMANTIC_COLUMNS, SemanticVectorStore
from mvp.src.taste_profile import build_taste_profile


CONDITION_C_DIR = Path("mvp/artifacts/experiments/condition_c_goodreads_crossmedia")
GOODREADS_SEMANTIC_CACHE_PATH = GOODREADS_SEMANTIC_VECTOR_DIR / "goodreads_semantic_vectors.csv"
GOODREADS_PROFILE_PATH = CONDITION_C_DIR / "goodreads_taste_profile.csv"
GOODREADS_METADATA_CACHE_PATH = CONDITION_C_DIR / "goodreads_metadata_cache.csv"
GROUP_MAPPING_PATH = CONDITION_C_DIR / "crossmedia_group_mapping.json"
CONFIG_PATH = CONDITION_C_DIR / "config.json"
CV_RESULTS_PATH = CONDITION_C_DIR / "cv_results.json"
SELECTION_PATH = CONDITION_C_DIR / "architecture_selection.json"
MODEL_PATH = CONDITION_C_DIR / "condition_c_model.joblib"

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TMDB_LIKE_RATING_BINS = [-1.0, 2.0, 3.0, 4.0, 5.1]
TMDB_LIKE_RATING_LABELS = ["low", "mid", "high", "top"]


CROSS_MEDIA_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "themes",
            (
                "identity",
                "love_desire",
                "grief",
                "class",
                "family",
                "alienation",
                "coming_of_age",
                "power",
                "morality",
                "friendship",
            ),
        ),
        (
            "character_relationships",
            (
                "morally_ambiguous",
                "dysfunctional_relationships",
                "intimate_relationships",
                "outsider_protagonist",
                "ensemble",
                "anti_hero",
            ),
        ),
        (
            "narrative",
            (
                "character_driven",
                "plot_driven",
                "nonlinear",
                "ambiguous",
                "episodic",
                "conventional",
                "complex",
            ),
        ),
        (
            "pacing_energy",
            (
                "slow_burn",
                "contemplative",
                "moderate",
                "fast_paced",
                "high_energy",
            ),
        ),
        (
            "tone",
            (
                "melancholic",
                "darkly_comic",
                "romantic",
                "unsettling",
                "sentimental",
                "absurd",
                "ethereal",
                "bizarre",
                "joyful",
                "tense",
                "comforting",
                "playful",
            ),
        ),
        (
            "style_experimentalism",
            (
                "experimental",
                "minimalist",
                "maximalist",
                "atmospheric",
                "stylized",
                "surreal",
            ),
        ),
    ]
)

SHARED_DIMENSIONS: tuple[str, ...] = tuple(dimension for group in CROSS_MEDIA_GROUPS.values() for dimension in group)
EXCLUDED_DIMENSIONS: tuple[str, ...] = tuple(d for d in SEMANTIC_COLUMNS if d not in SHARED_DIMENSIONS)


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
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
    return " ".join(str(value).lower().split())


def _truncate(value: Any, limit: int = 700) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _present_or_na(value: Any) -> Any:
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except Exception:
        pass
    return value


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


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


def _parse_semantic_batch_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    text = None
    if "choices" in payload:
        message = payload["choices"][0].get("message", {})
        text = message.get("content")
    elif "output_text" in payload:
        text = payload["output_text"]
    if text is None:
        raise ValueError("OpenAI response did not contain text content.")
    parsed = json.loads(text)
    items = parsed.get("items", [])
    results = []
    for item in items:
        vector = item.get("vector", {})
        results.append(
            {
                "canonical_id": str(item.get("canonical_id")),
                **{dimension: max(0.0, min(1.0, float(vector.get(dimension, 0.0)))) for dimension in SEMANTIC_COLUMNS},
            }
        )
    return results


def load_combined_ingestion(path: str | Path) -> pd.DataFrame:
    frame = normalize_column_names(load_table(path))
    if "canonical_id" not in frame.columns and "source_id" in frame.columns:
        frame["canonical_id"] = frame["source_id"].astype(str)
    if "source_id" not in frame.columns and "canonical_id" in frame.columns:
        frame["source_id"] = frame["canonical_id"].astype(str)
    if "year" not in frame.columns and "release_year" in frame.columns:
        frame["year"] = frame["release_year"]
    if "release_year" not in frame.columns and "year" in frame.columns:
        frame["release_year"] = frame["year"]
    return frame


def get_goodreads_book_subset(frame: pd.DataFrame) -> pd.DataFrame:
    books = frame.loc[(frame["media_type"] == "book") & frame["rating"].notna()].copy()
    if "canonical_id" not in books.columns:
        books["canonical_id"] = books["source_id"].astype(str)
    if "source_id" not in books.columns:
        books["source_id"] = books["canonical_id"].astype(str)
    return books.reset_index(drop=True)


class GoogleBooksClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def search_volume(self, title: str, author: str | None = None, year: int | float | str | None = None) -> dict[str, Any] | None:
        parts = [f'intitle:"{title}"']
        if author and not pd.isna(author):
            parts.append(f'inauthor:"{author}"')
        params = {
            "q": " ".join(parts),
            "printType": "books",
            "orderBy": "relevance",
            "maxResults": 5,
        }
        if self.api_key:
            params["key"] = self.api_key
        url = f"https://www.googleapis.com/books/v1/volumes?{urlencode(params, quote_via=quote_plus)}"
        payload = _http_json(url, timeout=self.timeout)
        results = payload.get("items", [])
        if not results:
            return None

        title_norm = _normalize_text(title)
        author_norm = _normalize_text(author) if author else ""
        best = None
        best_score = -1.0
        for result in results:
            info = result.get("volumeInfo", {})
            candidate_title = _normalize_text(info.get("title") or "")
            candidate_authors = [_normalize_text(a) for a in info.get("authors", [])]
            score = 0.0
            if candidate_title == title_norm:
                score += 3.0
            if author_norm and author_norm in candidate_authors:
                score += 2.0
            published = info.get("publishedDate")
            published_year = None
            if published:
                try:
                    published_year = int(str(published)[:4])
                except Exception:
                    published_year = None
            if year is not None and pd.notna(year) and published_year is not None:
                score += max(0.0, 2.0 - abs(int(float(year)) - published_year))
            score += float(info.get("pageCount") or 0.0) / 1000.0
            score += float(info.get("ratingsCount") or 0.0) / 100.0
            if score > best_score:
                best_score = score
                best = result
        if best is None:
            return None
        best["_match_score"] = best_score
        return best

    def book_context(self, row: pd.Series) -> dict[str, Any]:
        match = self.search_volume(row.get("title"), row.get("creator"), row.get("year"))
        if not match:
            return {
                "google_books_title": pd.NA,
                "google_books_authors": pd.NA,
                "google_books_description": pd.NA,
                "google_books_categories": pd.NA,
                "google_books_published_year": pd.NA,
                "google_books_language": pd.NA,
                "google_books_page_count": pd.NA,
                "google_books_average_rating": pd.NA,
                "google_books_ratings_count": pd.NA,
                "google_books_maturity_rating": pd.NA,
                "google_books_publisher": pd.NA,
                "google_books_match_score": pd.NA,
            }

        info = match.get("volumeInfo", {})
        published = info.get("publishedDate")
        published_year = pd.NA
        if published:
            try:
                published_year = int(str(published)[:4])
            except Exception:
                published_year = pd.NA
        authors = info.get("authors", [])
        categories = info.get("categories", [])
        return {
            "google_books_title": _present_or_na(info.get("title") or match.get("id")),
            "google_books_authors": "|".join(authors) if authors else pd.NA,
            "google_books_description": _truncate(info.get("description") or info.get("subtitle")),
            "google_books_categories": "|".join(categories) if categories else pd.NA,
            "google_books_published_year": published_year,
            "google_books_language": _present_or_na(info.get("language")),
            "google_books_page_count": _present_or_na(info.get("pageCount")),
            "google_books_average_rating": _present_or_na(info.get("averageRating")),
            "google_books_ratings_count": _present_or_na(info.get("ratingsCount")),
            "google_books_maturity_rating": _present_or_na(info.get("maturityRating")),
            "google_books_publisher": _present_or_na(info.get("publisher")),
            "google_books_match_score": _present_or_na(match.get("_match_score")),
        }


def _goodreads_semantic_prompt_batch(rows: list[pd.Series]) -> str:
    payload = []
    for row in rows:
        payload.append(
            {
                "canonical_id": _json_safe(row.get("canonical_id")),
                "title": _json_safe(row.get("title")),
                "creator": _json_safe(row.get("creator")),
                "year": _json_safe(row.get("year")),
                "google_books_title": _json_safe(row.get("google_books_title")),
                "google_books_authors": _json_safe(row.get("google_books_authors")),
                "google_books_description": _json_safe(row.get("google_books_description")),
                "google_books_categories": _json_safe(row.get("google_books_categories")),
                "google_books_published_year": _json_safe(row.get("google_books_published_year")),
                "google_books_language": _json_safe(row.get("google_books_language")),
                "google_books_page_count": _json_safe(row.get("google_books_page_count")),
                "google_books_average_rating": _json_safe(row.get("google_books_average_rating")),
                "google_books_maturity_rating": _json_safe(row.get("google_books_maturity_rating")),
                "google_books_publisher": _json_safe(row.get("google_books_publisher")),
            }
        )
    return (
        "Classify each Goodreads book into the fixed cultural taste taxonomy shared with the film model. "
        "Use the book metadata only. Do not use the Goodreads rating or preference_weight fields. "
        "Return valid JSON only, as an object with an `items` array. "
        "Each item must contain `canonical_id` and a `vector` object with exactly the 62 keys below, "
        "each mapped to a float between 0 and 1. Keep the same order as the input items. "
        "Do not infer sensitive or personal traits. "
        f"Input items: {json.dumps(payload, ensure_ascii=False)}. "
        f"Keys: {list(SEMANTIC_COLUMNS)}"
    )


def _classify_goodreads_semantic_batch(
    batch: list[pd.Series],
    model: str,
    api_key: str,
) -> list[dict[str, Any]]:
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "You classify cultural works into a fixed 62-dimensional taxonomy. Output only valid JSON.",
            },
            {"role": "user", "content": _goodreads_semantic_prompt_batch(batch)},
        ],
    }
    response = _openai_post_json(
        "https://api.openai.com/v1/chat/completions",
        api_key,
        payload,
        timeout=180,
    )
    parsed_batch = _parse_semantic_batch_response(response)
    parsed_by_id = {item["canonical_id"]: item for item in parsed_batch}
    ordered = []
    for row in batch:
        canonical_id = str(row.get("canonical_id"))
        item = parsed_by_id.get(canonical_id)
        if item is None:
            ordered.append({"canonical_id": canonical_id, **{dimension: 0.0 for dimension in SEMANTIC_COLUMNS}})
        else:
            ordered.append(item)
    return ordered


def _enrich_goodreads_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if GOODREADS_METADATA_CACHE_PATH.exists():
        cached = pd.read_csv(GOODREADS_METADATA_CACHE_PATH)
        if "canonical_id" in cached.columns:
            current_ids = set(frame["canonical_id"].astype(str))
            cached_ids = set(cached["canonical_id"].astype(str))
            if cached_ids == current_ids and len(cached) == len(frame):
                return cached

    keys = get_api_keys()
    client = GoogleBooksClient(api_key=keys.google_books_api_key)
    rows = []
    for _, row in frame.iterrows():
        enriched = row.to_dict()
        enriched.update(client.book_context(row))
        rows.append(enriched)
        time.sleep(0.03)
    enriched = pd.DataFrame(rows)
    GOODREADS_METADATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(GOODREADS_METADATA_CACHE_PATH, index=False)
    return enriched


def classify_goodreads_semantic_vectors(
    books: pd.DataFrame,
    cache_path: str | Path = GOODREADS_SEMANTIC_CACHE_PATH,
    model: str = DEFAULT_OPENAI_MODEL,
    batch_size: int = 8,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    keys = get_api_keys()
    if not keys.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Goodreads semantic classification.")

    cache_path = Path(cache_path)
    store = SemanticVectorStore(cache_path)
    cached = store.load()

    frame = books.copy()
    if "canonical_id" not in frame.columns:
        frame["canonical_id"] = frame["source_id"].astype(str)
    frame["canonical_id"] = frame["canonical_id"].astype(str)
    frame = _enrich_goodreads_metadata(frame)

    if not cached.empty:
        cached = cached.loc[cached["canonical_id"].astype(str).isin(set(frame["canonical_id"].astype(str)))].copy().reset_index(drop=True)

    existing_ids = set(cached["canonical_id"].astype(str)) if not cached.empty else set()
    to_classify = frame.loc[~frame["canonical_id"].isin(existing_ids)].copy().reset_index(drop=True)
    rows = [row for _, row in to_classify.iterrows()]
    vectors: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    classified = 0
    total_batches = max(1, int(np.ceil(len(rows) / batch_size)))
    batches = [
        (batch_index, rows[batch_start : batch_start + batch_size])
        for batch_index, batch_start in enumerate(range(0, len(rows), batch_size), start=1)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_classify_goodreads_semantic_batch, batch, model, keys.openai_api_key): (batch_index, batch)
            for batch_index, batch in batches
        }
        for future in as_completed(future_map):
            batch_index, batch = future_map[future]
            try:
                print(f"classifying Goodreads semantic batch {batch_index}/{total_batches}", flush=True)
                parsed_batch = future.result()
                vectors.extend(parsed_batch)
                classified += len(parsed_batch)
                combined = pd.concat([cached, pd.DataFrame(vectors)], ignore_index=True) if not cached.empty else pd.DataFrame(vectors)
                store.save(combined)
            except Exception as exc:
                for row in batch:
                    failures.append(
                        {
                            "canonical_id": str(row["canonical_id"]),
                            "title": row.get("title"),
                            "creator": row.get("creator"),
                            "error": str(exc),
                        }
                    )

    combined = pd.concat([cached, pd.DataFrame(vectors)], ignore_index=True) if vectors or not cached.empty else cached
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["canonical_id"], keep="first")
        combined = combined.loc[combined["canonical_id"].astype(str).isin(set(frame["canonical_id"].astype(str)))].copy()
    store.save(combined)
    coverage = float(len(combined) / len(frame)) if len(frame) else 0.0
    report = {
        "semantic_classified": int(len(combined)),
        "semantic_total": int(len(frame)),
        "coverage": coverage,
        "newly_classified": int(classified),
        "failures": failures,
    }
    return combined, report, frame


def build_goodreads_taste_profile(
    books: pd.DataFrame,
    book_vectors: pd.DataFrame,
    dimensions: Sequence[str] | None = None,
) -> pd.DataFrame:
    profile = build_taste_profile(
        books,
        book_vectors,
        id_column="canonical_id" if "canonical_id" in books.columns else "source_id",
        target_col="preference_weight",
        dimensions=dimensions or SHARED_DIMENSIONS,
    )
    profile["effective_book_preference"] = (
        profile["pearson_preference_association"].fillna(0.0) * profile["evidence_confidence"].fillna(0.0)
    )
    return profile


def _weighted_alignment_score(
    film_features: pd.DataFrame,
    preference_map: pd.Series,
    dimensions: Sequence[str],
) -> pd.Series:
    matrix = film_features.reindex(columns=list(dimensions), fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weights = preference_map.reindex(list(dimensions)).fillna(0.0).to_numpy(dtype=float)
    numer = matrix.to_numpy(dtype=float) @ weights
    denom = matrix.sum(axis=1).to_numpy(dtype=float)
    denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
    return pd.Series(numer / denom, index=matrix.index)


def build_cross_media_feature_frame(
    films: pd.DataFrame,
    film_vectors: pd.DataFrame,
    profile: pd.DataFrame,
    architecture: str,
) -> pd.DataFrame:
    key_column = "canonical_id" if "canonical_id" in film_vectors.columns else "source_id"
    if key_column not in film_vectors.columns:
        raise KeyError("Film semantic vectors must include canonical_id or source_id.")
    merged = films.merge(
        film_vectors.loc[:, [key_column, *SEMANTIC_COLUMNS]].copy(),
        left_on="source_id",
        right_on=key_column,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_semantic"),
    )
    preference_map = profile.set_index("dimension")["effective_book_preference"]
    features = pd.DataFrame(index=merged.index)
    if architecture == "c1":
        features["book_alignment_global"] = _weighted_alignment_score(merged, preference_map, SHARED_DIMENSIONS)
    elif architecture == "c2":
        for group_name, dims in CROSS_MEDIA_GROUPS.items():
            features[f"book_alignment_{group_name}"] = _weighted_alignment_score(merged, preference_map, dims)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    features["source_id"] = merged["source_id"].astype(str).to_numpy()
    return features


def _rating_strata(ratings: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(ratings, errors="coerce"),
        bins=TMDB_LIKE_RATING_BINS,
        labels=TMDB_LIKE_RATING_LABELS,
        include_lowest=True,
    ).astype(str)


def _build_numeric_model_frame(
    df: pd.DataFrame,
    architecture: str,
    metadata_builder: ConventionalMetadataFeatureBuilder,
    film_vectors: pd.DataFrame,
    profile: pd.DataFrame,
) -> pd.DataFrame:
    base = metadata_builder.transform(df).reset_index(drop=True)
    cross_media = build_cross_media_feature_frame(df, film_vectors, profile, architecture).reset_index(drop=True)
    cross_media = cross_media.drop(columns=["source_id"], errors="ignore")
    combined = pd.concat([df.loc[:, ["source_id", "preference_weight"]].reset_index(drop=True), base, cross_media], axis=1)
    return combined


def _evaluate_architecture(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    architecture: str,
    film_vectors: pd.DataFrame,
    profile: pd.DataFrame,
) -> dict[str, Any]:
    metadata_builder = ConventionalMetadataFeatureBuilder().fit(train_df)
    train_matrix = _build_numeric_model_frame(train_df, architecture, metadata_builder, film_vectors, profile)
    val_matrix = _build_numeric_model_frame(val_df, architecture, metadata_builder, film_vectors, profile)
    feature_columns = [column for column in train_matrix.columns if column not in {"source_id", "preference_weight"}]
    bundle = fit_ridge_model(
        train_matrix,
        target_col="preference_weight",
        feature_columns=feature_columns,
        model_name=f"condition_c_{architecture}",
    )
    predictions = predict_with_bundle(bundle, val_matrix.loc[:, feature_columns])
    metrics = evaluate_regression(val_matrix["preference_weight"].to_numpy(dtype=float), predictions)
    return {
        "bundle": bundle,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "predictions": predictions,
        "alpha": float(bundle.model.alpha_),
    }


def _summarize_metric(values: list[float]) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    return {"mean": float(series.mean()), "std": float(series.std(ddof=0))}


def cross_validate_condition_c(
    train_films: pd.DataFrame,
    film_vectors: pd.DataFrame,
    profile: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Any]:
    strata = _rating_strata(train_films["rating"])
    n_splits = min(5, int(strata.value_counts().min()))
    if n_splits < 2:
        raise ValueError("Not enough observations to run stratified cross-validation.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows: list[dict[str, Any]] = []
    architecture_metrics = {"a1": [], "c1": [], "c2": []}

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(train_films, strata), start=1):
        fold_train = train_films.iloc[train_idx].copy().reset_index(drop=True)
        fold_val = train_films.iloc[val_idx].copy().reset_index(drop=True)

        metadata_builder = ConventionalMetadataFeatureBuilder().fit(fold_train)
        base_train = metadata_builder.transform(fold_train).reset_index(drop=True)
        base_val = metadata_builder.transform(fold_val).reset_index(drop=True)
        base_feature_columns = base_train.columns.tolist()

        a1_train = pd.concat([fold_train.loc[:, ["preference_weight"]].reset_index(drop=True), base_train], axis=1)
        a1_val = pd.concat([fold_val.loc[:, ["preference_weight"]].reset_index(drop=True), base_val], axis=1)
        a1_bundle = fit_ridge_model(
            a1_train,
            target_col="preference_weight",
            feature_columns=base_feature_columns,
            model_name="condition_c_a1_reference",
        )
        a1_predictions = predict_with_bundle(a1_bundle, a1_val)
        a1_metrics = evaluate_regression(a1_val["preference_weight"].to_numpy(dtype=float), a1_predictions)

        c1_train = _build_numeric_model_frame(fold_train, "c1", metadata_builder, film_vectors, profile)
        c1_val = _build_numeric_model_frame(fold_val, "c1", metadata_builder, film_vectors, profile)
        c1_feature_columns = [column for column in c1_train.columns if column not in {"source_id", "preference_weight"}]
        c1_bundle = fit_ridge_model(
            c1_train,
            target_col="preference_weight",
            feature_columns=c1_feature_columns,
            model_name="condition_c_c1",
        )
        c1_predictions = predict_with_bundle(c1_bundle, c1_val.loc[:, c1_feature_columns])
        c1_metrics = evaluate_regression(c1_val["preference_weight"].to_numpy(dtype=float), c1_predictions)

        c2_train = _build_numeric_model_frame(fold_train, "c2", metadata_builder, film_vectors, profile)
        c2_val = _build_numeric_model_frame(fold_val, "c2", metadata_builder, film_vectors, profile)
        c2_feature_columns = [column for column in c2_train.columns if column not in {"source_id", "preference_weight"}]
        c2_bundle = fit_ridge_model(
            c2_train,
            target_col="preference_weight",
            feature_columns=c2_feature_columns,
            model_name="condition_c_c2",
        )
        c2_predictions = predict_with_bundle(c2_bundle, c2_val.loc[:, c2_feature_columns])
        c2_metrics = evaluate_regression(c2_val["preference_weight"].to_numpy(dtype=float), c2_predictions)

        architecture_metrics["a1"].append(a1_metrics)
        architecture_metrics["c1"].append(c1_metrics)
        architecture_metrics["c2"].append(c2_metrics)

        fold_rows.append(
            {
                "fold": fold_index,
                "train_n": int(len(fold_train)),
                "val_n": int(len(fold_val)),
                "a1_alpha": float(a1_bundle.model.alpha_),
                "c1_alpha": float(c1_bundle.model.alpha_),
                "c2_alpha": float(c2_bundle.model.alpha_),
                "a1": a1_metrics,
                "c1": c1_metrics,
                "c2": c2_metrics,
            }
        )

    summary = {}
    for name, metric_rows in architecture_metrics.items():
        summary[name] = {
            metric: _summarize_metric([row[metric] for row in metric_rows])
            for metric in ["mae", "rmse", "pearson", "spearman", "ndcg_at_k"]
        }

    comparison = {
        "selection_metric": "mean_spearman",
        "tie_breaker": "mean_ndcg_at_k",
        "selected_architecture": None,
    }
    candidate_rows = []
    for name in ("c1", "c2"):
        candidate_rows.append(
            {
                "architecture": name,
                "mean_spearman": summary[name]["spearman"]["mean"],
                "mean_ndcg_at_k": summary[name]["ndcg_at_k"]["mean"],
                "mean_mae": summary[name]["mae"]["mean"],
                "mean_rmse": summary[name]["rmse"]["mean"],
            }
        )
    candidate_rows = sorted(candidate_rows, key=lambda row: (row["mean_spearman"], row["mean_ndcg_at_k"]), reverse=True)
    comparison["selected_architecture"] = candidate_rows[0]["architecture"]
    comparison["c1_vs_c2_rank"] = candidate_rows

    return {
        "folds": fold_rows,
        "summary": summary,
        "selection": comparison,
        "n_splits": int(n_splits),
        "random_state": int(random_state),
    }


def train_selected_condition_c_model(
    train_films: pd.DataFrame,
    film_vectors: pd.DataFrame,
    profile: pd.DataFrame,
    architecture: str,
) -> tuple[Any, pd.DataFrame]:
    metadata_builder = ConventionalMetadataFeatureBuilder().fit(train_films)
    train_matrix = _build_numeric_model_frame(train_films, architecture, metadata_builder, film_vectors, profile)
    feature_columns = [column for column in train_matrix.columns if column not in {"source_id", "preference_weight"}]
    bundle = fit_ridge_model(
        train_matrix,
        target_col="preference_weight",
        feature_columns=feature_columns,
        model_name=f"condition_c_{architecture}_final",
    )
    return bundle, train_matrix


def run_condition_c_checkpoint(raw_path: str | Path) -> dict[str, Any]:
    CONDITION_C_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_combined_ingestion(raw_path)
    books = get_goodreads_book_subset(raw)
    if len(books) == 0:
        raise RuntimeError("No Goodreads books were found in the combined ingestion.")

    semantic_vectors, semantic_report, enriched_books = classify_goodreads_semantic_vectors(
        books,
        cache_path=GOODREADS_SEMANTIC_CACHE_PATH,
    )
    book_profile = build_goodreads_taste_profile(enriched_books, semantic_vectors, dimensions=SHARED_DIMENSIONS)
    book_profile.to_csv(GOODREADS_PROFILE_PATH, index=False)

    profile_lookup = book_profile.set_index("dimension")
    shared_profile = profile_lookup.loc[list(SHARED_DIMENSIONS)].reset_index()
    shared_profile.to_csv(GOODREADS_PROFILE_PATH, index=False)

    group_mapping = {
        "shared_dimensions": list(SHARED_DIMENSIONS),
        "excluded_dimensions": list(EXCLUDED_DIMENSIONS),
        "groups": {group: list(dimensions) for group, dimensions in CROSS_MEDIA_GROUPS.items()},
    }
    GROUP_MAPPING_PATH.write_text(json.dumps(group_mapping, indent=2))

    film_condition_a = pd.read_csv("mvp/data/processed/condition_a_enriched.csv")
    film_vectors = pd.read_csv("mvp/artifacts/semantic_vectors/semantic_vectors.csv")
    split = pd.read_csv("mvp/data/processed/primary_holdout_split.csv")
    merged = split.merge(film_condition_a, on="source_id", how="inner", validate="one_to_one")
    if len(merged) != len(split):
        raise RuntimeError("Frozen split and Condition A data do not align.")

    film_vectors_subset = film_vectors.loc[film_vectors["canonical_id"].astype(str).isin(set(merged["source_id"].astype(str)))].copy()
    if len(film_vectors_subset) != len(merged):
        raise RuntimeError("Film semantic vectors do not align with the frozen film population.")

    train_films = merged.loc[merged["split"] == "train"].copy().reset_index(drop=True)
    test_films = merged.loc[merged["split"] == "test"].copy().reset_index(drop=True)

    cv_results = cross_validate_condition_c(train_films, film_vectors_subset, shared_profile)

    cv_results["data"] = {
        "raw_rows": int(len(raw)),
        "goodreads_books": int(len(books)),
        "goodreads_semantic_total": int(semantic_report["semantic_total"]),
        "goodreads_semantic_classified": int(semantic_report["semantic_classified"]),
        "goodreads_semantic_coverage": float(semantic_report["coverage"]),
        "film_population": int(len(merged)),
        "train_n": int(len(train_films)),
        "test_n": int(len(test_films)),
        "shared_dimensions": list(SHARED_DIMENSIONS),
        "excluded_dimensions": list(EXCLUDED_DIMENSIONS),
    }

    cv_results["profile_summary"] = {
        "dimensions": int(len(shared_profile)),
        "mean_evidence_count": float(shared_profile["evidence_count"].mean()),
        "mean_evidence_confidence": float(shared_profile["evidence_confidence"].mean()),
        "top_positive": shared_profile.sort_values("effective_book_preference", ascending=False)
        .head(5)[["dimension", "effective_book_preference", "evidence_count"]]
        .to_dict(orient="records"),
        "top_negative": shared_profile.sort_values("effective_book_preference", ascending=True)
        .head(5)[["dimension", "effective_book_preference", "evidence_count"]]
        .to_dict(orient="records"),
    }

    CV_RESULTS_PATH.write_text(json.dumps(cv_results, indent=2))

    selected_architecture = cv_results["selection"]["selected_architecture"]
    bundle, train_matrix = train_selected_condition_c_model(train_films, film_vectors_subset, shared_profile, selected_architecture)
    joblib.dump(bundle, MODEL_PATH)

    config = {
        "experiment_name": "condition_c_goodreads_crossmedia",
        "status": "checkpoint_pending_final_evaluation",
        "random_state": 42,
        "selection_metric": "mean_spearman",
        "selection_tie_breaker": "mean_ndcg_at_k",
        "ridgecv_alphas": list(np.logspace(-3, 3, 25)),
        "shared_dimensions": list(SHARED_DIMENSIONS),
        "excluded_dimensions": list(EXCLUDED_DIMENSIONS),
        "groups": {group: list(dimensions) for group, dimensions in CROSS_MEDIA_GROUPS.items()},
        "selected_architecture": selected_architecture,
        "train_feature_count": int(len(bundle.feature_columns)),
        "train_feature_columns": bundle.feature_columns,
        "model_path": str(MODEL_PATH),
        "profile_path": str(GOODREADS_PROFILE_PATH),
        "semantic_cache_path": str(GOODREADS_SEMANTIC_CACHE_PATH),
        "group_mapping_path": str(GROUP_MAPPING_PATH),
        "cv_results_path": str(CV_RESULTS_PATH),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    selection = {
        "selected_architecture": selected_architecture,
        "reason": (
            "Selected by training-only 5-fold stratified CV on the 393 frozen training films, "
            "using mean Spearman as the primary criterion and mean NDCG@10 as the tie-breaker."
        ),
        "cv_result_path": str(CV_RESULTS_PATH),
        "holdout_used_for_selection": False,
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2))

    checkpoint = {
        "semantic_coverage": semantic_report["coverage"],
        "semantic_total": semantic_report["semantic_total"],
        "semantic_classified": semantic_report["semantic_classified"],
        "semantic_failures": semantic_report["failures"],
        "shared_dimensions": list(SHARED_DIMENSIONS),
        "group_mapping_path": str(GROUP_MAPPING_PATH),
        "profile_summary": cv_results["profile_summary"],
        "cv_results_path": str(CV_RESULTS_PATH),
        "selected_architecture": selected_architecture,
        "selection_path": str(SELECTION_PATH),
        "holdout_used_for_selection": False,
        "model_path": str(MODEL_PATH),
        "config_path": str(CONFIG_PATH),
    }
    return checkpoint
