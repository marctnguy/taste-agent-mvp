from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mvp.src.candidates import generate_candidate_pool, load_condition_a_enriched, load_consumption_history, load_watched_override_registry
from mvp.src.config import get_langsmith_config
from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.generative_v4.qualification_chain import qualify_candidates
from mvp.src.mvp_deployment import load_training_population
from mvp.src.prepare import TMDBClient
from mvp.src.retrieval.catalog_retrieval import discover_catalog_for_request, embed_text
from mvp.src.semantics import SemanticVectorStore
from mvp.src.taste_profile import build_taste_profile


REPO_ROOT = Path(__file__).resolve().parents[3]
WATCHLIST_CACHE_DIR = Path("mvp/artifacts/watchlist_personalization/cache")
WATCHLIST_RESOLUTION_CACHE_PATH = WATCHLIST_CACHE_DIR / "full_watchlist_resolved.csv"
_PREPARED_CASE_CACHE: dict[str, dict[str, Any]] = {}
SERVICE_VERSION = "reversible_history_watchlist_v2"
SLATE_SIZE = 5
SPARSE_WATCHLIST_LIMIT = 8


def _langsmith_project() -> str:
    config = get_langsmith_config()
    return config.project or "taste-agent-capstone"


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _stable_json_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame = frame.sort_index(axis=1)
        return {
            "__type__": "dataframe",
            "columns": frame.columns.tolist(),
            "records": [_stable_json_value(record) for record in frame.to_dict(orient="records")],
        }
    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "index": [str(item) for item in value.index.tolist()],
            "values": [_stable_json_value(item) for item in value.tolist()],
        }
    if isinstance(value, dict):
        return {str(key): _stable_json_value(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, set):
        return [_stable_json_value(item) for item in sorted(value, key=lambda item: repr(item))]
    if isinstance(value, np.generic):
        return _stable_json_value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {"__type__": "float", "value": "nan"}
        return {"__type__": "float", "value": repr(value)}
    if isinstance(value, (pd.Timestamp,)):
        return {"__type__": "timestamp", "value": value.isoformat()}
    if _is_missing(value):
        return None
    return value


def _normalize_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    return " ".join(str(value).split()).strip()


def _first_populated(*values: Any) -> Any:
    for value in values:
        if isinstance(value, list):
            cleaned = [item for item in value if not _is_missing(item) and str(item).strip()]
            if cleaned:
                return cleaned
            continue
        if not _is_missing(value) and str(value).strip():
            return value
    return None


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if not _is_missing(item) and str(item).strip()]
    if _is_missing(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.strip("[]")
    parts = [part.strip().strip("'\"") for part in text.split("|")] if "|" in text else [part.strip().strip("'\"") for part in text.split(",")]
    return [part for part in parts if part]


def _normalize_query(value: Any) -> str:
    return _normalize_text(value).lower()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if _is_missing(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.strip("[]")
    parts = [part.strip().strip("'\"") for part in text.split("|")] if "|" in text else [part.strip().strip("'\"") for part in text.split(",")]
    return [part for part in parts if part]


def _json_frame_fingerprint(frame: pd.DataFrame, *, columns: list[str]) -> str:
    if frame is None or frame.empty:
        return "empty"
    subset = frame.copy()
    for column in columns:
        if column not in subset.columns:
            subset[column] = pd.NA
    subset = subset.loc[:, columns].copy()
    for column in subset.columns:
        subset[column] = subset[column].map(_stable_json_value)
    payload = subset.sort_values(columns, kind="stable").to_dict(orient="records")
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _request_fingerprint(
    query_text: str,
    profile: Any | None,
    watchlist: pd.DataFrame | None,
    requested_count: int,
    config: dict[str, Any] | None,
) -> str:
    profile_payload = _stable_json_value(profile)
    watchlist_payload = _stable_json_value(watchlist if isinstance(watchlist, pd.DataFrame) else pd.DataFrame())
    config_payload = config or {}
    payload = {
        "query": _normalize_query(query_text),
        "profile": profile_payload,
        "watchlist": watchlist_payload,
        "requested_count": int(requested_count),
        "config": config_payload,
        "service_version": SERVICE_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _film_document_text(row: pd.Series | dict[str, Any]) -> str:
    payload = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    def _value(name: str) -> str:
        value = payload.get(name)
        if isinstance(value, list):
            return " | ".join(str(item) for item in value if str(item).strip())
        if _is_missing(value):
            return ""
        return str(value)

    lines = [f"Title: {_normalize_text(_value('title'))}"]
    year = _first_populated(payload.get("year"), payload.get("release_year"))
    if not _is_missing(year):
        try:
            lines.append(f"Release year: {int(float(year))}")
        except Exception:
            pass
    genres = _normalize_list(_first_populated(payload.get("genres"), payload.get("tmdb_genres")))
    if genres:
        lines.append(f"Genres: {_normalize_text(' | '.join(genres))}")
    language = _first_populated(payload.get("original_language"), payload.get("tmdb_original_language"))
    if language:
        lines.append(f"Original language: {_normalize_text(language)}")
    countries = _normalize_list(_first_populated(payload.get("production_countries"), payload.get("tmdb_production_countries")))
    if countries:
        lines.append(f"Production countries: {_normalize_text(' | '.join(countries))}")
    overview = _first_populated(payload.get("overview"), payload.get("tmdb_overview"))
    if not _is_missing(overview):
        lines.append(f"Overview: {_normalize_text(overview)}")
    return "\n".join(lines)


def _build_film_documents(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_id", "title", "document_text", "document_hash", "document_version"])
    ordered = frame.copy().reset_index(drop=True)
    if "source_id" not in ordered.columns:
        raise KeyError("Film documents require source_id.")
    rows: list[dict[str, Any]] = []
    for _, row in ordered.sort_values("source_id", kind="stable").iterrows():
        document = _film_document_text(row)
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "title": row.get("title"),
                "document_text": document,
                "document_hash": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                "document_version": "watchlist_film_doc_v1",
            }
        )
    return pd.DataFrame(rows)


def _cosine_scores(candidate_matrix: np.ndarray, reference_matrix: np.ndarray) -> np.ndarray:
    if candidate_matrix.size == 0 or reference_matrix.size == 0:
        return np.zeros((candidate_matrix.shape[0],), dtype=float)
    candidate_norm = candidate_matrix / np.clip(np.linalg.norm(candidate_matrix, axis=1, keepdims=True), 1e-12, None)
    reference_norm = reference_matrix / np.clip(np.linalg.norm(reference_matrix, axis=1, keepdims=True), 1e-12, None)
    return (candidate_norm @ reference_norm.T).max(axis=1)


def _topk_mean_similarities(candidate_matrix: np.ndarray, reference_matrix: np.ndarray, top_k: int = 3) -> np.ndarray:
    if candidate_matrix.size == 0 or reference_matrix.size == 0:
        return np.full((candidate_matrix.shape[0],), np.nan, dtype=float)
    candidate_norm = candidate_matrix / np.clip(np.linalg.norm(candidate_matrix, axis=1, keepdims=True), 1e-12, None)
    reference_norm = reference_matrix / np.clip(np.linalg.norm(reference_matrix, axis=1, keepdims=True), 1e-12, None)
    similarities = candidate_norm @ reference_norm.T
    topk_means = []
    for row in similarities:
        top = np.sort(row)[::-1][:top_k]
        topk_means.append(float(np.mean(top)) if len(top) else np.nan)
    return np.asarray(topk_means, dtype=float)


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("emb_")]

    def _column_key(column: str) -> tuple[int, str]:
        suffix = column.removeprefix("emb_")
        try:
            return int(suffix.split("_", 1)[0]), column
        except Exception:
            return (10**9, column)

    return sorted(columns, key=_column_key)


def _percentile_scores(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.rank(method="average", pct=True)


def _score_request_relevance(frame: pd.DataFrame, request_text: str, *, active: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"request_relevance_method": "no_candidates", "request_embedding": None}
    scored = frame.copy().reset_index(drop=True)
    emb_cols = _embedding_columns(scored)
    if not active or not emb_cols:
        scored["request_relevance"] = 0.0
        scored["request_relevance_rank"] = np.arange(1, len(scored) + 1)
        return scored, {"request_relevance_method": "disabled" if not active else "missing_embeddings", "request_embedding": None}
    query_embedding = np.asarray(embed_text(request_text), dtype=float)
    matrix = scored.loc[:, emb_cols].to_numpy(dtype=float)
    relevance = _cosine_scores(matrix, query_embedding.reshape(1, -1))
    scored["request_relevance"] = relevance
    scored["request_relevance_rank"] = pd.Series(relevance).rank(method="first", ascending=False).astype(int)
    return scored.sort_values(["request_relevance", "title", "source_id"], ascending=[False, True, True], na_position="last").reset_index(drop=True), {
        "request_relevance_method": "cosine",
        "request_embedding": query_embedding,
    }


def _score_direct_taste(frame: pd.DataFrame, historical: dict[str, Any]) -> pd.DataFrame:
    scored = frame.copy().reset_index(drop=True)
    if scored.empty:
        return scored
    emb_cols = historical.get("emb_cols") or _embedding_columns(scored)
    positive = historical.get("positive", pd.DataFrame())
    negative = historical.get("negative", pd.DataFrame())
    if not emb_cols or positive is None or positive.empty:
        scored["direct_taste_score"] = 0.0
        scored["direct_taste_signal_status"] = "insufficient_personalization"
        scored["positive_neighbors"] = [[] for _ in range(len(scored))]
        scored["negative_neighbors"] = [[] for _ in range(len(scored))]
        return scored
    candidate_matrix = scored.loc[:, emb_cols].to_numpy(dtype=float)
    pos_matrix = positive.loc[:, emb_cols].to_numpy(dtype=float) if not positive.empty else np.empty((0, len(emb_cols)))
    neg_matrix = negative.loc[:, emb_cols].to_numpy(dtype=float) if not negative.empty else np.empty((0, len(emb_cols)))
    pos_scores = _topk_mean_similarities(candidate_matrix, pos_matrix, top_k=3)
    neg_scores = _topk_mean_similarities(candidate_matrix, neg_matrix, top_k=3)
    positive_records = positive.loc[:, [column for column in ["source_id", "title", "year"] if column in positive.columns] + emb_cols].copy()
    negative_records = negative.loc[:, [column for column in ["source_id", "title", "year"] if column in negative.columns] + emb_cols].copy() if not negative.empty else negative.copy()

    def _top_neighbors(candidate_row: np.ndarray, reference_frame: pd.DataFrame) -> list[dict[str, Any]]:
        if reference_frame is None or reference_frame.empty:
            return []
        reference_matrix = reference_frame.loc[:, emb_cols].to_numpy(dtype=float)
        if reference_matrix.size == 0:
            return []
        candidate_norm = candidate_row / np.clip(np.linalg.norm(candidate_row), 1e-12, None)
        reference_norm = reference_matrix / np.clip(np.linalg.norm(reference_matrix, axis=1, keepdims=True), 1e-12, None)
        similarities = reference_norm @ candidate_norm
        top_indices = np.argsort(similarities)[::-1][:3]
        neighbors: list[dict[str, Any]] = []
        for index in top_indices:
            row = reference_frame.iloc[int(index)]
            neighbors.append(
                {
                    "source_id": str(row.get("source_id", "")),
                    "title": row.get("title"),
                    "year": int(float(row["year"])) if "year" in reference_frame.columns and not _is_missing(row.get("year")) else None,
                    "similarity": float(similarities[int(index)]),
                }
            )
        return neighbors

    positive_neighbors = [_top_neighbors(candidate_row, positive_records) for candidate_row in candidate_matrix]
    negative_neighbors = [_top_neighbors(candidate_row, negative_records) for candidate_row in candidate_matrix] if not negative.empty else [[] for _ in range(len(scored))]
    if negative.empty or not np.isfinite(neg_scores).any():
        scored["direct_taste_score"] = np.where(np.isfinite(pos_scores), pos_scores, 0.0)
        scored["direct_taste_signal_status"] = np.where(np.isfinite(pos_scores), "reduced_signal", "insufficient_personalization")
    else:
        scored["direct_taste_score"] = np.where(np.isfinite(pos_scores) & np.isfinite(neg_scores), pos_scores - neg_scores, np.where(np.isfinite(pos_scores), pos_scores, 0.0))
        scored["direct_taste_signal_status"] = np.where(np.isfinite(pos_scores) & np.isfinite(neg_scores), "balanced", np.where(np.isfinite(pos_scores), "reduced_signal", "insufficient_personalization"))
    scored["positive_neighbors"] = positive_neighbors
    scored["negative_neighbors"] = negative_neighbors
    return scored


def _score_novelty(frame: pd.DataFrame, watched_embeddings: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy().reset_index(drop=True)
    if scored.empty:
        return scored
    if watched_embeddings is None or watched_embeddings.empty:
        scored["novelty_score"] = np.nan
        return scored
    candidate_cols = _embedding_columns(scored)
    watched_cols = _embedding_columns(watched_embeddings)
    if not candidate_cols or not watched_cols:
        scored["novelty_score"] = np.nan
        return scored
    candidate_matrix = scored.loc[:, candidate_cols].to_numpy(dtype=float)
    watched_matrix = watched_embeddings.loc[:, watched_cols].to_numpy(dtype=float)
    scores = 1.0 - _cosine_scores(candidate_matrix, watched_matrix)
    scored["novelty_score"] = scores
    return scored


def _prepare_watchlist_frame(watchlist: pd.DataFrame, client: TMDBClient | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if watchlist is None or watchlist.empty:
        return pd.DataFrame(columns=["source_id"]), {"resolved_watchlist_candidates": 0, "unresolved_watchlist_candidates": 0}
    if "source_id" in watchlist.columns and "tmdb_overview" in watchlist.columns:
        frame = watchlist.copy().reset_index(drop=True)
        frame["source_id"] = frame["source_id"].astype(str)
        return frame, {"resolved_watchlist_candidates": int(len(frame)), "unresolved_watchlist_candidates": 0}
    if client is None:
        return pd.DataFrame(columns=["source_id"]), {"resolved_watchlist_candidates": 0, "unresolved_watchlist_candidates": int(len(watchlist))}

    from mvp.src.retrieval.catalog_retrieval import _enrich_movie_candidate

    rows: list[dict[str, Any]] = []
    unresolved = 0
    for _, entry in watchlist.iterrows():
        title = entry.get("title")
        year = entry.get("year")
        try:
            match = client.search_movie(str(title), year)
        except Exception:
            match = None
        if not match:
            unresolved += 1
            continue
        try:
            candidate = _enrich_movie_candidate(client, match)
        except Exception:
            candidate = {}
        if not candidate:
            unresolved += 1
            continue
        candidate["candidate_sources"] = [f"watchlist:{entry.get('watchlist_rank', '')}"]
        candidate["candidate_source_ranks"] = [str(int(entry.get("watchlist_rank", 0) or 0) or 1)]
        candidate["route_memberships"] = ["watchlist"]
        candidate["watchlist_rank"] = int(entry.get("watchlist_rank", 0) or 0) or None
        candidate["watchlist_uri"] = entry.get("Letterboxd URI")
        rows.append(candidate)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["source_id"]), {"resolved_watchlist_candidates": 0, "unresolved_watchlist_candidates": int(unresolved)}
    frame["source_id"] = frame["source_id"].astype(str)
    frame = frame.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    return frame, {"resolved_watchlist_candidates": int(len(frame)), "unresolved_watchlist_candidates": int(unresolved)}


def _coalesce_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame.copy().reset_index(drop=True) for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    combined["source_id"] = combined["source_id"].astype(str)
    grouped_rows: list[dict[str, Any]] = []
    for source_id, group in combined.groupby("source_id", sort=False):
        ranked = group.copy().reset_index(drop=True)
        ranked["_order"] = np.arange(len(ranked))
        ranked["_content_score"] = ranked.apply(
            lambda row: len(
                _normalize_text(
                    " ".join(
                        [
                            "" if _is_missing(_first_populated(row.get("overview"), row.get("tmdb_overview"))) else str(_first_populated(row.get("overview"), row.get("tmdb_overview"))),
                            "" if _is_missing(row.get("document_text")) else str(row.get("document_text")),
                        ]
                    )
                )
            ),
            axis=1,
        )
        ranked = ranked.sort_values(["_content_score", "_order"], ascending=[False, False], na_position="last").reset_index(drop=True)
        row = ranked.iloc[0].drop(labels=["_order", "_content_score"], errors="ignore").to_dict()
        for _, other in ranked.iloc[1:].iterrows():
            other_payload = other.drop(labels=["_order", "_content_score"], errors="ignore").to_dict()
            for column, value in other_payload.items():
                if column in {"candidate_sources", "candidate_source_ranks", "route_memberships"}:
                    merged_values = _as_list(row.get(column))
                    merged_values.extend(_as_list(value))
                    row[column] = _dedupe(merged_values)
                    continue
                if column in {"overview", "tmdb_overview"}:
                    if _is_missing(row.get(column)) and not _is_missing(value):
                        row[column] = value
                    continue
                if column in {"genres", "tmdb_genres", "production_countries", "tmdb_production_countries"}:
                    merged_values = _as_list(row.get(column))
                    merged_values.extend(_as_list(value))
                    row[column] = _dedupe(merged_values)
                    continue
                if _is_missing(row.get(column)) and not _is_missing(value):
                    row[column] = value
        memberships = _as_list(row.get("route_memberships")) or _as_list(row.get("candidate_sources"))
        row["route_memberships"] = _dedupe(memberships or ["request_external"])
        row["route_origin"] = "external" if any(member != "watchlist" for member in row["route_memberships"]) else "watchlist"
        for column in ["candidate_sources", "candidate_source_ranks"]:
            if column in row:
                row[column] = _dedupe(_as_list(row.get(column)))
        if "watchlist_rank" in group.columns:
            ranks = pd.to_numeric(group["watchlist_rank"], errors="coerce").dropna()
            row["watchlist_rank"] = int(ranks.min()) if not ranks.empty else pd.NA
        grouped_rows.append(row)
    merged = pd.DataFrame(grouped_rows)
    merged["source_id"] = merged["source_id"].astype(str)
    return merged.reset_index(drop=True)


def _apply_exclusions(frame: pd.DataFrame, consumed_keys: dict[str, set[str]]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame.copy().reset_index(drop=True)
    filtered["source_id"] = filtered["source_id"].astype(str)
    keep_mask: list[bool] = []
    for _, row in filtered.iterrows():
        source_id = str(row.get("source_id", ""))
        tmdb_id = row.get("tmdb_id")
        title_year = f"{_normalize_query(row.get('title'))}__{'' if pd.isna(row.get('year')) else int(float(row.get('year')))}"
        tmdb_key = None
        if tmdb_id is not None and not pd.isna(tmdb_id):
            try:
                tmdb_key = str(int(float(tmdb_id)))
            except Exception:
                tmdb_key = None
        excluded = source_id in consumed_keys.get("source", set()) or title_year in consumed_keys.get("title_year", set()) or (tmdb_key is not None and tmdb_key in consumed_keys.get("tmdb", set()))
        keep_mask.append(not excluded)
    return filtered.loc[keep_mask].reset_index(drop=True)


def _consumed_title_year(frame: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    if frame is None or frame.empty:
        return keys
    for _, row in frame.iterrows():
        title = _normalize_query(row.get("title"))
        year = row.get("year")
        year_token = "" if year is None or pd.isna(year) else str(int(float(year)))
        keys.add(f"{title}__{year_token}")
    return keys


def _prepare_historical_taste(training_population: pd.DataFrame, train_embeddings: pd.DataFrame) -> dict[str, Any]:
    merged = training_population.merge(train_embeddings, on="source_id", how="inner", validate="one_to_one")
    ratings = pd.to_numeric(merged["rating"], errors="coerce")
    positives = merged.loc[ratings >= 4.0].copy().reset_index(drop=True)
    negatives = merged.loc[ratings <= 2.5].copy().reset_index(drop=True)
    semantic_vectors = SemanticVectorStore("mvp/artifacts/semantic_vectors").load()
    taste_profile = build_taste_profile(merged, semantic_vectors, target_col="preference_weight")
    emb_cols = [column for column in train_embeddings.columns if column.startswith("emb_")]
    return {
        "positive": positives,
        "negative": negatives,
        "emb_cols": emb_cols,
        "positive_count": int(len(positives)),
        "negative_count": int(len(negatives)),
        "taste_profile": taste_profile,
    }


def _build_watched_history_embeddings() -> pd.DataFrame:
    history = load_consumption_history()
    condition_a = load_condition_a_enriched()
    if history.empty or condition_a.empty:
        return pd.DataFrame()
    history = history.copy()
    history["source_id"] = history["source_id"].astype(str)
    merged = history.merge(condition_a, on="source_id", how="inner", validate="one_to_one")
    if merged.empty:
        return pd.DataFrame()
    documents = _build_film_documents(merged)
    from mvp.src.retrieval.catalog_retrieval import embed_documents

    embeddings, _, _ = embed_documents(documents)
    if not embeddings.empty:
        embeddings["source_id"] = embeddings["source_id"].astype(str)
    return embeddings


def _history_seed_bucket(row: pd.Series) -> tuple[str, str]:
    year = row.get("year") if not _is_missing(row.get("year")) else row.get("release_year")
    decade = "unknown"
    if not _is_missing(year):
        try:
            decade = f"{int(float(year)) // 10 * 10}s"
        except Exception:
            decade = "unknown"
    genres = _normalize_list(_first_populated(row.get("genres"), row.get("tmdb_genres")))
    genre = genres[0].lower() if genres else "unknown"
    return decade, genre


def _select_diverse_history_seeds(frame: pd.DataFrame, *, limit: int = 18) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=frame.columns if frame is not None else [])
    ranked = frame.copy().reset_index(drop=True)
    if "rating" in ranked.columns:
        ranked["rating"] = pd.to_numeric(ranked["rating"], errors="coerce")
    if "year" in ranked.columns:
        ranked["year"] = pd.to_numeric(ranked["year"], errors="coerce")
    sort_columns = [column for column in ["rating", "year", "title", "source_id"] if column in ranked.columns]
    if sort_columns:
        ranked = ranked.sort_values(sort_columns, ascending=[False, True, True, True][: len(sort_columns)], na_position="last").reset_index(drop=True)
    selected_indices: list[int] = []
    seen_buckets: set[tuple[str, str]] = set()
    for index, row in ranked.iterrows():
        bucket = _history_seed_bucket(row)
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        selected_indices.append(int(index))
        if len(selected_indices) >= limit:
            break
    if len(selected_indices) < min(limit, len(ranked)):
        for index in ranked.index.tolist():
            if int(index) in selected_indices:
                continue
            selected_indices.append(int(index))
            if len(selected_indices) >= limit:
                break
    return ranked.loc[selected_indices].reset_index(drop=True)


def _build_qualification_pool(frame: pd.DataFrame, route_ids: dict[str, set[str]], *, limit: int = 48) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"qualification_pool_count": 0, "qualification_pool_truncated": False, "qualification_pool_route_counts": {}}
    candidates = frame.copy().reset_index(drop=True)
    route_priorities = [
        ("request", 18),
        ("history", 18),
        ("full_watchlist", 12),
        ("sparse_watchlist", 6),
    ]
    selected_ids: list[str] = []
    route_counts: dict[str, int] = {}
    for route_name, quota in route_priorities:
        ids = route_ids.get(route_name, set())
        if not ids:
            route_counts[route_name] = 0
            continue
        route_frame = candidates.loc[candidates["source_id"].astype(str).isin(ids)].copy().reset_index(drop=True)
        route_frame = route_frame.sort_values(
            [column for column in ["request_relevance", "direct_taste_score", "novelty_score", "source_id"] if column in route_frame.columns],
            ascending=[False, False, False, True][: len([column for column in ["request_relevance", "direct_taste_score", "novelty_score", "source_id"] if column in route_frame.columns])],
            na_position="last",
        )
        picked = 0
        for source_id in route_frame["source_id"].astype(str).tolist():
            if source_id in selected_ids:
                continue
            selected_ids.append(source_id)
            picked += 1
            if len(selected_ids) >= limit or picked >= quota:
                break
        route_counts[route_name] = picked
        if len(selected_ids) >= limit:
            break
    if len(selected_ids) < limit:
        fallback = candidates.sort_values(
            [column for column in ["request_relevance", "direct_taste_score", "novelty_score", "source_id"] if column in candidates.columns],
            ascending=[False, False, False, True][: len([column for column in ["request_relevance", "direct_taste_score", "novelty_score", "source_id"] if column in candidates.columns])],
            na_position="last",
        )
        for source_id in fallback["source_id"].astype(str).tolist():
            if source_id in selected_ids:
                continue
            selected_ids.append(source_id)
            if len(selected_ids) >= limit:
                break
    pool = candidates.loc[candidates["source_id"].astype(str).isin(selected_ids)].copy().reset_index(drop=True)
    return pool, {
        "qualification_pool_count": int(len(pool)),
        "qualification_pool_truncated": bool(len(candidates) > len(pool)),
        "qualification_pool_route_counts": route_counts,
    }


def _scenario_allowed_ids(route_ids: dict[str, set[str]], scenario_label: str, *, variant: str) -> set[str]:
    if scenario_label == "full_watchlist":
        base = set().union(route_ids.get("request", set()), route_ids.get("full_watchlist", set()))
    elif scenario_label == "sparse_watchlist":
        base = set().union(route_ids.get("request", set()), route_ids.get("sparse_watchlist", set()))
    elif scenario_label == "empty_watchlist":
        base = set(route_ids.get("request", set()))
    else:
        raise KeyError(f"Unsupported scenario_label: {scenario_label}")
    if variant == "B":
        base |= set(route_ids.get("history", set()))
    return base


def _qualifed_subset(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.loc[frame["qualification_status"].astype(str).isin({"strong", "partial"})].copy().reset_index(drop=True)


def _qualification_rank(value: str) -> int:
    return {"strong": 2, "partial": 1}.get(str(value), 0)


def _policy_score(row: pd.Series, *, request_mode: str, variant: str, novelty_requested: bool) -> float:
    direct = float(row.get("direct_taste_percentile", row.get("direct_taste_score", 0.0)) or 0.0)
    relevance = float(row.get("request_relevance_percentile", row.get("request_relevance", 0.0)) or 0.0)
    novelty = float(row.get("novelty_percentile", row.get("novelty_score", 0.0)) or 0.0)
    if variant == "A":
        return relevance
    if novelty_requested or request_mode == "novelty":
        return 0.5 * direct + 0.5 * novelty
    if request_mode == "generic":
        return direct
    return 0.5 * relevance + 0.5 * direct


def _attach_selection_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected = frame.copy().reset_index(drop=True)
    selected["qualification_rank"] = selected["qualification_status"].map(_qualification_rank)
    if "request_relevance" in selected.columns:
        selected["request_relevance_percentile"] = _percentile_scores(selected, "request_relevance")
    if "direct_taste_score" in selected.columns:
        selected["direct_taste_percentile"] = _percentile_scores(selected, "direct_taste_score")
    if "novelty_score" in selected.columns:
        selected["novelty_percentile"] = _percentile_scores(selected, "novelty_score")
    return selected


def _catalog_priority(row: pd.Series) -> tuple[int, int, str, str]:
    route_priority = {"request_external": 0, "history_external": 1, "watchlist": 2}
    memberships = row.get("route_memberships", [])
    memberships_list = _as_list(memberships) if not isinstance(memberships, list) else memberships
    membership_rank = min(route_priority.get(str(item), 99) for item in memberships_list) if memberships_list else 99
    source_rank_values = _as_list(row.get("candidate_source_ranks"))
    source_rank = 9999
    for value in source_rank_values:
        try:
            source_rank = min(source_rank, int(float(value)))
        except Exception:
            continue
    if pd.notna(row.get("watchlist_rank")):
        try:
            source_rank = min(source_rank, int(float(row.get("watchlist_rank"))))
        except Exception:
            pass
    return membership_rank, source_rank, _normalize_query(row.get("title")), str(row.get("source_id"))


def _select_view(frame: pd.DataFrame, *, request_mode: str, variant: str, requested_count: int, novelty_requested: bool) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected = frame.copy().reset_index(drop=True)
    selected["qualification_rank"] = selected["qualification_status"].map(_qualification_rank)
    if "request_relevance_percentile" not in selected.columns:
        selected["request_relevance_percentile"] = _percentile_scores(selected, "request_relevance") if "request_relevance" in selected.columns else 0.0
    if "direct_taste_percentile" not in selected.columns:
        selected["direct_taste_percentile"] = _percentile_scores(selected, "direct_taste_score") if "direct_taste_score" in selected.columns else 0.0
    if "novelty_percentile" not in selected.columns:
        selected["novelty_percentile"] = _percentile_scores(selected, "novelty_score") if "novelty_score" in selected.columns else 0.0
    selected["selection_score"] = selected.apply(lambda row: _policy_score(row, request_mode=request_mode, variant=variant, novelty_requested=novelty_requested), axis=1)
    selected["catalog_priority"] = selected.apply(lambda row: _catalog_priority(row), axis=1)
    if variant == "A" and request_mode == "generic":
        selected = selected.sort_values(
            ["qualification_rank", "catalog_priority", "title", "source_id"],
            ascending=[False, True, True, True],
            na_position="last",
        )
    else:
        selected = selected.sort_values(
            ["qualification_rank", "selection_score", "catalog_priority", "title", "source_id"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )
    return selected.head(requested_count).reset_index(drop=True)


def _build_recommendation_row(row: pd.Series, *, variant: str, scenario_label: str) -> dict[str, Any]:
    return {
        "candidate_id": str(row.get("candidate_id") or row.get("source_id") or ""),
        "title": row.get("title"),
        "year": int(float(row.get("year"))) if pd.notna(row.get("year")) else None,
        "tmdb_id": int(float(row.get("tmdb_id"))) if pd.notna(row.get("tmdb_id")) else None,
        "genres": _as_list(row.get("genres") or row.get("tmdb_genres")),
        "original_language": row.get("original_language") or row.get("tmdb_original_language"),
        "production_countries": _as_list(row.get("production_countries") or row.get("tmdb_production_countries")),
        "predicted_preference": row.get("selection_score"),
        "request_relevance": row.get("request_relevance"),
        "direct_taste_score": row.get("direct_taste_score"),
        "direct_taste_signal_status": row.get("direct_taste_signal_status"),
        "positive_neighbors": row.get("positive_neighbors", []),
        "negative_neighbors": row.get("negative_neighbors", []),
        "novelty_score": row.get("novelty_score"),
        "qualification_status": row.get("qualification_status"),
        "request_match": row.get("request_match"),
        "caveat": row.get("caveat"),
        "qualification_reason": row.get("qualification_reason"),
        "taste_signals": row.get("supported_request_aspects", []),
        "supported_required_aspects": row.get("supported_required_aspects", []),
        "supported_preferred_aspects": row.get("supported_preferred_aspects", []),
        "unsupported_required_aspects": row.get("unsupported_required_aspects", []),
        "unsupported_preferred_aspects": row.get("unsupported_preferred_aspects", []),
        "violated_semantic_exclusions": row.get("violated_semantic_exclusions", []),
        "grounded_evidence": row.get("grounded_evidence", []),
        "grounded_evidence_details": row.get("grounded_evidence_details", []),
        "route_memberships": row.get("route_memberships", []),
        "route_origin": row.get("route_origin"),
        "candidate_sources": row.get("candidate_sources", []),
        "candidate_source_ranks": row.get("candidate_source_ranks", []),
        "scenario_label": scenario_label,
        "variant": variant,
    }


@dataclass(frozen=True)
class ReversibleHistoryWatchlistService:
    prompts: pd.DataFrame
    inputs: dict[str, Any]
    condition_a: pd.DataFrame
    split: pd.DataFrame
    train_embeddings: pd.DataFrame
    watchlist: pd.DataFrame
    consumed_keys: dict[str, set[str]]
    tmdb_client: Any
    historical: dict[str, Any]
    watched_history_embeddings: pd.DataFrame
    full_watchlist_resolved: pd.DataFrame
    sparse_watchlist_resolved: pd.DataFrame
    empty_watchlist_resolved: pd.DataFrame
    preflight: dict[str, Any]

    @staticmethod
    def _load_cached_watchlist_resolution() -> pd.DataFrame | None:
        if not WATCHLIST_RESOLUTION_CACHE_PATH.exists():
            return None
        try:
            frame = pd.read_csv(WATCHLIST_RESOLUTION_CACHE_PATH)
        except Exception:
            return None
        if "source_id" in frame.columns:
            frame["source_id"] = frame["source_id"].astype(str)
        return frame

    @classmethod
    def from_environment(cls) -> "ReversibleHistoryWatchlistService":
        from mvp.src.config import load_runtime_env

        load_runtime_env()
        prompts = cls._load_prompts()
        inputs = cls._load_inputs()
        condition_a = inputs["condition_a"]
        split = inputs["split"]
        train_embeddings = inputs["train_embeddings"]
        watchlist = inputs["watchlist"]
        consumed_keys = cls._build_consumed_keys(condition_a)
        training_population = load_training_population()
        if len(training_population) != 393:
            raise RuntimeError("Frozen training population changed.")
        if len(split.loc[split["split"] == "train"]) != 393 or len(split.loc[split["split"] == "test"]) != 99:
            raise RuntimeError("Frozen 393/99 split changed.")
        if set(training_population["source_id"].astype(str)) != set(train_embeddings["source_id"].astype(str)):
            raise RuntimeError("Training embeddings are not aligned with the frozen training split.")

        tmdb_client = cls._configure_tmdb_client()
        historical = _prepare_historical_taste(training_population, train_embeddings)
        watched_history_embeddings = _build_watched_history_embeddings()
        cached_watchlist = cls._load_cached_watchlist_resolution()
        if cached_watchlist is not None and not cached_watchlist.empty:
            full_watchlist_resolved = cached_watchlist
        else:
            full_watchlist_resolved, _ = cls._prepare_watchlist_resolution(watchlist, tmdb_client, consumed_keys)
            WATCHLIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            full_watchlist_resolved.to_csv(WATCHLIST_RESOLUTION_CACHE_PATH, index=False)
        sparse_watchlist_resolved = full_watchlist_resolved.head(SPARSE_WATCHLIST_LIMIT).copy().reset_index(drop=True)
        empty_watchlist_resolved = pd.DataFrame(columns=full_watchlist_resolved.columns if not full_watchlist_resolved.empty else ["source_id"])

        return cls(
            prompts=prompts,
            inputs=inputs,
            condition_a=condition_a,
            split=split,
            train_embeddings=train_embeddings,
            watchlist=watchlist,
            consumed_keys=consumed_keys,
            tmdb_client=tmdb_client,
            historical=historical,
            watched_history_embeddings=watched_history_embeddings,
            full_watchlist_resolved=full_watchlist_resolved,
            sparse_watchlist_resolved=sparse_watchlist_resolved,
            empty_watchlist_resolved=empty_watchlist_resolved,
            preflight={"services": {}},
        )

    @staticmethod
    def _load_prompts() -> pd.DataFrame:
        manifest_path = REPO_ROOT / "evaluation/hitl/runtime_v4_final/05_run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        source_path = REPO_ROOT / manifest["source_prompt_path"]
        frame = pd.read_csv(source_path)
        return frame.loc[frame["hitl_id"].astype(str).str.fullmatch(r"H\d\d")].copy().reset_index(drop=True)

    @staticmethod
    def _load_inputs() -> dict[str, Any]:
        watchlist_zip = REPO_ROOT.parent / "letterboxd-marctguy-2026-09-17-17-00-utc.zip"
        if watchlist_zip.exists():
            import zipfile

            with zipfile.ZipFile(watchlist_zip) as archive:
                with archive.open("watchlist.csv") as handle:
                    watchlist = pd.read_csv(handle)
            watchlist = watchlist.rename(columns={"Date": "source_record_date", "Name": "title", "Year": "year"})
            watchlist["source_record_date"] = pd.to_datetime(watchlist["source_record_date"], errors="coerce")
            watchlist["year"] = pd.to_numeric(watchlist["year"], errors="coerce").astype("Int64")
            watchlist["watchlist_rank"] = np.arange(1, len(watchlist) + 1)
        else:
            watchlist = pd.DataFrame(columns=["source_record_date", "title", "year", "watchlist_rank"])
        return {
            "condition_a": load_condition_a_enriched(REPO_ROOT / "mvp/data/processed/condition_a_enriched.csv"),
            "split": pd.read_csv(REPO_ROOT / "mvp/data/processed/primary_holdout_split.csv"),
            "train_embeddings": pd.read_csv(REPO_ROOT / "mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv"),
            "watchlist": watchlist,
        }

    @staticmethod
    def _build_consumed_keys(condition_a: pd.DataFrame) -> dict[str, set[str]]:
        consumed = load_consumption_history().copy()
        overrides = load_watched_override_registry()
        if not overrides.empty:
            consumed = pd.concat([consumed, overrides], ignore_index=True, sort=False)
        if not condition_a.empty:
            consumed = pd.concat([consumed, condition_a.copy()], ignore_index=True, sort=False)
        consumed["source_id"] = consumed["source_id"].astype(str)
        consumed_tmdb = set(pd.to_numeric(consumed.get("tmdb_id"), errors="coerce").dropna().astype(int).astype(str))
        consumed_source = set(consumed["source_id"].astype(str))
        consumed_title_year = _consumed_title_year(consumed)
        return {"tmdb": consumed_tmdb, "source": consumed_source, "title_year": consumed_title_year}

    @staticmethod
    def _configure_tmdb_client():
        from mvp.src.config import get_api_keys

        keys = get_api_keys()
        if not keys.tmdb_api_key:
            return None
        return TMDBClient(api_key=keys.tmdb_api_key)

    @staticmethod
    def _prepare_watchlist_resolution(watchlist: pd.DataFrame, client, consumed_keys: dict[str, set[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
        resolved, report = _prepare_watchlist_frame(watchlist, client)
        if resolved.empty:
            return resolved, report
        resolved = _apply_exclusions(resolved, consumed_keys)
        resolved = resolved.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
        return resolved, report

    def prompt_row(self, hitl_id: str) -> pd.Series:
        match = self.prompts.loc[self.prompts["hitl_id"].astype(str) == str(hitl_id)]
        if match.empty:
            raise KeyError(f"Unknown hitl_id: {hitl_id}")
        return match.iloc[0]

    def prompt_row_for_query(self, query_text: str) -> pd.Series:
        normalized = _normalize_query(query_text)
        if not normalized:
            raise KeyError("Empty query_text.")
        prompt_matches = self.prompts.loc[
            self.prompts["prompt"].astype(str).map(lambda value: _normalize_query(value)) == normalized
        ]
        if prompt_matches.empty:
            raise KeyError(f"Unknown query_text: {query_text}")
        return prompt_matches.iloc[0]

    def _historical_profile(self, profile: Any | None) -> dict[str, Any]:
        if profile is None:
            return self.historical
        if isinstance(profile, dict):
            if {"positive", "negative", "emb_cols"} <= set(profile):
                return profile
            if "taste_profile" in profile:
                return {**self.historical, **profile}
        if isinstance(profile, pd.DataFrame):
            if "source_id" in profile.columns and "rating" in profile.columns:
                merged = profile.copy()
                merged["source_id"] = merged["source_id"].astype(str)
                train = merged.merge(self.train_embeddings, on="source_id", how="inner", validate="one_to_one")
                ratings = pd.to_numeric(train["rating"], errors="coerce")
                return {
                    "positive": train.loc[ratings >= 4.0].copy().reset_index(drop=True),
                    "negative": train.loc[ratings <= 2.5].copy().reset_index(drop=True),
                    "emb_cols": [column for column in self.train_embeddings.columns if column.startswith("emb_")],
                    "positive_count": int((ratings >= 4.0).sum()),
                    "negative_count": int((ratings <= 2.5).sum()),
                    "taste_profile": self.historical.get("taste_profile"),
                }
        return self.historical

    def _resolve_watchlist(self, watchlist: pd.DataFrame | None) -> pd.DataFrame:
        if watchlist is None:
            return self.full_watchlist_resolved
        if "source_id" in watchlist.columns and "tmdb_overview" in watchlist.columns:
            frame = watchlist.copy().reset_index(drop=True)
            frame["source_id"] = frame["source_id"].astype(str)
            return frame
        resolved, _ = self._prepare_watchlist_resolution(watchlist, self.tmdb_client, self.consumed_keys)
        return resolved

    def _build_candidate_pool(self, request, *, profile: Any | None, watchlist: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any], dict[str, set[str]]]:
        active_profile = self._historical_profile(profile)
        active_watchlist = self._resolve_watchlist(watchlist)
        request_text = request.spec.semantic_query_text if request.spec and request.spec.semantic_query_text else request.query
        positive = active_profile.get("positive", pd.DataFrame())
        history_seed_ids: list[str] = []
        history_pool = pd.DataFrame(columns=positive.columns if isinstance(positive, pd.DataFrame) else [])
        if isinstance(positive, pd.DataFrame) and not positive.empty:
            seed_frame = positive.copy().reset_index(drop=True)
            if "rating" in seed_frame.columns:
                seed_frame["rating"] = pd.to_numeric(seed_frame["rating"], errors="coerce")
            history_pool = _select_diverse_history_seeds(seed_frame, limit=min(24, len(seed_frame)))
            history_seed_ids = history_pool["source_id"].astype(str).tolist() if not history_pool.empty else []

        request_pool = pd.DataFrame()
        try:
            request_pool, request_pool_report = generate_candidate_pool(candidate_limit=300, include_seeded_recommendations=False)
        except Exception:
            request_pool = pd.DataFrame()
            request_pool_report = {"candidate_limit": 300, "before_deduplication": 0, "after_deduplication": 0, "before_exclusion": 0, "removed_watched": 0, "after_exclusion": 0, "strategy_counts": {}, "exclusion_report": {}}

        watched_keys = set().union(
            self.consumed_keys.get("source", set()),
            self.consumed_keys.get("tmdb", set()),
            self.consumed_keys.get("title_year", set()),
        )
        request_result = discover_catalog_for_request(request, client=self.tmdb_client, candidate_pool=request_pool, watched_ids=watched_keys)
        history_result = discover_catalog_for_request(request, client=self.tmdb_client, candidate_pool=history_pool, watched_ids=watched_keys)

        request_frame = request_result.catalog_frame.copy()
        history_frame = history_result.catalog_frame.copy()
        watchlist_frame, watchlist_report = _prepare_watchlist_frame(active_watchlist, self.tmdb_client)
        if not watchlist_frame.empty and "route_memberships" not in watchlist_frame.columns:
            watchlist_frame["route_memberships"] = [["watchlist"] for _ in range(len(watchlist_frame))]
        full_watchlist_frame = watchlist_frame.copy()
        sparse_watchlist_frame = watchlist_frame.head(SPARSE_WATCHLIST_LIMIT).copy().reset_index(drop=True)
        empty_watchlist_frame = pd.DataFrame(columns=watchlist_frame.columns if not watchlist_frame.empty else ["source_id"])

        request_frame["route_memberships"] = [["request_external"] for _ in range(len(request_frame))]
        history_frame["route_memberships"] = [["history_external"] for _ in range(len(history_frame))]
        if not request_frame.empty:
            request_frame["route_origin"] = "external"
        if not history_frame.empty:
            history_frame["route_origin"] = "external"
        if not full_watchlist_frame.empty:
            full_watchlist_frame["route_origin"] = "watchlist"
        if not sparse_watchlist_frame.empty:
            sparse_watchlist_frame["route_origin"] = "watchlist"
        if not empty_watchlist_frame.empty:
            empty_watchlist_frame["route_origin"] = "watchlist"

        master = _coalesce_frames(request_frame, history_frame, full_watchlist_frame, sparse_watchlist_frame)
        master = _apply_exclusions(master, self.consumed_keys)
        if master.empty:
            raise RuntimeError("No candidates could be constructed.")
        existing_embedding_columns = _embedding_columns(master)
        if existing_embedding_columns:
            master = master.drop(columns=existing_embedding_columns)

        documents = _build_film_documents(master)
        from mvp.src.retrieval.catalog_retrieval import embed_documents

        embeddings, _, embedding_report = embed_documents(documents)
        embeddings["source_id"] = embeddings["source_id"].astype(str)
        embeddings = embeddings.loc[:, [column for column in embeddings.columns if column.startswith("emb_") or column in {"source_id", "document_hash", "embedding_model", "embedding_dim", "status"}]].copy()
        master = master.merge(embeddings, on="source_id", how="left")
        if "candidate_id" not in master.columns:
            master["candidate_id"] = master["source_id"].astype(str)

        active_request_text = request_text if request.request_mode != "generic" else ""
        master, relevance_report = _score_request_relevance(master, active_request_text, active=request.request_mode != "generic" or bool(request.spec and request.spec.semantic_query_text))
        master = _score_direct_taste(master, active_profile)
        master = _score_novelty(master, self.watched_history_embeddings)
        route_ids = {
            "request": set(request_frame["source_id"].astype(str).tolist()) if not request_frame.empty else set(),
            "history": set(history_frame["source_id"].astype(str).tolist()) if not history_frame.empty else set(),
            "full_watchlist": set(full_watchlist_frame["source_id"].astype(str).tolist()) if not full_watchlist_frame.empty else set(),
            "sparse_watchlist": set(sparse_watchlist_frame["source_id"].astype(str).tolist()) if not sparse_watchlist_frame.empty else set(),
            "empty_watchlist": set(),
        }
        route_reports = {
            "request_pool_report": request_pool_report,
            "request_external": request_result.augmentation_report,
            "history_external": history_result.augmentation_report,
            "history_seed_ids": history_seed_ids,
            "history_seed_count": int(len(history_seed_ids)),
            "watchlist_full": watchlist_report,
            "watchlist_sparse": {"resolved_watchlist_candidates": int(len(sparse_watchlist_frame)), "unresolved_watchlist_candidates": int(len(active_watchlist) - len(sparse_watchlist_frame))},
            "watchlist_empty": {"resolved_watchlist_candidates": 0, "unresolved_watchlist_candidates": 0},
            "embedding_report": embedding_report,
            "request_relevance_report": relevance_report,
        }
        return master, route_reports, route_ids

    def prepare_case(
        self,
        query_text: str,
        *,
        scenario_label: str = "full_watchlist",
        variant: str = "A",
        requested_count: int = SLATE_SIZE,
        profile: Any | None = None,
        watchlist: pd.DataFrame | None = None,
        config: dict[str, Any] | None = None,
        hitl_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_profile = self._historical_profile(profile)
        resolved_watchlist = self._resolve_watchlist(watchlist)
        cache_config = dict(config or {})
        cache_config["watched_state"] = {
            "source_ids": self._consumed_source_ids(),
            "tmdb_ids": self._consumed_tmdb_ids(),
            "title_years": sorted(self.consumed_keys.get("title_year", set())),
        }
        fingerprint = _request_fingerprint(query_text, resolved_profile, resolved_watchlist, requested_count, cache_config)
        cached = _PREPARED_CASE_CACHE.get(fingerprint)
        if cached is not None:
            return cached

        request = understand_request(query_text, client=self.tmdb_client)
        master, route_reports, route_ids = self._build_candidate_pool(request, profile=resolved_profile, watchlist=resolved_watchlist)
        qualification_pool, qualification_pool_report = _build_qualification_pool(master, route_ids, limit=48)
        qualified_master, qualification_records = qualify_candidates(request, qualification_pool, max_candidates=48)
        qualification_usage = qualified_master.attrs.get("llm_usage", {}) if hasattr(qualified_master, "attrs") else {}
        qualified_master = qualified_master.copy().reset_index(drop=True)
        if "candidate_id" not in qualified_master.columns and "source_id" in qualified_master.columns:
            qualified_master["candidate_id"] = qualified_master["source_id"].astype(str)

        scenario_views: dict[str, Any] = {}
        for current_scenario in ("full_watchlist", "sparse_watchlist", "empty_watchlist"):
            allowed_ids_a = _scenario_allowed_ids(route_ids, current_scenario, variant="A")
            allowed_ids_b = _scenario_allowed_ids(route_ids, current_scenario, variant="B")
            universe_ids = _dedupe(sorted(set(allowed_ids_a).union(allowed_ids_b)))
            scenario_candidates = master.loc[master["source_id"].astype(str).isin(universe_ids)].copy().reset_index(drop=True)
            scenario_qualified_all = qualified_master.loc[qualified_master["source_id"].astype(str).isin(universe_ids)].copy().reset_index(drop=True)
            scenario_qualified_all = _attach_selection_percentiles(scenario_qualified_all)
            scenario_qualified_a = scenario_qualified_all.loc[scenario_qualified_all["source_id"].astype(str).isin(allowed_ids_a)].copy().reset_index(drop=True)
            scenario_qualified_b = scenario_qualified_all.loc[scenario_qualified_all["source_id"].astype(str).isin(allowed_ids_b)].copy().reset_index(drop=True)
            selected_a = _select_view(
                scenario_qualified_a,
                request_mode=request.request_mode,
                variant="A",
                requested_count=requested_count,
                novelty_requested=request.novelty_requested,
            )
            selected_b = _select_view(
                scenario_qualified_b,
                request_mode=request.request_mode,
                variant="B",
                requested_count=requested_count,
                novelty_requested=request.novelty_requested,
            )
            scenario_views[current_scenario] = {
                "candidate_ids": scenario_candidates["source_id"].astype(str).tolist(),
                "candidate_count": int(len(scenario_candidates)),
                "qualification_pool_count": int(len(qualification_pool)),
                "qualification_pool_truncated": bool(qualification_pool_report.get("qualification_pool_truncated", False)),
                "qualification_pool_route_counts": qualification_pool_report.get("qualification_pool_route_counts", {}),
                "A": {
                    "eligible_count": int(len(scenario_qualified_a)),
                    "selected": selected_a,
                    "selected_ids": selected_a["source_id"].astype(str).tolist() if not selected_a.empty else [],
                    "selected_titles": selected_a["title"].astype(str).tolist() if not selected_a.empty else [],
                },
                "B": {
                    "eligible_count": int(len(scenario_qualified_b)),
                    "selected": selected_b,
                    "selected_ids": selected_b["source_id"].astype(str).tolist() if not selected_b.empty else [],
                    "selected_titles": selected_b["title"].astype(str).tolist() if not selected_b.empty else [],
                },
            }

        prepared = {
            "case_id": hitl_id or fingerprint[:12],
            "hitl_id": hitl_id or None,
            "query": query_text,
            "prompt": query_text,
            "scenario_label": scenario_label,
            "variant": variant,
            "requested_count": int(requested_count),
            "request": request,
            "parsed_request": request.model_dump(),
            "controlled_request_spec": request.spec.model_dump() if request.spec else {},
            "route_reports": route_reports,
            "route_ids": route_ids,
            "qualification_pool": qualification_pool,
            "qualification_pool_report": qualification_pool_report,
            "candidate_union": master,
            "qualified_candidates": qualified_master,
            "qualification_records": qualification_records,
            "llm_usage": qualification_usage,
            "scenario_views": scenario_views,
            "fingerprint": fingerprint,
        }
        _PREPARED_CASE_CACHE[fingerprint] = prepared
        return prepared

    def recommend(
        self,
        query_text: str,
        *,
        scenario_label: str = "full_watchlist",
        variant: str = "A",
        requested_count: int = SLATE_SIZE,
        profile: Any | None = None,
        watchlist: pd.DataFrame | None = None,
        config: dict[str, Any] | None = None,
        hitl_id: str | None = None,
    ) -> dict[str, Any]:
        prepared = self.prepare_case(
            query_text,
            scenario_label=scenario_label,
            variant=variant,
            requested_count=requested_count,
            profile=profile,
            watchlist=watchlist,
            config=config,
            hitl_id=hitl_id,
        )
        scenario = prepared["scenario_views"][scenario_label]
        selected = scenario[variant]["selected"]
        selected_records = [_json_ready(row) for row in selected.to_dict(orient="records")]
        selection_report = {
            "variant": variant,
            "scenario_label": scenario_label,
            "candidate_count": int(scenario.get("candidate_count", 0)),
            "eligible_count": int(scenario.get(variant, {}).get("eligible_count", 0)),
            "selected_count": int(len(selected_records)),
            "requested_count": int(requested_count),
            "slate_size": int(requested_count),
        }
        response = {
            "intent": _json_ready(prepared["parsed_request"].get("intent", {})),
            "recommendations": [
                {
                    **{
                        "candidate_id": str(record.get("candidate_id") or record.get("source_id") or ""),
                        "title": record.get("title"),
                        "year": record.get("year"),
                        "tmdb_id": record.get("tmdb_id"),
                        "genres": record.get("genres", []),
                        "original_language": record.get("original_language"),
                        "production_countries": record.get("production_countries", []),
                        "predicted_preference": record.get("selection_score", record.get("predicted_preference")),
                        "request_relevance": record.get("request_relevance"),
                        "direct_taste_score": record.get("direct_taste_score"),
                        "novelty_score": record.get("novelty_score"),
                        "qualification_status": record.get("qualification_status"),
                        "request_match": record.get("request_match"),
                        "caveat": record.get("caveat"),
                        "qualification_reason": record.get("qualification_reason"),
                        "taste_signals": record.get("taste_signals", []),
                        "supported_required_aspects": record.get("supported_required_aspects", []),
                        "supported_preferred_aspects": record.get("supported_preferred_aspects", []),
                        "unsupported_required_aspects": record.get("unsupported_required_aspects", []),
                        "unsupported_preferred_aspects": record.get("unsupported_preferred_aspects", []),
                        "violated_semantic_exclusions": record.get("violated_semantic_exclusions", []),
                        "grounded_evidence": record.get("grounded_evidence", []),
                        "grounded_evidence_details": record.get("grounded_evidence_details", []),
                        "route_memberships": record.get("route_memberships", []),
                        "route_origin": record.get("route_origin"),
                        "candidate_sources": record.get("candidate_sources", []),
                        "candidate_source_ranks": record.get("candidate_source_ranks", []),
                    }
                }
                for record in selected_records
            ],
            "response_summary": f"{prepared['case_id']} {scenario_label} {variant} slate with {len(selected_records)} selections.",
            "methodology_note": f"Prepared reversible-history/watchlist selection for {prepared['case_id']} using {variant} policy under {scenario_label}.",
        }
        selected_count = int(selection_report["selected_count"])
        eligible_count = int(selection_report["eligible_count"])
        requested = int(selection_report["requested_count"])
        validation_passed = bool(selected_count > 0 and selected_count == min(requested, eligible_count))
        payload = {
            "case_id": prepared["case_id"],
            "hitl_id": prepared.get("hitl_id"),
            "prompt": prepared["prompt"],
            "scenario_label": scenario_label,
            "variant": variant,
            "requested_count": requested,
            "response": response,
            "runtime_metadata": {
                "runtime_version": SERVICE_VERSION,
                "intent_prompt_version": "intent_v4",
                "recommendation_prompt_version": "recommendation_v4",
                "scenario_label": scenario_label,
                "variant": variant,
                "requested_count": requested,
                "candidate_universe_count": int(len(prepared["candidate_union"])),
                "post_constraint_candidate_count": eligible_count,
                "contextual_retrieval_used": True,
                "qualification_candidate_count": eligible_count,
                "qualified_candidate_count": eligible_count,
                "selected_count": selected_count,
                "semantic_classified_count": eligible_count,
                "llm_call_count": int(prepared.get("llm_usage", {}).get("llm_call_count", 0) or 0),
                "runtime_llm_calls": prepared.get("llm_usage", {}).get("llm_call_count"),
                "runtime_input_tokens": prepared.get("llm_usage", {}).get("runtime_input_tokens"),
                "runtime_output_tokens": prepared.get("llm_usage", {}).get("runtime_output_tokens"),
                "runtime_total_tokens": prepared.get("llm_usage", {}).get("runtime_total_tokens"),
                "generation_status": "generated" if selected_count else "no_match",
                "validation_passed": validation_passed,
                "repair_attempted": False,
                "fallback_used": int(prepared.get("llm_usage", {}).get("llm_call_count", 0) or 0) == 0,
                "hard_filters_applied": True,
                "recommendation_count": selected_count,
                "source_watchlist_count": int(len(self._scenario_resolved_watchlist(scenario_label))),
            },
            "selection_report": selection_report,
            "debug": {
                "prompt": prepared["prompt"],
                "parsed_request": _json_ready(prepared.get("parsed_request", {})),
                "controlled_request_spec": _json_ready(prepared.get("controlled_request_spec", {})),
                "request_corrections": [],
                "route_reports": _json_ready(prepared.get("route_reports", {})),
                "candidate_context": _json_ready(prepared["qualified_candidates"].to_dict(orient="records")),
                "selected_candidates": selected_records,
                "consumed_source_ids": self._consumed_source_ids(),
                "consumed_tmdb_ids": self._consumed_tmdb_ids(),
                "reference_title": _json_ready(prepared.get("parsed_request", {}).get("reference_title")),
                "prepared_fingerprint": prepared["fingerprint"],
            },
            "validation_passed": validation_passed,
            "validation_errors": [] if validation_passed else ["requested slate was not fully satisfied"],
        }
        return payload

    def build_dataset_examples(self) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        for _, row in self.prompts.iterrows():
            case_id = str(row["hitl_id"])
            examples.append(
                {
                    "case_id": case_id,
                    "inputs": {
                        "case_id": case_id,
                        "hitl_id": case_id,
                        "prompt": row["prompt"],
                        "query": row["prompt"],
                    },
                    "outputs": {
                        "test_purpose": row.get("test_purpose"),
                        "prompt": row["prompt"],
                        "taste_fit_1_5": row.get("taste_fit_1_5"),
                        "request_fit_1_5": row.get("request_fit_1_5"),
                        "discovery_value_1_5": row.get("discovery_value_1_5"),
                        "explanation_usefulness_1_5": row.get("explanation_usefulness_1_5"),
                        "would_actually_watch": row.get("would_actually_watch"),
                        "already_knew_titles": row.get("already_knew_titles"),
                        "heard_of_titles": row.get("heard_of_titles"),
                        "new_to_me_titles": row.get("new_to_me_titles"),
                        "best_recommendation": row.get("best_recommendation"),
                        "worst_recommendation": row.get("worst_recommendation"),
                        "qualitative_feedback": row.get("qualitative_feedback"),
                    },
                    "metadata": {
                        "case_id": case_id,
                        "hitl_id": case_id,
                        "test_purpose": row.get("test_purpose"),
                    },
                }
            )
        return examples

    def _scenario_resolved_watchlist(self, scenario_label: str) -> pd.DataFrame:
        if scenario_label == "full_watchlist":
            return self.full_watchlist_resolved
        if scenario_label == "empty_watchlist":
            return self.empty_watchlist_resolved
        if scenario_label == "sparse_watchlist":
            return self.sparse_watchlist_resolved
        raise KeyError(f"Unsupported scenario_label: {scenario_label}")

    def _consumed_source_ids(self) -> list[str]:
        source_ids = set()
        for key in ("source", "source_ids"):
            source_ids.update(str(item) for item in self.consumed_keys.get(key, set()))
        return sorted(item for item in source_ids if item.strip())

    def _consumed_tmdb_ids(self) -> list[str]:
        tmdb_ids = set()
        for key in ("tmdb", "tmdb_ids"):
            tmdb_ids.update(str(item) for item in self.consumed_keys.get(key, set()))
        return sorted(item for item in tmdb_ids if item.strip())

    def run_case(self, hitl_id: str, *, scenario_label: str = "full_watchlist", variant: str = "A") -> dict[str, Any]:
        prompt_row = self.prompt_row(hitl_id)
        return self.recommend(
            str(prompt_row["prompt"]),
            scenario_label=scenario_label,
            variant=variant,
            hitl_id=hitl_id,
        )

    def run_full_comparison(self) -> dict[str, Any]:
        if self.prompts.empty:
            raise RuntimeError("No prompts available for comparison.")
        first = self.prompts.iloc[0]
        return self.recommend(str(first["prompt"]), hitl_id=str(first["hitl_id"]))


@lru_cache(maxsize=1)
def load_reversible_history_watchlist_service() -> ReversibleHistoryWatchlistService:
    return ReversibleHistoryWatchlistService.from_environment()
