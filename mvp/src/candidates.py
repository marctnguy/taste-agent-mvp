from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from mvp.src.config import get_api_keys
from mvp.src.data import load_table, normalize_column_names
from mvp.src.mvp_deployment import (
    MODEL_DIR,
    build_film_documents,
    build_taste_evidence,
    load_model_bundle,
    load_or_generate_candidate_embeddings,
    score_candidates,
)
from mvp.src.prepare import TMDBClient, classify_semantic_vectors
from mvp.src.semantics import SemanticVectorStore
from mvp.src.taste_profile import build_taste_profile


RAW_INGESTION_PATH = Path("mvp/data/raw/taste-agent-combined-ingestion.csv")
CONDITION_A_PATH = Path("mvp/data/processed/condition_a_enriched.csv")
SEMANTIC_CACHE_PATH = Path("mvp/artifacts/semantic_vectors/semantic_vectors.csv")
RECOMMENDATION_RUNS_DIR = Path("mvp/artifacts/recommendation_runs")
RECOMMENDATION_CACHE_DIR = MODEL_DIR / "candidate_embedding_cache"
RECOMMENDATION_CACHE_PATH = RECOMMENDATION_CACHE_DIR / "candidate_embeddings.csv"
TMDB_API_BASE = "https://api.themoviedb.org/3"

DEFAULT_CANDIDATE_LIMIT = 300
DEFAULT_TOP_K = 20
POPULAR_PAGES = 3
TOP_RATED_PAGES = 3
RECENT_PAGES = 2
GENRE_PAGES = 1
SEED_PAGES = 1
RECENT_YEARS_BACK = 4

GENRE_DISCOVERY_PLAN: tuple[tuple[int, str], ...] = (
    (28, "action"),
    (12, "adventure"),
    (16, "animation"),
    (35, "comedy"),
    (80, "crime"),
    (99, "documentary"),
    (18, "drama"),
    (27, "horror"),
    (9648, "mystery"),
    (10749, "romance"),
    (878, "science fiction"),
    (53, "thriller"),
)


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
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
    return " ".join(str(value).strip().lower().split())


def _normalize_title(value: Any) -> str:
    return re.sub(r"\s+", " ", _normalize_text(value))


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [part.strip() for part in re.split(r"[|,/;]+", text) if part.strip()]
    return [str(value)]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    return str(value)


def validate_semantic_taxonomy_alignment(frame: pd.DataFrame, *, label: str = "semantic frame") -> None:
    missing = [dimension for dimension in SEMANTIC_COLUMNS if dimension not in frame.columns]
    extra = [
        column
        for column in frame.columns
        if column not in SEMANTIC_COLUMNS and column not in {"canonical_id", "source_id"}
    ]
    if missing:
        raise ValueError(f"{label} is missing semantic dimensions: {missing}")
    if extra:
        raise ValueError(f"{label} contains unexpected semantic columns: {extra}")


def _unique_list(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result = result.loc[:, ~result.columns.duplicated()].copy()
    return result


def load_consumption_history(raw_path: str | Path = RAW_INGESTION_PATH) -> pd.DataFrame:
    frame = normalize_column_names(load_table(raw_path))
    frame = _ensure_columns(frame)
    if "media_type" in frame.columns:
        frame = frame.loc[frame["media_type"].astype(str).str.lower() == "film"].copy()
    if "source_id" not in frame.columns and "canonical_id" in frame.columns:
        frame["source_id"] = frame["canonical_id"].astype(str)
    if "canonical_id" not in frame.columns and "source_id" in frame.columns:
        frame["canonical_id"] = frame["source_id"].astype(str)
    if "year" not in frame.columns and "release_year" in frame.columns:
        frame["year"] = frame["release_year"]
    if "release_year" not in frame.columns and "year" in frame.columns:
        frame["release_year"] = frame["year"]
    frame["source_id"] = frame["source_id"].astype(str)
    frame["canonical_id"] = frame["canonical_id"].astype(str)
    return frame.reset_index(drop=True)


def load_watched_source_ids(raw_path: str | Path = RAW_INGESTION_PATH) -> list[str]:
    history = load_consumption_history(raw_path)
    if history.empty or "source_id" not in history.columns:
        return []
    watched = (
        history["source_id"]
        .astype(str)
        .loc[lambda series: series.str.strip() != ""]
        .drop_duplicates()
        .tolist()
    )
    return watched


def load_condition_a_enriched(path: str | Path = CONDITION_A_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _ensure_columns(frame)
    if "source_id" in frame.columns:
        frame["source_id"] = frame["source_id"].astype(str)
    if "tmdb_id" in frame.columns:
        frame["tmdb_id"] = frame["tmdb_id"].astype("Int64")
    return frame


def _tmdb_request(client: TMDBClient, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = {"api_key": client.api_key, "language": client.language}
    if params:
        query.update(params)
    url = f"{TMDB_API_BASE}{endpoint}?{urlencode(query, quote_via=quote_plus)}"
    return _http_json(url, timeout=client.timeout)


def _discover_movies(client: TMDBClient, endpoint: str, params: dict[str, Any], pages: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        payload = _tmdb_request(client, endpoint, {**params, "page": page})
        rows.extend(payload.get("results", []))
    return rows


def _discover_popular(client: TMDBClient, pages: int = POPULAR_PAGES) -> list[dict[str, Any]]:
    return _discover_movies(client, "/movie/popular", {}, pages)


def _discover_top_rated(client: TMDBClient, pages: int = TOP_RATED_PAGES) -> list[dict[str, Any]]:
    return _discover_movies(client, "/movie/top_rated", {}, pages)


def _discover_recent(client: TMDBClient, pages: int = RECENT_PAGES, years_back: int = RECENT_YEARS_BACK) -> list[dict[str, Any]]:
    reference_year = datetime.now(timezone.utc).year
    params = {
        "sort_by": "primary_release_date.desc",
        "primary_release_date.gte": f"{reference_year - years_back}-01-01",
        "primary_release_date.lte": f"{reference_year}-12-31",
        "vote_count.gte": 25,
    }
    return _discover_movies(client, "/discover/movie", params, pages)


def _discover_genre_diverse(client: TMDBClient, pages: int = GENRE_PAGES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for genre_id, _genre_name in GENRE_DISCOVERY_PLAN:
        params = {
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "vote_count.gte": 25,
        }
        rows.extend(_discover_movies(client, "/discover/movie", params, pages))
    return rows


def _discover_seeded_recommendations(
    client: TMDBClient,
    seed_tmdb_ids: Sequence[int],
    pages: int = SEED_PAGES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tmdb_id in seed_tmdb_ids:
        for page in range(1, pages + 1):
            payload = _tmdb_request(client, f"/movie/{int(tmdb_id)}/recommendations", {"page": page})
            rows.extend(payload.get("results", []))
    return rows


def _extract_movie_summary(movie: dict[str, Any]) -> dict[str, Any]:
    release_date = movie.get("release_date") or ""
    release_year = pd.NA
    if release_date:
        try:
            release_year = int(str(release_date)[:4])
        except Exception:
            release_year = pd.NA
    return {
        "tmdb_id": movie.get("id"),
        "title": movie.get("title") or movie.get("original_title") or pd.NA,
        "year": release_year,
        "release_year": release_year,
        "tmdb_original_language": movie.get("original_language") or pd.NA,
        "tmdb_overview": movie.get("overview") or pd.NA,
        "source_vote_average": movie.get("vote_average"),
        "source_popularity": movie.get("popularity"),
    }


def _enrich_movie_details(client: TMDBClient, movie: dict[str, Any]) -> dict[str, Any] | None:
    movie_id = movie.get("id")
    if movie_id is None:
        return None
    try:
        details = client.movie_details(int(movie_id))
    except Exception:
        details = {}
    summary = _extract_movie_summary(movie)
    if not summary["title"] or pd.isna(summary["title"]):
        return None

    genres = [genre.get("name") for genre in details.get("genres", []) if genre.get("name")]
    countries = [
        country.get("iso_3166_1") or country.get("name")
        for country in details.get("production_countries", [])
        if country.get("iso_3166_1") or country.get("name")
    ]
    release_date = details.get("release_date") or movie.get("release_date")
    release_year = summary["year"]
    if release_date:
        try:
            release_year = int(str(release_date)[:4])
        except Exception:
            pass
    return {
        "source_id": str(movie_id),
        "tmdb_id": int(movie_id),
        "canonical_id": str(movie_id),
        "title": summary["title"],
        "release_year": release_year,
        "year": release_year,
        "tmdb_genres": "|".join(genres) if genres else pd.NA,
        "tmdb_original_language": details.get("original_language") or movie.get("original_language") or pd.NA,
        "tmdb_production_countries": "|".join(countries) if countries else pd.NA,
        "tmdb_overview": details.get("overview") or movie.get("overview") or pd.NA,
        "candidate_sources": [],
        "candidate_source_ranks": [],
        "tmdb_movie_status": details.get("status") or movie.get("status") or pd.NA,
    }


def _merge_candidate_record(
    bucket: dict[str, dict[str, Any]],
    movie: dict[str, Any],
    source_label: str,
    source_rank: int,
) -> None:
    movie_id = movie.get("id")
    if movie_id is None:
        return
    key = str(movie_id)
    if key not in bucket:
        candidate = _extract_movie_summary(movie)
        candidate.update(
            {
                "source_id": key,
                "tmdb_id": int(movie_id),
                "canonical_id": key,
                "candidate_sources": [source_label],
                "candidate_source_ranks": [int(source_rank)],
            }
        )
        bucket[key] = candidate
        return
    candidate = bucket[key]
    candidate["candidate_sources"] = _unique_list([*(candidate.get("candidate_sources", [])), source_label])
    candidate["candidate_source_ranks"] = _unique_list(
        [*(str(rank) for rank in candidate.get("candidate_source_ranks", [])), str(int(source_rank))]
    )


def _seed_tmdb_ids(consumed: pd.DataFrame, enriched: pd.DataFrame, limit: int = 5) -> list[int]:
    if consumed.empty:
        return []
    ranked = consumed.copy()
    if "preference_weight" in ranked.columns:
        ranked["preference_weight"] = pd.to_numeric(ranked["preference_weight"], errors="coerce")
    if "rating" in ranked.columns:
        ranked["rating"] = pd.to_numeric(ranked["rating"], errors="coerce")
    ranked = ranked.sort_values(
        [column for column in ["rating", "preference_weight", "year"] if column in ranked.columns],
        ascending=[False, False, False][: len([column for column in ["rating", "preference_weight", "year"] if column in ranked.columns])],
        na_position="last",
    )
    enriched_lookup = enriched.loc[:, [column for column in ["source_id", "tmdb_id"] if column in enriched.columns]].copy()
    if enriched_lookup.empty:
        return []
    enriched_lookup["source_id"] = enriched_lookup["source_id"].astype(str)
    enriched_lookup["tmdb_id"] = pd.to_numeric(enriched_lookup["tmdb_id"], errors="coerce")
    merged = ranked.merge(enriched_lookup, on="source_id", how="left")
    seed_ids = [int(value) for value in merged["tmdb_id"].dropna().astype(int).tolist()[:limit]]
    return seed_ids


def build_watched_exclusions(
    consumed_films: pd.DataFrame,
    enriched_films: pd.DataFrame,
    client: TMDBClient | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched_lookup = enriched_films.loc[:, [column for column in ["source_id", "tmdb_id"] if column in enriched_films.columns]].copy()
    if not enriched_lookup.empty:
        enriched_lookup["source_id"] = enriched_lookup["source_id"].astype(str)
        enriched_lookup["tmdb_id"] = pd.to_numeric(enriched_lookup["tmdb_id"], errors="coerce")
    enriched_lookup = enriched_lookup.set_index("source_id") if not enriched_lookup.empty else pd.DataFrame()

    exclusions: list[dict[str, Any]] = []
    tmdb_matches = 0
    title_year_fallbacks = 0
    for _, row in consumed_films.iterrows():
        source_id = str(row.get("source_id", ""))
        title = row.get("title")
        year = row.get("year") if pd.notna(row.get("year")) else row.get("release_year")
        tmdb_id = pd.NA
        match_method = "title_year_fallback"
        if not isinstance(enriched_lookup, pd.DataFrame) and source_id in enriched_lookup.index:
            pass
        if isinstance(enriched_lookup, pd.DataFrame) and not enriched_lookup.empty and source_id in enriched_lookup.index:
            matched = enriched_lookup.loc[source_id]
            if isinstance(matched, pd.Series):
                tmdb_id = matched.get("tmdb_id", pd.NA)
            else:
                tmdb_id = matched.iloc[0].get("tmdb_id", pd.NA)
            if pd.notna(tmdb_id):
                match_method = "enriched_tmdb"
        if pd.isna(tmdb_id) and client is not None and title:
            try:
                result = client.search_movie(str(title), year)
            except Exception:
                result = None
            if result and result.get("id") is not None:
                tmdb_id = int(result["id"])
                match_method = "tmdb_search"
        if pd.notna(tmdb_id):
            tmdb_matches += 1
        else:
            title_year_fallbacks += 1
        normalized_title = _normalize_title(title)
        normalized_title_year = f"{normalized_title}__{'' if pd.isna(year) else int(float(year))}"
        exclusions.append(
            {
                "source_id": source_id,
                "title": title,
                "year": year,
                "tmdb_id": int(tmdb_id) if pd.notna(tmdb_id) else pd.NA,
                "match_method": match_method,
                "normalized_title": normalized_title,
                "normalized_title_year": normalized_title_year,
                "exclusion_key": str(int(tmdb_id)) if pd.notna(tmdb_id) else normalized_title_year,
            }
        )

    frame = pd.DataFrame(exclusions).drop_duplicates(subset=["exclusion_key"], keep="first").reset_index(drop=True)
    report = {
        "consumed_film_rows": int(len(consumed_films)),
        "tmdb_exclusions": int(tmdb_matches),
        "fallback_exclusions": int(title_year_fallbacks),
        "unique_exclusion_keys": int(len(frame)),
    }
    return frame, report


def _candidate_source_order() -> list[tuple[str, str]]:
    return [
        ("popular", "popular"),
        ("top_rated", "top_rated"),
        ("recent", "recent"),
        ("genre_diverse", "genre_diverse"),
        ("seeded_recommendations", "seeded_recommendations"),
    ]


def generate_candidate_pool(
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    raw_path: str | Path = RAW_INGESTION_PATH,
    condition_a_path: str | Path = CONDITION_A_PATH,
    include_seeded_recommendations: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = get_api_keys()
    if not keys.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required to generate the live candidate pool.")

    client = TMDBClient(api_key=keys.tmdb_api_key)
    consumed = load_consumption_history(raw_path)
    enriched = load_condition_a_enriched(condition_a_path)
    exclusions, exclusion_report = build_watched_exclusions(consumed, enriched, client=client)
    watched_tmdb_ids = set(pd.to_numeric(exclusions["tmdb_id"], errors="coerce").dropna().astype(int).tolist())
    watched_title_year = set(exclusions["normalized_title_year"].astype(str).tolist())

    buckets: dict[str, dict[str, Any]] = OrderedDict()
    strategy_counts: dict[str, int] = {}
    strategy_details: list[dict[str, Any]] = []

    discovered_sources: list[tuple[str, list[dict[str, Any]]]] = []
    discovered_sources.append(("popular", _discover_popular(client, pages=POPULAR_PAGES)))
    discovered_sources.append(("top_rated", _discover_top_rated(client, pages=TOP_RATED_PAGES)))
    discovered_sources.append(("recent", _discover_recent(client, pages=RECENT_PAGES)))
    discovered_sources.append(("genre_diverse", _discover_genre_diverse(client, pages=GENRE_PAGES)))
    if include_seeded_recommendations:
        seed_tmdb_ids = _seed_tmdb_ids(consumed, enriched, limit=5)
        if seed_tmdb_ids:
            discovered_sources.append(("seeded_recommendations", _discover_seeded_recommendations(client, seed_tmdb_ids, pages=SEED_PAGES)))

    before_dedup = 0
    for source_label, movies in discovered_sources:
        strategy_counts[source_label] = strategy_counts.get(source_label, 0) + len(movies)
        before_dedup += len(movies)
        for rank, movie in enumerate(movies, start=1):
            _merge_candidate_record(buckets, movie, source_label, rank)
    deduped = list(buckets.values())
    dedup_count = int(len(deduped))
    if candidate_limit and dedup_count > candidate_limit:
        deduped = deduped[:candidate_limit]

    candidate_frame = pd.DataFrame(deduped)
    if candidate_frame.empty:
        return candidate_frame, {
            "candidate_limit": int(candidate_limit),
            "before_deduplication": int(before_dedup),
            "after_deduplication": 0,
            "before_exclusion": 0,
            "removed_watched": 0,
            "after_exclusion": 0,
            "strategy_counts": strategy_counts,
            "exclusion_report": exclusion_report,
        }

    candidate_frame["source_id"] = candidate_frame["source_id"].astype(str)
    candidate_frame["tmdb_id"] = pd.to_numeric(candidate_frame["tmdb_id"], errors="coerce").astype("Int64")
    candidate_frame["normalized_title"] = candidate_frame["title"].apply(_normalize_title)
    candidate_frame["normalized_title_year"] = candidate_frame.apply(
        lambda row: f"{row['normalized_title']}__{'' if pd.isna(row.get('year')) else int(float(row.get('year')))}",
        axis=1,
    )
    candidate_frame["is_watched_tmdb"] = candidate_frame["tmdb_id"].astype(str).isin({str(value) for value in watched_tmdb_ids})
    candidate_frame["is_watched_title_year"] = candidate_frame["normalized_title_year"].astype(str).isin(watched_title_year)
    before_exclusion = int(len(candidate_frame))
    candidate_frame = candidate_frame.loc[~(candidate_frame["is_watched_tmdb"] | candidate_frame["is_watched_title_year"])].copy().reset_index(drop=True)
    removed_watched = before_exclusion - len(candidate_frame)

    enriched_rows: list[dict[str, Any]] = []
    enrichment_failures = 0
    for row in candidate_frame.to_dict(orient="records"):
        movie_payload = {
            "id": int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else None,
            "title": row.get("title"),
        }
        if pd.notna(row.get("year")):
            movie_payload["release_date"] = f"{int(float(row['year']))}-01-01"
        enriched = _enrich_movie_details(client, movie_payload) or {}
        if not enriched:
            enrichment_failures += 1
            enriched = {
                "source_id": str(row.get("source_id")),
                "tmdb_id": int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else pd.NA,
                "canonical_id": str(row.get("source_id")),
                "title": row.get("title"),
                "release_year": row.get("release_year"),
                "year": row.get("year"),
                "tmdb_genres": pd.NA,
                "tmdb_original_language": pd.NA,
                "tmdb_production_countries": pd.NA,
                "tmdb_overview": pd.NA,
            }
        else:
            for field in ["tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"]:
                if field not in enriched:
                    enriched[field] = pd.NA
        enriched["candidate_sources"] = _unique_list([str(value) for value in row.get("candidate_sources", [])])
        enriched["candidate_source_ranks"] = _unique_list([str(value) for value in row.get("candidate_source_ranks", [])])
        enriched_rows.append(enriched)

    candidate_frame = pd.DataFrame(enriched_rows).reset_index(drop=True)
    candidate_frame = candidate_frame.loc[
        :,
        [
            "source_id",
            "tmdb_id",
            "title",
            "year",
            "release_year",
            "tmdb_genres",
            "tmdb_original_language",
            "tmdb_production_countries",
            "tmdb_overview",
            "candidate_sources",
            "candidate_source_ranks",
        ]
    ].reset_index(drop=True)

    report = {
        "candidate_limit": int(candidate_limit),
        "before_deduplication": int(before_dedup),
        "after_deduplication": int(dedup_count),
        "before_exclusion": int(before_exclusion),
        "removed_watched": int(removed_watched),
        "after_exclusion": int(len(candidate_frame)),
        "strategy_counts": strategy_counts,
        "exclusion_report": exclusion_report,
        "watched_tmdb_ids": int(len(watched_tmdb_ids)),
        "watched_title_year_keys": int(len(watched_title_year)),
        "enrichment_failures": int(enrichment_failures),
    }
    return candidate_frame, report


def _safe_listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _distribution_from_pipe_column(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series.dropna().astype(str):
        for token in [part.strip() for part in value.split("|") if part.strip()]:
            counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _top_recommendation_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "genre_distribution": {},
            "decade_distribution": {},
            "language_distribution": {},
            "country_distribution": {},
            "predicted_preference": {},
        }
    decades = frame["year"].apply(lambda value: f"{int(value)//10*10}s" if pd.notna(value) else "unknown")
    return {
        "genre_distribution": _distribution_from_pipe_column(frame.get("tmdb_genres", pd.Series(dtype=str))),
        "decade_distribution": dict(decades.value_counts().sort_index()),
        "language_distribution": dict(frame.get("tmdb_original_language", pd.Series(dtype=str)).value_counts().sort_values(ascending=False)),
        "country_distribution": _distribution_from_pipe_column(frame.get("tmdb_production_countries", pd.Series(dtype=str))),
        "predicted_preference": {
            "mean": float(frame["predicted_preference"].mean()),
            "min": float(frame["predicted_preference"].min()),
            "max": float(frame["predicted_preference"].max()),
        },
    }


def _load_taste_profile_for_explanations(semantic_cache_path: str | Path = SEMANTIC_CACHE_PATH) -> pd.DataFrame:
    semantic_vectors = SemanticVectorStore(semantic_cache_path).load()
    history = load_consumption_history(RAW_INGESTION_PATH)
    if "rating" not in history.columns:
        raise KeyError("Consumption history is missing rating.")
    history = history.copy()
    return build_taste_profile(history, semantic_vectors)


def _build_top_recommendation_rows(
    scored: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    top_k: int,
    candidate_semantic_cache_path: str | Path = SEMANTIC_CACHE_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    top = scored.head(top_k).copy()
    if top.empty:
        return top, {"semantic_classified": 0, "coverage": 0.0, "failures": []}

    top = top.merge(
        candidate_frame.loc[:, [column for column in ["source_id", "candidate_sources", "tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"] if column in candidate_frame.columns]],
        on="source_id",
        how="left",
        suffixes=("", "_candidate"),
    )
    top["canonical_id"] = top["source_id"].astype(str)
    semantic_frame, semantic_report = classify_semantic_vectors(
        top,
        cache_path=candidate_semantic_cache_path,
    )
    semantic_frame = semantic_frame.rename(columns={"canonical_id": "source_id"})
    merged = top.merge(semantic_frame, on="source_id", how="left", suffixes=("", "_semantic"))
    taste_profile = _load_taste_profile_for_explanations()
    merged["candidate_sources"] = merged["candidate_sources"].apply(lambda value: _json_list(value) if not isinstance(value, list) else value)
    merged["taste_evidence"] = merged.apply(lambda row: build_taste_evidence(row, taste_profile), axis=1)
    merged["metadata"] = merged.apply(
        lambda row: {
            "genres": _json_list(row.get("tmdb_genres")),
            "original_language": row.get("tmdb_original_language"),
            "production_countries": _json_list(row.get("tmdb_production_countries")),
        },
        axis=1,
    )
    merged["model"] = merged.apply(
        lambda _row: {"architecture": "B3", "status": "exploratory_candidate_mvp"},
        axis=1,
    )
    return merged, semantic_report


def _serialize_candidate_pool(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ["candidate_sources", "candidate_source_ranks"]:
        if column in output.columns:
            output[column] = output[column].apply(lambda value: json.dumps(value if isinstance(value, list) else _json_list(value), ensure_ascii=False))
    return output


def run_recommendation_live(
    top_k: int = DEFAULT_TOP_K,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    model_dir: str | Path = MODEL_DIR,
    raw_path: str | Path = RAW_INGESTION_PATH,
    condition_a_path: str | Path = CONDITION_A_PATH,
    semantic_cache_path: str | Path = SEMANTIC_CACHE_PATH,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    model_bundle, _, _ = load_model_bundle(model_dir)

    candidate_frame, candidate_report = generate_candidate_pool(
        candidate_limit=candidate_limit,
        raw_path=raw_path,
        condition_a_path=condition_a_path,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RECOMMENDATION_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate_pool_path = run_dir / "candidate_pool.csv"
    ranked_candidates_path = run_dir / "ranked_candidates.csv"
    top_recommendations_path = run_dir / "top_recommendations.json"
    diagnostics_path = run_dir / "candidate_diagnostics.json"
    run_config_path = run_dir / "run_config.json"

    candidate_pool = candidate_frame.copy()
    _serialize_candidate_pool(candidate_pool).to_csv(candidate_pool_path, index=False)

    if not candidate_frame.empty:
        documents = build_film_documents(candidate_frame)
        embeddings, manifest, embedding_report = load_or_generate_candidate_embeddings(
            documents,
            cache_path=RECOMMENDATION_CACHE_PATH,
        )
        candidate_frame = candidate_frame.merge(
            embeddings.loc[:, [column for column in embeddings.columns if column != "title"]],
            on="source_id",
            how="left",
            suffixes=("", "_embedding"),
        )
    else:
        embedding_report = {"cache_status": "miss", "cache_hits": 0, "cache_misses": 0, "coverage": 0.0, "successful": 0, "failed": 0}

    scored = score_candidates(
        candidate_frame,
        model_dir=model_dir,
        watched_ids=None,
        top_k=max(top_k, len(candidate_frame)),
    )
    top_rows, semantic_report = _build_top_recommendation_rows(
        scored,
        candidate_frame,
        top_k,
        candidate_semantic_cache_path=run_dir / "top_recommendation_semantic_vectors.csv",
    )
    ranked_candidates = scored.merge(
        top_rows.loc[:, [column for column in ["source_id", "taste_evidence", "metadata", "model"] if column in top_rows.columns]],
        on="source_id",
        how="left",
    )

    ranked_candidates.to_csv(ranked_candidates_path, index=False)

    top_output = []
    for _, row in top_rows.head(top_k).iterrows():
        top_output.append(
            {
                "tmdb_id": int(row["tmdb_id"]) if pd.notna(row["tmdb_id"]) else None,
                "title": row["title"],
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "rank": int(row["rank"]),
                "predicted_preference": float(row["predicted_preference"]),
                "candidate_sources": row["candidate_sources"],
                "metadata": row["metadata"],
                "taste_evidence": row["taste_evidence"],
                "model": row["model"],
            }
        )
    top_recommendations_path.write_text(json.dumps(top_output, indent=2, ensure_ascii=False, default=_json_default))

    diagnostics = {
        "candidate_pool": candidate_report,
        "embedding": embedding_report,
        "semantic_classification": semantic_report,
        "top_k_diagnostics": _top_recommendation_diagnostics(top_rows.head(top_k)),
        "selected_model": {
            "architecture": model_bundle.architecture,
            "status": model_bundle.model_status,
            "selected_pca_dim": model_bundle.selected_pca_dim,
            "selected_alpha": model_bundle.selected_alpha,
        },
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=_json_default))

    run_config = {
        "run_id": run_id,
        "candidate_generation_strategies": [label for label, _ in _candidate_source_order()],
        "candidate_limit": int(candidate_limit),
        "counts_before_deduplication": candidate_report["before_deduplication"],
        "counts_after_deduplication": candidate_report["after_deduplication"],
        "watched_exclusions": candidate_report["removed_watched"],
        "enrichment_failures": int(candidate_report.get("enrichment_failures", 0)),
        "embedding_cache_hits": int(embedding_report.get("cache_hits", 0)),
        "embedding_cache_misses": int(embedding_report.get("cache_misses", 0)),
        "final_scorable_count": int(len(candidate_frame)),
        "top_k": int(top_k),
        "deployment_model": {
            "architecture": model_bundle.architecture,
            "status": model_bundle.model_status,
            "selected_pca_dim": model_bundle.selected_pca_dim,
            "selected_alpha": model_bundle.selected_alpha,
        },
    }
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False, default=_json_default))

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "candidate_pool_path": str(candidate_pool_path),
        "ranked_candidates_path": str(ranked_candidates_path),
        "top_recommendations_path": str(top_recommendations_path),
        "candidate_diagnostics_path": str(diagnostics_path),
        "run_config_path": str(run_config_path),
        "top_recommendations": top_output,
        "candidate_report": candidate_report,
        "embedding_report": embedding_report,
        "semantic_report": semantic_report,
    }
