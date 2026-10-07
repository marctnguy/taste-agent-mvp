from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from mvp.src.candidates import generate_candidate_pool, load_consumption_history
from mvp.src.config import get_api_keys
from mvp.src.data import load_table, normalize_column_names
from mvp.src.generative_v4.intent_chain import RequestUnderstanding
from mvp.src.generative_v4.schemas import StructuredConstraints
from mvp.src.mvp_deployment import CANDIDATE_EMBEDDING_CACHE_PATH, load_or_generate_candidate_embeddings
from mvp.src.prepare import TMDBClient


TMDB_API_BASE = "https://api.themoviedb.org/3"
FILM_DOC_VERSION = "film_doc_v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GENRE_NAME_TO_ID = {
    "Action": 28,
    "Adventure": 12,
    "Animation": 16,
    "Comedy": 35,
    "Crime": 80,
    "Documentary": 99,
    "Drama": 18,
    "Family": 10751,
    "Fantasy": 14,
    "History": 36,
    "Horror": 27,
    "Music": 10402,
    "Mystery": 9648,
    "Romance": 10749,
    "Science Fiction": 878,
    "TV Movie": 10770,
    "Thriller": 53,
    "War": 10752,
    "Western": 37,
}


@dataclass(frozen=True)
class CatalogRetrievalDiagnostics:
    candidate_universe_count: int
    post_constraint_candidate_count: int
    contextual_retrieval_used: bool
    candidate_context_size: int
    shortlist_size: int
    retrieval_mode: str
    request_relevance_mode: str
    reference_status: str
    reference_tmdb_id: int | None
    reference_title: str | None
    request_relevance_method: str
    query_aware_augmentation_used: bool
    watched_excluded_count: int
    candidate_limit: int
    b3_applicability_mode: str


@dataclass(frozen=True)
class CatalogRetrievalResult:
    request: RequestUnderstanding
    catalog_frame: pd.DataFrame
    documents: pd.DataFrame
    request_embedding: np.ndarray | None
    retrieval_diagnostics: CatalogRetrievalDiagnostics
    augmentation_report: dict[str, Any]


@dataclass(frozen=True)
class QueryAwareAugmentation:
    frame: pd.DataFrame
    report: dict[str, Any]


SEQUEL_ELIGIBILITY_RULES: tuple[dict[str, Any], ...] = (
    {
        "candidate_tmdb_id": 21521,
        "candidate_title": "Project A - Part II",
        "candidate_year": 1987,
        "required_prerequisites": [{"title": "Project A", "year": 1983}],
    },
    {
        "candidate_tmdb_id": 9714,
        "candidate_title": "Home Alone 3",
        "candidate_year": 1997,
        "required_prerequisites": [{"title": "Home Alone 2: Lost in New York", "year": 1992}],
    },
    {
        "candidate_tmdb_id": 1084244,
        "candidate_title": "Toy Story 5",
        "candidate_year": 2026,
        "required_prerequisites": [
            {"tmdb_id": 862, "title": "Toy Story", "year": 1995},
            {"tmdb_id": 863, "title": "Toy Story 2", "year": 1999},
            {"tmdb_id": 10193, "title": "Toy Story 3", "year": 2010},
            {"tmdb_id": 301528, "title": "Toy Story 4", "year": 2019},
        ],
    },
)


def _normalize_year(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _title_year_key(title: Any, year: Any) -> str | None:
    normalized_title = _normalize_text(title)
    normalized_year = _normalize_year(year)
    if not normalized_title or normalized_year is None:
        return None
    return f"{normalized_title}__{normalized_year}"


def _candidate_identity_keys(row: pd.Series | dict[str, Any]) -> set[str]:
    payload = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    keys: set[str] = set()
    tmdb_id = payload.get("tmdb_id")
    if tmdb_id is not None and not pd.isna(tmdb_id):
        try:
            keys.add(str(int(float(tmdb_id))))
        except Exception:
            pass
    source_id = payload.get("source_id")
    if source_id is not None and str(source_id).strip():
        keys.add(str(source_id).strip())
    title_year = _title_year_key(payload.get("title"), payload.get("year") or payload.get("release_year"))
    if title_year:
        keys.add(title_year)
    return keys


@lru_cache(maxsize=1)
def _series_prerequisite_rules() -> tuple[dict[str, Any], ...]:
    return SEQUEL_ELIGIBILITY_RULES


def _prerequisite_satisfied(prerequisite: dict[str, Any], watched_keys: set[str]) -> bool:
    tmdb_id = prerequisite.get("tmdb_id")
    if tmdb_id is not None and str(tmdb_id) in watched_keys:
        return True
    title = prerequisite.get("title")
    year = prerequisite.get("year")
    title_year = _title_year_key(title, year)
    return bool(title_year and title_year in watched_keys)


def _apply_series_prerequisite_filters(frame: pd.DataFrame, watched_keys: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"series_prerequisite_blocked": 0, "series_prerequisite_blocked_titles": []}
    filtered = frame.copy().reset_index(drop=True)
    blocked_titles: list[str] = []
    keep_mask: list[bool] = []
    rules = _series_prerequisite_rules()
    for _, row in filtered.iterrows():
        row_keys = _candidate_identity_keys(row)
        rule = next(
            (
                item
                for item in rules
                if (
                    (item.get("candidate_tmdb_id") is not None and str(item["candidate_tmdb_id"]) in row_keys)
                    or (
                        item.get("candidate_title")
                        and item.get("candidate_year") is not None
                        and _title_year_key(item["candidate_title"], item["candidate_year"]) in row_keys
                    )
                )
            ),
            None,
        )
        if rule is None:
            keep_mask.append(True)
            continue
        required = rule.get("required_prerequisites", [])
        satisfied = all(_prerequisite_satisfied(prereq, watched_keys) for prereq in required)
        keep_mask.append(bool(satisfied))
        if not satisfied:
            blocked_titles.append(str(row.get("title") or rule.get("candidate_title") or ""))
    if len(keep_mask) != len(filtered):
        keep_mask = [True for _ in range(len(filtered))]
    kept = filtered.loc[keep_mask].reset_index(drop=True)
    return kept, {"series_prerequisite_blocked": int(len(blocked_titles)), "series_prerequisite_blocked_titles": blocked_titles}


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _tmdb_request(client: TMDBClient, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = {"api_key": client.api_key, "language": client.language}
    if params:
        query.update(params)
    url = f"{TMDB_API_BASE}{endpoint}?{urlencode(query, quote_via=quote_plus)}"
    return _http_json(url, timeout=client.timeout)


def _request_spec(request: RequestUnderstanding):
    return request.spec


def _request_structured_constraints(request: RequestUnderstanding):
    spec = _request_spec(request)
    if spec is not None:
        return spec.structured_constraints
    return StructuredConstraints(
        required_languages=list(getattr(request.intent, "requested_languages", []) or []),
        required_countries=list(getattr(request.intent, "requested_countries", []) or []),
        required_genres=list(getattr(request.intent, "requested_genres", []) or []),
        required_decades=list(getattr(request.intent, "requested_decades", []) or []),
    )


def _request_semantic_concepts(request: RequestUnderstanding) -> list[str]:
    spec = _request_spec(request)
    if spec is None:
        return []
    return [concept.concept for concept in spec.semantic_requirements if concept.concept]


def _request_semantic_exclusions(request: RequestUnderstanding) -> list[str]:
    spec = _request_spec(request)
    if spec is None:
        return []
    return [concept.concept for concept in spec.semantic_exclusions if concept.concept]


def _request_has_query_aware_signal(request: RequestUnderstanding) -> bool:
    spec = _request_spec(request)
    if spec is None:
        return False
    return bool(
        spec.semantic_query_text.strip()
        or spec.reference.title
        or spec.structured_constraints.required_languages
        or spec.structured_constraints.required_countries
        or spec.structured_constraints.required_genres
        or spec.structured_constraints.required_decades
        or spec.structured_constraints.min_year is not None
        or spec.structured_constraints.max_year is not None
    )


def _discover_movies(client: TMDBClient, endpoint: str, params: dict[str, Any], pages: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        payload = _tmdb_request(client, endpoint, {**params, "page": page})
        rows.extend(payload.get("results", []))
    return rows


def _enrich_movie_candidate(client: TMDBClient, movie: dict[str, Any]) -> dict[str, Any]:
    movie_id = movie.get("id")
    title = movie.get("title") or movie.get("original_title")
    if movie_id is None or not title:
        return {}
    try:
        details = client.movie_details(int(movie_id))
    except Exception:
        details = {}
    genre_names = [genre.get("name") for genre in details.get("genres", []) if genre.get("name")]
    countries = [
        country.get("iso_3166_1") or country.get("name")
        for country in details.get("production_countries", [])
        if country.get("iso_3166_1") or country.get("name")
    ]
    release_date = details.get("release_date") or movie.get("release_date") or ""
    release_year = pd.NA
    if release_date:
        try:
            release_year = int(str(release_date)[:4])
        except Exception:
            release_year = pd.NA
    return {
        "source_id": str(movie_id),
        "tmdb_id": int(movie_id),
        "title": title,
        "year": release_year,
        "release_year": release_year,
        "tmdb_genres": "|".join(genre_names) if genre_names else pd.NA,
        "tmdb_original_language": details.get("original_language") or movie.get("original_language") or pd.NA,
        "tmdb_production_countries": "|".join(countries) if countries else pd.NA,
        "tmdb_overview": details.get("overview") or movie.get("overview") or pd.NA,
        "candidate_sources": [],
        "candidate_source_ranks": [],
    }


def _parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.strip("[]")
    parts = [part.strip().strip("'\"") for part in text.split("|")] if "|" in text else [part.strip().strip("'\"") for part in text.split(",")]
    return [part for part in parts if part]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_canonical_film_document(row: pd.Series) -> str:
    lines = [f"Title: {_normalize_text(row.get('title'))}"]
    if pd.notna(row.get("tmdb_id")):
        lines.append(f"TMDB ID: {int(row.get('tmdb_id'))}")
    if pd.notna(row.get("release_year")):
        lines.append(f"Release year: {int(float(row.get('release_year')))}")
    if not pd.isna(row.get("tmdb_genres")):
        lines.append(f"Genres: {_normalize_text(row.get('tmdb_genres')).replace('|', ' | ')}")
    if not pd.isna(row.get("tmdb_original_language")):
        lines.append(f"Original language: {_normalize_text(row.get('tmdb_original_language'))}")
    if not pd.isna(row.get("tmdb_production_countries")):
        lines.append(f"Production countries: {_normalize_text(row.get('tmdb_production_countries')).replace('|', ' | ')}")
    if not pd.isna(row.get("tmdb_overview")):
        lines.append(f"Overview: {_normalize_text(row.get('tmdb_overview'))}")
    provenance = _parse_listish(row.get("candidate_sources"))
    if provenance:
        lines.append(f"Candidate provenance: {' | '.join(provenance)}")
    return "\n".join(lines)


def build_canonical_film_documents(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    if "source_id" not in ordered.columns:
        raise KeyError("Film documents require a source_id column.")
    ordered["source_id"] = ordered["source_id"].astype(str)
    rows: list[dict[str, Any]] = []
    for _, row in ordered.sort_values("source_id").iterrows():
        document = build_canonical_film_document(row)
        rows.append(
            {
                "source_id": str(row["source_id"]),
                "tmdb_id": int(row["tmdb_id"]) if pd.notna(row.get("tmdb_id")) else pd.NA,
                "title": row.get("title"),
                "document_text": document,
                "document_hash": _sha256(document),
                "document_version": FILM_DOC_VERSION,
            }
        )
    return pd.DataFrame(rows)


def _openai_text_embedding(text: str, api_key: str, model: str = OPENAI_EMBEDDING_MODEL) -> np.ndarray:
    payload = {"model": model, "input": [text]}
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    if not data:
        raise RuntimeError("OpenAI embedding response was empty.")
    vector = np.asarray(data[0]["embedding"], dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _local_text_embedding(text: str, dimensions: int = 1536) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=float)
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dimensions
        sign = 1.0 if int(digest[8:16], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _embedding_frame_from_vectors(documents: pd.DataFrame, vectors: list[np.ndarray], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if documents.empty or not vectors:
        empty = pd.DataFrame(columns=["source_id", "title", "document_hash", "embedding_model", "embedding_dim", "status"])
        return empty, empty.copy(), {
            "provider": "local",
            "model": model_name,
            "embedding_dim": 0,
            "coverage": 0.0,
            "successful": 0,
            "failed": 0,
            "cache_status": "local_fallback",
            "cache_hits": 0,
            "cache_misses": 0,
        }
    vector_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for row, vector in zip(documents.to_dict(orient="records"), vectors, strict=True):
        vector_row = {
            "source_id": str(row["source_id"]),
            "title": row.get("title"),
            "document_hash": row["document_hash"],
            "embedding_model": model_name,
            "embedding_dim": len(vector),
            "status": "success",
        }
        vector_row.update({f"emb_{index:04d}": float(value) for index, value in enumerate(vector)})
        vector_rows.append(vector_row)
        manifest_rows.append(
            {
                "source_id": str(row["source_id"]),
                "title": row.get("title"),
                "document_hash": row["document_hash"],
                "embedding_model": model_name,
                "embedding_dim": len(vector),
                "status": "success",
                "error": "",
            }
        )
    embeddings = pd.DataFrame(vector_rows)
    manifest = pd.DataFrame(manifest_rows)
    report = {
        "provider": "local",
        "model": model_name,
        "embedding_dim": int(vectors[0].shape[0]) if vectors else 0,
        "coverage": float(len(embeddings) / len(documents)) if len(documents) else 0.0,
        "successful": int(len(embeddings)),
        "failed": 0,
        "cache_status": "local_fallback",
        "cache_hits": 0,
        "cache_misses": int(len(documents)),
    }
    return embeddings, manifest, report


def embed_documents(documents: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if documents.empty:
        empty = pd.DataFrame(columns=["source_id", "title", "document_hash", "embedding_model", "embedding_dim", "status"])
        return empty, empty.copy(), {
            "provider": "local",
            "model": "local-hashing-1536",
            "embedding_dim": 0,
            "coverage": 0.0,
            "successful": 0,
            "failed": 0,
            "cache_status": "empty",
            "cache_hits": 0,
            "cache_misses": 0,
        }
    keys = get_api_keys()
    if keys.openai_api_key:
        try:
            return load_or_generate_candidate_embeddings(documents, cache_path=CANDIDATE_EMBEDDING_CACHE_PATH)
        except Exception:
            pass
    vectors = [_local_text_embedding(text) for text in documents["document_text"].tolist()]
    return _embedding_frame_from_vectors(documents, vectors, "local-hashing-1536")


def embed_text(text: str) -> np.ndarray:
    keys = get_api_keys()
    if keys.openai_api_key:
        try:
            return _openai_text_embedding(text, keys.openai_api_key)
        except Exception:
            pass
    return _local_text_embedding(text)


def _candidate_embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def _augment_with_reference_candidates(request: RequestUnderstanding, client: TMDBClient | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if request.reference_status != "resolved" or request.reference_tmdb_id is None:
        return pd.DataFrame(), {"reference_candidates": 0}
    if client is None:
        keys = get_api_keys()
        if not keys.tmdb_api_key:
            return pd.DataFrame(), {"reference_candidates": 0}
        client = TMDBClient(api_key=keys.tmdb_api_key)

    rows: list[dict[str, Any]] = []
    try:
        payload = _tmdb_request(client, f"/movie/{int(request.reference_tmdb_id)}/recommendations", {"page": 1})
        results = payload.get("results", [])
    except Exception:
        results = []
    for movie in results:
        candidate = _enrich_movie_candidate(client, movie)
        if not candidate:
            continue
        candidate["candidate_sources"] = ["reference_recommendations"]
        candidate["candidate_source_ranks"] = ["1"]
        rows.append(candidate)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["source_id"])
    return frame, {"reference_candidates": int(len(frame))}


def _search_keywords(client: TMDBClient, query: str) -> list[dict[str, Any]]:
    try:
        payload = _tmdb_request(client, "/search/keyword", {"query": query, "page": 1})
    except Exception:
        return []
    results = payload.get("results", [])
    return [item for item in results if isinstance(item, dict)]


def _augment_with_semantic_keyword_candidates(request: RequestUnderstanding, client: TMDBClient | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = _request_spec(request)
    if spec is None or client is None:
        return pd.DataFrame(), {"semantic_keyword_candidates": 0, "resolved_keywords": []}
    concepts = _request_semantic_concepts(request)
    if not concepts:
        return pd.DataFrame(), {"semantic_keyword_candidates": 0, "resolved_keywords": []}

    resolved_keywords: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for concept in concepts:
        results = _search_keywords(client, concept.replace("_", " "))
        if not results:
            continue
        best = results[0]
        keyword_id = best.get("id")
        keyword_name = best.get("name") or concept.replace("_", " ")
        if keyword_id is None:
            continue
        resolved_keywords.append({"concept": concept, "tmdb_keyword_id": int(keyword_id), "tmdb_keyword_name": keyword_name, "provenance": "tmdb_keyword_search"})
        try:
            movies = _discover_movies(
                client,
                "/discover/movie",
                {"with_keywords": int(keyword_id), "sort_by": "popularity.desc", "vote_count.gte": 25},
                pages=1,
            )
        except Exception:
            movies = []
        rows = []
        for rank, movie in enumerate(movies, start=1):
            candidate = _enrich_movie_candidate(client, movie)
            if not candidate:
                continue
            candidate["candidate_sources"] = [f"keyword:{concept}"]
            candidate["candidate_source_ranks"] = [str(rank)]
            rows.append(candidate)
        if rows:
            frames.append(pd.DataFrame(rows))

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if "source_id" not in combined.columns or combined.empty:
            combined = pd.DataFrame(columns=["source_id"])
        else:
            combined["source_id"] = combined["source_id"].astype(str)
            combined = combined.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=["source_id"])
    return combined, {"semantic_keyword_candidates": int(len(combined)), "resolved_keywords": resolved_keywords}


def _augment_with_constraint_candidates(request: RequestUnderstanding, client: TMDBClient | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if client is None:
        keys = get_api_keys()
        if not keys.tmdb_api_key:
            return pd.DataFrame(), {"constraint_candidates": 0}
        client = TMDBClient(api_key=keys.tmdb_api_key)

    constraints = _request_structured_constraints(request)
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    if getattr(constraints, "required_genres", None):
        for genre in constraints.required_genres:
            genre_id = GENRE_NAME_TO_ID.get(genre)
            if genre_id is None:
                continue
            try:
                movies = _discover_movies(
                    client,
                    "/discover/movie",
                    {"with_genres": genre_id, "sort_by": "popularity.desc", "vote_count.gte": 25},
                    pages=1,
                )
            except Exception:
                movies = []
            rows = []
            for rank, movie in enumerate(movies, start=1):
                candidate = _enrich_movie_candidate(client, movie)
                if not candidate:
                    continue
                candidate.update(
                    {
                        "candidate_sources": [f"genre:{genre}"],
                        "candidate_source_ranks": [str(rank)],
                    }
                )
                rows.append(candidate)
            frame = pd.DataFrame(rows)
            counts[f"genre:{genre}"] = int(len(frame))
            frames.append(frame)

    if getattr(constraints, "required_languages", None):
        for language in constraints.required_languages:
            try:
                movies = _discover_movies(
                    client,
                    "/discover/movie",
                    {"with_original_language": language, "sort_by": "popularity.desc", "vote_count.gte": 25},
                    pages=1,
                )
            except Exception:
                movies = []
            rows = []
            for rank, movie in enumerate(movies, start=1):
                candidate = _enrich_movie_candidate(client, movie)
                if not candidate:
                    continue
                candidate.update(
                    {
                        "candidate_sources": [f"language:{language}"],
                        "candidate_source_ranks": [str(rank)],
                    }
                )
                rows.append(candidate)
            frame = pd.DataFrame(rows)
            counts[f"language:{language}"] = int(len(frame))
            frames.append(frame)

    if getattr(constraints, "required_decades", None):
        for decade in constraints.required_decades:
            try:
                decade_year = int(str(decade).replace("s", ""))
            except Exception:
                continue
            params = {
                "primary_release_date.gte": f"{decade_year}-01-01",
                "primary_release_date.lte": f"{decade_year + 9}-12-31",
                "sort_by": "primary_release_date.desc",
                "vote_count.gte": 25,
            }
            try:
                movies = _discover_movies(client, "/discover/movie", params, pages=1)
            except Exception:
                movies = []
            rows = []
            for rank, movie in enumerate(movies, start=1):
                candidate = _enrich_movie_candidate(client, movie)
                if not candidate:
                    continue
                candidate.update(
                    {
                        "candidate_sources": [f"decade:{decade}"],
                        "candidate_source_ranks": [str(rank)],
                    }
                )
                rows.append(candidate)
            frame = pd.DataFrame(rows)
            counts[f"decade:{decade}"] = int(len(frame))
            frames.append(frame)

    if getattr(constraints, "required_countries", None):
        for country in constraints.required_countries:
            try:
                movies = _discover_movies(
                    client,
                    "/discover/movie",
                    {"with_origin_country": country, "sort_by": "popularity.desc", "vote_count.gte": 25},
                    pages=1,
                )
            except Exception:
                movies = []
            rows = []
            for rank, movie in enumerate(movies, start=1):
                candidate = _enrich_movie_candidate(client, movie)
                if not candidate:
                    continue
                candidate.update(
                    {
                        "candidate_sources": [f"country:{country}"],
                        "candidate_source_ranks": [str(rank)],
                    }
                )
                rows.append(candidate)
            frame = pd.DataFrame(rows)
            counts[f"country:{country}"] = int(len(frame))
            frames.append(frame)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty or "source_id" not in combined.columns:
            combined = pd.DataFrame(columns=["source_id"])
        else:
            combined["source_id"] = combined["source_id"].astype(str)
            combined = combined.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    else:
        combined = pd.DataFrame(columns=["source_id"])
    return combined, counts


def build_query_aware_augmentation(
    request: RequestUnderstanding,
    client: TMDBClient | None,
) -> QueryAwareAugmentation:
    """Build request-only catalog augmentation once for reuse across route pools.

    Reference, structured-constraint, and semantic-keyword augmentation depend only
    on the parsed request and TMDB client, not on the base candidate pool. Keeping
    this result immutable-by-convention lets request/history routes reuse the same
    external lookup result while preserving their independent base pools.
    """
    augmentation_frames: list[pd.DataFrame] = []
    augmentation_report: dict[str, Any] = {"query_aware_augmentation_used": False}

    if _request_has_query_aware_signal(request):
        ref_frame, ref_report = _augment_with_reference_candidates(request, client)
        if not ref_frame.empty:
            augmentation_frames.append(ref_frame)
            augmentation_report.update(ref_report)
            augmentation_report["query_aware_augmentation_used"] = True

        constraint_frame, constraint_report = _augment_with_constraint_candidates(
            request, client
        )
        if not constraint_frame.empty:
            augmentation_frames.append(constraint_frame)
            augmentation_report.update(constraint_report)
            augmentation_report["query_aware_augmentation_used"] = True

        keyword_frame, keyword_report = _augment_with_semantic_keyword_candidates(
            request, client
        )
        if not keyword_frame.empty:
            augmentation_frames.append(keyword_frame)
            augmentation_report.update(keyword_report)
            augmentation_report["query_aware_augmentation_used"] = True

    if augmentation_frames:
        frame = pd.concat(augmentation_frames, ignore_index=True)
        if "source_id" in frame.columns:
            frame["source_id"] = frame["source_id"].astype(str)
    else:
        frame = pd.DataFrame(columns=["source_id"])

    return QueryAwareAugmentation(frame=frame, report=augmentation_report)



def _filter_hard_constraints(frame: pd.DataFrame, request: RequestUnderstanding) -> pd.DataFrame:
    spec = _request_spec(request)
    constraints = spec.structured_constraints if spec is not None else None
    filtered = frame.copy()
    if constraints and constraints.required_languages and "tmdb_original_language" in filtered.columns:
        allowed = {value.lower() for value in constraints.required_languages}
        filtered = filtered.loc[filtered["tmdb_original_language"].fillna("").astype(str).str.lower().isin(allowed)]
    if constraints and constraints.required_countries and "tmdb_production_countries" in filtered.columns:
        allowed = {value.lower() for value in constraints.required_countries}
        filtered = filtered.loc[
            filtered["tmdb_production_countries"].apply(
                lambda value: bool({str(item).lower() for item in _parse_listish(value)}.intersection(allowed))
            )
        ]
    if constraints and constraints.required_decades and "year" in filtered.columns:
        decades = {f"{value}" for value in constraints.required_decades}
        filtered = filtered.loc[
            filtered["year"].apply(
                lambda value: f"{int(value) // 10 * 10}s" in decades if pd.notna(value) else False
            )
        ]
    if constraints and constraints.required_genres and "tmdb_genres" in filtered.columns:
        allowed = {value.lower() for value in constraints.required_genres}
        filtered = filtered.loc[
            filtered["tmdb_genres"].apply(
                lambda value: bool(_parse_listish(value)) and bool({str(item).lower() for item in _parse_listish(value)}.intersection(allowed))
            )
        ]
    if constraints and constraints.excluded_genres and "tmdb_genres" in filtered.columns:
        excluded = {value.lower() for value in constraints.excluded_genres}
        filtered = filtered.loc[
            ~filtered["tmdb_genres"].apply(
                lambda value: not bool(_parse_listish(value)) or bool({str(item).lower() for item in _parse_listish(value)}.intersection(excluded))
            )
        ]
    if constraints and constraints.excluded_languages and "tmdb_original_language" in filtered.columns:
        excluded = {value.lower() for value in constraints.excluded_languages}
        filtered = filtered.loc[~filtered["tmdb_original_language"].fillna("").astype(str).str.lower().isin(excluded)]
    if constraints and constraints.excluded_countries and "tmdb_production_countries" in filtered.columns:
        excluded = {value.lower() for value in constraints.excluded_countries}
        filtered = filtered.loc[
            ~filtered["tmdb_production_countries"].apply(
                lambda value: bool({str(item).lower() for item in _parse_listish(value)}.intersection(excluded))
            )
        ]
    if constraints and constraints.excluded_decades and "year" in filtered.columns:
        decades = {f"{value}" for value in constraints.excluded_decades}
        filtered = filtered.loc[
            ~filtered["year"].apply(
                lambda value: f"{int(value) // 10 * 10}s" in decades if pd.notna(value) else False
            )
        ]
    spec = _request_spec(request)
    semantic_exclusions = {concept.concept.lower() for concept in spec.semantic_exclusions} if spec is not None else set()
    if semantic_exclusions:
        def _excluded(row: pd.Series) -> bool:
            haystack = " ".join(
                [
                    str(row.get("title") or "").lower(),
                    " ".join(_parse_listish(row.get("tmdb_genres"))).lower(),
                    str(row.get("tmdb_original_language") or "").lower(),
                    " ".join(_parse_listish(row.get("tmdb_production_countries"))).lower(),
                    " ".join(_parse_listish(row.get("candidate_sources"))).lower(),
                    str(row.get("tmdb_overview") or "").lower(),
                ]
            )
            return not any(exclusion in haystack for exclusion in semantic_exclusions)
        filtered = filtered.loc[filtered.apply(_excluded, axis=1)]
    return filtered.reset_index(drop=True)


def _request_embedding_and_scores(frame: pd.DataFrame, request: RequestUnderstanding) -> tuple[np.ndarray | None, pd.DataFrame, str]:
    if frame.empty:
        return None, frame.copy(), "no_candidates"
    spec = _request_spec(request)
    if not _request_has_query_aware_signal(request):
        frame = frame.copy()
        frame["request_relevance"] = 0.0
        frame["request_relevance_score"] = 0.0
        frame["request_relevance_rank"] = np.arange(1, len(frame) + 1)
        if "predicted_preference" not in frame.columns:
            frame["predicted_preference"] = np.nan
        return None, frame, "broad"

    semantic_query_text = spec.semantic_query_text if spec is not None else request.query
    query_embedding = embed_text(semantic_query_text or request.query)
    emb_cols = _candidate_embedding_columns(frame)
    if not emb_cols:
        frame = frame.copy()
        frame["request_relevance"] = 0.0
        frame["request_relevance_score"] = 0.0
        frame["request_relevance_rank"] = np.arange(1, len(frame) + 1)
        if "predicted_preference" not in frame.columns:
            frame["predicted_preference"] = np.nan
        return query_embedding, frame, "embedding_unavailable"

    matrix = frame.loc[:, emb_cols].to_numpy(dtype=float)
    relevance = _cosine_similarity(matrix, query_embedding.reshape(1, -1)).reshape(-1)
    frame = frame.copy()
    if "predicted_preference" not in frame.columns:
        frame["predicted_preference"] = np.nan
    frame["request_relevance"] = relevance
    frame["request_relevance_score"] = relevance
    frame["request_relevance_rank"] = frame["request_relevance"].rank(method="first", ascending=False).astype(int)
    frame = frame.sort_values(["request_relevance", "predicted_preference", "source_id"], ascending=[False, False, True]).reset_index(drop=True)
    frame["request_relevance_rank"] = np.arange(1, len(frame) + 1)
    return query_embedding, frame, "query_aware_embedding" if get_api_keys().openai_api_key else "local_embedding"


def _b3_applicability(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    emb_cols = _candidate_embedding_columns(frame)
    if not emb_cols:
        frame["b3_applicability"] = pd.NA
        frame["b3_applicability_distance"] = pd.NA
        return frame
    candidate_matrix = frame.loc[:, emb_cols].to_numpy(dtype=float)
    try:
        from mvp.src.mvp_deployment import load_training_embeddings
    except Exception:
        frame["b3_applicability"] = pd.NA
        frame["b3_applicability_distance"] = pd.NA
        return frame
    training_embeddings = load_training_embeddings()
    training_cols = [column for column in training_embeddings.columns if column.startswith("emb_")]
    if not training_cols:
        frame["b3_applicability"] = pd.NA
        frame["b3_applicability_distance"] = pd.NA
        return frame
    training_matrix = training_embeddings.loc[:, training_cols].to_numpy(dtype=float)
    if candidate_matrix.shape[1] != training_matrix.shape[1]:
        width = min(candidate_matrix.shape[1], training_matrix.shape[1])
        candidate_matrix = candidate_matrix[:, :width]
        training_matrix = training_matrix[:, :width]
    similarities = _cosine_similarity(candidate_matrix, training_matrix)
    nearest_similarity = similarities.max(axis=1)
    frame["b3_applicability"] = 1.0 - nearest_similarity
    frame["b3_applicability_distance"] = 1.0 - nearest_similarity
    return frame


def discover_catalog_for_request(
    request: RequestUnderstanding,
    *,
    candidate_limit: int = 300,
    candidate_context_size: int = 50,
    candidate_pool: pd.DataFrame | None = None,
    watched_ids: Iterable[str] | None = None,
    client: TMDBClient | None = None,
    precomputed_augmentation: QueryAwareAugmentation | None = None,
) -> CatalogRetrievalResult:
    keys = get_api_keys()
    client = client or (TMDBClient(api_key=keys.tmdb_api_key) if keys.tmdb_api_key else None)

    if candidate_pool is None:
        broad_pool, pool_report = generate_candidate_pool(candidate_limit=candidate_limit)
        candidate_pool = broad_pool.copy()
    else:
        candidate_pool = candidate_pool.copy().reset_index(drop=True)
        pool_report = {"candidate_limit": int(candidate_limit), "provided_pool": int(len(candidate_pool))}

    if "source_id" not in candidate_pool.columns:
        candidate_pool["source_id"] = pd.Series(dtype=str)
    candidate_pool["source_id"] = candidate_pool["source_id"].astype(str)
    if watched_ids is not None and "source_id" in candidate_pool.columns:
        watched = {str(value) for value in watched_ids if str(value).strip()}
        watched_tmdb: set[str] = set()
        for value in watched:
            match = re.search(r"(?:tmdb:)?(\d+)$", value)
            if match:
                watched_tmdb.add(match.group(1))
        watched_title_year = {value for value in watched if "__" in value}
        title_year_mask = candidate_pool.apply(lambda row: _title_year_key(row.get("title"), row.get("year") or row.get("release_year")) in watched_title_year, axis=1)
        if "tmdb_id" in candidate_pool.columns:
            tmdb_mask = candidate_pool["tmdb_id"].astype(str).isin(watched_tmdb)
        else:
            tmdb_mask = pd.Series([False] * len(candidate_pool), index=candidate_pool.index)
        candidate_pool = candidate_pool.loc[~candidate_pool["source_id"].isin(watched) & ~tmdb_mask & ~title_year_mask].copy().reset_index(drop=True)

    augmentation = (
        precomputed_augmentation
        if precomputed_augmentation is not None
        else build_query_aware_augmentation(request, client)
    )
    augmentation_report = dict(augmentation.report)
    augmentation_frame = augmentation.frame.copy()
    if not augmentation_frame.empty:
        candidate_pool = pd.concat(
            [candidate_pool, augmentation_frame], ignore_index=True
        )
        candidate_pool["source_id"] = candidate_pool["source_id"].astype(str)

    candidate_pool = candidate_pool.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)
    watched_key_set = {str(value) for value in watched_ids if str(value).strip()} if watched_ids is not None else set()
    candidate_pool, sequel_report = _apply_series_prerequisite_filters(candidate_pool, watched_key_set)
    if "rank" not in candidate_pool.columns:
        candidate_pool["rank"] = np.arange(1, len(candidate_pool) + 1)
    else:
        candidate_pool["rank"] = pd.to_numeric(candidate_pool["rank"], errors="coerce")
        missing_rank = candidate_pool["rank"].isna()
        if missing_rank.any():
            candidate_pool.loc[missing_rank, "rank"] = np.arange(1, len(candidate_pool) + 1)[missing_rank.to_numpy()]
    if "raw_rank" not in candidate_pool.columns:
        candidate_pool["raw_rank"] = candidate_pool["rank"]
    else:
        candidate_pool["raw_rank"] = pd.to_numeric(candidate_pool["raw_rank"], errors="coerce")
        candidate_pool["raw_rank"] = candidate_pool["raw_rank"].fillna(candidate_pool["rank"])
    candidate_universe_count = int(len(candidate_pool))
    filtered_pool = _filter_hard_constraints(candidate_pool, request)

    filtered_pool = filtered_pool.loc[:, [column for column in filtered_pool.columns if not column.startswith("emb_")]].copy()
    documents = build_canonical_film_documents(filtered_pool)
    embeddings, _, embedding_report = embed_documents(documents)
    embeddings = embeddings.loc[:, [column for column in embeddings.columns if column.startswith("emb_") or column in {"source_id", "document_hash", "document_version", "embedding_model", "embedding_dim", "status"}]].copy()
    embeddings["source_id"] = embeddings["source_id"].astype(str)

    candidate_pool = filtered_pool.merge(embeddings, on="source_id", how="left", suffixes=("", "_embedding"))
    query_embedding, candidate_pool, request_relevance_method = _request_embedding_and_scores(candidate_pool, request)
    candidate_pool = _b3_applicability(candidate_pool)

    post_constraint_candidate_count = int(len(candidate_pool))
    contextual_retrieval_used = _request_has_query_aware_signal(request)
    retrieval_mode = "query_aware" if contextual_retrieval_used else "broad"
    shortlist_size = min(candidate_context_size, len(candidate_pool))

    diagnostics = CatalogRetrievalDiagnostics(
        candidate_universe_count=candidate_universe_count,
        post_constraint_candidate_count=post_constraint_candidate_count,
        contextual_retrieval_used=contextual_retrieval_used,
        candidate_context_size=candidate_context_size,
        shortlist_size=shortlist_size,
        retrieval_mode=retrieval_mode,
        request_relevance_mode=request.request_relevance_mode,
        reference_status=request.reference_status,
        reference_tmdb_id=request.reference_tmdb_id,
        reference_title=request.reference_title,
        request_relevance_method=request_relevance_method,
        query_aware_augmentation_used=bool(augmentation_report.get("query_aware_augmentation_used", False)),
        watched_excluded_count=int(len(watched_ids)) if watched_ids is not None else 0,
        candidate_limit=candidate_limit,
        b3_applicability_mode="nearest_neighbor_to_frozen_training_embeddings",
    )

    return CatalogRetrievalResult(
        request=request,
        catalog_frame=candidate_pool.reset_index(drop=True),
        documents=documents,
        request_embedding=query_embedding,
        retrieval_diagnostics=diagnostics,
        augmentation_report={**pool_report, **augmentation_report, **embedding_report, **sequel_report},
    )
