from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode, quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from mvp.src.config import get_api_keys
from mvp.src.data import load_table, normalize_column_names, rating_to_preference_weight
from mvp.src.semantics import SEMANTIC_COLUMNS, SemanticVectorStore, coerce_semantic_frame


RATED_FILM_FILTER = lambda df: (df["media_type"] == "film") & df["rating"].notna()  # noqa: E731


PRIMARY_SPLIT_PATH = Path("mvp/data/processed/primary_holdout_split.csv")
CONDITION_A_PATH = Path("mvp/data/processed/condition_a_enriched.csv")
SEMANTIC_CACHE_PATH = Path("mvp/artifacts/semantic_vectors/semantic_vectors.csv")
NON_FILM_EXCLUSIONS_PATH = Path("mvp/data/processed/non_film_exclusions.csv")
ENRICHMENT_REPORT_PATH = Path("mvp/data/processed/enrichment_report.json")
TMDB_API_BASE = "https://api.themoviedb.org/3"
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def load_letterboxd_ingestion(path: str | Path) -> pd.DataFrame:
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


def get_rated_film_subset(frame: pd.DataFrame) -> pd.DataFrame:
    rated = frame.loc[RATED_FILM_FILTER(frame)].copy()
    if "canonical_id" not in rated.columns:
        rated["canonical_id"] = rated["source_id"].astype(str)
    if "source_id" not in rated.columns:
        rated["source_id"] = rated["canonical_id"].astype(str)
    if "year" not in rated.columns and "release_year" in rated.columns:
        rated["year"] = rated["release_year"]
    if "release_year" not in rated.columns and "year" in rated.columns:
        rated["release_year"] = rated["year"]
    return rated.reset_index(drop=True)


def load_non_film_exclusions(path: str | Path = NON_FILM_EXCLUSIONS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["source_id", "title", "tmdb_media_type", "reason"])
    exclusions = pd.read_csv(path)
    if "source_id" not in exclusions.columns:
        raise KeyError("Non-film exclusions file must include a source_id column.")
    exclusions["source_id"] = exclusions["source_id"].astype(str)
    return exclusions


def filter_experimental_population(
    rated_films: pd.DataFrame,
    exclusions_path: str | Path = NON_FILM_EXCLUSIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclusions = load_non_film_exclusions(exclusions_path)
    if exclusions.empty:
        return rated_films.copy().reset_index(drop=True), exclusions
    excluded_ids = set(exclusions["source_id"].astype(str))
    filtered = rated_films.loc[~rated_films["source_id"].astype(str).isin(excluded_ids)].copy()
    return filtered.reset_index(drop=True), exclusions


def _frozen_split_bins(ratings: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(ratings, errors="coerce"),
        bins=[-1.0, 2.0, 3.0, 4.0, 5.1],
        labels=["low", "mid", "high", "top"],
        include_lowest=True,
    ).astype(str)


def freeze_primary_holdout(
    rated_films: pd.DataFrame,
    split_path: str | Path = PRIMARY_SPLIT_PATH,
    id_column: str = "source_id",
    rating_column: str = "rating",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.model_selection import train_test_split

    split_path = Path(split_path)
    frame = rated_films.copy()
    if id_column not in frame.columns:
        raise KeyError(f"Missing split key column: {id_column}")
    if rating_column not in frame.columns:
        raise KeyError(f"Missing rating column: {rating_column}")

    if split_path.exists():
        existing = pd.read_csv(split_path)
        current_ids = set(frame[id_column].astype(str))
        existing_ids = set(existing[id_column].astype(str))
        verified = existing_ids == current_ids and len(existing) == len(frame)
        if verified:
            return existing, {
                "status": "verified",
                "train_count": int((existing["split"] == "train").sum()),
                "test_count": int((existing["split"] == "test").sum()),
            }

    strata = _frozen_split_bins(frame[rating_column])
    train_idx, test_idx = train_test_split(
        frame.index,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
        stratify=strata,
    )
    split = pd.DataFrame(
        {
            id_column: pd.concat(
                [frame.loc[train_idx, id_column], frame.loc[test_idx, id_column]], ignore_index=True
            ).astype(str),
            "split": ["train"] * len(train_idx) + ["test"] * len(test_idx),
            "stratum": pd.concat([strata.loc[train_idx], strata.loc[test_idx]], ignore_index=True).astype(str),
        }
    )
    split.to_csv(split_path, index=False)
    return split, {
        "status": "frozen",
        "train_count": int((split["split"] == "train").sum()),
        "test_count": int((split["split"] == "test").sum()),
    }


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().split())


@dataclass
class TMDBClient:
    api_key: str
    timeout: int = 30
    language: str = "en-US"

    def search_movie(self, title: str, year: int | float | str | None = None) -> dict[str, Any] | None:
        params = {"api_key": self.api_key, "query": title, "language": self.language}
        if year is not None and pd.notna(year):
            params["year"] = int(float(year))
        url = f"{TMDB_API_BASE}/search/movie?{urlencode(params, quote_via=quote_plus)}"
        payload = _http_json(url, timeout=self.timeout)
        results = payload.get("results", [])
        if not results:
            return None

        title_norm = _normalize_text(title)
        best = None
        best_score = -1.0
        for result in results:
            candidate_title = _normalize_text(result.get("title") or result.get("original_title") or "")
            score = 0.0
            if candidate_title == title_norm:
                score += 3.0
            if _normalize_text(result.get("original_title") or "") == title_norm:
                score += 1.0
            result_year = None
            release_date = result.get("release_date")
            if release_date:
                try:
                    result_year = int(release_date[:4])
                except Exception:
                    result_year = None
            if year is not None and pd.notna(year) and result_year is not None:
                score += max(0.0, 2.0 - abs(int(float(year)) - result_year))
            score += float(result.get("popularity") or 0.0) / 100.0
            if score > best_score:
                best_score = score
                best = result
        if best is None:
            return None
        best["_match_score"] = best_score
        return best

    def movie_details(self, movie_id: int) -> dict[str, Any]:
        params = {"api_key": self.api_key, "language": self.language}
        url = f"{TMDB_API_BASE}/movie/{movie_id}?{urlencode(params, quote_via=quote_plus)}"
        return _http_json(url, timeout=self.timeout)

    def enrich_work(self, row: pd.Series) -> dict[str, Any]:
        match = self.search_movie(row.get("title"), row.get("year"))
        if not match:
            return {
                "tmdb_match_status": "unmatched",
                "tmdb_id": pd.NA,
                "tmdb_title": pd.NA,
                "tmdb_release_year": pd.NA,
                "tmdb_genres": pd.NA,
                "tmdb_original_language": pd.NA,
                "tmdb_production_countries": pd.NA,
                "tmdb_overview": pd.NA,
                "tmdb_match_score": pd.NA,
            }

        details = self.movie_details(int(match["id"]))
        genre_names = [genre.get("name") for genre in details.get("genres", []) if genre.get("name")]
        countries = [
            country.get("iso_3166_1") or country.get("name")
            for country in details.get("production_countries", [])
            if country.get("iso_3166_1") or country.get("name")
        ]
        release_date = details.get("release_date") or match.get("release_date")
        release_year = pd.NA
        if release_date:
            try:
                release_year = int(str(release_date)[:4])
            except Exception:
                release_year = pd.NA
        return {
            "tmdb_match_status": "matched",
            "tmdb_id": details.get("id") or match.get("id"),
            "tmdb_title": details.get("title") or match.get("title"),
            "tmdb_release_year": release_year,
            "tmdb_genres": "|".join(genre_names) if genre_names else pd.NA,
            "tmdb_original_language": details.get("original_language") or match.get("original_language"),
            "tmdb_production_countries": "|".join(countries) if countries else pd.NA,
            "tmdb_overview": details.get("overview") or match.get("overview") or pd.NA,
            "tmdb_match_score": match.get("_match_score", pd.NA),
        }


def enrich_condition_a(
    rated_films: pd.DataFrame,
    output_path: str | Path = CONDITION_A_PATH,
    cache_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = get_api_keys()
    if not keys.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required for Condition A enrichment.")
    client = TMDBClient(api_key=keys.tmdb_api_key)

    output_path = Path(output_path)
    cache_path = Path(cache_path) if cache_path else output_path
    cache = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()

    frame = rated_films.copy()
    if "canonical_id" not in frame.columns and "source_id" in frame.columns:
        frame["canonical_id"] = frame["source_id"].astype(str)
    frame["source_id"] = frame.get("source_id", frame["canonical_id"]).astype(str)

    if not cache.empty and "source_id" in cache.columns:
        cached_ids = set(cache["source_id"].astype(str))
        current_ids = set(frame["source_id"].astype(str))
        if cached_ids == current_ids and len(cache) == len(frame):
            matched = int(cache.get("tmdb_match_status", pd.Series(dtype=str)).eq("matched").sum())
            report = {
                "rated_films": int(len(frame)),
                "matched": matched,
                "coverage": float(matched / len(frame)) if len(frame) else 0.0,
                "failures": [],
            }
            return cache, report
        frame = frame.merge(
            cache.loc[:, [column for column in ["source_id", "tmdb_match_status", "tmdb_id", "tmdb_title", "tmdb_release_year", "tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview", "tmdb_match_score"] if column in cache.columns]],
            on="source_id",
            how="left",
            suffixes=("", "_cache"),
        )

    rows = []
    matched = 0
    failures: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        cached = row.to_dict()
        if pd.notna(cached.get("tmdb_match_status")):
            rows.append(cached)
            matched += int(cached.get("tmdb_match_status") == "matched")
            continue
        enriched = client.enrich_work(row)
        merged = {**cached, **enriched}
        rows.append(merged)
        if enriched["tmdb_match_status"] == "matched":
            matched += 1
        else:
            failures.append(
                {
                    "source_id": str(row["source_id"]),
                    "title": row.get("title"),
                    "year": row.get("year"),
                }
            )
        time.sleep(0.05)

    enriched_df = pd.DataFrame(rows)
    for column in ["tmdb_genres", "tmdb_original_language", "tmdb_production_countries", "tmdb_overview"]:
        if column not in enriched_df.columns:
            enriched_df[column] = pd.NA
    year_series = pd.to_numeric(
        enriched_df.get("release_year", enriched_df.get("year")), errors="coerce"
    )
    enriched_df["release_decade"] = year_series.floordiv(10).mul(10).astype("Int64").astype(str) + "s"
    enriched_df.loc[year_series.isna(), "release_decade"] = pd.NA
    enriched_df.to_csv(output_path, index=False)

    report = {
        "condition_a": {
            "rated_films": int(len(rated_films)),
            "matched": int(matched),
            "coverage": float(matched / len(rated_films)) if len(rated_films) else 0.0,
            "failures": failures,
        }
    }
    return enriched_df, report["condition_a"]


def _openai_post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
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


def _semantic_prompt(row: pd.Series) -> str:
    metadata = {
        "title": row.get("title"),
        "year": row.get("year"),
        "tmdb_genres": row.get("tmdb_genres"),
        "tmdb_original_language": row.get("tmdb_original_language"),
        "tmdb_production_countries": row.get("tmdb_production_countries"),
        "tmdb_overview": row.get("tmdb_overview"),
        "release_decade": row.get("release_decade"),
    }
    return (
        "Classify this film into the fixed 62-dimension cultural taste taxonomy. "
        "Return a JSON object with exactly the following keys and float values between 0 and 1. "
        "Do not infer sensitive or personal traits. "
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}. "
        f"Keys: {list(SEMANTIC_COLUMNS)}"
    )


def _semantic_prompt_batch(rows: list[pd.Series]) -> str:
    payload = []
    for row in rows:
        payload.append(
            {
                "canonical_id": str(row.get("canonical_id")),
                "title": row.get("title"),
                "year": row.get("year"),
                "tmdb_genres": row.get("tmdb_genres"),
                "tmdb_original_language": row.get("tmdb_original_language"),
                "tmdb_production_countries": row.get("tmdb_production_countries"),
                "tmdb_overview": row.get("tmdb_overview"),
                "release_decade": row.get("release_decade"),
            }
        )
    return (
        "Classify each film into the fixed 62-dimension cultural taste taxonomy. "
        "Return valid JSON only, as an object with an `items` array. "
        "Each item must contain `canonical_id` and a `vector` object with exactly the 62 keys below, "
        "each mapped to a float between 0 and 1. Keep the same order as the input items. "
        "Do not infer sensitive or personal traits. "
        f"Input items: {json.dumps(payload, ensure_ascii=False)}. "
        f"Keys: {list(SEMANTIC_COLUMNS)}"
    )


def _parse_semantic_response(payload: dict[str, Any]) -> dict[str, float]:
    text = None
    if "choices" in payload:
        message = payload["choices"][0].get("message", {})
        text = message.get("content")
    elif "output_text" in payload:
        text = payload["output_text"]
    if text is None:
        raise ValueError("OpenAI response did not contain text content.")
    parsed = json.loads(text)
    vector = {dimension: float(parsed.get(dimension, 0.0)) for dimension in SEMANTIC_COLUMNS}
    return {k: max(0.0, min(1.0, float(v))) for k, v in vector.items()}


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


def _classify_semantic_batch(
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
            {"role": "user", "content": _semantic_prompt_batch(batch)},
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
            ordered.append(
                {
                    "canonical_id": canonical_id,
                    **{dimension: 0.0 for dimension in SEMANTIC_COLUMNS},
                }
            )
        else:
            ordered.append(item)
    return ordered


def classify_semantic_vectors(
    enriched_films: pd.DataFrame,
    cache_path: str | Path = SEMANTIC_CACHE_PATH,
    model: str = DEFAULT_OPENAI_MODEL,
    batch_size: int = 8,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = get_api_keys()
    if not keys.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for semantic classification.")

    cache_path = Path(cache_path)
    store = SemanticVectorStore(cache_path)
    cached = store.load()
    if "canonical_id" not in enriched_films.columns:
        raise KeyError("Enriched films must include canonical_id.")

    frame = enriched_films.copy()
    frame["canonical_id"] = frame["canonical_id"].astype(str)
    if not cached.empty:
        cached = cached.loc[cached["canonical_id"].astype(str).isin(set(frame["canonical_id"].astype(str)))].copy().reset_index(drop=True)
    existing_ids = set(cached["canonical_id"].astype(str)) if not cached.empty else set()

    to_classify = frame[~frame["canonical_id"].isin(existing_ids)].copy().reset_index(drop=True)
    vectors: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    classified = 0
    rows = [row for _, row in to_classify.iterrows()]
    total_batches = max(1, int(np.ceil(len(rows) / batch_size)))
    batches = [(batch_index, rows[batch_start : batch_start + batch_size]) for batch_index, batch_start in enumerate(range(0, len(rows), batch_size), start=1)]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_classify_semantic_batch, batch, model, keys.openai_api_key): (batch_index, batch)
            for batch_index, batch in batches
        }
        for future in as_completed(future_map):
            batch_index, batch = future_map[future]
            try:
                print(f"classifying semantic batch {batch_index}/{total_batches}", flush=True)
                parsed_batch = future.result()
                vectors.extend(parsed_batch)
                classified += len(parsed_batch)
                if cached.empty:
                    combined = pd.DataFrame(vectors)
                else:
                    combined = pd.concat([cached, pd.DataFrame(vectors)], ignore_index=True)
                store.save(combined)
            except Exception as exc:
                for row in batch:
                    failures.append({"canonical_id": str(row["canonical_id"]), "title": row.get("title"), "error": str(exc)})

    if cached.empty and not vectors:
        combined = cached
    elif cached.empty:
        combined = pd.DataFrame(vectors)
    elif not vectors:
        combined = cached
    else:
        combined = pd.concat([cached, pd.DataFrame(vectors)], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["canonical_id"], keep="first")
        combined = combined.loc[combined["canonical_id"].astype(str).isin(set(frame["canonical_id"].astype(str)))].copy()
    store.save(combined)
    coverage = float(len(combined) / len(frame)) if len(frame) else 0.0
    return combined, {
        "semantic_classified": int(len(combined)),
        "semantic_total": int(len(frame)),
        "coverage": coverage,
        "newly_classified": int(classified),
        "failures": failures,
    }


def run_enrichment_preparation(
    raw_path: str | Path,
    split_path: str | Path = PRIMARY_SPLIT_PATH,
    condition_a_path: str | Path = CONDITION_A_PATH,
    semantic_cache_path: str | Path = SEMANTIC_CACHE_PATH,
    semantic_batch_size: int = 8,
    semantic_max_workers: int = 4,
) -> dict[str, Any]:
    raw = load_letterboxd_ingestion(raw_path)
    rated = get_rated_film_subset(raw)
    experimental, exclusions = filter_experimental_population(rated)

    split, split_report = freeze_primary_holdout(experimental, split_path=split_path)
    enriched, condition_a_report = enrich_condition_a(experimental, output_path=condition_a_path)
    semantic_vectors, condition_b_report = classify_semantic_vectors(
        enriched,
        cache_path=semantic_cache_path,
        batch_size=semantic_batch_size,
        max_workers=semantic_max_workers,
    )

    report = {
        "holdout": split_report,
        "condition_a": condition_a_report,
        "condition_b": condition_b_report,
        "raw_rated_films": int(len(rated)),
        "experimental_films": int(len(experimental)),
        "excluded_non_film_works": int(len(exclusions)),
        "excluded_non_film_source_ids": exclusions["source_id"].astype(str).tolist() if not exclusions.empty else [],
        "semantic_vector_rows": int(len(semantic_vectors)),
    }
    ENRICHMENT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENRICHMENT_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report
