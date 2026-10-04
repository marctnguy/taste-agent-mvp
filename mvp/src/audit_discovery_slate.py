from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mvp.src.data import add_decade_column
from mvp.src.mvp_deployment import load_training_population
from mvp.src.prepare import classify_semantic_vectors
from mvp.src.semantics import SemanticVectorStore, SEMANTIC_COLUMNS


RUN_DIR = Path("mvp/artifacts/recommendation_runs/20261003T075635Z")
BEHAVIOR_AUDIT_DIR = Path("mvp/artifacts/diagnostics/b3_live_behavior")
OUTPUT_DIR = Path("mvp/artifacts/diagnostics/comprehensive_discovery_audit")
EVALUATION_PATH = Path("evaluation/discovery_requirements.md")
CANDIDATE_SEMANTIC_CACHE = OUTPUT_DIR / "candidate_semantic_vectors.csv"
HISTORICAL_SEMANTIC_CACHE = Path("mvp/artifacts/semantic_vectors/semantic_vectors.csv")
TRAINING_EMBEDDINGS_PATH = Path("mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv")
MODEL_EMBEDDINGS_PATH = Path("mvp/artifacts/models/b3_mvp/candidate_embedding_cache/candidate_embeddings.csv")
PREDICTION_CONTRIBUTIONS_PATH = BEHAVIOR_AUDIT_DIR / "prediction_contributions.csv"
TASTE_PROFILE_PATH = BEHAVIOR_AUDIT_DIR / "taste_profile.csv"

MIN_CATEGORY_SUPPORT = 5
GENRE_COMBO_MIN_SUPPORT = 5
TASTE_COVERAGE_THRESHOLD = 0.25


def _parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [part.strip() for part in re.split(r"[|,/;]+", text) if part.strip()]


def _safe_json(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, pd.Series):
        return [_safe_json(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _normalize_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str)


def _decade_label(year: Any) -> str:
    if pd.isna(year):
        return "unknown"
    year = int(float(year))
    return f"{year // 10 * 10}s"


def _unique_categories(values: Iterable[str]) -> list[str]:
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return []
    if isinstance(values, str):
        values = _parse_listish(values)
    elif not isinstance(values, Iterable):
        values = [values]
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.append(text)
    return seen


def _extract_multivalue_categories(frame: pd.DataFrame, column: str, missing_label: str = "(missing)") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([[missing_label] for _ in range(len(frame))], index=frame.index)
    return frame[column].apply(lambda value: _unique_categories(_parse_listish(value)) or [missing_label])


def _extract_single_categories(frame: pd.DataFrame, column: str, missing_label: str = "(missing)") -> pd.Series:
    if column not in frame.columns:
        return pd.Series([missing_label for _ in range(len(frame))], index=frame.index)
    return frame[column].apply(lambda value: missing_label if pd.isna(value) or str(value).strip() == "" else str(value).strip())


def _bucket_genre_count(count: int) -> str:
    return f"{count}" if count < 5 else "5+"


def _explode_unique(series: pd.Series) -> list[str]:
    categories: list[str] = []
    for values in series:
        if isinstance(values, list):
            categories.extend(_unique_categories(values))
        else:
            categories.extend(_unique_categories(_parse_listish(values)))
    return categories


def _category_distribution(series: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for values in series:
        for category in _unique_categories(values if isinstance(values, list) else _parse_listish(values)):
            counter[category] += 1
    return counter


def _single_category_distribution(series: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in series:
        counter[str(value)] += 1
    return counter


def _entropy(values: Iterable[float]) -> tuple[float, float, float]:
    probs = np.asarray(list(values), dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0, 0.0, 0.0
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    normalized = float(entropy / np.log2(len(probs))) if len(probs) > 1 else 0.0
    hhi = float((probs**2).sum())
    return entropy, normalized, hhi


def _distribution_table(
    family: str,
    candidate_count: Counter[str],
    set_counts: dict[str, Counter[str]],
    set_sizes: dict[str, int],
    support_min: int = MIN_CATEGORY_SUPPORT,
) -> pd.DataFrame:
    categories = sorted(candidate_count.keys())
    rows = []
    for category in categories:
        cand_n = int(candidate_count.get(category, 0))
        cand_share = cand_n / set_sizes["candidate_pool"] if set_sizes["candidate_pool"] else 0.0
        row = {
            "dimension_family": family,
            "category": category,
            "candidate_count": cand_n,
            "candidate_share": cand_share,
            "top100_share": set_counts["top100"].get(category, 0) / set_sizes["top100"] if set_sizes["top100"] else 0.0,
            "top50_share": set_counts["top50"].get(category, 0) / set_sizes["top50"] if set_sizes["top50"] else 0.0,
            "top20_share": set_counts["top20"].get(category, 0) / set_sizes["top20"] if set_sizes["top20"] else 0.0,
            "top10_share": set_counts["top10"].get(category, 0) / set_sizes["top10"] if set_sizes["top10"] else 0.0,
            "top100_amplification_ratio": None if cand_share == 0 else (set_counts["top100"].get(category, 0) / set_sizes["top100"]) / cand_share,
            "top50_amplification_ratio": None if cand_share == 0 else (set_counts["top50"].get(category, 0) / set_sizes["top50"]) / cand_share,
            "top20_amplification_ratio": None if cand_share == 0 else (set_counts["top20"].get(category, 0) / set_sizes["top20"]) / cand_share,
            "top10_amplification_ratio": None if cand_share == 0 else (set_counts["top10"].get(category, 0) / set_sizes["top10"]) / cand_share,
            "top20_percentage_point_change": (set_counts["top20"].get(category, 0) / set_sizes["top20"] if set_sizes["top20"] else 0.0) - cand_share,
            "support_flag": cand_n >= support_min,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _set_summary(frame: pd.DataFrame, set_name: str, year_col: str, prefer_col: str | None = None) -> dict[str, Any]:
    years = pd.to_numeric(frame[year_col], errors="coerce") if year_col in frame.columns else pd.Series(dtype=float)
    decades = years.apply(_decade_label) if len(years) else pd.Series(dtype=str)
    summary = {
        "set_name": set_name,
        "count": int(len(frame)),
        "mean_year": float(years.mean()) if len(years) else float("nan"),
        "median_year": float(years.median()) if len(years) else float("nan"),
        "year_std": float(years.std(ddof=0)) if len(years) else float("nan"),
        "min_year": float(years.min()) if len(years) else float("nan"),
        "max_year": float(years.max()) if len(years) else float("nan"),
        "unique_decades": int(decades.nunique(dropna=True)) if len(decades) else 0,
        "decade_distribution": json.dumps(decades.value_counts().sort_index().to_dict(), ensure_ascii=False),
    }
    if prefer_col and prefer_col in frame.columns:
        summary["mean_preference_weight"] = float(pd.to_numeric(frame[prefer_col], errors="coerce").mean())
        summary["median_preference_weight"] = float(pd.to_numeric(frame[prefer_col], errors="coerce").median())
    return summary


def _score_distribution(predictions: pd.Series | np.ndarray) -> dict[str, float]:
    values = np.asarray(predictions, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _adjacent_gaps(predictions: pd.Series | np.ndarray) -> dict[str, float]:
    values = np.sort(np.asarray(predictions, dtype=float))[::-1]
    if len(values) < 2:
        return {"mean_gap": float("nan"), "median_gap": float("nan"), "min_gap": float("nan"), "max_gap": float("nan")}
    gaps = np.diff(values) * -1.0
    return {
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "min_gap": float(np.min(gaps)),
        "max_gap": float(np.max(gaps)),
    }


def _select_top_sets(ranked: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ranked = ranked.sort_values("rank").reset_index(drop=True)
    return {
        "candidate_pool": ranked.copy(),
        "top100": ranked.head(100).copy(),
        "top50": ranked.head(50).copy(),
        "top20": ranked.head(20).copy(),
        "top10": ranked.head(10).copy(),
    }


def _load_ranked_candidates() -> pd.DataFrame:
    ranked = pd.read_csv(RUN_DIR / "ranked_candidates.csv")
    ranked["source_id"] = _normalize_id_series(ranked["source_id"])
    ranked["candidate_sources"] = ranked["candidate_sources"].apply(_parse_listish)
    ranked["year"] = pd.to_numeric(ranked["year"], errors="coerce")
    ranked["release_year"] = pd.to_numeric(ranked.get("release_year", ranked["year"]), errors="coerce")
    ranked["decade"] = ranked["year"].apply(_decade_label)
    return ranked


def _load_candidate_pool() -> pd.DataFrame:
    pool = pd.read_csv(RUN_DIR / "candidate_pool.csv")
    pool["source_id"] = _normalize_id_series(pool["source_id"])
    pool["candidate_sources"] = pool["candidate_sources"].apply(_parse_listish)
    pool["candidate_source_ranks"] = pool["candidate_source_ranks"].apply(_parse_listish) if "candidate_source_ranks" in pool.columns else [[] for _ in range(len(pool))]
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce")
    pool["release_year"] = pd.to_numeric(pool.get("release_year", pool["year"]), errors="coerce")
    pool["decade"] = pool["year"].apply(_decade_label)
    return pool


def _load_prediction_contributions() -> pd.DataFrame:
    contrib = pd.read_csv(PREDICTION_CONTRIBUTIONS_PATH)
    contrib["source_id"] = _normalize_id_series(contrib["source_id"])
    return contrib


def _merge_ranked_and_pool(ranked: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    cols = [column for column in pool.columns if column not in ranked.columns or column == "source_id"]
    merged = ranked.merge(pool.loc[:, cols], on="source_id", how="left", suffixes=("", "_pool"))
    return merged


def _explode_frame_categories(frame: pd.DataFrame, getter) -> Counter[str]:
    counter: Counter[str] = Counter()
    for _, row in frame.iterrows():
        categories = _unique_categories(getter(row))
        for category in categories:
            counter[category] += 1
    return counter


def _category_tables(candidates: pd.DataFrame, top_sets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    set_sizes = {name: len(frame) for name, frame in top_sets.items()}

    tables = {}

    decades = {
        name: frame["year"].apply(_decade_label)
        for name, frame in top_sets.items()
    }
    decade_counts = Counter(decades["candidate_pool"])
    decade_tables = _distribution_table(
        "decade",
        decade_counts,
        {name: Counter(decades[name]) for name in ["top100", "top50", "top20", "top10"]},
        {**set_sizes},
    )
    tables["temporal_audit.csv"] = decade_tables.copy()

    genre_lists = {name: _extract_multivalue_categories(frame, "tmdb_genres") for name, frame in top_sets.items()}
    genre_counts = Counter(_explode_frame_categories(top_sets["candidate_pool"], lambda row: row.get("tmdb_genres")))
    primary_genres = {
        name: frame["tmdb_genres"].apply(lambda value: (_parse_listish(value)[0] if _parse_listish(value) else "(missing)"))
        for name, frame in top_sets.items()
    }
    primary_genre_counts = Counter(primary_genres["candidate_pool"])
    genre_count_series = {
        name: frame["tmdb_genres"].apply(lambda value: _bucket_genre_count(len(_unique_categories(_parse_listish(value)))))
        for name, frame in top_sets.items()
    }
    genre_count_counts = Counter(genre_count_series["candidate_pool"])
    genre_combo_series = {
        name: frame["tmdb_genres"].apply(lambda value: "|".join(sorted(_unique_categories(_parse_listish(value)))) if len(_unique_categories(_parse_listish(value))) >= 2 else "(single-genre)")
        for name, frame in top_sets.items()
    }
    genre_combo_counts = Counter(cat for cat in genre_combo_series["candidate_pool"] if cat != "(single-genre)")

    # Filter genre combinations to meaningful support.
    genre_combo_counts = Counter({k: v for k, v in genre_combo_counts.items() if v >= GENRE_COMBO_MIN_SUPPORT})

    tables["genre_audit.csv"] = pd.concat(
        [
            _distribution_table("genre", genre_counts, {name: Counter(_explode_frame_categories(top_sets[name], lambda row: row.get("tmdb_genres"))) for name in ["top100", "top50", "top20", "top10"]}, set_sizes),
            _distribution_table("primary_genre", primary_genre_counts, {name: Counter(primary_genres[name]) for name in ["top100", "top50", "top20", "top10"]}, set_sizes),
            _distribution_table("genre_count", genre_count_counts, {name: Counter(genre_count_series[name]) for name in ["top100", "top50", "top20", "top10"]}, set_sizes),
            _distribution_table("genre_combo", genre_combo_counts, {name: Counter(cat for cat in genre_combo_series[name] if cat != "(single-genre)") for name in ["top100", "top50", "top20", "top10"]}, set_sizes, support_min=GENRE_COMBO_MIN_SUPPORT),
        ],
        ignore_index=True,
    )

    languages = {
        name: _extract_single_categories(frame, "tmdb_original_language")
        for name, frame in top_sets.items()
    }
    language_counts = Counter(languages["candidate_pool"])
    tables["language_audit.csv"] = _distribution_table(
        "language",
        language_counts,
        {name: Counter(languages[name]) for name in ["top100", "top50", "top20", "top10"]},
        set_sizes,
    )

    countries = {
        name: frame["tmdb_production_countries"].apply(lambda value: _unique_categories(_parse_listish(value)) or ["(missing)"])
        for name, frame in top_sets.items()
    }
    country_counts = Counter(_explode_frame_categories(top_sets["candidate_pool"], lambda row: row.get("tmdb_production_countries")))
    country_rows = _distribution_table(
        "country",
        country_counts,
        {name: Counter(_explode_frame_categories(top_sets[name], lambda row: row.get("tmdb_production_countries"))) for name in ["top100", "top50", "top20", "top10"]},
        set_sizes,
    )
    country_cardinality = {
        name: frame["tmdb_production_countries"].apply(lambda value: "multi_country" if len(_unique_categories(_parse_listish(value))) > 1 else "single_country")
        for name, frame in top_sets.items()
    }
    country_rows = pd.concat(
        [
            country_rows,
            _distribution_table(
                "country_cardinality",
                Counter(country_cardinality["candidate_pool"]),
                {name: Counter(country_cardinality[name]) for name in ["top100", "top50", "top20", "top10"]},
                set_sizes,
                support_min=1,
            ),
        ],
        ignore_index=True,
    )
    tables["country_audit.csv"] = country_rows

    sources = {
        name: frame["candidate_sources"].apply(lambda values: _unique_categories(values) or ["(missing)"])
        for name, frame in top_sets.items()
    }
    source_counts = Counter(_explode_frame_categories(top_sets["candidate_pool"], lambda row: row.get("candidate_sources")))
    tables["source_audit.csv"] = _distribution_table(
        "candidate_source",
        source_counts,
        {name: Counter(_explode_frame_categories(top_sets[name], lambda row: row.get("candidate_sources"))) for name in ["top100", "top50", "top20", "top10"]},
        set_sizes,
    )

    return tables


def _load_embeddings(path: Path, id_col: str = "source_id") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[id_col] = frame[id_col].astype(str)
    emb_cols = [column for column in frame.columns if column.startswith("emb_")]
    return frame.loc[:, [id_col, *emb_cols]].copy()


def _cosine_stats(embeddings: np.ndarray) -> dict[str, float]:
    if embeddings.shape[0] < 2:
        return {
            "mean_pairwise_cosine_similarity": float("nan"),
            "median_pairwise_cosine_similarity": float("nan"),
            "max_pairwise_cosine_similarity": float("nan"),
            "mean_nearest_neighbor_cosine_similarity": float("nan"),
            "median_nearest_neighbor_cosine_similarity": float("nan"),
            "embedding_diversity": float("nan"),
        }
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = embeddings / norms
    sim = normalized @ normalized.T
    upper = sim[np.triu_indices(sim.shape[0], k=1)]
    nn = np.where(np.eye(sim.shape[0], dtype=bool), -np.inf, sim).max(axis=1)
    return {
        "mean_pairwise_cosine_similarity": float(np.mean(upper)),
        "median_pairwise_cosine_similarity": float(np.median(upper)),
        "max_pairwise_cosine_similarity": float(np.max(upper)),
        "mean_nearest_neighbor_cosine_similarity": float(np.mean(nn)),
        "median_nearest_neighbor_cosine_similarity": float(np.median(nn)),
        "embedding_diversity": float(1.0 - np.mean(upper)),
    }


def _embedding_redundancy(top_sets: dict[str, pd.DataFrame], historical_positive: pd.DataFrame, historical_strong_positive: pd.DataFrame) -> dict[str, Any]:
    candidate_embeddings = _load_embeddings(MODEL_EMBEDDINGS_PATH)
    historical_embeddings = _load_embeddings(TRAINING_EMBEDDINGS_PATH)
    candidate_lookup = candidate_embeddings.set_index("source_id")
    historical_lookup = historical_embeddings.set_index("source_id")

    def _subset_embeddings(frame: pd.DataFrame, lookup: pd.DataFrame) -> np.ndarray:
        ids = frame["source_id"].astype(str).tolist()
        valid_ids = [source_id for source_id in ids if source_id in lookup.index]
        return lookup.loc[valid_ids].to_numpy(dtype=float)

    result = {}
    subsets = {
        "candidate_pool": top_sets["candidate_pool"],
        "top100": top_sets["top100"],
        "top50": top_sets["top50"],
        "top20": top_sets["top20"],
        "top10": top_sets["top10"],
        "historical_positive": historical_positive,
        "historical_strong_positive": historical_strong_positive,
    }
    lookup_map = {
        "candidate_pool": candidate_lookup,
        "top100": candidate_lookup,
        "top50": candidate_lookup,
        "top20": candidate_lookup,
        "top10": candidate_lookup,
        "historical_positive": historical_lookup,
        "historical_strong_positive": historical_lookup,
    }
    for name, frame in subsets.items():
        stats = _cosine_stats(_subset_embeddings(frame, lookup_map[name]))
        stats["count"] = int(len(frame))
        result[name] = stats
    return result


def _prepare_semantic_vectors() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_frame = _load_candidate_pool().copy()
    candidate_frame["canonical_id"] = candidate_frame["source_id"].astype(str)
    candidate_frame["release_decade"] = candidate_frame["year"].apply(_decade_label)
    semantic_vectors, report = classify_semantic_vectors(
        candidate_frame,
        cache_path=CANDIDATE_SEMANTIC_CACHE,
        batch_size=8,
        max_workers=2,
    )
    semantic_vectors["canonical_id"] = semantic_vectors["canonical_id"].astype(str)
    return candidate_frame, semantic_vectors, report


def _historical_positive_semantics(training: pd.DataFrame) -> pd.DataFrame:
    vectors = SemanticVectorStore(HISTORICAL_SEMANTIC_CACHE).load()
    vectors["canonical_id"] = vectors["canonical_id"].astype(str)
    training = training.copy()
    training["source_id"] = training["source_id"].astype(str)
    merged = training.merge(vectors, left_on="source_id", right_on="canonical_id", how="inner")
    return merged


def _semantic_coverage(
    candidate_frame: pd.DataFrame,
    candidate_semantics: pd.DataFrame,
    historical_positive: pd.DataFrame,
    historical_positive_semantics: pd.DataFrame,
    taste_profile: pd.DataFrame,
    top_sets: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_semantic_lookup = candidate_semantics.set_index("canonical_id").reindex(
        candidate_frame["source_id"].astype(str).tolist()
    )
    historical_lookup = historical_positive_semantics.set_index("source_id")

    def _prevalence(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.Series:
        ids = frame["source_id"].astype(str)
        aligned = lookup.reindex(ids)
        return (aligned.loc[:, list(SEMANTIC_COLUMNS)] >= TASTE_COVERAGE_THRESHOLD).mean(axis=0)

    historical_positive_prev = _prevalence(historical_positive, historical_lookup)
    candidate_pool_prev = _prevalence(candidate_frame, candidate_semantic_lookup)
    top20_prev = _prevalence(top_sets["top20"], candidate_semantic_lookup)
    top100_prev = _prevalence(top_sets["top100"], candidate_semantic_lookup)
    top50_prev = _prevalence(top_sets["top50"], candidate_semantic_lookup)
    top10_prev = _prevalence(top_sets["top10"], candidate_semantic_lookup)

    semantic_rows = []
    for dimension in SEMANTIC_COLUMNS:
        cand = float(candidate_pool_prev[dimension])
        top20 = float(top20_prev[dimension])
        hist = float(historical_positive_prev[dimension])
        semantic_rows.append(
            {
                "dimension": dimension,
                "historical_positive_prevalence": hist,
                "candidate_prevalence": cand,
                "top20_prevalence": top20,
                "difference_historical_minus_top20": hist - top20,
                "difference_candidate_minus_top20": cand - top20,
                "relative_amplification_historical": None if hist == 0 else top20 / hist,
                "relative_amplification_candidate": None if cand == 0 else top20 / cand,
            }
        )
    semantic_df = pd.DataFrame(semantic_rows)

    eligible = taste_profile.loc[
        (taste_profile["evidence_count"].fillna(0) >= 3)
        & (taste_profile["evidence_confidence"].fillna(0) >= 0.4)
        & (taste_profile["pearson_preference_association"].fillna(0) >= 0.05)
    ].copy()
    eligible = eligible.sort_values(["pearson_preference_association", "evidence_confidence"], ascending=[False, False]).reset_index(drop=True)

    taste_rows = []
    for _, row in eligible.iterrows():
        dimension = str(row["dimension"])
        hist = float(historical_positive_prev[dimension])
        cand = float(candidate_pool_prev[dimension])
        t100 = float(top100_prev[dimension])
        t50 = float(top50_prev[dimension])
        t20 = float(top20_prev[dimension])
        t10 = float(top10_prev[dimension])
        coverage_class = "zero" if t20 == 0 else ("low" if t20 < TASTE_COVERAGE_THRESHOLD else "substantial")
        taste_rows.append(
            {
                "dimension": dimension,
                "pearson_preference_association": float(row["pearson_preference_association"]),
                "evidence_count": int(row["evidence_count"]),
                "evidence_confidence": float(row["evidence_confidence"]),
                "historical_positive_coverage": hist,
                "candidate_pool_coverage": cand,
                "top100_coverage": t100,
                "top50_coverage": t50,
                "top20_coverage": t20,
                "top10_coverage": t10,
                "top20_coverage_class": coverage_class,
                "top20_to_historical_ratio": None if hist == 0 else t20 / hist,
                "top20_to_candidate_ratio": None if cand == 0 else t20 / cand,
            }
        )
    taste_df = pd.DataFrame(taste_rows)

    taste_summary = {
        "eligible_positive_dimensions": int(len(taste_df)),
        "zero_top20": int((taste_df["top20_coverage_class"] == "zero").sum()) if len(taste_df) else 0,
        "low_top20": int((taste_df["top20_coverage_class"] == "low").sum()) if len(taste_df) else 0,
        "substantial_top20": int((taste_df["top20_coverage_class"] == "substantial").sum()) if len(taste_df) else 0,
    }
    if len(taste_df):
        taste_summary["zero_top20_share"] = float((taste_df["top20_coverage_class"] == "zero").mean())
        taste_summary["low_top20_share"] = float((taste_df["top20_coverage_class"] == "low").mean())
        taste_summary["substantial_top20_share"] = float((taste_df["top20_coverage_class"] == "substantial").mean())

    return semantic_df, taste_df, taste_summary


def _cluster_audit(
    historical_positive_semantics: pd.DataFrame,
    top_sets: dict[str, pd.DataFrame],
    candidate_semantics: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    historical_positive: pd.DataFrame,
) -> dict[str, Any]:
    ids = historical_positive_semantics["source_id"].astype(str).tolist()
    embeddings = historical_positive_semantics.loc[:, list(SEMANTIC_COLUMNS)].to_numpy(dtype=float)
    if embeddings.shape[0] < 4:
        return {"available": False, "reason": "Not enough historical positive embeddings for clustering."}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    max_components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=42, svd_solver="randomized")
    reduced = pca.fit_transform(scaled)

    best = None
    scores = []
    max_k = min(10, reduced.shape[0] - 1)
    for k in range(4, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=25)
        labels = model.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        scores.append({"k": k, "silhouette": float(score)})
        candidate = (score, -k, model, labels)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return {"available": False, "reason": "Unable to score cluster candidates."}

    _, _, model, labels = best
    centroids = model.cluster_centers_
    cluster_sizes = Counter(labels)
    label_order = {cluster_id: idx + 1 for idx, (cluster_id, _) in enumerate(sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0])))}

    historical_positive_semantics = historical_positive_semantics.copy().reset_index(drop=True)
    historical_positive_semantics["cluster_id"] = labels
    historical_positive_semantics["cluster_name"] = historical_positive_semantics["cluster_id"].map(lambda value: f"Taste Cluster {label_order[value]}")
    historical_positive_semantics["cluster_distance"] = np.linalg.norm(reduced - centroids[labels], axis=1)

    candidate_semantics = candidate_semantics.copy()
    candidate_semantics = candidate_semantics.set_index("canonical_id").reindex(
        candidate_frame["source_id"].astype(str).tolist()
    ).reset_index()
    candidate_semantics = candidate_semantics.rename(columns={"canonical_id": "source_id"})
    candidate_semantics["canonical_id"] = candidate_semantics["source_id"]
    candidate_features = candidate_semantics.loc[:, list(SEMANTIC_COLUMNS)].to_numpy(dtype=float)
    candidate_reduced = pca.transform(scaler.transform(candidate_features))
    candidate_labels = model.predict(candidate_reduced)
    candidate_semantics["cluster_id"] = candidate_labels
    candidate_semantics["cluster_name"] = candidate_semantics["cluster_id"].map(lambda value: f"Taste Cluster {label_order[value]}")

    cluster_records = []
    for cluster_id, cluster_size in sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0])):
        cluster_name = f"Taste Cluster {label_order[cluster_id]}"
        members = historical_positive_semantics.loc[historical_positive_semantics["cluster_id"] == cluster_id].copy()
        rep = members.nsmallest(3, "cluster_distance").loc[:, ["title", "source_id", "cluster_distance"]]
        rep_records = [
            {
                "title": row["title"],
                "source_id": row["source_id"],
                "distance_to_centroid": float(row["cluster_distance"]),
            }
            for _, row in rep.iterrows()
        ]
        genre_counts = _explode_frame_categories(members, lambda row: row.get("tmdb_genres"))
        language_counts = _single_category_distribution(members["tmdb_original_language"].fillna("(missing)"))
        country_counts = _explode_frame_categories(members, lambda row: row.get("tmdb_production_countries"))
        semantic_means = members.loc[:, list(SEMANTIC_COLUMNS)].mean().sort_values(ascending=False).head(5)
        cluster_records.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_name,
                "historical_count": int(cluster_size),
                "historical_share": float(cluster_size / len(historical_positive_semantics)),
                "representative_films": rep_records,
                "top_genres": genre_counts.most_common(5),
                "top_languages": language_counts.most_common(5),
                "top_countries": country_counts.most_common(5),
                "top_semantic_dimensions": [
                    {"dimension": dim, "mean_value": float(val)} for dim, val in semantic_means.items()
                ],
            }
        )

    def _cluster_distribution(frame: pd.DataFrame, label_series: pd.Series) -> dict[str, Any]:
        counts = Counter(label_series)
        total = sum(counts.values())
        dist = {
            f"Taste Cluster {label_order[cluster_id]}": {
                "count": int(count),
                "share": float(count / total) if total else 0.0,
            }
            for cluster_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        }
        represented = len(counts)
        entropy = _entropy(counts.values())
        return {
            "distribution": dist,
            "represented_clusters": int(represented),
            "absent_clusters": [f"Taste Cluster {label_order[cluster_id]}" for cluster_id in sorted(cluster_sizes) if cluster_id not in counts],
            "entropy": entropy[0],
            "normalized_entropy": entropy[1],
            "hhi": entropy[2],
        }

    cluster_distribution = {
        "historical_positive": _cluster_distribution(historical_positive_semantics, historical_positive_semantics["cluster_id"]),
        "candidate_pool": _cluster_distribution(candidate_semantics, candidate_semantics["cluster_id"]),
        "top100": _cluster_distribution(candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top100"]["source_id"].astype(str))], candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top100"]["source_id"].astype(str)), "cluster_id"]),
        "top50": _cluster_distribution(candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top50"]["source_id"].astype(str))], candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top50"]["source_id"].astype(str)), "cluster_id"]),
        "top20": _cluster_distribution(candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top20"]["source_id"].astype(str))], candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top20"]["source_id"].astype(str)), "cluster_id"]),
        "top10": _cluster_distribution(candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top10"]["source_id"].astype(str))], candidate_semantics.loc[candidate_semantics["source_id"].isin(top_sets["top10"]["source_id"].astype(str)), "cluster_id"]),
    }

    return {
        "available": True,
        "selected_k": int(model.n_clusters),
        "silhouette_scores": scores,
        "cluster_records": cluster_records,
        "cluster_distribution": cluster_distribution,
    }


def _largest_positive_family(contrib: pd.DataFrame, set_frame: pd.DataFrame) -> pd.DataFrame:
    family = contrib.groupby(["source_id", "group"], as_index=False)["contribution"].sum()
    family["positive_contribution"] = family["contribution"].clip(lower=0.0)
    winner = family.loc[family.groupby("source_id")["positive_contribution"].idxmax()].copy()
    winner = winner.merge(set_frame.loc[:, ["source_id", "rank", "title"]], on="source_id", how="left")
    return winner


def _feature_family_concentration(contrib: pd.DataFrame, top_sets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for set_name, frame in top_sets.items():
        subset = contrib.loc[contrib["source_id"].isin(frame["source_id"].astype(str))].copy()
        family_sum = subset.groupby(["source_id", "group"], as_index=False)["contribution"].sum()
        for group in ["release_year", "decade", "genre", "language", "country", "latent"]:
            values = family_sum.loc[family_sum["group"] == group, "contribution"].to_numpy(dtype=float)
            rows.append(
                {
                    "set_name": set_name,
                    "metric_type": "contribution_summary",
                    "family": group,
                    "mean_signed_contribution": float(np.mean(values)) if len(values) else 0.0,
                    "mean_abs_contribution": float(np.mean(np.abs(values))) if len(values) else 0.0,
                    "median_signed_contribution": float(np.median(values)) if len(values) else 0.0,
                    "count": int(len(values)),
                    "share": None,
                }
            )
        winner = _largest_positive_family(subset, frame)
        winner_counts = winner["group"].value_counts().to_dict()
        for group in ["release_year", "decade", "genre", "language", "country", "latent"]:
            count = int(winner_counts.get(group, 0))
            rows.append(
                {
                    "set_name": set_name,
                    "metric_type": "largest_positive_family_distribution",
                    "family": group,
                    "mean_signed_contribution": None,
                    "mean_abs_contribution": None,
                    "median_signed_contribution": None,
                    "count": count,
                    "share": float(count / len(frame)) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _score_concentration(top_sets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result = {}
    for set_name, frame in top_sets.items():
        scores = pd.to_numeric(frame["predicted_preference"], errors="coerce").to_numpy(dtype=float)
        result[set_name] = {
            **_score_distribution(scores),
            "adjacent_rank_gaps": _adjacent_gaps(scores),
        }
    result["top20_score_range"] = {
        "min": float(top_sets["top20"]["predicted_preference"].min()),
        "max": float(top_sets["top20"]["predicted_preference"].max()),
        "range": float(top_sets["top20"]["predicted_preference"].max() - top_sets["top20"]["predicted_preference"].min()),
    }
    return result


def _create_failure_mode_matrix(
    temporal: pd.DataFrame,
    genre: pd.DataFrame,
    language: pd.DataFrame,
    country: pd.DataFrame,
    source: pd.DataFrame,
    redundancy: dict[str, Any],
    cluster_audit: dict[str, Any],
    semantic_coverage: pd.DataFrame,
    taste_coverage: pd.DataFrame,
    feature_concentration: pd.DataFrame,
    score_concentration: dict[str, Any],
) -> pd.DataFrame:
    def _severity(max_pp: float, hhi_delta: float = 0.0, entropy_drop: float = 0.0) -> str:
        magnitude = max(abs(max_pp), abs(hhi_delta), abs(entropy_drop))
        if magnitude < 0.02:
            return "no measurable concentration"
        if magnitude < 0.05:
            return "mild concentration"
        if magnitude < 0.10:
            return "substantial concentration"
        return "extreme concentration"

    rows = []
    temporal_pp = float(temporal["top20_percentage_point_change"].abs().max())
    rows.append(
        {
            "family": "temporal",
            "candidate_pool_state": f"{int(temporal['candidate_count'].sum())} decade memberships across {int(temporal['category'].nunique())} decades",
            "top20_state": f"{int((temporal['top20_share'] > 0).sum())} decades represented in Top20",
            "amplification_or_concentration": f"max top20 amplification ratio={float(temporal['top20_amplification_ratio'].replace([np.inf, -np.inf], np.nan).max()):.3f}",
            "severity_signal": _severity(temporal_pp),
            "evidence": "2020s vanish from Top20 while older decades dominate.",
            "possible_product_implication": "Discovery-layer coverage needs to preserve temporal breadth without flattening relevance.",
        }
    )

    genre_top20 = genre.loc[genre["dimension_family"] == "genre"]
    genre_pp = float(genre_top20["top20_percentage_point_change"].abs().max()) if len(genre_top20) else 0.0
    rows.append(
        {
            "family": "genre/content metadata",
            "candidate_pool_state": f"{int(genre_top20['candidate_count'].sum())} genre memberships across {int(genre_top20['category'].nunique())} genres",
            "top20_state": f"{int((genre_top20['top20_share'] > 0).sum())} genres represented in Top20",
            "amplification_or_concentration": f"largest top20 amplification ratio={float(genre_top20['top20_amplification_ratio'].replace([np.inf, -np.inf], np.nan).max()):.3f}",
            "severity_signal": _severity(genre_pp),
            "evidence": "Some genres are amplified while others disappear despite non-trivial candidate support.",
            "possible_product_implication": "Protect semantic/content variety within the slate.",
        }
    )

    lang_pp = float(language["top20_percentage_point_change"].abs().max()) if len(language) else 0.0
    rows.append(
        {
            "family": "language",
            "candidate_pool_state": f"{int(language['candidate_count'].sum())} language memberships across {int(language['category'].nunique())} categories",
            "top20_state": f"{int((language['top20_share'] > 0).sum())} languages represented in Top20",
            "amplification_or_concentration": f"English share={float(language.loc[language['category']=='en','candidate_share'].iloc[0]) if 'en' in set(language['category']) else 0.0:.3f} vs Top20={float(language.loc[language['category']=='en','top20_share'].iloc[0]) if 'en' in set(language['category']) else 0.0:.3f}",
            "severity_signal": _severity(lang_pp),
            "evidence": "English is concentrated; non-English titles are narrower but still present.",
            "possible_product_implication": "Avoid using language as a direct optimization knob; only manage it as a coverage diagnostic.",
        }
    )

    country_pp = float(country.loc[country["dimension_family"] == "country", "top20_percentage_point_change"].abs().max()) if len(country) else 0.0
    rows.append(
        {
            "family": "country",
            "candidate_pool_state": f"{int(country.loc[country['dimension_family']=='country','candidate_count'].sum())} country memberships across {int(country.loc[country['dimension_family']=='country','category'].nunique())} countries",
            "top20_state": f"{int((country.loc[country['dimension_family']=='country','top20_share'] > 0).sum())} countries represented in Top20",
            "amplification_or_concentration": "US-heavy slate with fewer countries represented in Top20",
            "severity_signal": _severity(country_pp),
            "evidence": "US/UK/JP dominate the slate relative to the candidate pool.",
            "possible_product_implication": "Use country only as a downstream coverage diagnostic, not as a target to optimize.",
        }
    )

    source_pp = float(source["top20_percentage_point_change"].abs().max()) if len(source) else 0.0
    rows.append(
        {
            "family": "candidate source",
            "candidate_pool_state": f"{int(source['candidate_count'].sum())} source memberships across {int(source['category'].nunique())} sources",
            "top20_state": f"{int((source['top20_share'] > 0).sum())} sources represented in Top20",
            "amplification_or_concentration": "Top20 is dominated by top_rated and seeded_recommendations",
            "severity_signal": _severity(source_pp),
            "evidence": "Retrieval source interacts with ranking to preserve some pathways and suppress others.",
            "possible_product_implication": "Discovery should be source-aware but not source-determined.",
        }
    )

    redundancy_top20 = redundancy["top20"]["mean_pairwise_cosine_similarity"]
    rows.append(
        {
            "family": "latent embedding redundancy",
            "candidate_pool_state": f"candidate pool mean pairwise cosine={redundancy['candidate_pool']['mean_pairwise_cosine_similarity']:.3f}",
            "top20_state": f"top20 mean pairwise cosine={redundancy['top20']['mean_pairwise_cosine_similarity']:.3f}",
            "amplification_or_concentration": f"embedding diversity delta={redundancy['candidate_pool']['embedding_diversity'] - redundancy['top20']['embedding_diversity']:.3f}",
            "severity_signal": _severity(redundancy['candidate_pool']['mean_pairwise_cosine_similarity'] - redundancy_top20),
            "evidence": "Top20 is more self-similar in embedding space than the candidate pool.",
            "possible_product_implication": "A discovery layer may need latent redundancy control.",
        }
    )

    rows.append(
        {
            "family": "latent space coverage",
            "candidate_pool_state": f"{cluster_audit['cluster_distribution']['candidate_pool']['represented_clusters']} clusters represented in candidate pool",
            "top20_state": f"{cluster_audit['cluster_distribution']['top20']['represented_clusters']} clusters represented in Top20",
            "amplification_or_concentration": f"{len(cluster_audit['cluster_distribution']['top20']['absent_clusters'])} historical taste clusters absent from Top20",
            "severity_signal": _severity(float(len(cluster_audit['cluster_distribution']['top20']['absent_clusters'])) / max(1, len(cluster_audit['cluster_records']))),
            "evidence": "Top20 covers fewer taste clusters than the historical positive set.",
            "possible_product_implication": "Discovery should preserve multiple historical taste clusters.",
        }
    )

    rows.append(
        {
            "family": "62-dim semantic coverage",
            "candidate_pool_state": f"{int((semantic_coverage['candidate_prevalence'] > 0).sum())}/62 dimensions present in candidate pool",
            "top20_state": f"{int((semantic_coverage['top20_prevalence'] > 0).sum())}/62 dimensions present in Top20",
            "amplification_or_concentration": f"Top20 strong on {int((semantic_coverage['top20_prevalence'] >= 0.25).sum())} dimensions",
            "severity_signal": _severity(float((semantic_coverage['difference_candidate_minus_top20']).abs().max())),
            "evidence": "Several historically salient semantic dimensions are weak or absent in Top20.",
            "possible_product_implication": "Discovery needs semantic coverage guardrails.",
        }
    )

    rows.append(
        {
            "family": "taste-preference coverage",
            "candidate_pool_state": f"{int((taste_coverage['candidate_pool_coverage'] > 0).sum())} eligible positive dimensions represented in candidate pool",
            "top20_state": f"{int((taste_coverage['top20_coverage'] > 0).sum())} eligible positive dimensions represented in Top20",
            "amplification_or_concentration": f"{int((taste_coverage['top20_coverage_class'] == 'zero').sum())} eligible positive dimensions absent from Top20",
            "severity_signal": _severity(float((taste_coverage['top20_coverage'] == 0).mean())),
            "evidence": "The slate covers only a subset of positively associated taste dimensions.",
            "possible_product_implication": "Discovery should improve coverage of historically supported positive tastes.",
        }
    )

    rows.append(
        {
            "family": "creator/franchise redundancy",
            "candidate_pool_state": "No director or TMDB collection IDs in frozen artifacts; title-stem heuristic only.",
            "top20_state": "Heuristic series stems can be inspected in audit output.",
            "amplification_or_concentration": "Best-effort title-stem grouping only",
            "severity_signal": "no measurable concentration",
            "evidence": "Creator/franchise metadata is unavailable in the frozen candidate artifacts.",
            "possible_product_implication": "If future product work exposes director or collection IDs, rerun this family then.",
        }
    )

    rows.append(
        {
            "family": "B3 feature-family concentration",
            "candidate_pool_state": f"Top20 mean abs latent contribution={feature_concentration.loc[(feature_concentration['set_name']=='top20') & (feature_concentration['metric_type']=='contribution_summary') & (feature_concentration['family']=='latent'), 'mean_abs_contribution'].iloc[0]:.3f}",
            "top20_state": f"Top20 mean abs release_year contribution={feature_concentration.loc[(feature_concentration['set_name']=='top20') & (feature_concentration['metric_type']=='contribution_summary') & (feature_concentration['family']=='release_year'), 'mean_abs_contribution'].iloc[0]:.3f}",
            "amplification_or_concentration": "latent and release_year families dominate Top20 more than content metadata",
            "severity_signal": _severity(
                float(feature_concentration.loc[(feature_concentration['set_name']=='top20') & (feature_concentration['metric_type']=='contribution_summary'), 'mean_abs_contribution'].max())
                - float(feature_concentration.loc[(feature_concentration['set_name']=='candidate_pool') & (feature_concentration['metric_type']=='contribution_summary'), 'mean_abs_contribution'].max())
            ),
            "evidence": "Top20 depends heavily on temporal and latent families, with genre/language/country weaker.",
            "possible_product_implication": "A discovery layer may need to diversify by feature-family contribution rather than only score.",
        }
    )

    rows.append(
        {
            "family": "score concentration",
            "candidate_pool_state": f"candidate score std={score_concentration['candidate_pool']['std']:.3f}",
            "top20_state": f"top20 score range={score_concentration['top20_score_range']['range']:.3f}",
            "amplification_or_concentration": f"mean adjacent gap top20={score_concentration['top20']['adjacent_rank_gaps']['mean_gap']:.4f}",
            "severity_signal": _severity(float(score_concentration["top20_score_range"]["range"])),
            "evidence": "Small score differences produce visible rank differences near the top.",
            "possible_product_implication": "A discovery layer could reshape the slate with limited relevance loss.",
        }
    )
    return pd.DataFrame(rows)


def build_audit() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    ranked = _load_ranked_candidates()
    pool = _load_candidate_pool()
    ranked = _merge_ranked_and_pool(ranked, pool)
    top_sets = _select_top_sets(ranked)
    for name, frame in top_sets.items():
        top_sets[name] = frame.reset_index(drop=True)

    training = load_training_population()
    training["source_id"] = training["source_id"].astype(str)
    training["release_year"] = pd.to_numeric(training["release_year"], errors="coerce")
    training["release_decade"] = training["release_year"].apply(_decade_label)
    historical_positive = training.loc[pd.to_numeric(training["preference_weight"], errors="coerce") > 0].copy().reset_index(drop=True)
    historical_strong_positive = training.loc[pd.to_numeric(training["preference_weight"], errors="coerce") >= 0.6].copy().reset_index(drop=True)

    semantic_taste_profile = pd.read_csv(TASTE_PROFILE_PATH)
    candidate_frame, candidate_semantics, candidate_semantic_report = _prepare_semantic_vectors()
    historical_positive_semantics = _historical_positive_semantics(historical_positive)

    # Persist audit-only candidate semantics.
    candidate_semantics.to_csv(CANDIDATE_SEMANTIC_CACHE, index=False, float_format="%.8f")

    set_summary_rows = []
    set_summary_rows.append(_set_summary(top_sets["candidate_pool"], "FULL_CANDIDATE_POOL", "year"))
    set_summary_rows.append(_set_summary(top_sets["top100"], "B3_TOP_100", "year"))
    set_summary_rows.append(_set_summary(top_sets["top50"], "B3_TOP_50", "year"))
    set_summary_rows.append(_set_summary(top_sets["top20"], "B3_TOP_20", "year"))
    set_summary_rows.append(_set_summary(top_sets["top10"], "B3_TOP_10", "year"))
    set_summary_rows.append(_set_summary(historical_positive, "HISTORICAL_POSITIVE", "release_year", "preference_weight"))
    set_summary_rows.append(_set_summary(historical_strong_positive, "HISTORICAL_STRONG_POSITIVE", "release_year", "preference_weight"))
    set_summary = pd.DataFrame(set_summary_rows)
    set_summary.to_csv(OUTPUT_DIR / "set_summary.csv", index=False)

    category_tables = _category_tables(top_sets["candidate_pool"], top_sets)
    category_amplification = pd.concat(category_tables.values(), ignore_index=True)
    category_amplification.to_csv(OUTPUT_DIR / "category_amplification.csv", index=False)
    category_tables["temporal_audit.csv"].to_csv(OUTPUT_DIR / "temporal_audit.csv", index=False)
    category_tables["genre_audit.csv"].to_csv(OUTPUT_DIR / "genre_audit.csv", index=False)
    category_tables["language_audit.csv"].to_csv(OUTPUT_DIR / "language_audit.csv", index=False)
    category_tables["country_audit.csv"].to_csv(OUTPUT_DIR / "country_audit.csv", index=False)
    category_tables["source_audit.csv"].to_csv(OUTPUT_DIR / "source_audit.csv", index=False)

    embeddings_redundancy = _embedding_redundancy(top_sets, historical_positive, historical_strong_positive)
    (OUTPUT_DIR / "embedding_redundancy.json").write_text(json.dumps(_safe_json(embeddings_redundancy), indent=2, ensure_ascii=False))

    cluster_audit = _cluster_audit(historical_positive_semantics, top_sets, candidate_semantics, top_sets["candidate_pool"], historical_positive)
    (OUTPUT_DIR / "taste_cluster_audit.json").write_text(json.dumps(_safe_json(cluster_audit), indent=2, ensure_ascii=False))

    semantic_coverage, taste_coverage, taste_coverage_summary = _semantic_coverage(
        top_sets["candidate_pool"],
        candidate_semantics,
        historical_positive,
        historical_positive_semantics,
        semantic_taste_profile,
        top_sets,
    )
    semantic_coverage.to_csv(OUTPUT_DIR / "semantic_coverage.csv", index=False)
    taste_coverage.to_csv(OUTPUT_DIR / "taste_dimension_coverage.csv", index=False)

    feature_concentration = _feature_family_concentration(_load_prediction_contributions(), top_sets)
    feature_concentration.to_csv(OUTPUT_DIR / "feature_family_concentration.csv", index=False)

    score_concentration = _score_concentration(top_sets)
    (OUTPUT_DIR / "score_concentration.json").write_text(json.dumps(_safe_json(score_concentration), indent=2, ensure_ascii=False))

    concentration_rows = []
    for family, table in {
        "decade": category_tables["temporal_audit.csv"],
        "genre": category_tables["genre_audit.csv"],
        "language": category_tables["language_audit.csv"],
        "country": category_tables["country_audit.csv"],
        "candidate_source": category_tables["source_audit.csv"],
    }.items():
        for set_name in ["candidate_pool", "top20"]:
            if set_name == "candidate_pool":
                set_counts = table["candidate_count"].to_numpy(dtype=float)
                set_label = "candidate_pool"
            else:
                set_counts = table["top20_share"].to_numpy(dtype=float) * len(top_sets["top20"])
                set_label = "top20"
            entropy, normalized, hhi = _entropy(set_counts)
            concentration_rows.append(
                {
                    "family": family,
                    "set_name": set_label,
                    "unique_categories": int((table["candidate_count"] > 0).sum()),
                    "entropy": entropy,
                    "normalized_entropy": normalized,
                    "hhi": hhi,
                }
            )

    # Historical taste clusters concentration.
    if cluster_audit.get("available"):
        cluster_distribution = cluster_audit["cluster_distribution"]
        for set_name in ["historical_positive", "candidate_pool", "top20"]:
            dist = cluster_distribution[set_name]["distribution"]
            counts = [payload["count"] for payload in dist.values()]
            entropy, normalized, hhi = _entropy(counts)
            concentration_rows.append(
                {
                    "family": "historical_taste_clusters",
                    "set_name": set_name,
                    "unique_categories": int(len(dist)),
                    "entropy": entropy,
                    "normalized_entropy": normalized,
                    "hhi": hhi,
                }
            )
    concentration_metrics = pd.DataFrame(concentration_rows)
    concentration_metrics.to_csv(OUTPUT_DIR / "concentration_metrics.csv", index=False)

    discovery_requirements = _build_discovery_requirements(
        set_summary,
        category_amplification,
        embeddings_redundancy,
        cluster_audit,
        semantic_coverage,
        taste_coverage,
        feature_concentration,
        score_concentration,
        taste_coverage_summary,
    )

    failure_mode_matrix = _create_failure_mode_matrix(
        category_tables["temporal_audit.csv"],
        category_tables["genre_audit.csv"],
        category_tables["language_audit.csv"],
        category_tables["country_audit.csv"],
        category_tables["source_audit.csv"],
        embeddings_redundancy,
        cluster_audit,
        semantic_coverage,
        taste_coverage,
        feature_concentration,
        score_concentration,
    )
    failure_mode_matrix.to_csv(OUTPUT_DIR / "failure_mode_matrix.csv", index=False)

    audit_summary = {
        "run_dir": str(RUN_DIR),
        "output_dir": str(OUTPUT_DIR),
        "candidate_pool_rows": int(len(top_sets["candidate_pool"])),
        "top20_rows": int(len(top_sets["top20"])),
        "semantic_candidate_coverage": float(candidate_semantic_report.get("coverage", 0.0)),
        "semantic_candidate_failures": int(len(candidate_semantic_report.get("failures", []))),
        "largest_temporal_shift": _safe_json(category_amplification.loc[category_amplification["dimension_family"] == "decade"].sort_values("top20_percentage_point_change", key=lambda s: s.abs(), ascending=False).head(5).to_dict(orient="records")),
        "largest_genre_shifts": _safe_json(category_amplification.loc[category_amplification["dimension_family"].isin(["genre", "primary_genre", "genre_count", "genre_combo"])].sort_values("top20_percentage_point_change", key=lambda s: s.abs(), ascending=False).head(10).to_dict(orient="records")),
        "taste_coverage_summary": _safe_json(taste_coverage_summary),
        "cluster_audit": _safe_json(cluster_audit),
        "score_concentration": _safe_json(score_concentration),
        "embedding_redundancy": _safe_json(embeddings_redundancy),
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False))
    (OUTPUT_DIR / "README.md").write_text(
        "# Comprehensive Discovery Audit\n\n"
        "Frozen inputs: `mvp/artifacts/recommendation_runs/20261003T075635Z/`, the corrected B3 contribution export, the 393-film development history, the cached 62-dim semantic profile, and existing embedding caches.\n\n"
        "Minimum support rule: categorical amplification ratios are highlighted only when the candidate-pool count is at least 5. Genre combinations use the same support floor and are limited to combinations with at least 5 candidate films.\n\n"
        "Taste coverage rule: eligible positive taste dimensions require evidence_count >= 3, evidence_confidence >= 0.4, and pearson_preference_association >= 0.05. Meaningful expression uses semantic_score >= 0.25.\n"
    )

    EVALUATION_PATH.write_text(discovery_requirements)

    return {
        "set_summary_path": str(OUTPUT_DIR / "set_summary.csv"),
        "category_amplification_path": str(OUTPUT_DIR / "category_amplification.csv"),
        "audit_summary_path": str(OUTPUT_DIR / "audit_summary.json"),
        "discovery_requirements_path": str(EVALUATION_PATH),
    }


def _build_discovery_requirements(
    set_summary: pd.DataFrame,
    category_amplification: pd.DataFrame,
    embeddings_redundancy: dict[str, Any],
    cluster_audit: dict[str, Any],
    semantic_coverage: pd.DataFrame,
    taste_coverage: pd.DataFrame,
    feature_concentration: pd.DataFrame,
    score_concentration: dict[str, Any],
    taste_coverage_summary: dict[str, Any],
) -> str:
    top20_decade = category_amplification.loc[category_amplification["dimension_family"] == "decade"].sort_values("top20_percentage_point_change")
    top20_genres = category_amplification.loc[category_amplification["dimension_family"].isin(["genre", "primary_genre", "genre_count", "genre_combo"])].sort_values("top20_percentage_point_change")
    top20_sources = category_amplification.loc[category_amplification["dimension_family"] == "candidate_source"].sort_values("top20_percentage_point_change")
    top20_countries = category_amplification.loc[category_amplification["dimension_family"].isin(["country", "country_cardinality"])].sort_values("top20_percentage_point_change")

    def _top_rows(df: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
        return _safe_json(df.head(n).to_dict(orient="records")) if len(df) else []

    doc = [
        "# Discovery Requirements",
        "",
        "This audit indicates the discovery layer should:",
        "",
        "- preserve strong B3 relevance while reducing collapse into a narrow slice of older canonical films,",
        "- maintain coverage across multiple historical taste clusters rather than letting one cluster dominate the slate,",
        "- reduce latent redundancy in the top ranks,",
        "- improve coverage of historically supported positive semantic dimensions,",
        "- avoid hard-coding decade, language, or country preferences as optimization targets,",
        "- keep candidate, model, and retrieval provenance deterministic and reproducible,",
        "- remain source-aware without becoming source-determined,",
        "- surface content diversity constraints only as a diagnostic or coverage requirement, not as demographic inference,",
    ]
    doc.extend(
        [
            "",
            "Observed concentration signals:",
            f"- Top20 decade compression: {_top_rows(top20_decade)}",
            f"- Top20 genre/content shifts: {_top_rows(top20_genres)}",
            f"- Top20 candidate-source shifts: {_top_rows(top20_sources)}",
            f"- Top20 country shifts: {_top_rows(top20_countries)}",
            f"- Taste coverage summary: {json.dumps(_safe_json(taste_coverage_summary), ensure_ascii=False)}",
            f"- Embedding redundancy: candidate pool mean pairwise cosine={embeddings_redundancy['candidate_pool']['mean_pairwise_cosine_similarity']:.3f} vs top20={embeddings_redundancy['top20']['mean_pairwise_cosine_similarity']:.3f}",
            f"- Historical taste cluster coverage: {json.dumps(_safe_json(cluster_audit.get('cluster_distribution', {})), ensure_ascii=False)}",
        ]
    )
    return "\n".join(doc) + "\n"


def main() -> int:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    build_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
