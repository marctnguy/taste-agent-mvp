from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mvp.src.data import load_table
from mvp.src.mvp_deployment import load_training_population
from mvp.src.semantics import SEMANTIC_COLUMNS, SemanticVectorStore


RUN_DIR = Path("mvp/artifacts/recommendation_runs/20261003T075635Z")
OUTPUT_DIR = RUN_DIR / "discovery_slate_concentration_reranking"
AUDIT_DIR = Path("mvp/artifacts/diagnostics/comprehensive_discovery_audit")
HISTORICAL_SEMANTIC_PATH = Path("mvp/artifacts/semantic_vectors/semantic_vectors.csv")
CANDIDATE_EMBEDDINGS_PATH = Path("mvp/artifacts/models/b3_mvp/candidate_embedding_cache/candidate_embeddings.csv")
TASTE_PROFILE_PATH = Path("mvp/artifacts/diagnostics/b3_live_behavior/taste_profile.csv")

TOP_K = 20
LAMBDAS: "OrderedDict[str, float]" = OrderedDict(
    [
        ("RAW_B3", 0.0),
        ("CONCENTRATION_005", 0.05),
        ("CONCENTRATION_010", 0.10),
        ("CONCENTRATION_020", 0.20),
    ]
)

RUN_METADATA = {
    "ranking_model": "B3",
    "ranking_model_status": "exploratory_candidate_mvp",
    "slate_strategy": "distribution_aware_concentration_reranking",
    "slate_strategy_status": "exploratory_product_layer",
    "direct_optimization_dimensions": [
        "decade_concentration",
        "genre_concentration",
        "language_concentration",
        "country_concentration",
        "retrieval_provenance_concentration",
    ],
    "directly_optimized_for_recency": False,
    "directly_optimized_for_specific_genres": False,
    "directly_optimized_for_specific_languages": False,
    "directly_optimized_for_specific_countries": False,
    "embedding_similarity_affects_ranking": False,
    "mmr_used": False,
}

FAMILY_ORDER = ("decade", "genre", "language", "country", "source")


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


def _unique_categories(values: Iterable[str] | Any) -> list[str]:
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


def _load_ranked_candidates() -> pd.DataFrame:
    ranked = pd.read_csv(RUN_DIR / "ranked_candidates.csv")
    pool = pd.read_csv(RUN_DIR / "candidate_pool.csv")
    if "source_id" not in ranked.columns:
        raise KeyError("ranked_candidates.csv must include source_id.")
    ranked["source_id"] = ranked["source_id"].astype(str)
    pool["source_id"] = pool["source_id"].astype(str)
    if "candidate_source_ranks" not in pool.columns:
        pool["candidate_source_ranks"] = [[] for _ in range(len(pool))]
    merged = ranked.merge(
        pool.loc[:, ["source_id", "candidate_source_ranks"]],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    merged["candidate_sources"] = merged["candidate_sources"].apply(_parse_listish)
    merged["candidate_source_ranks"] = merged["candidate_source_ranks"].apply(_parse_listish)
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
    merged["release_year"] = pd.to_numeric(merged.get("release_year", merged["year"]), errors="coerce")
    merged["raw_b3_rank"] = pd.to_numeric(merged["rank"], errors="coerce").astype(int)
    merged["decade"] = merged["year"].apply(_year_to_decade)
    return merged


def _year_to_decade(year: Any) -> str:
    if pd.isna(year):
        return "(missing)"
    year = int(float(year))
    return f"{year // 10 * 10}s"


def _family_labels(row: pd.Series, family: str) -> list[str]:
    def _first_present(*columns: str) -> Any:
        for column in columns:
            if column in row.index:
                value = row.get(column)
                if value is None:
                    continue
                if isinstance(value, float) and np.isnan(value):
                    continue
                if isinstance(value, (list, tuple, set, dict)):
                    return value
                if pd.notna(value):
                    return value
        return None

    if family == "decade":
        return [_year_to_decade(row.get("year"))]
    if family == "genre":
        return _unique_categories(_parse_listish(_first_present("tmdb_genres", "genres"))) or ["(missing)"]
    if family == "language":
        value = _first_present("tmdb_original_language", "original_language")
        return [str(value).strip()] if pd.notna(value) and str(value).strip() else ["(missing)"]
    if family == "country":
        return _unique_categories(_parse_listish(_first_present("tmdb_production_countries", "production_countries"))) or ["(missing)"]
    if family == "source":
        return _unique_categories(_parse_listish(_first_present("candidate_sources"))) or ["(missing)"]
    raise KeyError(f"Unsupported family: {family}")


def _family_mass(labels: list[str]) -> dict[str, float]:
    unique = _unique_categories(labels)
    if not unique:
        return {"(missing)": 1.0}
    mass = 1.0 / len(unique)
    return {label: mass for label in unique}


def _family_state(frame: pd.DataFrame, family: str) -> tuple[Counter[str], float]:
    counts: Counter[str] = Counter()
    total_mass = 0.0
    for _, row in frame.iterrows():
        for label, mass in _family_mass(_family_labels(row, family)).items():
            counts[label] += float(mass)
            total_mass += float(mass)
    return counts, total_mass


def _hhi(counts: Counter[str], total_mass: float | None = None) -> float:
    total = float(sum(counts.values())) if total_mass is None else float(total_mass)
    if total <= 0:
        return 0.0
    shares = np.asarray([float(value) / total for value in counts.values() if float(value) > 0], dtype=float)
    if shares.size == 0:
        return 0.0
    return float(np.sum(shares**2))


def _entropy(counts: Counter[str], total_mass: float | None = None) -> tuple[float, float, float]:
    total = float(sum(counts.values())) if total_mass is None else float(total_mass)
    if total <= 0:
        return 0.0, 0.0, 0.0
    shares = np.asarray([float(value) / total for value in counts.values() if float(value) > 0], dtype=float)
    if shares.size == 0:
        return 0.0, 0.0, 0.0
    entropy = float(-(shares * np.log2(shares)).sum())
    normalized = float(entropy / np.log2(len(shares))) if len(shares) > 1 else 0.0
    hhi = float(np.sum(shares**2))
    return entropy, normalized, hhi


def _delta_hhi(current_counts: Counter[str], current_mass: float, addition: dict[str, float]) -> float:
    before = _hhi(current_counts, current_mass)
    updated = Counter(current_counts)
    for label, mass in addition.items():
        updated[label] += float(mass)
    after = _hhi(updated, current_mass + 1.0)
    return float(after - before)


def _reference_scales(frame: pd.DataFrame, raw_top: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    scales: dict[str, float] = {}
    raw_prefix = raw_top.head(TOP_K).reset_index(drop=True)
    for family in FAMILY_ORDER:
        positive_deltas: list[float] = []
        for prefix_size in range(1, len(raw_prefix)):
            current = raw_prefix.iloc[:prefix_size]
            current_counts, current_mass = _family_state(current, family)
            remaining = frame.loc[~frame["source_id"].isin(current["source_id"])].copy()
            for _, candidate in remaining.iterrows():
                delta = _delta_hhi(current_counts, current_mass, _family_mass(_family_labels(candidate, family)))
                if delta > 0:
                    positive_deltas.append(delta)
        scale = float(max(positive_deltas)) if positive_deltas else 1.0
        scales[family] = scale if scale > 0 else 1.0
        arr = np.asarray(positive_deltas, dtype=float)
        rows.append(
            {
                "family": family,
                "reference_scale": scales[family],
                "positive_delta_count": int(arr.size),
                "positive_delta_mean": float(arr.mean()) if arr.size else 0.0,
                "positive_delta_median": float(np.median(arr)) if arr.size else 0.0,
                "positive_delta_p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
                "positive_delta_max": float(arr.max()) if arr.size else 0.0,
                "positive_delta_min": float(arr.min()) if arr.size else 0.0,
            }
        )
    return scales, pd.DataFrame(rows)


def _select_raw_b3(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.sort_values(["raw_b3_rank", "predicted_preference", "source_id"], ascending=[True, False, True]).head(TOP_K).copy().reset_index(drop=True)
    rows = []
    for discovery_rank, (_, row) in enumerate(selected.iterrows(), start=1):
        rows.append(
            _build_selected_row(
                row,
                slate_name="RAW_B3",
                lambda_value=0.0,
                discovery_rank=discovery_rank,
                discovery_score=float(row["relevance_normalized"]),
                concentration_penalty=0.0,
                family_penalties={family: 0.0 for family in FAMILY_ORDER},
                family_deltas={family: 0.0 for family in FAMILY_ORDER},
            )
        )
    return pd.DataFrame(rows).sort_values("discovery_rank").reset_index(drop=True)


def _score_candidate(
    candidate: pd.Series,
    current_slate: pd.DataFrame,
    scales: dict[str, float],
    lambda_value: float,
) -> dict[str, Any]:
    penalties: dict[str, float] = {}
    deltas: dict[str, float] = {}
    current_mass_cache: dict[str, tuple[Counter[str], float]] = {}
    for family in FAMILY_ORDER:
        current_counts, current_mass = current_mass_cache.get(family) or _family_state(current_slate, family)
        current_mass_cache[family] = (current_counts, current_mass)
        delta = _delta_hhi(current_counts, current_mass, _family_mass(_family_labels(candidate, family)))
        deltas[family] = float(delta)
        penalties[family] = float(max(delta, 0.0) / max(scales.get(family, 1.0), 1e-12))
    concentration_penalty = float(np.mean(list(penalties.values()))) if penalties else 0.0
    return {
        "concentration_penalty": concentration_penalty,
        "family_penalties": penalties,
        "family_deltas": deltas,
        "discovery_score": float(candidate["relevance_normalized"] - lambda_value * concentration_penalty),
    }


def _select_greedy(frame: pd.DataFrame, lambda_value: float, scales: dict[str, float], slate_name: str) -> pd.DataFrame:
    remaining = frame.copy().reset_index(drop=True)
    raw_first = remaining.sort_values(["raw_b3_rank", "predicted_preference", "source_id"], ascending=[True, False, True]).iloc[0]
    selected_rows = []
    selected_rows.append(
        _build_selected_row(
            raw_first,
            slate_name=slate_name,
            lambda_value=lambda_value,
            discovery_rank=1,
            discovery_score=float(raw_first["relevance_normalized"]),
            concentration_penalty=0.0,
            family_penalties={family: 0.0 for family in FAMILY_ORDER},
            family_deltas={family: 0.0 for family in FAMILY_ORDER},
        )
    )
    selected_ids = {str(raw_first["source_id"])}
    for discovery_rank in range(2, TOP_K + 1):
        current_slate = remaining.loc[remaining["source_id"].isin(selected_ids)].copy()
        candidates = remaining.loc[~remaining["source_id"].isin(selected_ids)].copy()
        score_rows = []
        for _, candidate in candidates.iterrows():
            score = _score_candidate(candidate, current_slate, scales, lambda_value)
            score_rows.append(
                {
                    "source_id": candidate["source_id"],
                    "predicted_preference": float(candidate["predicted_preference"]),
                    "relevance_normalized": float(candidate["relevance_normalized"]),
                    "raw_b3_rank": int(candidate["raw_b3_rank"]),
                    **score,
                }
            )
        score_df = pd.DataFrame(score_rows)
        score_df = score_df.sort_values(
            ["discovery_score", "relevance_normalized", "predicted_preference", "raw_b3_rank", "source_id"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
        chosen = score_df.iloc[0]
        chosen_row = remaining.loc[remaining["source_id"] == chosen["source_id"]].iloc[0]
        selected_ids.add(str(chosen["source_id"]))
        selected_rows.append(
            _build_selected_row(
                chosen_row,
                slate_name=slate_name,
                lambda_value=lambda_value,
                discovery_rank=discovery_rank,
                discovery_score=float(chosen["discovery_score"]),
                concentration_penalty=float(chosen["concentration_penalty"]),
                family_penalties={family: float(chosen["family_penalties"][family]) for family in FAMILY_ORDER},
                family_deltas={family: float(chosen["family_deltas"][family]) for family in FAMILY_ORDER},
            )
        )
    return pd.DataFrame(selected_rows).sort_values("discovery_rank").reset_index(drop=True)


def _build_selected_row(
    candidate: pd.Series,
    slate_name: str,
    lambda_value: float,
    discovery_rank: int,
    discovery_score: float,
    concentration_penalty: float,
    family_penalties: dict[str, float],
    family_deltas: dict[str, float],
) -> dict[str, Any]:
    return {
        "slate_name": slate_name,
        "discovery_lambda": float(lambda_value),
        "discovery_rank": int(discovery_rank),
        "raw_b3_rank": int(candidate["raw_b3_rank"]),
        "source_id": str(candidate["source_id"]),
        "tmdb_id": candidate.get("tmdb_id"),
        "title": candidate.get("title"),
        "year": _safe_json(candidate.get("year")),
        "predicted_preference": float(candidate["predicted_preference"]),
        "relevance_normalized": float(candidate["relevance_normalized"]),
        "discovery_score": float(discovery_score),
        "concentration_penalty": float(concentration_penalty),
        "decade_penalty": float(family_penalties["decade"]),
        "genre_penalty": float(family_penalties["genre"]),
        "language_penalty": float(family_penalties["language"]),
        "country_penalty": float(family_penalties["country"]),
        "source_penalty": float(family_penalties["source"]),
        "decade_delta_hhi": float(family_deltas["decade"]),
        "genre_delta_hhi": float(family_deltas["genre"]),
        "language_delta_hhi": float(family_deltas["language"]),
        "country_delta_hhi": float(family_deltas["country"]),
        "source_delta_hhi": float(family_deltas["source"]),
        "decade": candidate.get("decade"),
        "tmdb_genres": json.dumps(_parse_listish(candidate.get("tmdb_genres")), ensure_ascii=False),
        "genres": json.dumps(_parse_listish(candidate.get("tmdb_genres")), ensure_ascii=False),
        "tmdb_original_language": candidate.get("tmdb_original_language"),
        "original_language": candidate.get("tmdb_original_language"),
        "tmdb_production_countries": json.dumps(_parse_listish(candidate.get("tmdb_production_countries")), ensure_ascii=False),
        "production_countries": json.dumps(_parse_listish(candidate.get("tmdb_production_countries")), ensure_ascii=False),
        "candidate_sources": json.dumps(candidate.get("candidate_sources") if isinstance(candidate.get("candidate_sources"), list) else _parse_listish(candidate.get("candidate_sources")), ensure_ascii=False),
        "candidate_source_ranks": json.dumps(candidate.get("candidate_source_ranks") if isinstance(candidate.get("candidate_source_ranks"), list) else _parse_listish(candidate.get("candidate_source_ranks")), ensure_ascii=False),
    }


def _normalize_relevance(frame: pd.DataFrame) -> pd.DataFrame:
    pred = pd.to_numeric(frame["predicted_preference"], errors="coerce")
    min_pred = float(pred.min())
    max_pred = float(pred.max())
    if math.isclose(max_pred, min_pred):
        normalized = pd.Series(0.0, index=frame.index)
    else:
        normalized = (pred - min_pred) / (max_pred - min_pred)
    frame = frame.copy()
    frame["relevance_normalized"] = normalized.fillna(0.0)
    frame["raw_b3_rank"] = pd.to_numeric(frame["raw_b3_rank"], errors="coerce").astype(int)
    frame["predicted_preference"] = pred.astype(float)
    return frame


def _distribution_counts(frame: pd.DataFrame, family: str) -> tuple[Counter[str], float]:
    counts: Counter[str] = Counter()
    total_mass = 0.0
    for _, row in frame.iterrows():
        for label, mass in _family_mass(_family_labels(row, family)).items():
            counts[label] += float(mass)
            total_mass += float(mass)
    return counts, total_mass


def _distribution_frame(frame: pd.DataFrame, slate_name: str, family: str) -> pd.DataFrame:
    counts, total_mass = _distribution_counts(frame, family)
    rows = []
    for label, mass in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "slate_name": slate_name,
                "family": family,
                "category": label,
                "mass": float(mass),
                "share": float(mass / total_mass) if total_mass else 0.0,
                "total_mass": float(total_mass),
            }
        )
    return pd.DataFrame(rows)


def _family_summary(frame: pd.DataFrame, slate_name: str) -> pd.DataFrame:
    rows = []
    for family in FAMILY_ORDER:
        counts, total_mass = _distribution_counts(frame, family)
        entropy, normalized_entropy, hhi = _entropy(counts, total_mass)
        rows.append(
            {
                "slate_name": slate_name,
                "family": family,
                "unique_categories": int(sum(1 for value in counts.values() if value > 0)),
                "entropy": float(entropy),
                "normalized_entropy": float(normalized_entropy),
                "hhi": float(hhi),
            }
        )
    return pd.DataFrame(rows)


def _embedding_stats(frame: pd.DataFrame, embeddings: pd.DataFrame) -> dict[str, float]:
    emb_cols = [column for column in embeddings.columns if column.startswith("emb_")]
    lookup = embeddings.loc[:, ["source_id", *emb_cols]].set_index("source_id")
    ids = [str(value) for value in frame["source_id"].tolist() if str(value) in lookup.index]
    if len(ids) < 2:
        return {
            "mean_pairwise_cosine_similarity": float("nan"),
            "median_pairwise_cosine_similarity": float("nan"),
            "max_pairwise_cosine_similarity": float("nan"),
            "mean_nearest_neighbor_cosine_similarity": float("nan"),
            "median_nearest_neighbor_cosine_similarity": float("nan"),
            "embedding_diversity": float("nan"),
        }
    matrix = lookup.loc[ids].to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / norms
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


def _fit_cluster_model(historical_positive_semantics: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    embeddings = historical_positive_semantics.loc[:, list(SEMANTIC_COLUMNS)].to_numpy(dtype=float)
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
        scores.append({"k": int(k), "silhouette": float(score)})
        candidate = (score, -k, model, labels)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to fit a cluster model for historical taste diagnostics.")

    _, _, model, labels = best
    cluster_sizes = Counter(labels)
    label_order = {cluster_id: idx + 1 for idx, (cluster_id, _) in enumerate(sorted(cluster_sizes.items(), key=lambda item: (-item[1], item[0])))}

    historical = historical_positive_semantics.copy().reset_index(drop=True)
    historical["cluster_id"] = labels
    historical["cluster_name"] = historical["cluster_id"].map(lambda value: f"Taste Cluster {label_order[value]}")
    historical["cluster_distance"] = np.linalg.norm(reduced - model.cluster_centers_[labels], axis=1)

    return {
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "cluster_sizes": cluster_sizes,
        "label_order": label_order,
        "selected_k": int(model.n_clusters),
        "silhouette_scores": scores,
        "historical_positive": historical,
    }, historical


def _cluster_distribution(frame: pd.DataFrame, cluster_model: dict[str, Any], selected: pd.DataFrame) -> dict[str, Any]:
    scaler = cluster_model["scaler"]
    pca = cluster_model["pca"]
    model = cluster_model["model"]
    label_order = cluster_model["label_order"]
    features = selected.loc[:, list(SEMANTIC_COLUMNS)].to_numpy(dtype=float)
    reduced = pca.transform(scaler.transform(features))
    labels = model.predict(reduced)
    selected = selected.copy().reset_index(drop=True)
    selected["cluster_id"] = labels
    selected["cluster_name"] = selected["cluster_id"].map(lambda value: f"Taste Cluster {label_order[value]}")

    counts = Counter(labels)
    total = sum(counts.values())
    dist = {
        f"Taste Cluster {label_order[cluster_id]}": {"count": int(count), "share": float(count / total) if total else 0.0}
        for cluster_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }
    entropy, normalized_entropy, hhi = _entropy(Counter({key: value["count"] for key, value in dist.items()}))
    return {
        "distribution": dist,
        "represented_clusters": int(len(counts)),
        "absent_clusters": [f"Taste Cluster {label_order[cluster_id]}" for cluster_id in sorted(cluster_model["cluster_sizes"]) if cluster_id not in counts],
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "hhi": float(hhi),
    }, selected


def _semantic_coverage(selected: pd.DataFrame, candidate_semantics: pd.DataFrame, taste_profile: pd.DataFrame) -> dict[str, Any]:
    candidate_lookup = candidate_semantics.set_index("canonical_id")
    selected_ids = selected["source_id"].astype(str).tolist()
    selected_vectors = candidate_lookup.reindex(selected_ids)
    if selected_vectors.empty:
        raise RuntimeError("No selected semantic vectors were found.")
    threshold = 0.25
    semantic_represented = int((selected_vectors.loc[:, list(SEMANTIC_COLUMNS)] >= threshold).any(axis=0).sum())

    eligible = taste_profile.loc[
        (pd.to_numeric(taste_profile["evidence_count"], errors="coerce").fillna(0) >= 3)
        & (pd.to_numeric(taste_profile["evidence_confidence"], errors="coerce").fillna(0) >= 0.4)
        & (pd.to_numeric(taste_profile["pearson_preference_association"], errors="coerce").fillna(0) >= 0.05)
    ].copy()
    selected_strength = (selected_vectors.loc[:, list(SEMANTIC_COLUMNS)] >= threshold).mean(axis=0)
    taste_rows = []
    for _, row in eligible.sort_values(["pearson_preference_association", "evidence_confidence"], ascending=[False, False]).iterrows():
        dimension = str(row["dimension"])
        taste_rows.append(
            {
                "dimension": dimension,
                "pearson_preference_association": float(row["pearson_preference_association"]),
                "evidence_count": int(row["evidence_count"]),
                "evidence_confidence": float(row["evidence_confidence"]),
                "selected_coverage": float(selected_strength[dimension]),
                "represented": bool(selected_strength[dimension] > 0),
            }
        )
    taste_df = pd.DataFrame(taste_rows)
    return {
        "semantic_dimensions_represented": semantic_represented,
        "taste_dimensions_represented": int((taste_df["selected_coverage"] > 0).sum()) if len(taste_df) else 0,
        "taste_coverage": taste_df,
    }


def _selection_diagnostics(
    slate_name: str,
    slate: pd.DataFrame,
    raw_slate: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    embeddings: pd.DataFrame,
    candidate_semantics: pd.DataFrame,
    cluster_model: dict[str, Any],
    taste_profile: pd.DataFrame,
) -> dict[str, Any]:
    raw_pred = pd.to_numeric(raw_slate["predicted_preference"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(slate["predicted_preference"], errors="coerce").to_numpy(dtype=float)
    raw_relevance = pd.to_numeric(raw_slate["relevance_normalized"], errors="coerce").to_numpy(dtype=float)
    relevance = pd.to_numeric(slate["relevance_normalized"], errors="coerce").to_numpy(dtype=float)
    diag = {
        "slate_name": slate_name,
        "selected_count": int(len(slate)),
        "mean_predicted_preference": float(np.mean(pred)),
        "median_predicted_preference": float(np.median(pred)),
        "min_predicted_preference": float(np.min(pred)),
        "max_predicted_preference": float(np.max(pred)),
        "cumulative_predicted_preference": float(np.sum(pred)),
        "mean_normalized_relevance": float(np.mean(relevance)),
        "mean_raw_b3_rank": float(pd.to_numeric(slate["raw_b3_rank"], errors="coerce").mean()),
        "median_raw_b3_rank": float(pd.to_numeric(slate["raw_b3_rank"], errors="coerce").median()),
        "worst_raw_b3_rank": int(pd.to_numeric(slate["raw_b3_rank"], errors="coerce").max()),
        "embedding": _embedding_stats(slate, embeddings),
    }
    cluster_distribution, clustered = _cluster_distribution(raw_slate, cluster_model, selected=_selected_semantics(candidate_semantics, slate))
    semantic_summary = _semantic_coverage(slate, candidate_semantics, taste_profile)
    diag["historical_taste_clusters"] = cluster_distribution
    diag["semantic"] = {
        "semantic_dimensions_represented": semantic_summary["semantic_dimensions_represented"],
        "taste_dimensions_represented": semantic_summary["taste_dimensions_represented"],
    }
    diag["taste_coverage"] = semantic_summary["taste_coverage"].to_dict(orient="records")
    diag["relevance_deltas_vs_raw"] = {
        "mean_relevance_delta": float(np.mean(relevance) - np.mean(raw_relevance)),
        "relative_mean_relevance_change": float((np.mean(relevance) - np.mean(raw_relevance)) / np.mean(raw_relevance)) if np.mean(raw_relevance) else float("nan"),
        "cumulative_relevance_delta": float(np.sum(relevance) - np.sum(raw_relevance)),
    }
    return diag


def _selected_semantics(candidate_semantics: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    candidate_lookup = candidate_semantics.set_index("canonical_id")
    ids = slate["source_id"].astype(str).tolist()
    selected = candidate_lookup.reindex(ids).reset_index().rename(columns={"index": "canonical_id"})
    selected = selected.rename(columns={"canonical_id": "source_id"})
    return selected


def _promotion_tables(discovery_slate: pd.DataFrame, raw_slate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_ids = set(raw_slate["source_id"].astype(str))
    discovery_ids = set(discovery_slate["source_id"].astype(str))
    promoted = discovery_slate.loc[~discovery_slate["source_id"].astype(str).isin(raw_ids)].copy()
    removed = raw_slate.loc[~raw_slate["source_id"].astype(str).isin(discovery_ids)].copy()
    raw_by_rank = raw_slate.sort_values("discovery_rank").reset_index(drop=True).copy()
    displaced = raw_by_rank.loc[:, ["discovery_rank", "source_id", "title", "predicted_preference"]].rename(
        columns={
            "source_id": "displaced_source_id",
            "title": "displaced_title",
            "predicted_preference": "displaced_predicted_preference",
        }
    )
    promoted = promoted.merge(displaced, on="discovery_rank", how="left", validate="one_to_one")
    promoted["relevance_cost"] = promoted["displaced_predicted_preference"] - promoted["predicted_preference"]
    promoted["raw_rank"] = promoted["raw_b3_rank"]
    return promoted, removed, promoted.loc[:, [
        "discovery_rank",
        "source_id",
        "title",
        "year",
        "raw_b3_rank",
        "predicted_preference",
        "displaced_source_id",
        "displaced_title",
        "displaced_predicted_preference",
        "relevance_cost",
        "decade_penalty",
        "genre_penalty",
        "language_penalty",
        "country_penalty",
        "source_penalty",
        "candidate_sources",
        "genres",
        "original_language",
        "production_countries",
    ]]


def _slate_metrics(slate: pd.DataFrame, raw_slate: pd.DataFrame, slate_name: str) -> dict[str, Any]:
    pred = pd.to_numeric(slate["predicted_preference"], errors="coerce").to_numpy(dtype=float)
    raw_pred = pd.to_numeric(raw_slate["predicted_preference"], errors="coerce").to_numpy(dtype=float)
    relevance = pd.to_numeric(slate["relevance_normalized"], errors="coerce").to_numpy(dtype=float)
    raw_relevance = pd.to_numeric(raw_slate["relevance_normalized"], errors="coerce").to_numpy(dtype=float)
    return {
        "slate_name": slate_name,
        "mean_predicted_preference": float(np.mean(pred)),
        "median_predicted_preference": float(np.median(pred)),
        "min_predicted_preference": float(np.min(pred)),
        "max_predicted_preference": float(np.max(pred)),
        "cumulative_predicted_preference": float(np.sum(pred)),
        "mean_normalized_relevance": float(np.mean(relevance)),
        "mean_raw_b3_rank": float(pd.to_numeric(slate["raw_b3_rank"], errors="coerce").mean()),
        "median_raw_b3_rank": float(pd.to_numeric(slate["raw_b3_rank"], errors="coerce").median()),
        "worst_raw_b3_rank": int(pd.to_numeric(slate["raw_b3_rank"], errors="coerce").max()),
        "mean_relevance_delta": float(np.mean(relevance) - np.mean(raw_relevance)),
        "relative_mean_relevance_change": float((np.mean(relevance) - np.mean(raw_relevance)) / np.mean(raw_relevance)) if np.mean(raw_relevance) else float("nan"),
        "cumulative_relevance_delta": float(np.sum(relevance) - np.sum(raw_relevance)),
    }


def run_discovery_slate_experiment(output_dir: str | Path | None = None) -> dict[str, Any]:
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = _normalize_relevance(_load_ranked_candidates())
    raw_slate = _select_raw_b3(ranked)
    scales, scale_summary = _reference_scales(ranked, raw_slate)

    candidate_semantics = SemanticVectorStore(AUDIT_DIR / "candidate_semantic_vectors.csv").load()
    candidate_semantics["canonical_id"] = candidate_semantics["canonical_id"].astype(str)
    embeddings = pd.read_csv(CANDIDATE_EMBEDDINGS_PATH)
    embeddings["source_id"] = embeddings["source_id"].astype(str)
    taste_profile = pd.read_csv(TASTE_PROFILE_PATH)

    historical = load_training_population().copy()
    historical["source_id"] = historical["source_id"].astype(str)
    historical["preference_weight"] = pd.to_numeric(historical["preference_weight"], errors="coerce")
    historical_positive = historical.loc[historical["preference_weight"] > 0].copy()
    historical_semantics = SemanticVectorStore(HISTORICAL_SEMANTIC_PATH).load()
    historical_semantics["canonical_id"] = historical_semantics["canonical_id"].astype(str)
    historical_positive_semantics = historical_positive.merge(
        historical_semantics,
        left_on="source_id",
        right_on="canonical_id",
        how="inner",
        validate="one_to_one",
    )
    cluster_model, _ = _fit_cluster_model(historical_positive_semantics)

    slates: dict[str, pd.DataFrame] = {"RAW_B3": raw_slate.copy()}
    for slate_name, lambda_value in list(LAMBDAS.items())[1:]:
        slates[slate_name] = _select_greedy(ranked, lambda_value=lambda_value, scales=scales, slate_name=slate_name)

    for slate_name, slate in slates.items():
        slate_path = out_dir / f"{slate_name.lower()}_top20.csv"
        slate.to_csv(slate_path, index=False)

    selection_traces = pd.concat(list(slates.values()), ignore_index=True)
    selection_traces.to_csv(out_dir / "selection_traces.csv", index=False)
    scale_summary.to_csv(out_dir / "family_penalty_scales.csv", index=False)

    family_distributions = []
    family_summaries = []
    for slate_name, slate in slates.items():
        for family in FAMILY_ORDER:
            family_distributions.append(_distribution_frame(slate, slate_name, family))
        family_summaries.append(_family_summary(slate, slate_name))
    family_distribution_df = pd.concat(family_distributions, ignore_index=True)
    family_summary_df = pd.concat(family_summaries, ignore_index=True)
    raw_summary = family_summary_df.loc[family_summary_df["slate_name"] == "RAW_B3", ["family", "entropy", "hhi"]].rename(
        columns={"entropy": "raw_entropy", "hhi": "raw_hhi"}
    )
    family_summary_df = family_summary_df.merge(raw_summary, on="family", how="left", validate="many_to_one")
    family_summary_df["entropy_delta_vs_raw"] = family_summary_df["entropy"] - family_summary_df["raw_entropy"]
    family_summary_df["hhi_delta_vs_raw"] = family_summary_df["hhi"] - family_summary_df["raw_hhi"]
    family_summary_df.to_csv(out_dir / "family_concentration_summary.csv", index=False)
    family_distribution_df.to_csv(out_dir / "family_category_distributions.csv", index=False)

    slate_metrics = []
    safety_diag = {}
    raw_selected_semantics = _selected_semantics(candidate_semantics, raw_slate)
    raw_embedding_stats = _embedding_stats(raw_slate, embeddings)
    raw_cluster_distribution, _ = _cluster_distribution(raw_slate, cluster_model, raw_selected_semantics)
    raw_semantic_summary = _semantic_coverage(raw_slate, candidate_semantics, taste_profile)
    raw_safety = {
        "embedding": raw_embedding_stats,
        "historical_taste_clusters": raw_cluster_distribution,
        "semantic": {
            "semantic_dimensions_represented": raw_semantic_summary["semantic_dimensions_represented"],
            "taste_dimensions_represented": raw_semantic_summary["taste_dimensions_represented"],
        },
        "taste_coverage": raw_semantic_summary["taste_coverage"].to_dict(orient="records"),
    }
    safety_diag["RAW_B3"] = raw_safety

    for slate_name, slate in slates.items():
        metrics = _slate_metrics(slate, raw_slate, slate_name)
        slate_metrics.append(metrics)
        if slate_name != "RAW_B3":
            safety = _selection_diagnostics(slate_name, slate, raw_slate, ranked, embeddings, candidate_semantics, cluster_model, taste_profile)
            safety_diag[slate_name] = safety

    slate_metrics_df = pd.DataFrame(slate_metrics)
    slate_metrics_df.to_csv(out_dir / "slate_metrics.csv", index=False)

    promoted_010, removed_010, promoted_diag_010 = _promotion_tables(slates["CONCENTRATION_010"], raw_slate)
    promoted_010.to_csv(out_dir / "promoted_lambda_010.csv", index=False)
    removed_010.to_csv(out_dir / "removed_lambda_010.csv", index=False)
    promoted_diag_010.to_csv(out_dir / "score_gap_opportunity_lambda_010.csv", index=False)

    comparison = []
    raw_metrics = slate_metrics_df.loc[slate_metrics_df["slate_name"] == "RAW_B3"].iloc[0].to_dict()
    for _, row in slate_metrics_df.iterrows():
        comparison.append(
            {
                "slate_name": row["slate_name"],
                "lambda": float(LAMBDAS[row["slate_name"]]),
                "mean_predicted_preference": row["mean_predicted_preference"],
                "mean_predicted_preference_delta_vs_raw": float(row["mean_predicted_preference"] - raw_metrics["mean_predicted_preference"]),
                "median_predicted_preference": row["median_predicted_preference"],
                "min_predicted_preference": row["min_predicted_preference"],
                "max_predicted_preference": row["max_predicted_preference"],
                "cumulative_predicted_preference": row["cumulative_predicted_preference"],
                "mean_normalized_relevance": row["mean_normalized_relevance"],
                "mean_normalized_relevance_delta_vs_raw": float(row["mean_normalized_relevance"] - raw_metrics["mean_normalized_relevance"]),
                "mean_raw_b3_rank": row["mean_raw_b3_rank"],
                "median_raw_b3_rank": row["median_raw_b3_rank"],
                "worst_raw_b3_rank": row["worst_raw_b3_rank"],
                "mean_relevance_delta": row["mean_relevance_delta"],
                "relative_mean_relevance_change": row["relative_mean_relevance_change"],
                "cumulative_relevance_delta": row["cumulative_relevance_delta"],
            }
        )
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(out_dir / "slate_comparison.csv", index=False)

    run_config = {
        "run_dir": str(RUN_DIR),
        "output_dir": str(out_dir),
        "candidate_count": int(len(ranked)),
        "top_k": int(TOP_K),
        "lambda_scenarios": {key: float(value) for key, value in LAMBDAS.items()},
        "scales": _safe_json(scales),
        **RUN_METADATA,
        "reference_scale_summary_path": str(out_dir / "family_penalty_scales.csv"),
        "raw_slate_path": str(out_dir / "raw_b3_top20.csv"),
        "comparison_path": str(out_dir / "slate_comparison.csv"),
        "family_summary_path": str(out_dir / "family_concentration_summary.csv"),
        "family_distribution_path": str(out_dir / "family_category_distributions.csv"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))
    (out_dir / "safety_diagnostics.json").write_text(json.dumps(_safe_json(safety_diag), indent=2, ensure_ascii=False))

    readme = (
        "# Deterministic Discovery Slate\n\n"
        "Frozen input: `mvp/artifacts/recommendation_runs/20261003T075635Z/`.\n\n"
        "This run constructs four deterministic top-20 slates: `RAW_B3`, `CONCENTRATION_005`, `CONCENTRATION_010`, and `CONCENTRATION_020`.\n\n"
        "Penalty behavior:\n"
        "- only positive incremental concentration increases are penalized\n"
        "- multi-label families split mass equally across labels\n"
        "- no latent-space or semantic dimensions affect ranking\n"
        "- source is treated as retrieval provenance, not taste\n"
    )
    (out_dir / "README.md").write_text(readme)

    return {
        "output_dir": str(out_dir),
        "slates": {name: str(out_dir / f"{name.lower()}_top20.csv") for name in slates},
        "comparison_path": str(out_dir / "slate_comparison.csv"),
        "safety_diagnostics_path": str(out_dir / "safety_diagnostics.json"),
    }


def main() -> int:
    run_discovery_slate_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
