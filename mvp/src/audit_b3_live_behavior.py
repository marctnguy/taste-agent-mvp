from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mvp.src.data import add_decade_column, add_preference_weight_column
from mvp.src.candidates import load_consumption_history
from mvp.src.mvp_deployment import load_model_bundle, load_training_population
from mvp.src.taste_profile import build_taste_profile
from mvp.src.semantics import SemanticVectorStore


DEFAULT_RUN_DIR = Path("mvp/artifacts/recommendation_runs/20261003T075635Z")
DEFAULT_MODEL_DIR = Path("mvp/artifacts/models/b3_mvp")
DEFAULT_OUTPUT_DIR = Path("mvp/artifacts/diagnostics/b3_live_behavior")
DEFAULT_EVALUATION_PATH = Path("evaluation/qualitative_recommendation_feedback.md")


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [part.strip() for part in text.split("|") if part.strip()]


def _decade_label(year: Any) -> str:
    if pd.isna(year):
        return "unknown"
    year_int = int(float(year))
    return f"{year_int // 10 * 10}s"


def _feature_group(feature: str) -> str:
    if feature == "release_year_z":
        return "release_year"
    if feature.startswith("decade__"):
        return "decade"
    if feature.startswith("genre__"):
        return "genre"
    if feature.startswith("language__"):
        return "language"
    if feature.startswith("country__"):
        return "country"
    if feature.startswith("latent_pca_"):
        return "latent"
    return "other"


def _load_candidates(run_dir: Path) -> pd.DataFrame:
    ranked = pd.read_csv(run_dir / "ranked_candidates.csv")
    ranked["source_id"] = ranked["source_id"].astype(str)
    ranked["candidate_sources"] = ranked["candidate_sources"].apply(_parse_listish)
    ranked["decade"] = ranked["year"].apply(_decade_label)
    return ranked


def _top_contributions(feature_contrib_row: pd.Series, kind: str, n: int = 5) -> list[dict[str, Any]]:
    frame = feature_contrib_row.sort_values("contribution", ascending=(kind == "negative"))
    if kind == "positive":
        frame = frame[frame["contribution"] > 0]
    else:
        frame = frame[frame["contribution"] < 0]
    if frame.empty:
        frame = feature_contrib_row.sort_values("contribution", ascending=(kind == "negative"))
    out = []
    for _, row in frame.head(n).iterrows():
        out.append(
            {
                "feature": row["feature"],
                "group": row["group"],
                "value": float(row["feature_value"]),
                "coefficient": float(row["coefficient"]),
                "contribution": float(row["contribution"]),
            }
        )
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(x_num) & np.isfinite(y_num)
    x_vals = x_num[mask].to_numpy(dtype=float)
    y_vals = y_num[mask].to_numpy(dtype=float)
    if len(x_vals) < 2 or np.std(x_vals) == 0 or np.std(y_vals) == 0:
        return float("nan")
    return float(np.corrcoef(x_vals, y_vals)[0, 1])


def build_audit(
    run_dir: Path = DEFAULT_RUN_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    evaluation_path: Path = DEFAULT_EVALUATION_PATH,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)

    bundle, pca, metadata_builder = load_model_bundle(model_dir)
    ranked = _load_candidates(run_dir)
    candidate_embeddings = pd.read_csv(model_dir / "candidate_embedding_cache" / "candidate_embeddings.csv")
    candidate_embeddings["source_id"] = candidate_embeddings["source_id"].astype(str)
    emb_cols = [column for column in candidate_embeddings.columns if column.startswith("emb_")]

    merged = ranked.merge(candidate_embeddings, on="source_id", how="inner", suffixes=("", "_emb"))
    merged = merged.sort_values("rank").reset_index(drop=True)
    metadata = metadata_builder.transform(merged).reset_index(drop=True)
    latent = pca.transform(bundle.scaler.transform(merged.loc[:, emb_cols].to_numpy(dtype=float)))
    latent_df = pd.DataFrame(latent, columns=bundle.latent_feature_columns)
    features = pd.concat([metadata, latent_df], axis=1).reindex(columns=bundle.feature_columns, fill_value=0.0)

    predictions = bundle.ridge.predict(features.to_numpy(dtype=float))
    reconstructed = bundle.ridge.intercept_ + np.dot(features.to_numpy(dtype=float), bundle.ridge.coef_)
    if not np.allclose(predictions, reconstructed, atol=1e-10, rtol=1e-10):
        raise RuntimeError("Reconstructed predictions do not match the saved B3 predictions.")

    coef = pd.Series(bundle.ridge.coef_, index=bundle.feature_columns, dtype=float)
    contrib = features.mul(coef, axis=1)
    contrib["intercept"] = float(bundle.ridge.intercept_)
    contrib["reconstructed_prediction"] = reconstructed
    contrib["predicted_preference"] = predictions
    contrib["prediction_error"] = predictions - reconstructed

    group_map = {column: _feature_group(column) for column in bundle.feature_columns}
    long_rows: list[dict[str, Any]] = []
    for idx, row in merged.iterrows():
        for feature in bundle.feature_columns:
            value = float(features.loc[idx, feature])
            coefficient = float(coef[feature])
            contribution_value = float(value * coefficient)
            long_rows.append(
                {
                    "source_id": row["source_id"],
                    "tmdb_id": row["tmdb_id"],
                    "title": row["title"],
                    "rank": int(row["rank"]),
                    "year": int(row["year"]) if pd.notna(row["year"]) else pd.NA,
                    "feature": feature,
                    "group": group_map[feature],
                    "feature_value": value,
                    "coefficient": coefficient,
                    "contribution": contribution_value,
                    "predicted_preference": float(predictions[idx]),
                    "reconstructed_prediction": float(reconstructed[idx]),
                    "prediction_error": float(predictions[idx] - reconstructed[idx]),
                }
            )
    long_frame = pd.DataFrame(long_rows)
    long_frame.to_csv(output_dir / "prediction_contributions.csv", index=False, float_format="%.12f")

    candidate_summary = (
        long_frame.groupby(["source_id", "tmdb_id", "title", "rank", "year", "predicted_preference", "reconstructed_prediction", "prediction_error", "group"], as_index=False)["contribution"]
        .sum()
        .pivot_table(
            index=["source_id", "tmdb_id", "title", "rank", "year", "predicted_preference", "reconstructed_prediction", "prediction_error"],
            columns="group",
            values="contribution",
            fill_value=0.0,
        )
        .reset_index()
    )
    for column in ["release_year", "decade", "genre", "language", "country", "latent", "other"]:
        if column not in candidate_summary.columns:
            candidate_summary[column] = 0.0
    candidate_summary = candidate_summary.rename(
        columns={
            "release_year": "release_year_contribution",
            "decade": "decade_contribution",
            "genre": "genre_contribution",
            "language": "language_contribution",
            "country": "country_contribution",
            "latent": "latent_contribution",
        }
    )
    candidate_summary["metadata_total"] = candidate_summary[
        [
            "release_year_contribution",
            "decade_contribution",
            "genre_contribution",
            "language_contribution",
            "country_contribution",
        ]
    ].sum(axis=1)
    candidate_summary["total_grouped"] = candidate_summary[
        [
            "release_year_contribution",
            "decade_contribution",
            "genre_contribution",
            "language_contribution",
            "country_contribution",
            "latent_contribution",
            "other",
        ]
    ].sum(axis=1)
    candidate_summary["prediction_reconstruction_gap"] = candidate_summary["predicted_preference"] - candidate_summary["reconstructed_prediction"]
    candidate_summary["year_decade"] = candidate_summary["year"].apply(_decade_label)

    top20 = candidate_summary.nsmallest(20, "rank").copy()
    top20_records: list[dict[str, Any]] = []
    for _, row in top20.iterrows():
        feature_rows = long_frame.loc[long_frame["source_id"] == row["source_id"]].copy()
        top20_records.append(
            {
                "source_id": row["source_id"],
                "tmdb_id": int(row["tmdb_id"]),
                "title": row["title"],
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "rank": int(row["rank"]),
                "predicted_preference": float(row["predicted_preference"]),
                "reconstructed_prediction": float(row["reconstructed_prediction"]),
                "prediction_error": float(row["prediction_reconstruction_gap"]),
                "candidate_sources": _parse_listish(merged.loc[merged["source_id"] == row["source_id"], "candidate_sources"].iloc[0]),
                    "grouped_contributions": {
                    "release_year": float(row["release_year_contribution"]),
                    "decade": float(row["decade_contribution"]),
                    "genre": float(row["genre_contribution"]),
                    "language": float(row["language_contribution"]),
                    "country": float(row["country_contribution"]),
                    "latent": float(row["latent_contribution"]),
                    "metadata_total": float(row["metadata_total"]),
                    "intercept": float(bundle.ridge.intercept_),
                },
                "top_positive_features": _top_contributions(feature_rows, "positive"),
                "top_negative_features": _top_contributions(feature_rows, "negative"),
            }
        )
    (output_dir / "top20_prediction_attributions.json").write_text(json.dumps(top20_records, indent=2, ensure_ascii=False))

    decade_summary = candidate_summary.groupby("year_decade", dropna=False).agg(
        count=("source_id", "size"),
        mean_prediction=("predicted_preference", "mean"),
        median_prediction=("predicted_preference", "median"),
        max_prediction=("predicted_preference", "max"),
        mean_release_year_contribution=("release_year_contribution", "mean"),
        mean_decade_contribution=("decade_contribution", "mean"),
        mean_genre_contribution=("genre_contribution", "mean"),
        mean_language_contribution=("language_contribution", "mean"),
        mean_country_contribution=("country_contribution", "mean"),
        mean_latent_contribution=("latent_contribution", "mean"),
    ).reset_index()
    decade_summary.to_csv(output_dir / "decade_attribution_summary.csv", index=False, float_format="%.12f")

    training = load_training_population()
    training = add_decade_column(training, year_col="release_year", output_col="release_decade")
    training["release_decade"] = training["release_decade"].fillna("unknown")
    training_summary_rows = []
    rating_distribution = training.groupby("release_decade")["rating"].value_counts(dropna=False).rename("count").reset_index()
    for decade, group in training.groupby("release_decade", dropna=False):
        training_summary_rows.append(
            {
                "decade": decade,
                "count": int(len(group)),
                "mean_preference_weight": float(pd.to_numeric(group["preference_weight"], errors="coerce").mean()),
                "median_preference_weight": float(pd.to_numeric(group["preference_weight"], errors="coerce").median()),
                "mean_release_year": float(pd.to_numeric(group["release_year"], errors="coerce").mean()),
                "pearson_release_year_preference": _safe_corr(group["release_year"], group["preference_weight"]),
                "spearman_release_year_preference": _safe_corr(
                    pd.to_numeric(group["release_year"], errors="coerce").rank(),
                    pd.to_numeric(group["preference_weight"], errors="coerce").rank(),
                ),
                "rating_distribution": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in group["rating"].value_counts(dropna=False).sort_index().items()
                    },
                    sort_keys=True,
                ),
            }
        )
    training_summary = pd.DataFrame(training_summary_rows).sort_values("decade").reset_index(drop=True)
    training_summary.to_csv(output_dir / "training_decade_preference_summary.csv", index=False)

    latent_rows = []
    for latent_column in bundle.latent_feature_columns:
        values = pd.to_numeric(features[latent_column], errors="coerce").to_numpy(dtype=float)
        component_contribution = values * float(coef[latent_column])
        top_abs = pd.DataFrame(
            {
                "title": merged["title"].to_numpy(),
                "rank": merged["rank"].to_numpy(),
                "abs_contribution": np.abs(component_contribution),
            }
        ).sort_values("abs_contribution", ascending=False).head(3)[["title", "rank"]]
        latent_rows.append(
            {
                "component": latent_column,
                "ridge_coefficient": float(coef[latent_column]),
                "mean_contribution": float(np.mean(component_contribution)),
                "mean_abs_contribution": float(np.mean(np.abs(component_contribution))),
                "median_contribution": float(np.median(component_contribution)),
                "min_contribution": float(np.min(component_contribution)),
                "max_contribution": float(np.max(component_contribution)),
                "top_abs_titles": json.dumps(top_abs.to_dict(orient="records")),
            }
        )
    latent_summary = pd.DataFrame(latent_rows).sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)
    latent_summary.to_csv(output_dir / "latent_component_summary.csv", index=False, float_format="%.12f")

    semantic_vectors = SemanticVectorStore("mvp/artifacts/semantic_vectors").load()
    history = load_consumption_history("mvp/data/raw/taste-agent-combined-ingestion.csv")
    history = add_preference_weight_column(history)
    current_profile = build_taste_profile(history, semantic_vectors)
    merged_history = history.merge(semantic_vectors, on="canonical_id", how="inner", suffixes=("", "_semantic"))
    current_profile.to_csv(output_dir / "taste_profile.csv", index=False, float_format="%.12f")
    evidence_rows = []
    association_rows = []
    for _, row in current_profile.iterrows():
        dimension = str(row["dimension"])
        feature = pd.to_numeric(merged_history[dimension], errors="coerce").to_numpy(dtype=float)
        target = pd.to_numeric(merged_history["preference_weight"], errors="coerce").to_numpy(dtype=float) if "preference_weight" in merged_history.columns else pd.to_numeric(merged_history["rating"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(feature) & np.isfinite(target)
        x = feature[mask]
        y = target[mask]
        expected_count = int(np.sum(x >= 0.25))
        saved_count = int(mask.sum())
        expected_association = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
        association_rows.append(
            {
                "dimension": dimension,
                "population_used": "history x semantic_vectors inner join on canonical_id",
                "observations": int(mask.sum()),
                "missing_handling": "pairwise finite mask",
                "recomputed_association": expected_association,
                "saved_association": float(row["pearson_preference_association"]),
                "difference": float(expected_association - float(row["pearson_preference_association"])) if np.isfinite(expected_association) else float("nan"),
            }
        )
        evidence_rows.append(
            {
                "dimension": dimension,
                "expected_count": expected_count,
                "saved_count": saved_count,
                "difference": expected_count - saved_count,
                "observations": int(mask.sum()),
                "population_size": int(len(merged_history)),
            }
        )
    evidence_df = pd.DataFrame(evidence_rows)
    association_df = pd.DataFrame(association_rows)
    evidence_df.to_csv(output_dir / "taste_profile_evidence_audit.csv", index=False)
    association_df.to_csv(output_dir / "taste_profile_association_audit.csv", index=False)

    assoc_abs = association_df["recomputed_association"].abs()
    eligible = current_profile.assign(
        eligible=(current_profile["evidence_count"].fillna(0) >= 3) & (current_profile["evidence_confidence"].fillna(0) >= 0.4)
    )
    summary = {
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "evaluation_path": str(evaluation_path),
        "reconstruction": {
            "max_abs_error": float(np.max(np.abs(predictions - reconstructed))),
            "mean_abs_error": float(np.mean(np.abs(predictions - reconstructed))),
            "passed": bool(np.allclose(predictions, reconstructed, atol=1e-10, rtol=1e-10)),
        },
        "candidate_population": {
            "rows": int(len(merged)),
            "years_2010s": int((merged["year"].between(2010, 2019)).sum()),
            "years_2020s": int((merged["year"] >= 2020).sum()),
            "pre_2000": int((merged["year"] < 2000).sum()),
        },
        "top20": {
            "source_mix": ranked.head(20)["candidate_sources"].explode().value_counts().to_dict(),
            "decade_mix": ranked.head(20)["decade"].value_counts().to_dict(),
        },
        "attribution_split": {
            "metadata_mean": float(candidate_summary["metadata_total"].mean()),
            "latent_mean": float(candidate_summary["latent_contribution"].mean()),
            "metadata_vs_latent_correlation": float(candidate_summary["metadata_total"].corr(candidate_summary["latent_contribution"])),
        },
        "association_audit": {
            "abs_association_min": float(assoc_abs.min()),
            "abs_association_q25": float(assoc_abs.quantile(0.25)),
            "abs_association_median": float(assoc_abs.median()),
            "abs_association_q75": float(assoc_abs.quantile(0.75)),
            "abs_association_max": float(assoc_abs.max()),
            "eligible_positive_dimensions": int(((current_profile["pearson_preference_association"] > 0) & eligible["eligible"]).sum()),
            "eligible_negative_dimensions": int(((current_profile["pearson_preference_association"] < 0) & eligible["eligible"]).sum()),
        },
        "evidence_audit": {
            "all_dimensions_same_saved_count": bool((evidence_df["saved_count"] == evidence_df["saved_count"].iloc[0]).all()),
            "saved_count": int(evidence_df["saved_count"].iloc[0]) if not evidence_df.empty else 0,
        },
        "files": {
            "prediction_contributions": str(output_dir / "prediction_contributions.csv"),
            "top20_prediction_attributions": str(output_dir / "top20_prediction_attributions.json"),
            "decade_attribution_summary": str(output_dir / "decade_attribution_summary.csv"),
            "training_decade_preference_summary": str(output_dir / "training_decade_preference_summary.csv"),
            "latent_component_summary": str(output_dir / "latent_component_summary.csv"),
            "taste_profile": str(output_dir / "taste_profile.csv"),
            "taste_profile_evidence_audit": str(output_dir / "taste_profile_evidence_audit.csv"),
            "taste_profile_association_audit": str(output_dir / "taste_profile_association_audit.csv"),
        },
    }

    # Direct comparisons for the requested modern-film slices.
    candidate_summary["year_decade"] = candidate_summary["year"].apply(_decade_label)
    top_2010s = candidate_summary.loc[candidate_summary["year_decade"] == "2010s"].sort_values("rank").head(10)
    top_2020s = candidate_summary.loc[candidate_summary["year_decade"] == "2020s"].sort_values("rank").head(10)
    pre_2000 = candidate_summary.loc[candidate_summary["year"] < 2000]

    def _serialize_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for _, row in frame.iterrows():
            rows.append(
                {
                    "source_id": row["source_id"],
                    "tmdb_id": int(row["tmdb_id"]),
                    "title": row["title"],
                    "year": int(row["year"]) if pd.notna(row["year"]) else None,
                    "rank": int(row["rank"]),
                    "predicted_preference": float(row["predicted_preference"]),
                    "grouped_contributions": {
                        "release_year": float(row["release_year_contribution"]),
                        "decade": float(row["decade_contribution"]),
                        "genre": float(row["genre_contribution"]),
                        "language": float(row["language_contribution"]),
                        "country": float(row["country_contribution"]),
                        "latent": float(row["latent_contribution"]),
                        "metadata_total": float(row["metadata_total"]),
                        "intercept": float(bundle.ridge.intercept_),
                    },
                }
            )
        return rows

    comparison_frames = {
        "top20": candidate_summary.nsmallest(20, "rank"),
        "pre_2000": pre_2000,
        "2010s": candidate_summary.loc[candidate_summary["year_decade"] == "2010s"],
        "2020s": candidate_summary.loc[candidate_summary["year_decade"] == "2020s"],
    }
    comparison_summary = {}
    for label, frame in comparison_frames.items():
        comparison_summary[label] = {
            "count": int(len(frame)),
            "mean_grouped_contributions": {
                "release_year": float(frame["release_year_contribution"].mean()) if len(frame) else 0.0,
                "decade": float(frame["decade_contribution"].mean()) if len(frame) else 0.0,
                "genre": float(frame["genre_contribution"].mean()) if len(frame) else 0.0,
                "language": float(frame["language_contribution"].mean()) if len(frame) else 0.0,
                "country": float(frame["country_contribution"].mean()) if len(frame) else 0.0,
                "latent": float(frame["latent_contribution"].mean()) if len(frame) else 0.0,
                "metadata_total": float(frame["metadata_total"].mean()) if len(frame) else 0.0,
            },
        }
    modern_payload = {
        "top_2010s": _serialize_rows(top_2010s),
        "top_2020s": _serialize_rows(top_2020s),
        "comparison_summary": comparison_summary,
    }
    (output_dir / "modern_candidate_attributions.json").write_text(json.dumps(modern_payload, indent=2, ensure_ascii=False))

    evaluation_path.write_text(
        "# Qualitative Recommendation Feedback\n\n"
        "First live recommendation evaluation:\n\n"
        "The user perceived the raw Top-20 output as strongly skewed toward older canonical and male-centered cinema and as poorly representative of their perceived taste.\n\n"
        "Candidate-pool analysis subsequently showed that recent films were well represented in retrieval but disappeared during B3 ranking, motivating a model-behavior attribution audit.\n"
    )

    summary["modern_slices"] = {
        "top_2010s_count": int(len(top_2010s)),
        "top_2020s_count": int(len(top_2020s)),
        "comparison_summary": comparison_summary,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    (output_dir / "README.md").write_text(
        "# B3 Live Behavior Audit\n\n"
        "This directory contains a frozen attribution and evidence audit for the corrected live recommendation run.\n"
    )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-path", type=Path, default=DEFAULT_EVALUATION_PATH)
    args = parser.parse_args()
    summary = build_audit(args.run_dir, args.model_dir, args.output_dir, args.evaluation_path)
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
