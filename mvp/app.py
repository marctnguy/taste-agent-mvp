from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mvp.src.data import (  # noqa: E402
    add_preference_weight_column,
    chronological_split,
    ensure_directories,
    infer_work_id_column,
    load_table,
    normalize_column_names,
)
from mvp.src.config import load_runtime_env  # noqa: E402
from mvp.src.condition_c import run_condition_c_checkpoint  # noqa: E402
from mvp.src.exploratory_latent_crossmedia import run_exploratory_latent_crossmedia_checkpoint  # noqa: E402
from mvp.src.exploratory_latent_semantics import run_exploratory_latent_semantics_checkpoint  # noqa: E402
from mvp.src.generative.recommendation_chain import run_recommendation_agent  # noqa: E402
from mvp.src.generative.langsmith_evaluation import run_runtime_v1_baseline  # noqa: E402
from mvp.src.candidates import run_recommendation_live  # noqa: E402
from mvp.src.mvp_deployment import fit_mvp_b3_model  # noqa: E402
from mvp.src.prepare import run_enrichment_preparation  # noqa: E402
from mvp.src.models import (  # noqa: E402
    ConventionalMetadataFeatureBuilder,
    evaluate_regression,
    fit_ridge_model,
    predict_with_bundle,
    train_semantic_model,
)
from mvp.src.recommend import score_candidates  # noqa: E402
from mvp.src.semantics import SemanticVectorStore  # noqa: E402
from mvp.src.taste_profile import build_taste_profile, top_associations  # noqa: E402


def _load_history(path: str | Path) -> pd.DataFrame:
    history = normalize_column_names(load_table(path))
    return add_preference_weight_column(history)


def _load_ids(path: str | Path, id_column: str | None = None) -> list[str]:
    frame = normalize_column_names(load_table(path))
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    if "split" in frame.columns and frame["split"].astype(str).str.lower().isin({"train", "test"}).any():
        train_mask = frame["split"].astype(str).str.lower() == "train"
        if train_mask.any():
            frame = frame.loc[train_mask].copy()
    if id_column and id_column in frame.columns:
        column = id_column
    else:
        try:
            column = infer_work_id_column(frame)
        except KeyError:
            column = frame.columns[0]
    return frame[column].dropna().astype(str).tolist()


def cmd_profile(args: argparse.Namespace) -> int:
    history = _load_history(args.history)
    vectors = SemanticVectorStore(args.vectors).load()
    profile = build_taste_profile(history, vectors)

    print("\nTop positive associations")
    print(top_associations(profile, kind="positive", n=args.top_n).to_string(index=False))
    print("\nTop negative associations")
    print(top_associations(profile, kind="negative", n=args.top_n).to_string(index=False))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    history = _load_history(args.history)
    vectors = SemanticVectorStore(args.vectors).load()

    merged = history.merge(vectors, on=args.id_column, how="inner")
    train, test = chronological_split(merged, test_size=args.test_size, time_col=args.time_column)

    baseline_builder = ConventionalMetadataFeatureBuilder().fit(train)
    baseline_bundle = fit_ridge_model(
        train,
        target_col="preference_weight",
        feature_builder=baseline_builder,
        model_name="baseline_metadata",
    )
    semantic_bundle = train_semantic_model(
        train,
        target_col="preference_weight",
        model_name="semantic_ridge",
    )

    baseline_metrics = evaluate_regression(
        test["preference_weight"].to_numpy(),
        predict_with_bundle(baseline_bundle, test),
    )
    semantic_metrics = evaluate_regression(
        test["preference_weight"].to_numpy(),
        predict_with_bundle(semantic_bundle, test),
    )

    print("\nBaseline model")
    print(pd.Series(baseline_metrics).to_string())
    print("\nSemantic model")
    print(pd.Series(semantic_metrics).to_string())
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    candidates = normalize_column_names(load_table(args.candidates))
    watched_ids = _load_ids(args.watched_ids, args.watched_id_column) if args.watched_ids else None
    scored = score_candidates(
        candidates,
        model_dir=args.model_dir,
        watched_ids=watched_ids,
        top_k=args.top_k,
        candidate_semantic_vectors=args.candidate_semantic_vectors,
    )
    print(scored.to_json(orient="records", indent=2, date_format="iso"))
    return 0


def cmd_fit_mvp_model(args: argparse.Namespace) -> int:
    report = fit_mvp_b3_model(args.model_dir)
    print(json.dumps(report, indent=2))
    return 0


def cmd_recommend_live(args: argparse.Namespace) -> int:
    report = run_recommendation_live(
        top_k=args.top_k,
        candidate_limit=args.candidate_limit,
        model_dir=args.model_dir,
        raw_path=args.raw,
        condition_a_path=args.condition_a_path,
        semantic_cache_path=args.semantic_cache_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    report = run_enrichment_preparation(
        raw_path=args.raw,
        split_path=args.split_path,
        condition_a_path=args.condition_a_path,
        semantic_cache_path=args.semantic_cache_path,
        semantic_batch_size=args.semantic_batch_size,
        semantic_max_workers=args.semantic_max_workers,
    )
    print(pd.Series(report).to_string())
    return 0


def cmd_condition_c_checkpoint(args: argparse.Namespace) -> int:
    checkpoint = run_condition_c_checkpoint(args.raw)
    print(json.dumps(checkpoint, indent=2))
    return 0


def cmd_recommend_agent(args: argparse.Namespace) -> int:
    result = run_recommendation_agent(
        query=args.query,
        count=args.count,
        candidate_context_size=args.candidate_context_size,
        run_dir=args.run_dir,
        candidate_id=args.candidate_id,
        debug=args.debug,
    )
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_langsmith_evaluate(args: argparse.Namespace) -> int:
    summary = run_runtime_v1_baseline(
        results_dir=args.results_dir,
        dataset_name=args.dataset_name,
        experiment_prefix=args.experiment_prefix,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_exploratory_latent_semantics_checkpoint(args: argparse.Namespace) -> int:
    checkpoint = run_exploratory_latent_semantics_checkpoint()
    print(json.dumps(checkpoint, indent=2))
    return 0


def cmd_exploratory_latent_crossmedia_checkpoint(args: argparse.Namespace) -> int:
    checkpoint = run_exploratory_latent_crossmedia_checkpoint()
    print(json.dumps(checkpoint, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taste-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--history", required=True, help="Path to rated viewing history CSV/JSON/parquet.")
    common.add_argument(
        "--vectors",
        required=True,
        help="Path to cached semantic vectors CSV/JSON/parquet.",
    )
    common.add_argument("--id-column", default="canonical_id", help="Join key for works.")
    common.add_argument("--time-column", default="watched_at", help="Timestamp column for chronological split.")
    common.add_argument("--test-size", type=float, default=0.2, help="Held-out share for evaluation.")

    profile = subparsers.add_parser("profile", parents=[common], help="Build a descriptive taste profile.")
    profile.add_argument("--top-n", type=int, default=10)
    profile.set_defaults(func=cmd_profile)

    evaluate = subparsers.add_parser("evaluate", parents=[common], help="Train and compare the MVP models.")
    evaluate.set_defaults(func=cmd_evaluate)

    recommend = subparsers.add_parser("recommend", help="Score candidate films with the frozen B3 MVP model.")
    recommend.add_argument("--candidates", required=True, help="Path to candidate catalog CSV/JSON/parquet.")
    recommend.add_argument(
        "--model-dir",
        default="mvp/artifacts/models/b3_mvp",
        help="Path to the frozen B3 MVP model directory.",
    )
    recommend.add_argument(
        "--watched-ids",
        help="Optional path to a file containing watched works to exclude.",
    )
    recommend.add_argument(
        "--watched-id-column",
        default="source_id",
        help="Column to use from the watched-ids file when present.",
    )
    recommend.add_argument(
        "--candidate-semantic-vectors",
        help="Optional path to semantic vectors for explanatory taste evidence.",
    )
    recommend.add_argument("--top-k", type=int, default=10)
    recommend.set_defaults(func=cmd_recommend)

    fit_mvp = subparsers.add_parser(
        "fit-mvp-model",
        help="Fit the frozen B3 MVP deployment bundle on the 393-film training population.",
    )
    fit_mvp.add_argument(
        "--model-dir",
        default="mvp/artifacts/models/b3_mvp",
        help="Output directory for the deployable model bundle.",
    )
    fit_mvp.set_defaults(func=cmd_fit_mvp_model)

    recommend_live = subparsers.add_parser(
        "recommend-live",
        help="Generate a live TMDB candidate pool and rank it with the frozen B3 MVP model.",
    )
    recommend_live.add_argument("--top-k", type=int, default=20, help="Number of recommendations to return.")
    recommend_live.add_argument(
        "--candidate-limit",
        type=int,
        default=300,
        help="Approximate unique candidate limit before watched exclusion.",
    )
    recommend_live.add_argument(
        "--model-dir",
        default="mvp/artifacts/models/b3_mvp",
        help="Path to the frozen B3 MVP model directory.",
    )
    recommend_live.add_argument(
        "--raw",
        default="mvp/data/raw/taste-agent-combined-ingestion.csv",
        help="Path to the combined consumption history CSV.",
    )
    recommend_live.add_argument(
        "--condition-a-path",
        default="mvp/data/processed/condition_a_enriched.csv",
        help="Path to the enriched TMDB metadata CSV.",
    )
    recommend_live.add_argument(
        "--semantic-cache-path",
        default="mvp/artifacts/semantic_vectors/semantic_vectors.csv",
        help="Path to the semantic vector cache CSV.",
    )
    recommend_live.set_defaults(func=cmd_recommend_live)

    recommend_agent = subparsers.add_parser(
        "recommend-agent",
        help="Generate grounded recommendations with the LangChain contextual runtime.",
    )
    recommend_agent.add_argument("--query", required=True, help="Natural-language recommendation request.")
    recommend_agent.add_argument("--count", type=int, default=5, help="Maximum number of recommendations to return.")
    recommend_agent.add_argument(
        "--candidate-context-size",
        type=int,
        default=50,
        help="Number of frozen B3 candidates to supply to the contextual selection layer.",
    )
    recommend_agent.add_argument(
        "--run-dir",
        default="mvp/artifacts/recommendation_runs/20261003T075635Z",
        help="Frozen recommendation run directory.",
    )
    recommend_agent.add_argument(
        "--candidate-id",
        help="Optional candidate_id for EXPLAIN requests.",
    )
    recommend_agent.add_argument("--debug", action="store_true", help="Emit debug metadata in the JSON output.")
    recommend_agent.set_defaults(func=cmd_recommend_agent)

    langsmith_eval = subparsers.add_parser(
        "langsmith-evaluate",
        help="Run the frozen runtime_v1 LangSmith baseline evaluation.",
    )
    langsmith_eval.add_argument(
        "--results-dir",
        default="mvp/artifacts/generative/langsmith/runtime_v1_baseline",
        help="Directory for LangSmith experiment artifacts.",
    )
    langsmith_eval.add_argument(
        "--dataset-name",
        default="taste-agent-runtime-v1-eval",
        help="LangSmith dataset name.",
    )
    langsmith_eval.add_argument(
        "--experiment-prefix",
        default="taste-agent-runtime-v1-baseline",
        help="LangSmith experiment name prefix.",
    )
    langsmith_eval.set_defaults(func=cmd_langsmith_evaluate)

    prepare = subparsers.add_parser("prepare", help="Enrich Condition A and cache Condition B vectors.")
    prepare.add_argument(
        "--raw",
        default="mvp/data/raw/taste-agent-combined-ingestion.csv",
        help="Path to the combined ingestion CSV.",
    )
    prepare.add_argument(
        "--split-path",
        default="mvp/data/processed/primary_holdout_split.csv",
        help="Path to the frozen train/test split CSV.",
    )
    prepare.add_argument(
        "--condition-a-path",
        default="mvp/data/processed/condition_a_enriched.csv",
        help="Path to the enriched Condition A CSV.",
    )
    prepare.add_argument(
        "--semantic-cache-path",
        default="mvp/artifacts/semantic_vectors/semantic_vectors.csv",
        help="Path to the cached semantic vectors CSV.",
    )
    prepare.add_argument(
        "--semantic-batch-size",
        type=int,
        default=8,
        help="How many films to classify per OpenAI request.",
    )
    prepare.add_argument(
        "--semantic-max-workers",
        type=int,
        default=4,
        help="How many semantic classification requests to run in parallel.",
    )
    prepare.set_defaults(func=cmd_prepare)

    condition_c = subparsers.add_parser(
        "condition-c-checkpoint",
        help="Build the Goodreads semantic cache and Condition C training-only checkpoint.",
    )
    condition_c.add_argument(
        "--raw",
        default="mvp/data/raw/taste-agent-combined-ingestion.csv",
        help="Path to the combined ingestion CSV.",
    )
    condition_c.set_defaults(func=cmd_condition_c_checkpoint)

    exploratory = subparsers.add_parser(
        "exploratory-latent-semantics-checkpoint",
        help="Run the training-only exploratory latent semantics checkpoint.",
    )
    exploratory.set_defaults(func=cmd_exploratory_latent_semantics_checkpoint)

    crossmedia = subparsers.add_parser(
        "exploratory-latent-crossmedia-checkpoint",
        help="Run the training-only exploratory latent crossmedia checkpoint.",
    )
    crossmedia.set_defaults(func=cmd_exploratory_latent_crossmedia_checkpoint)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_runtime_env()
    ensure_directories()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
