from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from functools import partial
from typing import Any
from types import SimpleNamespace

import pandas as pd
from langsmith import Client
from langsmith.evaluation import evaluate

from evaluation.langsmith import evaluators
from mvp.src.config import get_api_keys, get_langsmith_config, load_runtime_env
from mvp.src.generative.context_builder import parse_intent
from mvp.src.generative.recommendation_chain import run_recommendation_agent
from mvp.src.generative.runtime_version import (
    RUNTIME_VERSION,
    runtime_manifest,
)


DEFAULT_DATASET_NAME = "taste-agent-runtime-v1-eval"
DEFAULT_EXPERIMENT_PREFIX = "taste-agent-runtime-v1-baseline"
DEFAULT_DATASET_PATH = Path("evaluation/langsmith/runtime_v1_dataset.json")
DEFAULT_PLAN_PATH = Path("evaluation/langsmith/runtime_v1_evaluation_plan.md")
DEFAULT_RESULTS_DIR = Path("mvp/artifacts/generative/langsmith/runtime_v1_baseline")
DEFAULT_V2_RESULTS_DIR = Path("mvp/artifacts/generative/langsmith/runtime_v2_iteration")
DEFAULT_V2_DATASET_NAME = "taste-agent-runtime-v1-eval"
DEFAULT_V2_EXPERIMENT_PREFIX = "taste-agent-runtime-v2-iteration"
DEFAULT_V2_RUNTIME_VERSION = "runtime_v2"
DEFAULT_V2_INTENT_PROMPT_VERSION = "intent_v2"
DEFAULT_V2_RECOMMENDATION_PROMPT_VERSION = "recommendation_v1"
DEFAULT_V2_MANIFEST_PATH = Path("mvp/artifacts/generative/runtime_v2_manifest.json")
DEFAULT_V4_DATASET_NAME = "taste-agent-runtime-v4-eval"
DEFAULT_V4_EXPERIMENT_PREFIX = "taste-agent-runtime-v4-hosted"
DEFAULT_V4_RESULTS_DIR = Path("mvp/artifacts/generative/langsmith/runtime_v4_hosted")
DEFAULT_V4_RUNTIME_VERSION = "runtime_v4"
DEFAULT_V4_INTENT_PROMPT_VERSION = "intent_v4"
DEFAULT_V4_RECOMMENDATION_PROMPT_VERSION = "recommendation_v4"
DEFAULT_V4_MANIFEST_PATH = Path("mvp/artifacts/generative/runtime_v4_manifest.json")


def ensure_langsmith_env() -> None:
    load_runtime_env()
    config = get_langsmith_config()
    api_key = config.api_key
    if api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", config.project or "taste-agent-capstone")
        if config.endpoint:
            os.environ.setdefault("LANGSMITH_ENDPOINT", config.endpoint)


def load_runtime_v1_cases(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def build_examples(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for case in cases:
        example_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"taste-agent-runtime-v1-eval:{case['case_id']}"))
        examples.append(
            {
                "id": example_id,
                "inputs": {
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "candidate_id": case.get("candidate_id"),
                },
                "outputs": {
                    "expected_intent": case["expected_intent"],
                    "expected_hard_constraints": case["expected_hard_constraints"],
                    "expected_behavior": case["expected_behavior"],
                    "evaluation_notes": case["evaluation_notes"],
                },
                "metadata": {
                    "case_id": case["case_id"],
                    "evaluation_notes": case["evaluation_notes"],
                },
            }
        )
    return examples


def _client() -> Client | None:
    keys = get_api_keys()
    if not keys.openai_api_key:
        return None
    api_key = get_langsmith_config().api_key
    if not api_key:
        return None
    return Client(api_key=api_key)


def _target(inputs: dict[str, Any]) -> dict[str, Any]:
    return _target_with_versions(inputs)


def _target_with_versions(
    inputs: dict[str, Any],
    *,
    runtime_version: str = RUNTIME_VERSION,
    intent_prompt_version: str = "intent_v1",
    recommendation_prompt_version: str = "recommendation_v1",
    source_recommendation_run: str = "20261003T075635Z",
) -> dict[str, Any]:
    result = run_recommendation_agent(
        query=str(inputs["query"]),
        candidate_id=inputs.get("candidate_id"),
        debug=True,
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
    )
    payload = result.model_dump()
    payload["case_id"] = inputs.get("case_id")
    payload["query"] = inputs.get("query")
    payload["candidate_id"] = inputs.get("candidate_id")
    return payload


def _example_namespace(case: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        inputs={
            "case_id": case["case_id"],
            "query": case["query"],
            "candidate_id": case.get("candidate_id"),
        },
        outputs={
            "expected_intent": case["expected_intent"],
            "expected_hard_constraints": case["expected_hard_constraints"],
            "expected_behavior": case["expected_behavior"],
            "evaluation_notes": case["evaluation_notes"],
        },
        metadata={
            "case_id": case["case_id"],
            "evaluation_notes": case["evaluation_notes"],
        },
    )


def _run_local_experiment(
    cases: list[dict[str, Any]],
    evaluators_list: list[Any],
    target: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in cases:
        payload = target(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "candidate_id": case.get("candidate_id"),
            }
        )
        run = SimpleNamespace(outputs={"output": payload})
        example = _example_namespace(case)
        run.id = f"run-{case['case_id']}"
        example.id = f"example-{case['case_id']}"
        row = {
            "case_id": case["case_id"],
            "query": case["query"],
            "candidate_id": case.get("candidate_id"),
            "expected_intent": case["expected_intent"],
            "expected_behavior": case["expected_behavior"],
            "generation_status": payload["runtime_metadata"]["generation_status"],
            "validation_passed": payload["runtime_metadata"]["validation_passed"],
            "repair_attempted": payload["runtime_metadata"]["repair_attempted"],
            "fallback_used": payload["runtime_metadata"]["fallback_used"],
            "llm_call_count": payload["runtime_metadata"]["llm_call_count"],
        }
        for evaluator in evaluators_list:
            evaluated = evaluator.evaluate_run(run, example)
            if hasattr(evaluated, "results"):
                for result in evaluated.results:
                    row[result.key] = result.score
                    row[f"{result.key}_comment"] = result.comment
            else:
                row[evaluator.__name__] = getattr(evaluated, "score", None)
        rows.append(row)
    return pd.DataFrame(rows)


def _dataset_metadata() -> dict[str, Any]:
    return {
        "runtime_version": RUNTIME_VERSION,
        "source_dataset": str(DEFAULT_DATASET_PATH),
        "source_recommendation_run": "20261003T075635Z",
    }


def _save_runtime_manifest(
    results_dir: Path,
    *,
    manifest_filename: str,
    root_manifest_path: Path,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    source_recommendation_run: str,
) -> Path:
    manifest_path = results_dir / manifest_filename
    manifest = runtime_manifest(
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    root_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


def _summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"row_count": int(len(frame))}
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            summary[column] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return summary


def _run_langsmith_evaluation(
    *,
    results_dir: str | Path,
    dataset_name: str,
    experiment_prefix: str,
    runtime_version: str,
    intent_prompt_version: str,
    recommendation_prompt_version: str,
    root_manifest_path: Path,
    manifest_filename: str,
    source_recommendation_run: str = "20261003T075635Z",
) -> dict[str, Any]:
    ensure_langsmith_env()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cases = load_runtime_v1_cases()
    examples = build_examples(cases)
    client = _client()
    data_source: str | list[dict[str, Any]] = examples
    dataset_status = "local_examples"

    if client is not None:
        try:
            try:
                client.create_dataset(
                    dataset_name=dataset_name,
                    description="Fixed runtime_v1 evaluation dataset for Taste Agent LangSmith baseline.",
                    metadata=_dataset_metadata(),
                )
            except Exception as exc:
                if "already exists" not in str(exc):
                    raise
            try:
                client.create_examples(dataset_name=dataset_name, examples=examples)
            except Exception as exc:
                if "example already exists" not in str(exc).lower() and "conflict" not in str(exc).lower():
                    raise
            data_source = dataset_name
            dataset_status = "langsmith_dataset"
        except Exception:
            data_source = examples
            dataset_status = "local_examples_fallback"

    evaluators_list = [
        evaluators.candidate_compliance,
        evaluators.candidate_identity_integrity,
        evaluators.recommendation_count,
        evaluators.duplicate_candidates,
        evaluators.intent_match,
        evaluators.hard_constraint_satisfaction,
        evaluators.exclusion_satisfaction,
        evaluators.taste_signal_grounding,
        evaluators.unsupported_taste_dimension,
        evaluators.goodreads_ranking_guardrail,
        evaluators.interaction_routing_contract,
        evaluators.candidate_context_required,
        evaluators.no_match_behavior,
        evaluators.sensitive_inference_guardrail,
        evaluators.request_relevance,
        evaluators.explanation_groundedness,
        evaluators.explanation_usefulness,
        evaluators.response_clarity,
    ]

    experiment_name = None
    experiment_id = None
    experiment_url = None
    target = partial(
        _target_with_versions,
        runtime_version=runtime_version,
        intent_prompt_version=intent_prompt_version,
        recommendation_prompt_version=recommendation_prompt_version,
        source_recommendation_run=source_recommendation_run,
    )
    if client is not None:
        try:
            experiment = evaluate(
                target,
                data=data_source,
                evaluators=evaluators_list,
                metadata={
                    "runtime_version": runtime_version,
                    "dataset_name": dataset_name,
                    "dataset_status": dataset_status,
                    "experiment_prefix": experiment_prefix,
                },
                experiment_prefix=experiment_prefix,
                description="Baseline LangSmith evaluation for runtime_v1.",
                client=client,
                blocking=True,
                upload_results=True,
                error_handling="log",
            )
            frame = experiment.to_pandas()
            experiment_name = getattr(experiment, "experiment_name", None)
            try:
                experiment_id = str(experiment.experiment_id)
            except Exception:
                experiment_id = None
            experiment_url = getattr(experiment, "url", None)
        except Exception:
            frame = _run_local_experiment(cases, evaluators_list, target)
            dataset_status = "local_manual_fallback"
    else:
        frame = _run_local_experiment(cases, evaluators_list, target)

    results_path = results_dir / "experiment_results.csv"
    frame.to_csv(results_path, index=False)
    summary = {
        "runtime_version": runtime_version,
        "dataset_name": dataset_name,
        "dataset_status": dataset_status,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "experiment_url": experiment_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "summary": _summarize_frame(frame),
        "results_path": str(results_path),
        "manifest_path": str(
            _save_runtime_manifest(
                results_dir,
                manifest_filename=manifest_filename,
                root_manifest_path=root_manifest_path,
                runtime_version=runtime_version,
                intent_prompt_version=intent_prompt_version,
                recommendation_prompt_version=recommendation_prompt_version,
                source_recommendation_run=source_recommendation_run,
            )
        ),
    }
    (results_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_runtime_v1_baseline(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
) -> dict[str, Any]:
    return _run_langsmith_evaluation(
        results_dir=results_dir,
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        runtime_version=RUNTIME_VERSION,
        intent_prompt_version="intent_v1",
        recommendation_prompt_version="recommendation_v1",
        root_manifest_path=Path("mvp/artifacts/generative/runtime_v1_manifest.json"),
        manifest_filename="runtime_v1_manifest.json",
    )


def run_runtime_v2_iteration(
    results_dir: str | Path = DEFAULT_V2_RESULTS_DIR,
    dataset_name: str = DEFAULT_V2_DATASET_NAME,
    experiment_prefix: str = DEFAULT_V2_EXPERIMENT_PREFIX,
) -> dict[str, Any]:
    return _run_langsmith_evaluation(
        results_dir=results_dir,
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        runtime_version=DEFAULT_V2_RUNTIME_VERSION,
        intent_prompt_version=DEFAULT_V2_INTENT_PROMPT_VERSION,
        recommendation_prompt_version=DEFAULT_V2_RECOMMENDATION_PROMPT_VERSION,
        root_manifest_path=DEFAULT_V2_MANIFEST_PATH,
        manifest_filename="runtime_v2_manifest.json",
    )


def run_runtime_v4_hosted(
    results_dir: str | Path = DEFAULT_V4_RESULTS_DIR,
    dataset_name: str = DEFAULT_V4_DATASET_NAME,
    experiment_prefix: str = DEFAULT_V4_EXPERIMENT_PREFIX,
) -> dict[str, Any]:
    return _run_runtime_v4_langsmith_evaluation(
        results_dir=results_dir,
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
    )


def _runtime_v4_target(inputs: dict[str, Any]) -> dict[str, Any]:
    from mvp.src.generative_v4.runtime import run_runtime_v4

    result = run_runtime_v4(
        query=str(inputs["query"]),
        candidate_id=inputs.get("candidate_id"),
        debug=True,
        runtime_version=DEFAULT_V4_RUNTIME_VERSION,
        intent_prompt_version=DEFAULT_V4_INTENT_PROMPT_VERSION,
        recommendation_prompt_version=DEFAULT_V4_RECOMMENDATION_PROMPT_VERSION,
    )
    payload = result.model_dump()
    payload["case_id"] = inputs.get("case_id")
    payload["query"] = inputs.get("query")
    payload["candidate_id"] = inputs.get("candidate_id")
    return payload


def _build_runtime_v4_examples(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for case in cases:
        example_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"taste-agent-runtime-v4-eval:{case['case_id']}"))
        intent = parse_intent(str(case["query"]))
        examples.append(
            {
                "id": example_id,
                "inputs": {
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "candidate_id": case.get("candidate_id"),
                },
                "outputs": {
                    "expected_intent": intent.intent_type,
                    "expected_behavior": case["expected_selection_behavior"],
                    "expected_selection_behavior": case["expected_selection_behavior"],
                    "expected_explanation_behavior": case["expected_explanation_behavior"],
                    "expected_retrieval_behavior": case["expected_retrieval_behavior"],
                    "expected_qualification_status": case["expected_qualification_status"],
                    "expected_hard_constraints": case["expected_hard_constraints"],
                    "evaluation_notes": case["evaluation_notes"],
                },
                "metadata": {
                    "case_id": case["case_id"],
                    "expected_request_mode": case["expected_request_mode"],
                    "expected_retrieval_behavior": case["expected_retrieval_behavior"],
                    "expected_qualification_status": case["expected_qualification_status"],
                    "expected_selection_behavior": case["expected_selection_behavior"],
                    "expected_explanation_behavior": case["expected_explanation_behavior"],
                    "evaluation_notes": case["evaluation_notes"],
                },
            }
        )
    return examples


def _run_runtime_v4_langsmith_evaluation(
    *,
    results_dir: str | Path,
    dataset_name: str,
    experiment_prefix: str,
) -> dict[str, Any]:
    ensure_langsmith_env()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cases = load_runtime_v1_cases(Path("evaluation/langsmith/runtime_v4_dataset.json"))
    examples = _build_runtime_v4_examples(cases)
    client = _client()
    data_source: str | list[dict[str, Any]] = examples
    dataset_status = "local_examples"

    if client is not None:
        try:
            try:
                client.create_dataset(
                    dataset_name=dataset_name,
                    description="Fixed runtime_v4 evaluation dataset for Taste Agent LangSmith hosted run.",
                    metadata={
                        "runtime_version": DEFAULT_V4_RUNTIME_VERSION,
                        "source_dataset": str(Path("evaluation/langsmith/runtime_v4_dataset.json")),
                    },
                )
            except Exception as exc:
                if "already exists" not in str(exc):
                    raise
            try:
                client.create_examples(dataset_name=dataset_name, examples=examples)
            except Exception as exc:
                if "example already exists" not in str(exc).lower() and "conflict" not in str(exc).lower():
                    raise
            data_source = dataset_name
            dataset_status = "langsmith_dataset"
        except Exception:
            data_source = examples
            dataset_status = "local_examples_fallback"

    evaluators_list = [
        evaluators.candidate_compliance,
        evaluators.candidate_identity_integrity,
        evaluators.duplicate_candidates,
        evaluators.intent_match,
        evaluators.hard_constraint_satisfaction,
        evaluators.exclusion_satisfaction,
        evaluators.taste_signal_grounding,
        evaluators.unsupported_taste_dimension,
        evaluators.goodreads_ranking_guardrail,
        evaluators.interaction_routing_contract,
        evaluators.sensitive_inference_guardrail,
        evaluators.request_relevance,
        evaluators.explanation_groundedness,
        evaluators.explanation_usefulness,
        evaluators.response_clarity,
    ]

    experiment_name = None
    experiment_id = None
    experiment_url = None
    if client is not None:
        try:
            experiment = evaluate(
                _runtime_v4_target,
                data=data_source,
                evaluators=evaluators_list,
                metadata={
                    "runtime_version": DEFAULT_V4_RUNTIME_VERSION,
                    "dataset_name": dataset_name,
                    "dataset_status": dataset_status,
                    "experiment_prefix": experiment_prefix,
                },
                experiment_prefix=experiment_prefix,
                description="Hosted LangSmith evaluation for runtime_v4.",
                client=client,
                blocking=True,
                upload_results=True,
                error_handling="log",
            )
            frame = experiment.to_pandas()
            experiment_name = getattr(experiment, "experiment_name", None)
            try:
                experiment_id = str(experiment.experiment_id)
            except Exception:
                experiment_id = None
            experiment_url = getattr(experiment, "url", None)
        except Exception:
            frame = _run_local_experiment(cases, evaluators_list, _runtime_v4_target)
            dataset_status = "local_manual_fallback"
    else:
        frame = _run_local_experiment(cases, evaluators_list, _runtime_v4_target)

    results_path = results_dir / "experiment_results.csv"
    frame.to_csv(results_path, index=False)
    summary = {
        "runtime_version": DEFAULT_V4_RUNTIME_VERSION,
        "dataset_name": dataset_name,
        "dataset_status": dataset_status,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "experiment_url": experiment_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "summary": _summarize_frame(frame),
        "results_path": str(results_path),
        "manifest_path": str(
            _save_runtime_manifest(
                results_dir,
                manifest_filename="runtime_v4_manifest.json",
                root_manifest_path=DEFAULT_V4_MANIFEST_PATH,
                runtime_version=DEFAULT_V4_RUNTIME_VERSION,
                intent_prompt_version=DEFAULT_V4_INTENT_PROMPT_VERSION,
                recommendation_prompt_version=DEFAULT_V4_RECOMMENDATION_PROMPT_VERSION,
                source_recommendation_run="dynamic_tmdb_runtime_v4",
            )
        ),
    }
    (results_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
