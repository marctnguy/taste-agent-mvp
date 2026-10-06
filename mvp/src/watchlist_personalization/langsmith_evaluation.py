from __future__ import annotations

import argparse
import json
import os
import uuid
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from langsmith import Client
from langsmith.evaluation import evaluate

from evaluation.langsmith import evaluators as base_evaluators
from mvp.src.config import get_api_keys, get_langsmith_config, load_runtime_env
from mvp.src.watchlist_personalization.reversible_history_watchlist_service import (
    ReversibleHistoryWatchlistService,
    load_reversible_history_watchlist_service,
)
from mvp.src.watchlist_personalization import langsmith_evaluators as watchlist_evaluators


DEFAULT_DATASET_NAME = "taste-agent-reversible-history-watchlist-h12"
DEFAULT_EXPERIMENT_PREFIX = "taste-agent-reversible-history-watchlist"
DEFAULT_RESULTS_DIR = Path("mvp/artifacts/generative/langsmith/reversible_history_watchlist")
DEFAULT_SUITE_FILENAME = "watchlist_langsmith_suite_summary.json"
DEFAULT_MANIFEST_FILENAME = "watchlist_langsmith_manifest.json"
DEFAULT_DATASET_FILENAME = "watchlist_langsmith_dataset.json"
DEFAULT_CHECKPOINT_FILENAME = "watchlist_langsmith_checkpoint.json"
DEFAULT_CANARY_FILENAME = "watchlist_langsmith_canary.json"
DEFAULT_HUMAN_REVIEW_FILENAME = "blinded_human_review_sheet.csv"
DEFAULT_UNBLINDING_KEY_FILENAME = "unblinding_key.csv"

EXPERIMENT_CONFIGS = [
    ("A", "full_watchlist"),
    ("B", "full_watchlist"),
    ("A", "empty_watchlist"),
    ("B", "empty_watchlist"),
]
CANARY_CASE_IDS = ["H03", "H11", "H12"]


def ensure_langsmith_env() -> None:
    load_runtime_env()
    config = get_langsmith_config()
    if config.api_key:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", config.project or "taste-agent-capstone")
        if config.endpoint:
            os.environ.setdefault("LANGSMITH_ENDPOINT", config.endpoint)


def _client() -> Client | None:
    keys = get_api_keys()
    config = get_langsmith_config()
    if not keys.openai_api_key or not config.api_key:
        return None
    return Client(api_key=config.api_key)


def _dataset_metadata() -> dict[str, Any]:
    return {
        "source_prompt_path": "evaluation/hitl/runtime_v3/hitl_runtime_v3_rating_template.csv",
        "source_prompt_set": "evaluation/hitl/runtime_v3/hitl_runtime_v3_rating_template.csv",
        "dataset_version": DEFAULT_DATASET_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _example_id(case_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{DEFAULT_DATASET_NAME}:{case_id}"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _checkpoint_path(results_dir: Path) -> Path:
    return results_dir / DEFAULT_CHECKPOINT_FILENAME


def _save_checkpoint(results_dir: Path, payload: dict[str, Any]) -> None:
    _write_json(_checkpoint_path(results_dir), payload)


def _is_auth_or_billing_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ["401", "403", "authentication", "auth", "billing", "quota", "insufficient", "payment", "permission denied"])


def _retry_with_backoff(operation, *, attempts: int = 3, base_delay: float = 1.0):
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or _is_auth_or_billing_failure(exc):
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))
    if last_exc is not None:
        raise last_exc


def _prompt_rows_by_id(service: ReversibleHistoryWatchlistService) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for _, row in service.prompts.iterrows():
        rows[str(row["hitl_id"])] = row
    return rows


def _grounded_explanation(payload: dict[str, Any]) -> str:
    selection = payload.get("selection_report", {}) or {}
    runtime = payload.get("runtime_metadata", {}) or {}
    parts = [
        f"status={runtime.get('generation_status')}",
        f"eligible={selection.get('eligible_count')}",
        f"selected={selection.get('selected_count')}",
        f"validated={payload.get('validation_passed')}",
    ]
    return "; ".join(parts)


def _build_review_artifacts(
    service: ReversibleHistoryWatchlistService,
    examples: list[dict[str, Any]],
    payloads_by_experiment: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_lookup = {str(case["inputs"]["case_id"]): str(case["inputs"]["prompt"]) for case in examples}
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for experiment_key, payloads in payloads_by_experiment.items():
        scenario_label, variant = experiment_key.split(":", 1)
        for case_id, payload in payloads.items():
            prompt = prompt_lookup.get(case_id, str(payload.get("prompt") or ""))
            debug_rows = payload.get("debug", {}).get("selected_candidates", []) or []
            for selected in debug_rows:
                candidate_id = str(selected.get("candidate_id") or selected.get("source_id") or "")
                if not candidate_id:
                    continue
                key = (prompt, candidate_id)
                record = grouped.setdefault(
                    key,
                    {
                        "prompt": prompt,
                        "candidate_id": candidate_id,
                        "title": selected.get("title"),
                        "year": selected.get("year"),
                        "synopsis": selected.get("overview") or selected.get("tmdb_overview") or selected.get("document_text") or "",
                        "grounded_explanation": selected.get("qualification_reason") or selected.get("selection_reason") or "",
                        "anon_label": f"anon-{len(grouped) + 1:04d}",
                        "variant_memberships": set(),
                        "scenario_memberships": set(),
                        "rank_memberships": set(),
                    },
                )
                record["variant_memberships"].add(variant)
                record["scenario_memberships"].add(scenario_label)
                rank_value = selected.get("selection_rank") or selected.get("rank") or ""
                if rank_value != "":
                    record["rank_memberships"].add(str(rank_value))
                if not record["grounded_explanation"]:
                    record["grounded_explanation"] = selected.get("qualification_reason") or selected.get("selection_reason") or ""
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for record in grouped.values():
        blinded_rows.append(
            {
                "anon_label": record["anon_label"],
                "prompt": record["prompt"],
                "title": record["title"],
                "year": record["year"],
                "synopsis": record["synopsis"],
                "grounded_explanation": record["grounded_explanation"],
                "rating_1_5": "",
                "fit_1_5": "",
            }
        )
        key_rows.append(
            {
                "anon_label": record["anon_label"],
                "candidate_id": record["candidate_id"],
                "prompt": record["prompt"],
                "variant_memberships": sorted(record["variant_memberships"]),
                "scenario_memberships": sorted(record["scenario_memberships"]),
                "rank_memberships": sorted(record["rank_memberships"]),
            }
        )
    return pd.DataFrame(blinded_rows), pd.DataFrame(key_rows)


def _create_dataset(client: Client, examples: list[dict[str, Any]], dataset_name: str) -> tuple[str, str | None]:
    dataset_id: str | None = None
    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Frozen H01-H12 watchlist personalization dataset.",
            metadata=_dataset_metadata(),
        )
        dataset_id = str(getattr(dataset, "id", None) or getattr(dataset, "dataset_id", None) or (dataset.get("id") if isinstance(dataset, dict) else None) or "") or None
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise
    try:
        client.create_examples(dataset_name=dataset_name, examples=examples)
    except Exception as exc:
        if "already exists" not in str(exc).lower() and "conflict" not in str(exc).lower():
            raise
    return dataset_name, dataset_id


def _build_examples(service: ReversibleHistoryWatchlistService) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for _, row in service.prompts.iterrows():
        case_id = str(row["hitl_id"])
        examples.append(
            {
                "id": _example_id(case_id),
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


def _precompute_payloads(
    service: ReversibleHistoryWatchlistService,
    examples: list[dict[str, Any]],
    *,
    variant: str,
    scenario_label: str,
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for case in examples:
        inputs = case["inputs"]
        hitl_id = str(inputs.get("hitl_id") or inputs.get("case_id") or "")
        query_text = str(inputs.get("prompt") or inputs.get("query") or "")
        payloads[hitl_id] = service.recommend(
            query_text,
            scenario_label=scenario_label,
            variant=variant,
            hitl_id=hitl_id,
        )
    return payloads


def _target_factory(
    service: ReversibleHistoryWatchlistService,
    *,
    variant: str,
    scenario_label: str,
    initial_cache: dict[str, dict[str, Any]] | None = None,
):
    cache: dict[str, dict[str, Any]] = dict(initial_cache or {})

    def _target(inputs: dict[str, Any]) -> dict[str, Any]:
        hitl_id = str(inputs.get("hitl_id") or inputs.get("case_id") or "")
        if hitl_id not in cache:
            query_text = str(inputs.get("prompt") or inputs.get("query") or "")
            cache[hitl_id] = service.recommend(query_text or hitl_id, hitl_id=hitl_id, variant=variant, scenario_label=scenario_label)
        return cache[hitl_id]

    return _target


def _experiment_key(variant: str, scenario_label: str) -> str:
    return f"{scenario_label}:{variant}"


def _is_complete_experiment_record(record: dict[str, Any]) -> bool:
    return bool(record.get("results_path")) and Path(str(record["results_path"])).exists()


def _evaluator_suite() -> list[Any]:
    return [
        base_evaluators.candidate_compliance,
        base_evaluators.candidate_identity_integrity,
        watchlist_evaluators.duplicate_candidates,
        watchlist_evaluators.watched_reference_leakage,
        watchlist_evaluators.structured_constraint_compliance,
        watchlist_evaluators.output_validity,
        watchlist_evaluators.requested_slate_completion,
    ]


def _run_local_experiment(
    cases: list[dict[str, Any]],
    evaluators_list: list[Any],
    payloads: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    from types import SimpleNamespace

    rows: list[dict[str, Any]] = []
    for case in cases:
        inputs = case["inputs"]
        payload = payloads[str(inputs.get("hitl_id") or inputs.get("case_id") or "")]
        run = SimpleNamespace(outputs={"output": payload})
        example = SimpleNamespace(inputs=case["inputs"], outputs=case["outputs"], metadata=case["metadata"])
        run.id = f"run-{case['inputs']['case_id']}"
        example.id = f"example-{case['inputs']['case_id']}"
        row: dict[str, Any] = {
            "case_id": case["inputs"]["case_id"],
            "hitl_id": case["inputs"]["hitl_id"],
            "variant": payload.get("variant"),
            "scenario_label": payload.get("scenario_label"),
            "llm_call_count": payload.get("runtime_metadata", {}).get("llm_call_count"),
            "selected_count": payload.get("selection_report", {}).get("selected_count"),
            "eligible_count": payload.get("selection_report", {}).get("eligible_count"),
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


def run_reversible_history_watchlist_langsmith_suite(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
) -> dict[str, Any]:
    ensure_langsmith_env()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_json(_checkpoint_path(results_dir))
    repo_root = Path(__file__).resolve().parents[3]

    service = load_reversible_history_watchlist_service()
    examples = _build_examples(service)
    prompt_rows = _prompt_rows_by_id(service)
    client = _client()
    data_source: str | list[dict[str, Any]] = examples
    dataset_status = "local_examples"
    dataset_id: str | None = None
    if client is not None:
        try:
            data_source, dataset_id = _create_dataset(client, examples, dataset_name)
            dataset_status = "langsmith_dataset"
        except Exception:
            data_source = examples
            dataset_status = "local_examples_fallback"
    (results_dir / DEFAULT_DATASET_FILENAME).write_text(json.dumps(examples, indent=2, ensure_ascii=False, default=str))

    evaluator_list = _evaluator_suite()
    experiment_results: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    payloads_by_experiment: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint.setdefault("experiments", {})
    checkpoint.setdefault("payloads", {})
    for variant, scenario_label in EXPERIMENT_CONFIGS:
        experiment_key = _experiment_key(variant, scenario_label)
        payloads = checkpoint["payloads"].get(experiment_key)
        if payloads is None:
            payloads = _precompute_payloads(service, examples, variant=variant, scenario_label=scenario_label)
            checkpoint["payloads"][experiment_key] = payloads
            _save_checkpoint(results_dir, checkpoint)
        payloads_by_experiment[experiment_key] = payloads
        results_path = results_dir / f"experiment_results_{scenario_label}_{variant}.csv"
        cached_record = checkpoint["experiments"].get(experiment_key)
        if cached_record and _is_complete_experiment_record(cached_record) and results_path.exists():
            frame = pd.read_csv(results_path)
            experiment_results.append(cached_record)
            suite_rows.extend(frame.to_dict(orient="records"))
            continue
        target = _target_factory(service, variant=variant, scenario_label=scenario_label, initial_cache=payloads)
        experiment_name = None
        experiment_id = None
        experiment_url = None
        if client is not None:
            try:
                def _run_evaluate():
                    return evaluate(
                        target,
                        data=data_source,
                        evaluators=evaluator_list,
                        metadata={
                            "dataset_name": dataset_name,
                            "dataset_status": dataset_status,
                            "experiment_prefix": experiment_prefix,
                            "variant": variant,
                            "scenario_label": scenario_label,
                        },
                        experiment_prefix=f"{experiment_prefix}-{scenario_label}-{variant}",
                        description="LangSmith evaluation for reversible history + watchlist selection.",
                        client=client,
                        blocking=True,
                        upload_results=True,
                        error_handling="log",
                    )

                experiment = _retry_with_backoff(_run_evaluate)
                frame = experiment.to_pandas()
                experiment_name = getattr(experiment, "experiment_name", None)
                try:
                    experiment_id = str(experiment.experiment_id)
                except Exception:
                    experiment_id = None
                experiment_url = getattr(experiment, "url", None)
            except Exception:
                frame = _run_local_experiment(examples, evaluator_list, payloads)
                dataset_status = "local_manual_fallback"
        else:
            frame = _run_local_experiment(examples, evaluator_list, payloads)

        results_path = results_dir / f"experiment_results_{scenario_label}_{variant}.csv"
        frame.to_csv(results_path, index=False)
        record = {
            "variant": variant,
            "scenario_label": scenario_label,
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "experiment_url": experiment_url,
            "results_path": str(results_path),
            "row_count": int(len(frame)),
        }
        experiment_results.append(record)
        suite_rows.extend(frame.to_dict(orient="records"))
        checkpoint["experiments"][experiment_key] = record
        _save_checkpoint(results_dir, checkpoint)

    blinded_sheet, unblinding_key = _build_review_artifacts(service, examples, payloads_by_experiment)
    blinded_sheet.to_csv(results_dir / DEFAULT_HUMAN_REVIEW_FILENAME, index=False)
    unblinding_key.to_csv(results_dir / DEFAULT_UNBLINDING_KEY_FILENAME, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "dataset_status": dataset_status,
        "experiment_prefix": experiment_prefix,
        "input_hashes": {
            "prompt_manifest": _sha256_path(repo_root / "evaluation/hitl/runtime_v4_final/05_run_manifest.json"),
            "prompt_source_set": _sha256_path(repo_root / "evaluation/hitl/runtime_v3/hitl_runtime_v3_rating_template.csv"),
        },
        "code_hashes": {
            "watchlist_service": _sha256_path(repo_root / "mvp/src/watchlist_personalization/reversible_history_watchlist_service.py"),
            "watchlist_langsmith_evaluation": _sha256_path(repo_root / "mvp/src/watchlist_personalization/langsmith_evaluation.py"),
            "watchlist_experiment_script": _sha256_path(repo_root / "evaluation/watchlist_personalization/run_reversible_history_watchlist_experiment.py"),
            "watchlist_isolated_experiment_script": _sha256_path(repo_root / "evaluation/watchlist_personalization/run_isolated_experiment.py"),
        },
        "case_ids": [example["inputs"]["case_id"] for example in examples],
        "example_ids": [example["id"] for example in examples],
        "experiments": experiment_results,
    }
    _write_json(results_dir / DEFAULT_MANIFEST_FILENAME, manifest)
    suite_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "dataset_status": dataset_status,
        "experiment_prefix": experiment_prefix,
        "experiments": experiment_results,
        "case_ids": [example["inputs"]["case_id"] for example in examples],
        "example_ids": [example["id"] for example in examples],
        "manifest_path": str((results_dir / DEFAULT_MANIFEST_FILENAME).resolve()),
        "human_review_sheet_path": str((results_dir / DEFAULT_HUMAN_REVIEW_FILENAME).resolve()),
        "unblinding_key_path": str((results_dir / DEFAULT_UNBLINDING_KEY_FILENAME).resolve()),
        "benchmark_authorized": bool(all(row.get("output_validity", 0) == 1 and row.get("requested_slate_completion", 0) == 1 for row in suite_rows)),
    }
    _write_json(results_dir / DEFAULT_SUITE_FILENAME, suite_summary)
    _write_json(results_dir / "watchlist_langsmith_results.json", {"rows": suite_rows, "blinded_sheet_rows": int(len(blinded_sheet)), "unblinding_key_rows": int(len(unblinding_key))})
    checkpoint["suite_summary"] = suite_summary
    checkpoint["completed_at"] = datetime.now(timezone.utc).isoformat()
    _save_checkpoint(results_dir, checkpoint)
    return suite_summary


def run_reversible_history_watchlist_pipeline(
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = DEFAULT_EXPERIMENT_PREFIX,
) -> dict[str, Any]:
    service = load_reversible_history_watchlist_service()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_json(_checkpoint_path(results_dir))
    prompt_rows = _prompt_rows_by_id(service)

    canary_reports: list[dict[str, Any]] = []
    for hitl_id in CANARY_CASE_IDS:
        if hitl_id not in prompt_rows:
            continue
        prompt = str(prompt_rows[hitl_id]["prompt"])
        canary = service.recommend(prompt, scenario_label="full_watchlist", variant="A", hitl_id=hitl_id)
        selection_report = canary.get("selection_report", {}) or {}
        runtime_metadata = canary.get("runtime_metadata", {}) or {}
        selected = canary.get("response", {}).get("recommendations", []) or []
        grounded_selected = [row for row in selected if row.get("qualification_reason")]
        report = {
            "hitl_id": hitl_id,
            "scenario_label": "full_watchlist",
            "variant": "A",
            "selected_count": int(selection_report.get("selected_count", 0) or 0),
            "eligible_count": int(selection_report.get("eligible_count", 0) or 0),
            "generation_status": runtime_metadata.get("generation_status"),
            "validation_passed": bool(canary.get("validation_passed", False)),
            "grounded_selected_count": int(len(grounded_selected)),
            "prompt": prompt,
        }
        if hitl_id == "H11" and (report["eligible_count"] < 1 or report["selected_count"] < 1):
            raise RuntimeError(json.dumps({"canary": report, "error": "H11 did not produce at least one grounded eligible result."}, indent=2, ensure_ascii=False, default=str))
        if hitl_id != "H11" and report["generation_status"] not in {"generated", "no_match"}:
            raise RuntimeError(json.dumps({"canary": report, "error": "Canary did not complete with a valid prepared result."}, indent=2, ensure_ascii=False, default=str))
        canary_reports.append(report)
    checkpoint["canary"] = canary_reports
    checkpoint["canary_completed_at"] = datetime.now(timezone.utc).isoformat()
    _save_checkpoint(results_dir, checkpoint)
    (results_dir / DEFAULT_CANARY_FILENAME).write_text(json.dumps(canary_reports, indent=2, ensure_ascii=False, default=str))

    suite = run_reversible_history_watchlist_langsmith_suite(
        results_dir=results_dir,
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
    )
    suite["canary"] = canary_reports
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reversible history watchlist evaluation pipeline.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--experiment-prefix", default=DEFAULT_EXPERIMENT_PREFIX)
    args = parser.parse_args()
    print(
        json.dumps(
            run_reversible_history_watchlist_pipeline(
                results_dir=args.results_dir,
                dataset_name=args.dataset_name,
                experiment_prefix=args.experiment_prefix,
            ),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
