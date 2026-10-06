from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from mvp.src.watchlist_personalization import langsmith_evaluation as runner


def _fake_service() -> SimpleNamespace:
    prompts = pd.DataFrame(
        [
            {
                "hitl_id": f"H{index:02d}",
                "prompt": f"Prompt {index:02d}",
                "test_purpose": "Frozen prompt",
                "taste_fit_1_5": "",
                "request_fit_1_5": "",
                "discovery_value_1_5": "",
                "explanation_usefulness_1_5": "",
                "would_actually_watch": "",
                "already_knew_titles": "",
                "heard_of_titles": "",
                "new_to_me_titles": "",
                "best_recommendation": "",
                "worst_recommendation": "",
                "qualitative_feedback": "",
            }
            for index in range(1, 13)
        ]
    )

    call_log: list[tuple[str, str, str]] = []

    def recommend(query_text: str, *, scenario_label: str = "full_watchlist", variant: str = "A", hitl_id: str | None = None):
        case_id = hitl_id or "H00"
        call_log.append((case_id, scenario_label, variant))
        candidate_id = f"film-{case_id}"
        payload = {
            "case_id": case_id,
            "hitl_id": case_id,
            "prompt": query_text,
            "scenario_label": scenario_label,
            "variant": variant,
            "requested_count": 1,
            "response": {
                "recommendations": [
                    {
                        "candidate_id": candidate_id,
                        "title": f"Title {case_id}",
                        "year": 2000,
                        "qualification_reason": "grounded",
                        "selection_rank": 1,
                    }
                ],
                "response_summary": f"{case_id} {variant} {scenario_label}",
            },
            "runtime_metadata": {
                "generation_status": "generated",
                "validation_passed": True,
                "llm_call_count": 1,
            },
            "selection_report": {
                "selected_count": 1,
                "eligible_count": 1,
                "requested_count": 1,
            },
            "validation_passed": True,
            "debug": {
                "selected_candidates": [
                    {
                        "candidate_id": candidate_id,
                        "title": f"Title {case_id}",
                        "year": 2000,
                        "overview": f"Synopsis for {case_id}",
                        "qualification_reason": "grounded",
                        "selection_rank": 1,
                    }
                ],
                "candidate_context": [
                    {
                        "candidate_id": candidate_id,
                        "title": f"Title {case_id}",
                        "year": 2000,
                        "overview": f"Synopsis for {case_id}",
                        "qualification_reason": "grounded",
                    }
                ],
                "route_reports": {},
                "parsed_request": {"intent": {}},
                "controlled_request_spec": {},
            },
        }
        return payload

    service = SimpleNamespace(prompts=prompts, recommend=recommend, prompt_row=lambda hitl_id: prompts.loc[prompts["hitl_id"] == hitl_id].iloc[0])
    service.call_log = call_log
    return service


def test_pipeline_writes_canary_checkpoint_and_resumes_without_recomputing_suite(monkeypatch, tmp_path):
    service = _fake_service()
    monkeypatch.setattr(runner, "load_reversible_history_watchlist_service", lambda: service)
    monkeypatch.setattr(runner, "_client", lambda: None)
    monkeypatch.setattr(runner, "_evaluator_suite", lambda: [])

    first = runner.run_reversible_history_watchlist_pipeline(results_dir=tmp_path, dataset_name="test-dataset", experiment_prefix="test-prefix")

    assert len(service.call_log) == 51
    assert (tmp_path / runner.DEFAULT_CHECKPOINT_FILENAME).exists()
    assert (tmp_path / runner.DEFAULT_CANARY_FILENAME).exists()
    assert (tmp_path / runner.DEFAULT_HUMAN_REVIEW_FILENAME).exists()
    assert (tmp_path / runner.DEFAULT_UNBLINDING_KEY_FILENAME).exists()
    assert (tmp_path / runner.DEFAULT_MANIFEST_FILENAME).exists()
    assert first["canary"][1]["hitl_id"] == "H11"

    checkpoint = json.loads((tmp_path / runner.DEFAULT_CHECKPOINT_FILENAME).read_text())
    assert checkpoint["canary_completed_at"]
    assert len(checkpoint["canary"]) == 3
    assert checkpoint["experiments"]

    second = runner.run_reversible_history_watchlist_pipeline(results_dir=tmp_path, dataset_name="test-dataset", experiment_prefix="test-prefix")

    assert len(service.call_log) == 54
    assert second["canary"][0]["hitl_id"] == "H03"

    blinded = pd.read_csv(tmp_path / runner.DEFAULT_HUMAN_REVIEW_FILENAME)
    key = pd.read_csv(tmp_path / runner.DEFAULT_UNBLINDING_KEY_FILENAME)
    assert set(blinded.columns) == {"anon_label", "prompt", "title", "year", "synopsis", "grounded_explanation", "rating_1_5", "fit_1_5"}
    assert blinded["rating_1_5"].isna().all()
    assert blinded["fit_1_5"].isna().all()
    assert len(blinded) == len(key) == 12
