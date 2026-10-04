from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone


RUNTIME_VERSION = "runtime_v1"
INTENT_PROMPT_VERSION = "intent_v1"
RECOMMENDATION_PROMPT_VERSION = "recommendation_v1"
GENERATION_MODEL = "gpt-4o-mini"
SOURCE_RECOMMENDATION_RUN = "20261003T075635Z"
RUNTIME_MANIFEST_PATH = Path("mvp/artifacts/generative/runtime_v1_manifest.json")
RUNTIME_V3_VERSION = "runtime_v3"
RUNTIME_V3_MANIFEST_PATH = Path("mvp/artifacts/generative/runtime_v3_manifest.json")


def runtime_manifest(
    *,
    runtime_version: str = RUNTIME_VERSION,
    intent_prompt_version: str = INTENT_PROMPT_VERSION,
    recommendation_prompt_version: str = RECOMMENDATION_PROMPT_VERSION,
    source_recommendation_run: str = SOURCE_RECOMMENDATION_RUN,
    created_at: str | None = None,
) -> dict[str, object]:
    return {
        "runtime_version": runtime_version,
        "intent_prompt_version": intent_prompt_version,
        "recommendation_prompt_version": recommendation_prompt_version,
        "generation_model": GENERATION_MODEL,
        "candidate_context_size": 50,
        "recommendation_count": 5,
        "ranking_model": "B3",
        "ranking_model_status": "exploratory_candidate_mvp",
        "taste_model": "62_dim_interpretable_profile",
        "goodreads_status": "descriptive_only",
        "source_recommendation_run": source_recommendation_run,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "b3_modified": False,
        "candidate_retrieval_modified": False,
        "semantic_taxonomy_modified": False,
        "goodreads_ranking_enabled": False,
    }
