from __future__ import annotations

import pandas as pd

from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.generative_v4.runtime import run_runtime_v4
from mvp.src.retrieval.catalog_retrieval import CatalogRetrievalDiagnostics, CatalogRetrievalResult


def test_runtime_abstains_cleanly_when_no_candidate_survives(monkeypatch, dummy_trace_factory) -> None:
    events: list[tuple[str, str, object | None]] = []
    monkeypatch.setattr("mvp.src.generative_v4.runtime.trace", dummy_trace_factory(events))

    request = understand_request("Recommend me something to watch.")
    diagnostics = CatalogRetrievalDiagnostics(
        candidate_universe_count=0,
        post_constraint_candidate_count=0,
        contextual_retrieval_used=False,
        candidate_context_size=50,
        shortlist_size=0,
        retrieval_mode="no_match",
        request_relevance_mode="broad",
        reference_status="absent",
        reference_tmdb_id=None,
        reference_title=None,
        request_relevance_method="no_candidates",
        query_aware_augmentation_used=False,
        watched_excluded_count=0,
        candidate_limit=120,
        b3_applicability_mode="nearest_neighbor_to_frozen_training_embeddings",
    )
    retrieval = CatalogRetrievalResult(
        request=request,
        catalog_frame=pd.DataFrame(),
        documents=pd.DataFrame(),
        request_embedding=None,
        retrieval_diagnostics=diagnostics,
        augmentation_report={},
    )
    monkeypatch.setattr("mvp.src.generative_v4.runtime.discover_catalog_for_request", lambda *args, **kwargs: retrieval)

    result = run_runtime_v4("Recommend me something to watch.")

    assert result.runtime_metadata.generation_status == "no_match"
    assert result.response.recommendations == []
    assert result.response.intent.free_text_context == "Recommend me something to watch."
    assert result.debug["request"]["query"] == "Recommend me something to watch."
    assert result.runtime_metadata.abstention_used is True
    assert result.runtime_metadata.selection_llm_used is False
    assert result.runtime_metadata.explanation_llm_used is False
    assert result.validation_passed is True
    assert [name for kind, name, _ in events if kind == "enter"] == [
        "taste_agent_request_v4",
        "intent_interpretation",
        "retrieval_plan",
        "interaction_routing",
        "catalog_retrieval",
        "semantic_retrieval",
        "hard_constraints",
        "semantic_qualification",
    ]
