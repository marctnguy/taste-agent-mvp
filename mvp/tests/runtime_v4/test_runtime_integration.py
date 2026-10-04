from __future__ import annotations

from pathlib import Path

import pandas as pd

from mvp.src.generative.schemas import RecommendationItem, RecommendationResponse
from mvp.src.generative_v4.explanation_chain import ExplanationResult
from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.generative_v4.runtime import run_runtime_v4
from mvp.src.generative_v4.schemas import QualificationRecord, SelectionRecord
from mvp.src.generative_v4.selection_chain import SelectionResult
from mvp.src.generative_v4.validators import ValidationResult
from mvp.src.retrieval.catalog_retrieval import CatalogRetrievalDiagnostics, CatalogRetrievalResult


def test_runtime_v4_stage_order_and_unchanged_artifacts(monkeypatch, dummy_trace_factory) -> None:
    events: list[tuple[str, str, object | None]] = []
    monkeypatch.setattr("mvp.src.generative_v4.runtime.trace", dummy_trace_factory(events))

    request = understand_request("I want a film about performers and fame").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    candidate_frame = pd.DataFrame(
        [
            {
                "source_id": "1002",
                "tmdb_id": 1002,
                "title": "Stage Lights",
                "year": 1998,
                "release_year": 1998,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "GB",
                "tmdb_overview": "A celebrated stage actress struggles with celebrity in the theatre world.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["2"],
                "predicted_preference": 0.55,
                "rank": 2,
                "request_relevance": 0.95,
                "b3_applicability_distance": 0.75,
            }
        ]
    )
    diagnostics = CatalogRetrievalDiagnostics(
        candidate_universe_count=1,
        post_constraint_candidate_count=1,
        contextual_retrieval_used=True,
        candidate_context_size=50,
        shortlist_size=1,
        retrieval_mode="query_aware",
        request_relevance_mode="query_aware",
        reference_status="absent",
        reference_tmdb_id=None,
        reference_title=None,
        request_relevance_method="embedding",
        query_aware_augmentation_used=True,
        watched_excluded_count=0,
        candidate_limit=120,
        b3_applicability_mode="nearest_neighbor_to_frozen_training_embeddings",
    )
    retrieval = CatalogRetrievalResult(
        request=request,
        catalog_frame=candidate_frame.copy(),
        documents=pd.DataFrame([{"source_id": "1002", "title": "Stage Lights", "document_text": "Title: stage lights"}]),
        request_embedding=None,
        retrieval_diagnostics=diagnostics,
        augmentation_report={"query_aware_augmentation_used": True},
    )

    def fake_qualify_candidates(request, frame, max_candidates=20):
        qualified = frame.copy()
        qualified.loc[:, "qualification_status"] = ["strong"]
        qualified.loc[:, "supported_request_aspects"] = [["performers", "fame"]]
        qualified.loc[:, "unsupported_request_aspects"] = [[]]
        qualified.loc[:, "grounded_evidence"] = [["stage actress", "celebrity"]]
        qualified.loc[:, "qualification_reason"] = ["Candidate has grounded evidence that directly supports the request."]
        qualified.loc[:, "request_match"] = ["Supported request aspects: performers, fame"]
        qualified.loc[:, "caveat"] = [None]
        records = [
            QualificationRecord(
                candidate_id="1002",
                qualification_status="strong",
                supported_request_aspects=["performers", "fame"],
                unsupported_request_aspects=[],
                grounded_evidence=["stage actress", "celebrity"],
                qualification_reason="Candidate has grounded evidence that directly supports the request.",
                request_match="Supported request aspects: performers, fame",
                caveat=None,
                llm_used=False,
            )
        ]
        return qualified, records

    def fake_select_candidates(request, qualified_frame, recommendation_count=5):
        selected = qualified_frame.copy()
        selected.loc[:, "selection_rank"] = [1]
        selected.loc[:, "selection_reason"] = [
            "Contextual selection prioritized request-fit first, then frozen B3 compatibility among qualified candidates."
        ]
        records = [
            SelectionRecord(
                candidate_id="1002",
                selection_rank=1,
                selection_reason="Contextual selection prioritized request-fit first, then frozen B3 compatibility among qualified candidates.",
                request_relevance=0.95,
                b3_predicted_preference=0.55,
                qualification_status="strong",
            )
        ]
        return SelectionResult(selected, records, {"selected_count": 1, "abstained": False, "novelty_requested": False, "request_mode": "contextual"})

    def fake_explain_selection(request, selected_frame):
        selected = selected_frame.copy()
        selected.loc[:, "semantic_evidence"] = [[{"dimension": "melancholic"}]]
        response = RecommendationResponse(
            intent=request.intent,
            recommendations=[
                RecommendationItem(
                    candidate_id="1002",
                    title="Stage Lights",
                    year=1998,
                    why_it_may_fit="Supported request aspects: performers, fame. Taste evidence: melancholic.",
                    taste_signals=["melancholic"],
                    request_match="Supported request aspects: performers, fame",
                    caveat=None,
                )
            ],
            response_summary="Returned 1 grounded recommendation from the qualified catalog.",
            methodology_note="Request relevance was checked first, then frozen B3 compatibility was used to order only the qualified candidates.",
        )
        return ExplanationResult(response=response, selected_frame=selected, semantic_report={"semantic_classified": 1, "coverage": 1.0})

    monkeypatch.setattr("mvp.src.generative_v4.runtime.discover_catalog_for_request", lambda *args, **kwargs: retrieval)
    monkeypatch.setattr("mvp.src.generative_v4.runtime.qualify_candidates", fake_qualify_candidates)
    monkeypatch.setattr("mvp.src.generative_v4.runtime.select_candidates", fake_select_candidates)
    monkeypatch.setattr("mvp.src.generative_v4.runtime.explain_selection", fake_explain_selection)
    monkeypatch.setattr("mvp.src.generative_v4.runtime.validate_runtime_v4_result", lambda result: ValidationResult(True, []))

    artifact_paths = [
        Path("mvp/artifacts/generative/runtime_v1_manifest.json"),
        Path("mvp/artifacts/generative/runtime_v2_manifest.json"),
        Path("mvp/artifacts/generative/runtime_v3_manifest.json"),
    ]
    before = {path: path.read_bytes() for path in artifact_paths if path.exists()}

    result = run_runtime_v4("I want a film about performers and fame")

    after = {path: path.read_bytes() for path in artifact_paths if path.exists()}
    assert result.response.recommendations[0].candidate_id == "1002"
    assert result.debug["candidate_context"][0]["candidate_id"] == "1002"
    assert result.runtime_metadata.interaction_mode == "recommend"
    assert result.runtime_metadata.abstention_used is False
    assert result.runtime_metadata.intent_interpretation_llm_used is False
    assert result.runtime_metadata.semantic_qualification_llm_used is False
    assert [name for kind, name, _ in events if kind == "enter"] == [
        "taste_agent_request_v4",
        "intent_interpretation",
        "retrieval_plan",
        "interaction_routing",
        "catalog_retrieval",
        "semantic_retrieval",
        "hard_constraints",
        "semantic_qualification",
        "b3_personalization",
        "novelty_diagnostic",
        "taste_evidence",
        "selection",
        "selection_validation",
        "explanation",
        "explanation_validation",
        "final_response",
    ]
    assert before == after
