from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evaluation.watchlist_personalization import langsmith_evaluators
from mvp.src.generative_v4.schemas import (
    NoveltyGoal,
    PersonalizationInstruction,
    QualificationRecord,
    ReferenceSpec,
    RequestSpec,
    RequestUnderstanding,
    RetrievalPlan,
    RecommendationIntent,
    SemanticConcept,
    SemanticRequirementGroup,
    StructuredConstraints,
)
from mvp.src.watchlist_personalization import langsmith_evaluation
from mvp.src.watchlist_personalization import reversible_history_watchlist_service as service_mod
from mvp.src.watchlist_personalization.reversible_history_watchlist_service import ReversibleHistoryWatchlistService


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    service_mod._PREPARED_CASE_CACHE.clear()


def _prompt_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hitl_id": f"H{index:02d}",
                "test_purpose": "Frozen prompt",
                "prompt": f"Prompt {index}",
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


def _request(
    query: str,
    *,
    mode: str = "contextual",
    novelty: bool = False,
    groups: list[SemanticRequirementGroup] | None = None,
    structured: StructuredConstraints | None = None,
    requested_taste_dimensions: list[str] | None = None,
    exclusions: list[str] | None = None,
) -> RequestUnderstanding:
    structured = structured or StructuredConstraints()
    groups = groups or []
    semantic_requirements = [
        concept
        for group in groups
        for concept in group.concepts
    ]
    intent = RecommendationIntent(
        intent_type="MOOD_THEME",
        requested_taste_dimensions=requested_taste_dimensions or [concept.concept for concept in semantic_requirements if concept.concept],
        requested_genres=list(structured.required_genres),
        requested_languages=list(structured.required_languages),
        requested_countries=list(structured.required_countries),
        requested_decades=list(structured.required_decades),
        exclusions=exclusions or [concept.concept for concept in semantic_requirements if concept.concept and concept.importance != "required"],
        free_text_context=query,
    )
    spec = RequestSpec(
        intent_type="MOOD_THEME",
        mode="contextual" if mode != "generic" else "generic",
        structured_constraints=structured,
        semantic_requirement_groups=groups,
        semantic_requirements=semantic_requirements,
        semantic_exclusions=[],
        reference=ReferenceSpec(),
        novelty_goal=NoveltyGoal(enabled=novelty, priority="history_distance" if novelty else None),
        personalization_instruction=PersonalizationInstruction(),
        request_relevance_mode="query_aware" if (groups or novelty or structured.required_genres or structured.required_countries or structured.required_languages or structured.required_decades) else "broad",
        semantic_query_text=query,
        request_summary=query,
    )
    return RequestUnderstanding(
        query=query,
        intent=intent,
        spec=spec,
        request_mode="novelty" if novelty else mode,
        interaction_mode="recommend",
        novelty_requested=novelty,
        reference_status="absent",
        request_relevance_mode=spec.request_relevance_mode,
        retrieval_plan=RetrievalPlan(
            request_mode=spec.mode,
            structured_constraints=structured,
            semantic_query_text=query,
            semantic_concepts=[],
            keyword_resolutions=[],
            reference_seed=ReferenceSpec(),
            query_aware_expansion_required=True,
            supplemental_seed_sources=[],
        ),
    )


def _service() -> ReversibleHistoryWatchlistService:
    prompts = _prompt_frame()
    frame = pd.DataFrame(
        [
            {
                "source_id": "101",
                "tmdb_id": 101,
                "title": "Complete Film",
                "year": 2001,
                "genres": ["Drama"],
                "original_language": "en",
                "production_countries": ["FR"],
                "overview": "complete synopsis",
                "candidate_sources": ["request_external"],
                "candidate_source_ranks": ["1"],
                "route_memberships": ["request_external"],
                "route_origin": "external",
                "request_relevance": 0.9,
                "direct_taste_score": 0.8,
                "novelty_score": 0.1,
                "qualification_status": "strong",
                "supported_required_aspects": ["performers"],
                "supported_preferred_aspects": ["period_setting"],
                "unsupported_required_aspects": [],
                "unsupported_preferred_aspects": [],
                "violated_semantic_exclusions": [],
                "grounded_evidence": ["performers:overview:stage actress"],
                "grounded_evidence_details": [{"aspect_id": "performers", "field": "overview", "evidence": "stage actress"}],
                "request_match": "Supported request aspects: performers",
                "caveat": None,
                "qualification_reason": "Grounded evidence was sufficient.",
            }
        ]
    )
    return ReversibleHistoryWatchlistService(
        prompts=prompts,
        inputs={},
        condition_a=pd.DataFrame(),
        split=pd.DataFrame(),
        train_embeddings=pd.DataFrame(),
        watchlist=pd.DataFrame(),
        consumed_keys={"source": set(), "tmdb": set(), "title_year": set()},
        tmdb_client=None,
        historical={
            "positive": pd.DataFrame(),
            "negative": pd.DataFrame(),
            "emb_cols": [],
            "taste_profile": pd.DataFrame(),
        },
        watched_history_embeddings=pd.DataFrame(),
        full_watchlist_resolved=frame.copy(),
        sparse_watchlist_resolved=frame.copy(),
        empty_watchlist_resolved=pd.DataFrame(columns=frame.columns),
        preflight={"services": {"tmdb": {"state": "ok"}, "openai": {"state": "ok"}}},
    )


def _qualify_keep_only(frame: pd.DataFrame, keep_ids: set[str] | None = None):
    keep_ids = keep_ids or set(frame["source_id"].astype(str).tolist())
    rows = frame.loc[frame["source_id"].astype(str).isin(keep_ids)].copy().reset_index(drop=True)
    rows = rows.loc[rows["qualification_status"].astype(str).isin({"strong", "partial"})].copy().reset_index(drop=True)
    records = [
        QualificationRecord(
            candidate_id=str(row["source_id"]),
            qualification_status=row["qualification_status"],
            supported_required_aspects=list(row.get("supported_required_aspects", [])),
            unsupported_required_aspects=list(row.get("unsupported_required_aspects", [])),
            supported_request_aspects=list(row.get("supported_required_aspects", [])),
            unsupported_request_aspects=list(row.get("unsupported_required_aspects", [])),
            supported_preferred_aspects=list(row.get("supported_preferred_aspects", [])),
            unsupported_preferred_aspects=list(row.get("unsupported_preferred_aspects", [])),
            violated_semantic_exclusions=list(row.get("violated_semantic_exclusions", [])),
            grounded_evidence=list(row.get("grounded_evidence", [])),
            grounded_evidence_details=list(row.get("grounded_evidence_details", [])),
            qualification_reason=row.get("qualification_reason", ""),
            request_match=row.get("request_match"),
            caveat=row.get("caveat"),
            llm_used=False,
        )
        for _, row in rows.iterrows()
    ]
    rows.attrs["llm_usage"] = {"llm_call_count": 0, "runtime_input_tokens": 0, "runtime_output_tokens": 0, "runtime_total_tokens": 0}
    return rows, records


def test_unseen_query_works_without_benchmark_lookup_and_is_case_insensitive(monkeypatch):
    service = _service()
    calls: list[str] = []

    monkeypatch.setattr(ReversibleHistoryWatchlistService, "prompt_row_for_query", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("benchmark lookup should not be used")))
    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query, mode="generic"))
    monkeypatch.setattr(
        ReversibleHistoryWatchlistService,
        "_build_candidate_pool",
        lambda self, request, *, profile=None, watchlist=None: (
            service.full_watchlist_resolved.copy(),
            {"request_external": {"count": 1}},
            {"request": {"101"}, "history": set(), "full_watchlist": {"101"}, "sparse_watchlist": {"101"}, "empty_watchlist": set()},
        ),
    )
    monkeypatch.setattr(service_mod, "qualify_candidates", lambda request, frame, max_candidates=20: _qualify_keep_only(frame))

    first = service.recommend("Something fresh", scenario_label="full_watchlist", variant="A")
    second = service.recommend("SOMETHING FRESH", scenario_label="full_watchlist", variant="A")

    assert first["response"]["recommendations"][0]["candidate_id"] == "101"
    assert second["response"]["recommendations"][0]["candidate_id"] == "101"
    assert first["debug"]["prepared_fingerprint"] == second["debug"]["prepared_fingerprint"]


def test_request_groups_preferences_and_novelty_reach_downstream_components(monkeypatch):
    service = _service()
    captured: list[RequestUnderstanding] = []
    groups = [
        SemanticRequirementGroup(group_id="required", mode="all_of", concepts=[SemanticConcept(aspect_id="performers", concept="performers", importance="required")]),
        SemanticRequirementGroup(group_id="either", mode="any_of", concepts=[SemanticConcept(aspect_id="memory", concept="memory"), SemanticConcept(aspect_id="identity", concept="identity")]),
        SemanticRequirementGroup(group_id="preferred", mode="preferred", concepts=[SemanticConcept(aspect_id="period_setting", concept="period_setting", importance="preferred")]),
    ]

    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query, novelty=True, groups=groups))

    def _build(self, request, *, profile=None, watchlist=None):
        captured.append(request)
        return (
            service.full_watchlist_resolved.copy(),
            {"request_external": {"count": 1}, "history_external": {"count": 1}},
            {"request": {"101"}, "history": {"101"}, "full_watchlist": {"101"}, "sparse_watchlist": {"101"}, "empty_watchlist": set()},
        )

    monkeypatch.setattr(ReversibleHistoryWatchlistService, "_build_candidate_pool", _build)
    monkeypatch.setattr(service_mod, "qualify_candidates", lambda request, frame, max_candidates=20: _qualify_keep_only(frame))

    service.recommend("performers memory identity", scenario_label="full_watchlist", variant="B")

    assert captured
    spec = captured[0].spec
    assert [group.mode for group in spec.semantic_requirement_groups] == ["all_of", "any_of", "preferred"]
    assert spec.novelty_goal.enabled is True
    assert spec.request_relevance_mode == "query_aware"


def test_duplicate_metadata_preserves_complete_evidence_and_embeddings_stay_stable(monkeypatch):
    service = _service()
    request = _request("first query")
    call_count = {"value": 0}

    def _discover(req, client=None, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            frame = pd.DataFrame(
                [
                    {
                        "source_id": "201",
                        "title": "Duplicate Film",
                        "year": 2002,
                        "tmdb_id": 201,
                        "genres": ["Drama"],
                        "original_language": "en",
                        "production_countries": ["FR"],
                        "overview": "incomplete",
                        "candidate_sources": ["request_external"],
                        "candidate_source_ranks": ["1"],
                        "route_memberships": ["request_external"],
                    }
                ]
            )
        else:
            frame = pd.DataFrame(
                [
                    {
                        "source_id": "201",
                        "title": "Duplicate Film",
                        "year": 2002,
                        "tmdb_id": 201,
                        "genres": ["Drama"],
                        "original_language": "en",
                        "production_countries": ["FR"],
                        "overview": "complete synopsis with much more detail",
                        "candidate_sources": ["history_external"],
                        "candidate_source_ranks": ["1"],
                        "route_memberships": ["history_external"],
                    }
                ]
            )
        return SimpleNamespace(catalog_frame=frame, augmentation_report={"count": 1})

    captured_docs: list[list[str]] = []

    def _embed_documents(documents: pd.DataFrame):
        captured_docs.append(documents["document_text"].tolist())
        frame = documents.loc[:, ["source_id", "title", "document_hash"]].copy()
        frame["embedding_model"] = "local"
        frame["embedding_dim"] = 2
        frame["status"] = "success"
        frame["emb_0000"] = [1.0] * len(frame)
        frame["emb_0001"] = [0.0] * len(frame)
        return frame, frame.copy(), {"cache_status": "local", "cache_hits": 0, "cache_misses": len(frame)}

    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query, mode="generic"))
    monkeypatch.setattr(service_mod, "discover_catalog_for_request", _discover)
    monkeypatch.setattr(service_mod, "qualify_candidates", lambda request, frame, max_candidates=20: _qualify_keep_only(frame))
    monkeypatch.setattr("mvp.src.retrieval.catalog_retrieval.embed_documents", _embed_documents)
    monkeypatch.setattr(service_mod, "embed_text", lambda text: np.array([1.0, 0.0]))

    first = service._build_candidate_pool(request, profile=None, watchlist=None)
    service_mod._PREPARED_CASE_CACHE.clear()
    second = service._build_candidate_pool(_request("second query"), profile=None, watchlist=None)

    assert first[0].iloc[0]["overview"] == "complete synopsis with much more detail"
    assert second[0].iloc[0]["overview"] == "complete synopsis with much more detail"
    assert captured_docs[0] == captured_docs[1]


def test_preparation_and_qualification_happen_once_across_four_policy_views(monkeypatch):
    service = _service()
    prep_calls = {"build": 0, "qualify": 0}
    master = service.full_watchlist_resolved.copy()

    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query, mode="generic"))

    def _build(self, request, *, profile=None, watchlist=None):
        prep_calls["build"] += 1
        return (
            master.copy(),
            {"request_external": {"count": 1}},
            {"request": {"101"}, "history": set(), "full_watchlist": {"101"}, "sparse_watchlist": {"101"}, "empty_watchlist": set()},
        )

    def _qualify(request, frame, max_candidates=20):
        prep_calls["qualify"] += 1
        return _qualify_keep_only(frame)

    monkeypatch.setattr(ReversibleHistoryWatchlistService, "_build_candidate_pool", _build)
    monkeypatch.setattr(service_mod, "qualify_candidates", _qualify)

    service.recommend("same query", scenario_label="full_watchlist", variant="A")
    service.recommend("same query", scenario_label="full_watchlist", variant="B")
    service.recommend("same query", scenario_label="empty_watchlist", variant="A")
    service.recommend("same query", scenario_label="empty_watchlist", variant="B")

    assert prep_calls["build"] == 1
    assert prep_calls["qualify"] == 1


def test_empty_sparse_watchlists_and_missing_negative_ratings_work(monkeypatch):
    service = _service()
    master = pd.DataFrame(
        [
            {
                "source_id": "101",
                "title": "Taste Match",
                "year": 2001,
                "tmdb_id": 101,
                "genres": ["Drama"],
                "original_language": "en",
                "production_countries": ["FR"],
                "overview": "taste match",
                "candidate_sources": ["request_external"],
                "candidate_source_ranks": ["1"],
                "route_memberships": ["request_external"],
                "request_relevance": 0.4,
                "direct_taste_score": 0.75,
                "novelty_score": 0.2,
                "qualification_status": "strong",
                "supported_required_aspects": ["performers"],
                "supported_preferred_aspects": [],
                "unsupported_required_aspects": [],
                "unsupported_preferred_aspects": [],
                "violated_semantic_exclusions": [],
                "grounded_evidence": ["performers:overview:stage actress"],
                "grounded_evidence_details": [{"aspect_id": "performers", "field": "overview", "evidence": "stage actress"}],
                "request_match": "Supported request aspects: performers",
                "caveat": None,
                "qualification_reason": "Grounded evidence was sufficient.",
            }
        ]
    )

    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query, mode="generic"))
    monkeypatch.setattr(
        ReversibleHistoryWatchlistService,
        "_build_candidate_pool",
        lambda self, request, *, profile=None, watchlist=None: (
            master.copy(),
            {"request_external": {"count": 1}},
            {"request": {"101"}, "history": {"101"}, "full_watchlist": {"101"}, "sparse_watchlist": {"101"}, "empty_watchlist": set()},
        ),
    )
    monkeypatch.setattr(service_mod, "qualify_candidates", lambda request, frame, max_candidates=20: _qualify_keep_only(frame))

    payload = service.recommend("empty watchlist test", scenario_label="empty_watchlist", variant="B", requested_count=1)

    assert payload["selection_report"]["selected_count"] == 1
    assert payload["response"]["recommendations"][0]["direct_taste_score"] == pytest.approx(0.75)
    assert payload["response"]["recommendations"][0]["predicted_preference"] == pytest.approx(1.0)


def test_qualification_evidence_survives_and_unsupported_candidates_are_excluded(monkeypatch):
    service = _service()
    rows = pd.DataFrame(
        [
            {
                "source_id": "101",
                "title": "Supported Film",
                "year": 2001,
                "tmdb_id": 101,
                "genres": ["Drama"],
                "original_language": "en",
                "production_countries": ["FR"],
                "overview": "complete synopsis",
                "candidate_sources": ["request_external"],
                "candidate_source_ranks": ["1"],
                "route_memberships": ["request_external"],
                "request_relevance": 0.9,
                "direct_taste_score": 0.8,
                "novelty_score": 0.1,
                "qualification_status": "strong",
                "supported_required_aspects": ["performers"],
                "supported_preferred_aspects": ["period_setting"],
                "unsupported_required_aspects": [],
                "unsupported_preferred_aspects": [],
                "violated_semantic_exclusions": [],
                "grounded_evidence": ["performers:overview:stage actress"],
                "grounded_evidence_details": [{"aspect_id": "performers", "field": "overview", "evidence": "stage actress"}],
                "request_match": "Supported request aspects: performers",
                "caveat": None,
                "qualification_reason": "Grounded evidence was sufficient.",
            },
            {
                "source_id": "202",
                "title": "Unsupported Film",
                "year": 2005,
                "tmdb_id": 202,
                "genres": ["Comedy"],
                "original_language": "en",
                "production_countries": ["US"],
                "overview": "missing",
                "candidate_sources": ["history_external"],
                "candidate_source_ranks": ["1"],
                "route_memberships": ["history_external"],
                "request_relevance": 0.1,
                "direct_taste_score": 0.2,
                "novelty_score": 0.3,
                "qualification_status": "unsupported",
                "supported_required_aspects": [],
                "supported_preferred_aspects": [],
                "unsupported_required_aspects": ["performers"],
                "unsupported_preferred_aspects": [],
                "violated_semantic_exclusions": ["cheesy"],
                "grounded_evidence": [],
                "grounded_evidence_details": [],
                "request_match": None,
                "caveat": "unsupported",
                "qualification_reason": "No grounded evidence.",
            },
        ]
    )

    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query))
    monkeypatch.setattr(
        ReversibleHistoryWatchlistService,
        "_build_candidate_pool",
        lambda self, request, *, profile=None, watchlist=None: (
            rows.copy(),
            {"request_external": {"count": 1}},
            {"request": {"101", "202"}, "history": set(), "full_watchlist": {"101", "202"}, "sparse_watchlist": {"101"}, "empty_watchlist": set()},
        ),
    )
    monkeypatch.setattr(service_mod, "qualify_candidates", lambda request, frame, max_candidates=20: _qualify_keep_only(frame, {"101"}))

    payload = service.recommend("evidence test", scenario_label="full_watchlist", variant="A")

    assert [rec["candidate_id"] for rec in payload["response"]["recommendations"]] == ["101"]
    assert payload["response"]["recommendations"][0]["grounded_evidence_details"][0]["evidence"] == "stage actress"
    assert payload["debug"]["candidate_context"][0]["candidate_id"] == "101"


def test_qualification_id_mismatch_is_rejected(monkeypatch):
    from mvp.src.generative_v4.qualification_chain import QualificationOutput, _record_from_output, qualify_candidates as real_qualify

    request = _request("mismatch")
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "101",
                "source_id": "101",
                "title": "Film",
                "year": 2001,
                "tmdb_id": 101,
                "genres": ["Drama"],
                "original_language": "en",
                "production_countries": ["FR"],
                "overview": "overview",
                "request_relevance": 0.5,
                "predicted_preference": 0.5,
            }
        ]
    )

    def _fake_llm(request, candidate_rows):
        output = QualificationOutput(
            candidate_id="999",
            status="strong",
            supported_required_aspects=["performers"],
            unsupported_required_aspects=[],
            supported_preferred_aspects=[],
            unsupported_preferred_aspects=[],
            violated_semantic_exclusions=[],
            grounded_evidence=["performers:overview:stage actress"],
            grounded_evidence_details=[{"aspect_id": "performers", "field": "overview", "evidence": "stage actress"}],
            qualification_reason="ok",
            request_match="supported",
            caveat=None,
        )
        return [output], 1, 0, 0, 0

    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain._qualify_with_llm", _fake_llm)

    with pytest.raises(RuntimeError):
        real_qualify(request, frame, max_candidates=5)


def test_structured_constraint_and_completion_metrics_are_truthful():
    run = SimpleNamespace(
        id="run-1",
        outputs={
            "output": {
                "response": {
                    "recommendations": [
                        {"candidate_id": "101", "title": "Film", "year": 1999, "genres": ["Drama"], "original_language": "en", "production_countries": ["france"]}
                    ]
                },
                "runtime_metadata": {"variant": "A", "requested_count": 5},
                "selection_report": {"selected_count": 1, "eligible_count": 1, "slate_size": 5},
                "debug": {
                    "controlled_request_spec": {
                        "structured_constraints": {
                            "required_countries": ["france", "united kingdom"],
                            "excluded_countries": ["united states"],
                            "required_genres": [],
                            "required_languages": [],
                            "required_decades": [],
                            "excluded_genres": [],
                            "excluded_languages": [],
                            "excluded_decades": [],
                            "min_year": 1990,
                            "max_year": 2005,
                        },
                        "semantic_requirement_groups": [
                            {
                                "group_id": "country_alt",
                                "mode": "any_of",
                                "concepts": [{"concept": "france"}, {"concept": "united kingdom"}],
                            }
                        ],
                    }
                },
            }
        }
    )
    example = SimpleNamespace(id="example-1", inputs={"prompt": "Prompt"}, outputs={}, metadata={})

    structured = langsmith_evaluators.structured_constraint_compliance.evaluate_run(run, example)
    completion = langsmith_evaluators.requested_slate_completion.evaluate_run(run, example)
    invalid_run = SimpleNamespace(id="run-2", outputs={"output": {"response": {"recommendations": []}, "runtime_metadata": {"variant": "A", "requested_count": 5}, "selection_report": {"selected_count": 0, "eligible_count": 5, "slate_size": 5}}})
    invalid_completion = langsmith_evaluators.requested_slate_completion.evaluate_run(invalid_run, example)

    assert structured.score == 1.0
    assert completion.score == 1.0
    assert invalid_completion.score == 0.0


def test_export_failure_does_not_trigger_inference_again(monkeypatch, tmp_path):
    service = _service()
    calls: list[str] = []
    original_recommend = ReversibleHistoryWatchlistService.recommend

    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query))
    monkeypatch.setattr(
        ReversibleHistoryWatchlistService,
        "_build_candidate_pool",
        lambda self, request, *, profile=None, watchlist=None: (
            service.full_watchlist_resolved.copy(),
            {"request_external": {"count": 1}},
            {"request": {"101"}, "history": set(), "full_watchlist": {"101"}, "sparse_watchlist": {"101"}, "empty_watchlist": set()},
        ),
    )
    monkeypatch.setattr(service_mod, "qualify_candidates", lambda request, frame, max_candidates=20: _qualify_keep_only(frame))

    def _counting_recommend(self, *args, **kwargs):
        calls.append(kwargs.get("scenario_label", "full_watchlist"))
        return original_recommend(self, *args, **kwargs)

    monkeypatch.setattr(ReversibleHistoryWatchlistService, "recommend", _counting_recommend)
    monkeypatch.setattr(langsmith_evaluation, "load_reversible_history_watchlist_service", lambda: service)
    monkeypatch.setattr(langsmith_evaluation, "_client", lambda: object())
    monkeypatch.setattr(langsmith_evaluation, "evaluate", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("upload failed")))

    summary = langsmith_evaluation.run_reversible_history_watchlist_langsmith_suite(results_dir=tmp_path)

    assert summary["dataset_name"] == "taste-agent-reversible-history-watchlist-h12"
    assert len(calls) == 48
    assert (tmp_path / "watchlist_langsmith_dataset.json").exists()


def test_invalid_validation_cannot_authorize_benchmark_and_frozen_artifacts_do_not_change(monkeypatch, tmp_path):
    service = _service()
    frozen_path = service_mod.REPO_ROOT / "evaluation/watchlist_personalization/reversible_history_watchlist/artifacts/reversible_history_watchlist_manifest.json"
    before = hashlib.sha256(frozen_path.read_bytes()).hexdigest()

    monkeypatch.setattr(langsmith_evaluation, "load_reversible_history_watchlist_service", lambda: service)
    monkeypatch.setattr(service_mod, "understand_request", lambda query, client=None: _request(query))
    monkeypatch.setattr(
        ReversibleHistoryWatchlistService,
        "_build_candidate_pool",
        lambda self, request, *, profile=None, watchlist=None: (
            service.full_watchlist_resolved.copy(),
            {"request_external": {"count": 1}},
            {"request": {"101"}, "history": set(), "full_watchlist": {"101"}, "sparse_watchlist": {"101"}, "empty_watchlist": set()},
        ),
    )
    monkeypatch.setattr(service_mod, "qualify_candidates", lambda request, frame, max_candidates=20: _qualify_keep_only(frame))
    monkeypatch.setattr(ReversibleHistoryWatchlistService, "recommend", lambda self, *args, **kwargs: {"selection_report": {"selected_count": 0, "eligible_count": 5, "slate_size": 5}, "runtime_metadata": {"generation_status": "validation_failed"}, "validation_passed": False})

    with pytest.raises(RuntimeError):
        langsmith_evaluation.run_reversible_history_watchlist_pipeline(results_dir=tmp_path)

    after = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert before == after
