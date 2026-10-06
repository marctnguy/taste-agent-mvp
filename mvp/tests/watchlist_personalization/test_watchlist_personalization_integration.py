from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from langchain_core.messages import HumanMessage, SystemMessage

from mvp.src.generative.schemas import RecommendationIntent
from mvp.src.generative_v4.intent_chain import RequestUnderstanding
from mvp.src.generative_v4.qualification_chain import (
    CandidateQualification,
    GroundedEvidence,
    QualificationBatch,
    qualify_candidates,
)
from mvp.src.generative_v4.schemas import (
    NoveltyGoal,
    PersonalizationInstruction,
    QualificationRecord,
    ReferenceSpec,
    RequestSpec,
    SemanticConcept,
    SemanticRequirementGroup,
    StructuredConstraints,
)
from mvp.src.watchlist_personalization.reversible_history_watchlist_service import (
    ReversibleHistoryWatchlistService,
    _film_document_text,
    _score_direct_taste,
    _select_diverse_history_seeds,
    _request_fingerprint,
    _scenario_allowed_ids,
    _select_view,
)
from mvp.src.retrieval.catalog_retrieval import discover_catalog_for_request


@pytest.fixture(autouse=True)
def _disable_tracing() -> None:
    import os

    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def _request(query: str = "test query", *, structured: StructuredConstraints | None = None) -> RequestUnderstanding:
    groups = [
        SemanticRequirementGroup(
            group_id="required",
            mode="all_of",
            concepts=[SemanticConcept(aspect_id="performers", concept="performers", importance="required")],
        ),
        SemanticRequirementGroup(
            group_id="optional",
            mode="any_of",
            concepts=[
                SemanticConcept(aspect_id="fame", concept="fame", importance="required"),
                SemanticConcept(aspect_id="show_business", concept="show_business", importance="required"),
            ],
        ),
    ]
    structured = structured or StructuredConstraints(required_genres=["Drama"])
    spec = RequestSpec(
        intent_type="MOOD_THEME",
        mode="contextual",
        structured_constraints=structured,
        semantic_requirement_groups=groups,
        semantic_requirements=[concept for group in groups for concept in group.concepts],
        semantic_exclusions=[],
        reference=ReferenceSpec(title="La Bola Negra", resolution_status="unresolved"),
        novelty_goal=NoveltyGoal(enabled=False),
        personalization_instruction=PersonalizationInstruction(),
        request_relevance_mode="query_aware",
        semantic_query_text=query,
        request_summary=query,
    )
    intent = RecommendationIntent(
        intent_type="MOOD_THEME",
        requested_taste_dimensions=["performers", "fame", "show_business"],
        requested_genres=list(structured.required_genres),
        requested_languages=[],
        requested_countries=[],
        requested_decades=[],
        exclusions=[],
        free_text_context=query,
    )
    return RequestUnderstanding(
        query=query,
        intent=intent,
        spec=spec,
        request_mode="contextual",
        novelty_requested=False,
        reference_title="La Bola Negra",
        reference_status="unresolved",
        request_relevance_mode="query_aware",
    )


def _service() -> ReversibleHistoryWatchlistService:
    positive = pd.DataFrame(
        [
            {
                "source_id": "p1",
                "title": "Seed One",
                "year": 2001,
                "rating": 5.0,
                "emb_0000": 1.0,
                "emb_0001": 0.0,
            },
            {
                "source_id": "p2",
                "title": "Seed Two",
                "year": 1998,
                "rating": 4.0,
                "emb_0000": 0.0,
                "emb_0001": 1.0,
            },
        ]
    )
    negative = pd.DataFrame(
        [
            {
                "source_id": "n1",
                "title": "Seed Negative",
                "year": 2003,
                "rating": 1.0,
                "emb_0000": 0.0,
                "emb_0001": -1.0,
            }
        ]
    )
    return ReversibleHistoryWatchlistService(
        prompts=pd.DataFrame(),
        inputs={},
        condition_a=pd.DataFrame(),
        split=pd.DataFrame(),
        train_embeddings=pd.DataFrame(),
        watchlist=pd.DataFrame(),
        consumed_keys={"source": set(), "tmdb": set(), "title_year": set()},
        tmdb_client=None,
        historical={
            "positive": positive,
            "negative": negative,
            "emb_cols": ["emb_0000", "emb_0001"],
            "taste_profile": pd.DataFrame(),
        },
        watched_history_embeddings=pd.DataFrame(
            [{"source_id": "w1", "emb_0000": 1.0, "emb_0001": 0.0}]
        ),
        full_watchlist_resolved=pd.DataFrame(),
        sparse_watchlist_resolved=pd.DataFrame(),
        empty_watchlist_resolved=pd.DataFrame(),
        preflight={"services": {}},
    )


def test_real_qualifier_serializes_groups_preserves_scores_and_enforces_hard_constraints(monkeypatch):
    captured_messages: list[list[object]] = []

    class _FakeStructured:
        def invoke(self, messages):
            captured_messages.append(messages)
            items = [
                CandidateQualification(
                    candidate_id="100",
                    qualification_status="strong",
                    supported_required_aspects=["performers", "fame"],
                    unsupported_required_aspects=[],
                    supported_preferred_aspects=[],
                    unsupported_preferred_aspects=[],
                    violated_semantic_exclusions=[],
                    grounded_evidence=[GroundedEvidence(aspect_id="performers", field="overview", evidence="stage actress")],
                    reason="good",
                    required_caveat=None,
                ),
                CandidateQualification(
                    candidate_id="200",
                    qualification_status="strong",
                    supported_required_aspects=["performers"],
                    unsupported_required_aspects=[],
                    supported_preferred_aspects=[],
                    unsupported_preferred_aspects=[],
                    violated_semantic_exclusions=[],
                    grounded_evidence=[GroundedEvidence(aspect_id="performers", field="overview", evidence="stage actress")],
                    reason="bad but allowed by fake LLM",
                    required_caveat=None,
                ),
            ]
            return {
                "parsed": QualificationBatch(items=items),
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            }

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return _FakeStructured()

    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain.ChatOpenAI", _FakeLLM)
    monkeypatch.setattr(
        "mvp.src.generative_v4.qualification_chain.get_api_keys",
        lambda: SimpleNamespace(openai_api_key="test"),
    )

    request = _request()
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "100",
                "source_id": "100",
                "title": "Supported",
                "year": 2001,
                "tmdb_id": 100,
                "genres": ["Drama"],
                "original_language": "fr",
                "production_countries": ["FR"],
                "overview": "stage actress in a drama about performers",
                "candidate_sources": ["request_external"],
                "candidate_source_ranks": ["1"],
                "route_memberships": ["request_external"],
                "request_relevance": 0.9,
                "predicted_preference": 0.8,
                "direct_taste_score": 0.7,
                "novelty_score": 0.2,
            },
            {
                "candidate_id": "200",
                "source_id": "200",
                "title": "Hard Constraint Violation",
                "year": 2002,
                "tmdb_id": 200,
                "genres": ["Comedy"],
                "original_language": "fr",
                "production_countries": ["FR"],
                "overview": "still about performers",
                "candidate_sources": ["history_external"],
                "candidate_source_ranks": ["1"],
                "route_memberships": ["history_external"],
                "request_relevance": 0.8,
                "predicted_preference": 0.7,
                "direct_taste_score": 0.6,
                "novelty_score": 0.5,
            },
        ]
    )

    qualified, records = qualify_candidates(request, candidates, max_candidates=10)

    assert qualified["source_id"].astype(str).tolist() == ["100"]
    assert qualified.iloc[0]["direct_taste_score"] == pytest.approx(0.7)
    assert qualified.iloc[0]["route_memberships"] == ["request_external"]
    assert len(records) == 2
    assert {record.candidate_id for record in records} == {"100", "200"}
    assert any(record.qualification_status == "unsupported" and record.candidate_id == "200" for record in records)
    assert any("semantic_requirement_groups" in str(message) for message in captured_messages[0])


def test_build_candidate_pool_replaces_existing_embeddings_and_coalesces_missing_values(monkeypatch):
    service = _service()
    request = _request("seed one seed two")
    calls = {"count": 0}

    def _discover(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            frame = pd.DataFrame(
                [
                    {
                        "source_id": "201",
                        "title": "Duplicate Film",
                        "year": 2001,
                        "tmdb_id": 201,
                        "genres": ["Drama"],
                        "original_language": "fr",
                        "production_countries": ["FR"],
                        "overview": pd.NA,
                        "tmdb_overview": "request overview",
                        "candidate_sources": ["request_external"],
                        "candidate_source_ranks": ["1"],
                        "route_memberships": ["request_external"],
                        "emb_0000": 9.0,
                        "emb_0001": 9.0,
                    }
                ]
            )
        else:
            frame = pd.DataFrame(
                [
                    {
                        "source_id": "201",
                        "title": "Duplicate Film",
                        "year": 2001,
                        "tmdb_id": 201,
                        "genres": ["Drama"],
                        "original_language": "fr",
                        "production_countries": ["FR"],
                        "overview": "rich overview",
                        "tmdb_overview": pd.NA,
                        "candidate_sources": ["history_external"],
                        "candidate_source_ranks": ["1"],
                        "route_memberships": ["history_external"],
                        "emb_0000": 1.0,
                        "emb_0001": 0.0,
                    }
                ]
            )
        return SimpleNamespace(catalog_frame=frame, augmentation_report={"count": len(frame)})

    def _embed_documents(documents: pd.DataFrame):
        frame = documents.loc[:, ["source_id", "title", "document_hash"]].copy()
        frame["embedding_model"] = "test"
        frame["embedding_dim"] = 2
        frame["status"] = "success"
        frame["emb_0000"] = [1.0] * len(frame)
        frame["emb_0001"] = [0.0] * len(frame)
        return frame, frame.copy(), {"cache_status": "mock", "successful": len(frame), "failed": 0}

    monkeypatch.setattr("mvp.src.watchlist_personalization.reversible_history_watchlist_service.understand_request", lambda query, client=None: request)
    monkeypatch.setattr("mvp.src.watchlist_personalization.reversible_history_watchlist_service.discover_catalog_for_request", _discover)
    monkeypatch.setattr("mvp.src.retrieval.catalog_retrieval.embed_documents", _embed_documents)
    monkeypatch.setattr("mvp.src.watchlist_personalization.reversible_history_watchlist_service.embed_text", lambda text: np.array([1.0, 0.0]))

    master, reports, route_ids = service._build_candidate_pool(request, profile=None, watchlist=None)

    assert calls["count"] == 2
    assert master["source_id"].astype(str).tolist() == ["201"]
    assert master.iloc[0]["overview"] == "rich overview"
    assert [column for column in master.columns if column.startswith("emb_")] == ["emb_0000", "emb_0001"]
    assert set(master.iloc[0]["route_memberships"]) == {"request_external", "history_external"}
    assert route_ids["request"] == {"201"}
    assert route_ids["history"] == {"201"}
    assert reports["history_seed_ids"]


def test_variant_b_can_select_a_history_only_candidate_while_variant_a_cannot():
    route_ids = {
        "request": {"100"},
        "history": {"200"},
        "full_watchlist": {"300"},
        "sparse_watchlist": {"300"},
        "empty_watchlist": set(),
    }
    frame = pd.DataFrame(
        [
            {
                "source_id": "100",
                "title": "Request Film",
                "qualification_status": "strong",
                "request_relevance": 0.9,
                "direct_taste_score": 0.1,
                "novelty_score": 0.2,
                "route_memberships": ["request_external"],
            },
            {
                "source_id": "200",
                "title": "History Film",
                "qualification_status": "strong",
                "request_relevance": 0.3,
                "direct_taste_score": 0.95,
                "novelty_score": 0.8,
                "route_memberships": ["history_external"],
            },
            {
                "source_id": "300",
                "title": "Watchlist Film",
                "qualification_status": "strong",
                "request_relevance": 0.5,
                "direct_taste_score": 0.4,
                "novelty_score": 0.6,
                "route_memberships": ["watchlist"],
            },
        ]
    )

    allowed_a = _scenario_allowed_ids(route_ids, "full_watchlist", variant="A")
    allowed_b = _scenario_allowed_ids(route_ids, "full_watchlist", variant="B")
    selected_a = _select_view(frame.loc[frame["source_id"].isin(allowed_a)].copy(), request_mode="generic", variant="A", requested_count=2, novelty_requested=False)
    selected_b = _select_view(frame.loc[frame["source_id"].isin(allowed_b)].copy(), request_mode="generic", variant="B", requested_count=2, novelty_requested=False)

    assert "200" not in selected_a["source_id"].astype(str).tolist()
    assert "200" in selected_b["source_id"].astype(str).tolist()


def test_cache_fingerprint_changes_for_watchlist_content_and_close_float_profile_values():
    watchlist_a = pd.DataFrame([{"source_id": "1", "title": "Film A", "year": 2001, "rating": 4.0, "synopsis": "alpha"}])
    watchlist_b = pd.DataFrame([{"source_id": "1", "title": "Film A", "year": 2001, "rating": 4.0, "synopsis": "beta"}])
    profile_a = {"positive": pd.DataFrame([{"source_id": "p1", "weight": 0.1000000000000001}])}
    profile_b = {"positive": pd.DataFrame([{"source_id": "p1", "weight": 0.1000000000000002}])}

    fingerprint_a = _request_fingerprint("query", profile_a, watchlist_a, 5, {"variant": "A"})
    fingerprint_b = _request_fingerprint("query", profile_b, watchlist_a, 5, {"variant": "A"})
    fingerprint_c = _request_fingerprint("query", profile_a, watchlist_b, 5, {"variant": "A"})

    assert fingerprint_a != fingerprint_b
    assert fingerprint_a != fingerprint_c


def test_qualifier_batches_multi_valued_metadata_and_missing_year_are_handled(monkeypatch):
    batch_sizes: list[int] = []

    class _FakeStructured:
        def invoke(self, messages):
            candidate_block = messages[1].content.split("Candidate payloads:\n", 1)[1].strip()
            candidates = ast.literal_eval(candidate_block)
            batch_sizes.append(len(candidates))
            items = [
                CandidateQualification(
                    candidate_id=str(candidate["candidate_id"]),
                    qualification_status="strong",
                    supported_required_aspects=["performers"],
                    unsupported_required_aspects=[],
                    supported_preferred_aspects=[],
                    unsupported_preferred_aspects=[],
                    violated_semantic_exclusions=[],
                    grounded_evidence=[GroundedEvidence(aspect_id="performers", field="overview", evidence="stage actress")],
                    reason="grounded",
                    required_caveat=None,
                )
                for candidate in candidates
            ]
            return {
                "parsed": QualificationBatch(items=items),
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            }

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return _FakeStructured()

    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain.ChatOpenAI", _FakeLLM)
    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain.get_api_keys", lambda: SimpleNamespace(openai_api_key="test"))

    request = _request("performers in drama with a period setting", structured=StructuredConstraints(required_genres=["Drama"], min_year=2000))
    candidates = []
    for index in range(25):
        candidates.append(
            {
                "candidate_id": str(index),
                "source_id": str(index),
                "title": f"Film {index}",
                "year": None if index == 0 else 2001,
                "release_year": None if index == 0 else 2001,
                "tmdb_id": index,
                "genres": ["Drama", "Comedy"] if index == 0 else ["Drama"],
                "original_language": "fr",
                "production_countries": ["FR", "ES"] if index == 0 else ["FR"],
                "overview": "stage actress in a drama about performers",
                "candidate_sources": ["request_external"],
                "candidate_source_ranks": ["1"],
                "route_memberships": ["request_external"],
                "request_relevance": 0.9 - (index * 0.01),
                "predicted_preference": 0.8,
                "direct_taste_score": 0.7,
                "novelty_score": 0.2,
            }
        )

    qualified, records = qualify_candidates(request, pd.DataFrame(candidates), max_candidates=48)

    assert batch_sizes == [6, 6, 6, 6, 1]
    assert len(records) == 25
    assert len(qualified) == 24
    assert any(record.candidate_id == "0" and record.qualification_status == "unsupported" for record in records)


def test_qualifier_rejects_duplicate_candidate_ids_before_llm_parsing(monkeypatch):
    class _FakeStructured:
        def invoke(self, messages):
            raise AssertionError("LLM should not be invoked when candidate IDs are duplicated")

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return _FakeStructured()

    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain.ChatOpenAI", _FakeLLM)
    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain.get_api_keys", lambda: SimpleNamespace(openai_api_key="test"))

    request = _request("duplicate ids")
    frame = pd.DataFrame(
        [
            {"candidate_id": "dup", "source_id": "dup", "title": "One", "request_relevance": 0.9},
            {"candidate_id": "dup", "source_id": "dup", "title": "Two", "request_relevance": 0.8},
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate candidate IDs"):
        qualify_candidates(request, frame, max_candidates=48)


def test_qualifier_defaults_missing_predicted_preference_to_zero(monkeypatch):
    class _FakeStructured:
        def invoke(self, messages):
            return {
                "parsed": QualificationBatch(
                    items=[
                        CandidateQualification(
                            candidate_id="1",
                            qualification_status="strong",
                            supported_required_aspects=["performers"],
                            unsupported_required_aspects=[],
                            supported_preferred_aspects=[],
                            unsupported_preferred_aspects=[],
                            violated_semantic_exclusions=[],
                            grounded_evidence=[GroundedEvidence(aspect_id="performers", field="overview", evidence="stage actress")],
                            reason="grounded",
                            required_caveat=None,
                        )
                    ]
                ),
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            }

    class _FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, *args, **kwargs):
            return _FakeStructured()

    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain.ChatOpenAI", _FakeLLM)
    monkeypatch.setattr("mvp.src.generative_v4.qualification_chain.get_api_keys", lambda: SimpleNamespace(openai_api_key="test"))

    request = _request("missing predicted preference")
    frame = pd.DataFrame(
        [
                {
                    "candidate_id": "1",
                    "source_id": "1",
                    "title": "Film",
                    "request_relevance": 0.9,
                    "predicted_preference": None,
                    "genres": ["Drama"],
                    "original_language": "fr",
                    "production_countries": ["FR"],
                    "overview": "stage actress in a drama about performers",
                    "candidate_sources": ["request_external"],
                    "candidate_source_ranks": ["1"],
                    "route_memberships": ["request_external"],
                }
        ]
    )

    qualified, records = qualify_candidates(request, frame, max_candidates=48)

    assert records[0].candidate_id == "1"
    assert qualified.iloc[0]["predicted_preference"] == pytest.approx(0.0)


def test_score_direct_taste_handles_missing_negative_and_missing_positive_history():
    frame = pd.DataFrame(
        [
            {"source_id": "c1", "title": "Candidate", "emb_0000": 1.0, "emb_0001": 0.0},
        ]
    )
    positive = pd.DataFrame(
        [
            {"source_id": "p1", "title": "Positive", "year": 2001, "emb_0000": 1.0, "emb_0001": 0.0},
        ]
    )
    scored_reduced = _score_direct_taste(frame, {"positive": positive, "negative": pd.DataFrame(), "emb_cols": ["emb_0000", "emb_0001"]})
    assert np.isfinite(scored_reduced.loc[0, "direct_taste_score"])
    assert scored_reduced.loc[0, "direct_taste_signal_status"] == "reduced_signal"
    assert scored_reduced.loc[0, "positive_neighbors"][0]["source_id"] == "p1"
    assert scored_reduced.loc[0, "positive_neighbors"][0]["title"] == "Positive"
    assert scored_reduced.loc[0, "positive_neighbors"][0]["similarity"] == pytest.approx(1.0)
    assert scored_reduced.loc[0, "negative_neighbors"] == []

    scored_fallback = _score_direct_taste(frame, {"positive": pd.DataFrame(), "negative": pd.DataFrame(), "emb_cols": ["emb_0000", "emb_0001"]})
    assert scored_fallback.loc[0, "direct_taste_score"] == pytest.approx(0.0)
    assert scored_fallback.loc[0, "direct_taste_signal_status"] == "insufficient_personalization"


def test_film_document_text_uses_tmdb_aliases_when_primary_fields_are_missing():
    document = _film_document_text(
        {
            "source_id": "1",
            "title": "Alias Film",
            "year": pd.NA,
            "release_year": 1999,
            "overview": pd.NA,
            "tmdb_overview": "Alias synopsis",
            "genres": pd.NA,
            "tmdb_genres": ["Drama", "Comedy"],
            "production_countries": pd.NA,
            "tmdb_production_countries": ["FR", "ES"],
        }
    )

    assert "Alias synopsis" in document
    assert "Release year: 1999" in document
    assert "Drama" in document and "Comedy" in document


def test_build_candidate_pool_uses_explicit_history_seeds_and_disables_seeded_candidate_generation(monkeypatch):
    service = _service()
    request = _request("explicit history seeds")
    positive = pd.DataFrame(
        [
            {"source_id": "p1", "title": "Seed One", "year": 2001, "rating": 5.0, "tmdb_genres": "Drama", "emb_0000": 1.0},
            {"source_id": "p2", "title": "Seed Two", "year": 1998, "rating": 4.5, "tmdb_genres": "Comedy", "emb_0000": 1.0},
        ]
    )
    service = ReversibleHistoryWatchlistService(
        prompts=service.prompts,
        inputs=service.inputs,
        condition_a=service.condition_a,
        split=service.split,
        train_embeddings=service.train_embeddings,
        watchlist=service.watchlist,
        consumed_keys=service.consumed_keys,
        tmdb_client=None,
        historical={"positive": positive, "negative": pd.DataFrame(), "emb_cols": ["emb_0000"], "taste_profile": pd.DataFrame()},
        watched_history_embeddings=pd.DataFrame(),
        full_watchlist_resolved=service.full_watchlist_resolved,
        sparse_watchlist_resolved=service.sparse_watchlist_resolved,
        empty_watchlist_resolved=service.empty_watchlist_resolved,
        preflight=service.preflight,
    )

    pool_calls: list[bool] = []
    discover_calls: list[pd.DataFrame] = []

    def _fake_generate_candidate_pool(*, candidate_limit=300, include_seeded_recommendations=True, **kwargs):
        pool_calls.append(include_seeded_recommendations)
        frame = pd.DataFrame(
            [
                {"source_id": "r1", "title": "Request One", "year": 2000, "candidate_sources": ["request_external"], "candidate_source_ranks": ["1"], "route_memberships": ["request_external"], "request_relevance": 0.9},
                {"source_id": "r2", "title": "Request Two", "year": 2001, "candidate_sources": ["request_external"], "candidate_source_ranks": ["2"], "route_memberships": ["request_external"], "request_relevance": 0.8},
            ]
        )
        return frame, {"candidate_limit": candidate_limit, "after_exclusion": len(frame)}

    def _fake_discover(request, *, candidate_pool=None, watched_ids=None, client=None, candidate_limit=300, candidate_context_size=50):
        discover_calls.append(candidate_pool.copy().reset_index(drop=True))
        frame = candidate_pool.copy().reset_index(drop=True)
        frame["request_relevance"] = 0.9
        frame["direct_taste_score"] = 0.8
        frame["novelty_score"] = 0.1
        frame["candidate_sources"] = frame.get("candidate_sources", [["request_external"] for _ in range(len(frame))])
        frame["candidate_source_ranks"] = frame.get("candidate_source_ranks", [["1"] for _ in range(len(frame))])
        frame["route_memberships"] = frame.get("route_memberships", [["request_external"] for _ in range(len(frame))])
        return SimpleNamespace(catalog_frame=frame, augmentation_report={"candidate_pool": len(frame)})

    def _fake_embed_documents(documents):
        frame = documents.loc[:, ["source_id", "title", "document_hash"]].copy()
        frame["embedding_model"] = "test"
        frame["embedding_dim"] = 1
        frame["status"] = "success"
        frame["emb_0000"] = 1.0
        return frame, frame.copy(), {"cache_status": "mock", "successful": len(frame), "failed": 0}

    monkeypatch.setattr("mvp.src.watchlist_personalization.reversible_history_watchlist_service.generate_candidate_pool", _fake_generate_candidate_pool)
    monkeypatch.setattr("mvp.src.watchlist_personalization.reversible_history_watchlist_service.discover_catalog_for_request", _fake_discover)
    monkeypatch.setattr("mvp.src.retrieval.catalog_retrieval.embed_documents", _fake_embed_documents)
    monkeypatch.setattr("mvp.src.watchlist_personalization.reversible_history_watchlist_service.embed_text", lambda text: np.array([1.0]))

    master, reports, route_ids = service._build_candidate_pool(request, profile=None, watchlist=None)

    assert pool_calls == [False]
    assert discover_calls[0]["source_id"].astype(str).tolist() == ["r1", "r2"]
    assert discover_calls[1]["source_id"].astype(str).tolist() == ["p1", "p2"]
    assert route_ids["history"] == {"p1", "p2"}
    assert reports["history_seed_count"] == 2
    assert master["source_id"].astype(str).tolist()


def test_discover_catalog_handles_empty_candidate_pool_without_source_id(monkeypatch):
    request = _request("empty pool")
    empty_pool = pd.DataFrame()

    result = discover_catalog_for_request(request, candidate_pool=empty_pool, watched_ids=set(), client=None)

    assert "source_id" in result.catalog_frame.columns
    assert result.catalog_frame.empty
