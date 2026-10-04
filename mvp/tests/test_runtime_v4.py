from __future__ import annotations

import pandas as pd
import pytest

from mvp.src.generative.schemas import RecommendationItem, RecommendationResponse
from mvp.src.generative_v4.explanation_chain import ExplanationResult
from mvp.src.generative_v4 import intent_chain as intent_chain_module
from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.generative_v4.qualification_chain import qualify_candidates
from mvp.src.generative_v4.runtime import run_runtime_v4
from mvp.src.generative_v4.selection_chain import select_candidates
from mvp.src.generative_v4.schemas import RequestSpec, RequestUnderstanding
from mvp.src.generative_v4.validators import validate_runtime_v4_result
from mvp.src.retrieval.catalog_retrieval import CatalogRetrievalDiagnostics, CatalogRetrievalResult, discover_catalog_for_request


def _spec_signature(spec: RequestSpec) -> dict[str, object]:
    return {
        "intent_type": spec.intent_type,
        "mode": spec.mode,
        "structured_constraints": spec.structured_constraints.model_dump(),
        "semantic_requirement_groups": [group.model_dump() for group in spec.semantic_requirement_groups],
        "semantic_requirements": [concept.model_dump() for concept in spec.semantic_requirements],
        "semantic_exclusions": [concept.model_dump() for concept in spec.semantic_exclusions],
        "reference": spec.reference.model_dump(),
        "novelty_goal": spec.novelty_goal.model_dump(),
        "personalization_instruction": spec.personalization_instruction.model_dump(),
        "request_relevance_mode": spec.request_relevance_mode,
        "semantic_query_text": spec.semantic_query_text,
        "request_summary": spec.request_summary,
    }


def _lossy_llm_request_spec(query: str) -> tuple[RequestSpec, int, None, None, None]:
    spec = intent_chain_module._deterministic_request_spec(query)
    return (
        spec.model_copy(
            update={
                "mode": "generic",
                "semantic_requirement_groups": [],
                "semantic_requirements": [],
                "semantic_exclusions": [],
                "request_relevance_mode": "broad",
            }
        ),
        1,
        None,
        None,
        None,
    )


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": "1",
                "tmdb_id": 1,
                "title": "Seven Samurai",
                "year": 1954,
                "release_year": 1954,
                "tmdb_genres": "Action|Drama",
                "tmdb_original_language": "ja",
                "tmdb_production_countries": "JP",
                "tmdb_overview": "A samurai epic.",
                "candidate_sources": ["top_rated"],
                "candidate_source_ranks": ["1"],
                "predicted_preference": 0.95,
                "rank": 1,
                "emb_0000": 1.0,
                "emb_0001": 0.0,
            },
            {
                "source_id": "2",
                "tmdb_id": 2,
                "title": "Cinema Paradiso",
                "year": 1988,
                "release_year": 1988,
                "tmdb_genres": "Drama|Romance",
                "tmdb_original_language": "it",
                "tmdb_production_countries": "IT",
                "tmdb_overview": "A nostalgic story about cinema and memory.",
                "candidate_sources": ["top_rated"],
                "candidate_source_ranks": ["2"],
                "predicted_preference": 0.6,
                "rank": 2,
                "emb_0000": 0.0,
                "emb_0001": 1.0,
            },
        ]
    )


def test_understand_request_distinguishes_contextual_prompts() -> None:
    contextual = understand_request("I want a film about performers, fame and show business, preferably period.")
    assert contextual.request_mode == "contextual"
    assert contextual.interaction_mode == "recommend"
    assert contextual.reference_status == "absent"

    generic = understand_request("Recommend me something to watch.")
    assert generic.request_mode == "generic"
    assert generic.interaction_mode == "recommend"


def test_understand_request_routes_interaction_modes() -> None:
    explain_target = understand_request("Why are you recommending this?", candidate_id="346")
    assert explain_target.interaction_mode == "explain_target"

    explain_needs_target = understand_request("Why are you recommending this?")
    assert explain_needs_target.interaction_mode == "explain_needs_target"

    provenance = understand_request("Which film are you recommending because of my Goodreads history?")
    assert provenance.interaction_mode == "provenance"

    boundary = understand_request("What do these recommendations say about my personality and identity?")
    assert boundary.interaction_mode == "profile_boundary"


def test_request_spec_is_orthogonal_to_interaction_routing() -> None:
    assert "interaction_mode" not in RequestSpec.model_fields

    without_target = understand_request("Why are you recommending this?")
    with_target = understand_request("Why are you recommending this?", candidate_id="346")

    assert without_target.spec is not None
    assert with_target.spec is not None
    assert without_target.interaction_mode == "explain_needs_target"
    assert with_target.interaction_mode == "explain_target"
    assert _spec_signature(without_target.spec) == _spec_signature(with_target.spec)


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "I want a film about performers, fame, and show business, preferably with a period setting.",
            {"semantics": {"performers", "fame", "show_business"}, "preferred": {"period_setting"}, "request_mode": "contextual", "relevance_mode": "query_aware"},
        ),
        (
            "I just watched La Bola Negra and loved the Penelope Cruz character in it, I want something about performers, fame and/or show business preferably with a period setting.",
            {"semantics": {"performers", "fame", "show_business"}, "preferred": {"period_setting"}, "request_mode": "reference", "relevance_mode": "query_aware", "reference_title": "la bola negra"},
        ),
        (
            "I want something melancholic but still a little intimate.",
            {"semantics": {"melancholic", "intimate"}, "preferred": set(), "request_mode": "contextual", "relevance_mode": "query_aware"},
        ),
        (
            "Recommend something that is very high-B3 compatible, but it must not actually be about performers, fame, or show business.",
            {"exclusions": {"performers", "fame", "show_business"}, "request_mode": "contextual", "relevance_mode": "broad"},
        ),
        (
            "Surprise me with something different from my usual choices.",
            {"intent_type": "NOVELTY", "novelty_enabled": True, "request_mode": "novelty", "relevance_mode": "query_aware"},
        ),
        (
            "I want a French film tonight, but absolutely no drama.",
            {"languages": {"fr"}, "exclusions": {"drama"}, "request_mode": "contextual", "relevance_mode": "query_aware"},
        ),
    ],
)
def test_runtime_v4_preserves_semantic_request_spec_at_retrieval_boundary(monkeypatch, query, expected) -> None:
    captured: list[object] = []

    def fake_discover(request, *args, **kwargs):
        captured.append(request)
        diagnostics = CatalogRetrievalDiagnostics(
            candidate_universe_count=0,
            post_constraint_candidate_count=0,
            contextual_retrieval_used=request.request_mode != "generic",
            candidate_context_size=50,
            shortlist_size=0,
            retrieval_mode="query_aware" if request.request_mode != "generic" else "broad",
            request_relevance_mode=request.request_relevance_mode,
            reference_status=request.reference_status,
            reference_tmdb_id=request.reference_tmdb_id,
            reference_title=request.reference_title,
            request_relevance_method="stub",
            query_aware_augmentation_used=False,
            watched_excluded_count=0,
            candidate_limit=120,
            b3_applicability_mode="nearest_neighbor_to_frozen_training_embeddings",
        )
        return CatalogRetrievalResult(
            request=request,
            catalog_frame=pd.DataFrame(),
            documents=pd.DataFrame(),
            request_embedding=None,
            retrieval_diagnostics=diagnostics,
            augmentation_report={},
        )

    monkeypatch.setattr("mvp.src.generative_v4.intent_chain._llm_request_spec", lambda q: _lossy_llm_request_spec(q))
    monkeypatch.setattr("mvp.src.generative_v4.runtime.discover_catalog_for_request", fake_discover)

    result = run_runtime_v4(query)
    assert result.runtime_metadata.generation_status == "no_match"
    assert captured, "The production entrypoint should reach retrieval before returning."

    request = captured[0]
    assert request.interaction_mode == "recommend"
    assert request.request_mode == expected["request_mode"]
    assert request.request_relevance_mode == expected["relevance_mode"]
    assert request.spec is not None

    spec = request.spec
    semantic_requirements = {concept.aspect_id for concept in spec.semantic_requirements}
    semantic_exclusions = {concept.aspect_id for concept in spec.semantic_exclusions}
    grouped = {concept.aspect_id for group in spec.semantic_requirement_groups for concept in group.concepts}

    if "semantics" in expected:
        assert expected["semantics"].issubset(semantic_requirements)
        assert expected["semantics"].issubset(grouped)
    if "preferred" in expected:
        preferred = {concept.aspect_id for concept in spec.semantic_requirements if concept.importance == "preferred"}
        assert expected["preferred"].issubset(preferred)
    if "exclusions" in expected:
        assert expected["exclusions"].issubset(semantic_exclusions)
    if expected.get("intent_type"):
        assert request.intent.intent_type == expected["intent_type"]
    if expected.get("novelty_enabled"):
        assert spec.novelty_goal.enabled is True
    if "languages" in expected:
        assert set(spec.structured_constraints.required_languages) == expected["languages"]
    if "languages" in expected and "exclusions" in expected:
        assert "drama" in semantic_exclusions
    if "reference" in expected:
        assert spec.reference.title or request.reference_title
    if "reference_title" in expected:
        assert expected["reference_title"] in (spec.reference.title or "")


def test_negative_only_personalization_stays_broad_and_positive_aliases_survive() -> None:
    contextual = understand_request("I want something meditative and visually austere.")
    assert contextual.request_mode == "contextual"
    assert contextual.request_relevance_mode == "query_aware"
    assert contextual.spec is not None
    concepts = {concept.concept for concept in contextual.spec.semantic_requirements}
    assert {"contemplative", "austere"}.issubset(concepts)

    exclusion_only = understand_request("Recommend something that is very high-B3 compatible, but it must not actually be about performers, fame, or show business.")
    assert exclusion_only.request_mode == "contextual"
    assert exclusion_only.request_relevance_mode == "broad"
    assert exclusion_only.spec is not None
    assert exclusion_only.spec.semantic_query_text == ""
    exclusions = {concept.concept for concept in exclusion_only.spec.semantic_exclusions}
    assert {"performers", "fame", "show_business"}.issubset(exclusions)


def test_positive_and_negative_semantic_mentions_are_disjoint_after_normalization() -> None:
    request = understand_request("I want drama, but no drama.")
    assert request.spec is not None
    required = {concept.concept for concept in request.spec.semantic_requirements}
    excluded = {concept.concept for concept in request.spec.semantic_exclusions}

    assert "drama" not in required
    assert "drama" in excluded


def test_runtime_explain_target_uses_supplied_candidate_only(monkeypatch) -> None:
    monkeypatch.setattr("mvp.src.generative_v4.runtime.discover_catalog_for_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("catalog retrieval should not run")))
    monkeypatch.setattr("mvp.src.generative_v4.runtime.qualify_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("qualification should not run")))
    monkeypatch.setattr("mvp.src.generative_v4.runtime.select_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection should not run")))

    def fake_explain_selection(request, selected_frame):
        assert len(selected_frame) == 1
        assert str(selected_frame.iloc[0]["source_id"]) == "346"
        response = RecommendationResponse(
            intent=request.intent,
            recommendations=[
                RecommendationItem(
                    candidate_id="346",
                    title="Seven Samurai",
                    year=1954,
                    why_it_may_fit="Grounded explanation for the supplied target candidate.",
                    taste_signals=["historical"],
                    request_match=None,
                    caveat="Awards information is not grounded in the current film data.",
                )
            ],
            response_summary="Explaining the supplied target candidate with grounded evidence only.",
            methodology_note="No new recommendation slate was generated.",
        )
        selected = selected_frame.copy()
        selected.loc[:, "semantic_evidence"] = [[{"dimension": "historical"}]]
        selected.loc[:, "qualification_status"] = ["strong"]
        return ExplanationResult(response=response, selected_frame=selected, semantic_report={"semantic_classified": 1, "coverage": 1.0})

    monkeypatch.setattr("mvp.src.generative_v4.runtime.explain_selection", fake_explain_selection)

    result = run_runtime_v4("Why are you recommending this?", candidate_id="346")
    assert result.runtime_metadata.interaction_mode == "explain_target"
    assert result.runtime_metadata.generation_status == "generated"
    assert len(result.response.recommendations) == 1
    assert result.response.recommendations[0].candidate_id == "346"
    assert validate_runtime_v4_result(result).passed


def test_runtime_explain_without_target_clarifies(monkeypatch) -> None:
    monkeypatch.setattr("mvp.src.generative_v4.runtime.discover_catalog_for_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("catalog retrieval should not run")))
    monkeypatch.setattr("mvp.src.generative_v4.runtime.qualify_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("qualification should not run")))
    monkeypatch.setattr("mvp.src.generative_v4.runtime.select_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection should not run")))

    result = run_runtime_v4("Why are you recommending this?")
    assert result.runtime_metadata.interaction_mode == "explain_needs_target"
    assert result.runtime_metadata.generation_status == "clarification"
    assert result.response.recommendations == []
    assert "which recommendation" in result.response.response_summary.lower()
    assert validate_runtime_v4_result(result).passed


def test_runtime_provenance_and_profile_boundary_routes(monkeypatch) -> None:
    monkeypatch.setattr("mvp.src.generative_v4.runtime.discover_catalog_for_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("catalog retrieval should not run")))
    monkeypatch.setattr("mvp.src.generative_v4.runtime.qualify_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("qualification should not run")))
    monkeypatch.setattr("mvp.src.generative_v4.runtime.select_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection should not run")))

    provenance = run_runtime_v4("Which film are you recommending because of my Goodreads history?")
    assert provenance.runtime_metadata.interaction_mode == "provenance"
    assert provenance.response.recommendations == []
    assert "goodreads" in provenance.response.response_summary.lower()
    assert validate_runtime_v4_result(provenance).passed

    boundary = run_runtime_v4("What do these recommendations say about my personality and identity?")
    assert boundary.runtime_metadata.interaction_mode == "profile_boundary"
    assert boundary.response.recommendations == []
    assert "taste" in boundary.response.response_summary.lower()
    assert validate_runtime_v4_result(boundary).passed


def test_catalog_retrieval_excludes_watched_films(monkeypatch) -> None:
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval.generate_candidate_pool",
        lambda candidate_limit=300: (_candidate_frame().copy(), {"candidate_limit": candidate_limit}),
    )
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval.embed_documents",
        lambda documents: (
            pd.DataFrame(
                {
                    "source_id": documents["source_id"].astype(str),
                    "emb_0000": [1.0] * len(documents),
                    "emb_0001": [0.0] * len(documents),
                    "status": ["success"] * len(documents),
                }
            ),
            pd.DataFrame(),
            {"provider": "local", "model": "test", "coverage": 1.0, "successful": len(documents), "failed": 0, "cache_status": "test", "cache_hits": 0, "cache_misses": 0},
        ),
    )
    request = understand_request("Recommend me something to watch.")
    result = discover_catalog_for_request(request, watched_ids=["2"])
    assert set(result.catalog_frame["source_id"].astype(str).tolist()) == {"1"}


def test_qualification_rejects_associative_show_business_reasoning() -> None:
    request = understand_request("I want a film about performers, fame and show business, preferably with a period setting.")
    frame = _candidate_frame().iloc[[0]].copy()
    qualified, records = qualify_candidates(request, frame, max_candidates=10)
    assert qualified.empty
    assert records[0].qualification_status == "unsupported"


def test_partial_match_preserves_caveat() -> None:
    request = understand_request("I want something melancholic but still intimate.")
    frame = pd.DataFrame(
        [
            {
                "source_id": "10",
                "tmdb_id": 10,
                "title": "Melancholic Film",
                "year": 2000,
                "release_year": 2000,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A melancholic story of grief.",
                "candidate_sources": ["top_rated"],
                "candidate_source_ranks": ["1"],
                "predicted_preference": 0.4,
                "rank": 1,
                "document_text": "A melancholic story of grief.",
            }
        ]
    )
    qualified, records = qualify_candidates(request, frame, max_candidates=10)
    assert not qualified.empty
    assert records[0].qualification_status == "strong"
    assert records[0].caveat is None


def test_novelty_selection_uses_history_distance(monkeypatch) -> None:
    request = RequestUnderstanding(
        query="Show me something different from what I normally watch.",
        intent=understand_request("Show me something different from what I normally watch.").intent,
        request_mode="novelty",
        novelty_requested=True,
        reference_status="absent",
        request_relevance_mode="query_aware",
    )
    frame = pd.DataFrame(
        [
            {
                "source_id": "1",
                "tmdb_id": 1,
                "title": "Film A",
                "year": 2001,
                "release_year": 2001,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A film.",
                "candidate_sources": ["top_rated"],
                "candidate_source_ranks": ["1"],
                "predicted_preference": 0.0,
                "rank": 1,
                "emb_0000": 1.0,
                "emb_0001": 0.0,
                "qualification_status": "strong",
            },
            {
                "source_id": "2",
                "tmdb_id": 2,
                "title": "Film B",
                "year": 2002,
                "release_year": 2002,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "B film.",
                "candidate_sources": ["top_rated"],
                "candidate_source_ranks": ["2"],
                "predicted_preference": 0.2,
                "rank": 2,
                "emb_0000": 0.0,
                "emb_0001": 1.0,
                "qualification_status": "strong",
            },
        ]
    )
    monkeypatch.setattr(
        "mvp.src.generative_v4.selection_chain._build_history_embeddings",
        lambda: pd.DataFrame({"source_id": ["h1"], "emb_0000": [1.0], "emb_0001": [0.0]}),
    )
    result = select_candidates(request, frame, recommendation_count=1)
    assert result.selected_frame.iloc[0]["source_id"] == "2"


def test_runtime_abstains_when_no_candidate_survives(monkeypatch) -> None:
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
    monkeypatch.setattr("mvp.src.generative_v4.runtime.qualify_candidates", lambda *args, **kwargs: (pd.DataFrame(), []))

    result = run_runtime_v4("Recommend me something to watch.")
    assert result.runtime_metadata.generation_status == "no_match"
    assert result.response.recommendations == []
    assert validate_runtime_v4_result(result).passed


def test_reference_resolution_unresolved_does_not_invent_facts() -> None:
    request = understand_request("Recommend something like an imaginary film title that TMDB cannot resolve.")
    assert request.reference_status == "unresolved"
    assert request.reference_tmdb_id is None
