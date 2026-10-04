from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mvp.src.candidates import build_watched_exclusions
from mvp.src.generative_v4.intent_chain import RequestUnderstanding, understand_request
from mvp.src.generative_v4.qualification_chain import qualify_candidates
from mvp.src.retrieval.catalog_retrieval import discover_catalog_for_request


def _contextual_request(query: str, **intent_updates) -> RequestUnderstanding:
    request = understand_request(query)
    return request.model_copy(
        update={
            "request_mode": "contextual",
            "request_relevance_mode": "query_aware",
            "intent": request.intent.model_copy(update=intent_updates),
        }
    )


def _patch_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval._augment_with_constraint_candidates",
        lambda request, client: (pd.DataFrame(), {"constraint_candidates": 0}),
    )
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval.embed_documents",
        lambda documents: (
            pd.DataFrame(columns=["source_id", "title", "document_hash", "embedding_model", "embedding_dim", "status"]),
            pd.DataFrame(),
            {"provider": "local", "model": "test", "coverage": 0.0, "successful": 0, "failed": 0, "cache_status": "test", "cache_hits": 0, "cache_misses": 0},
        ),
    )
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval.embed_text",
        lambda text: np.array([1.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(
        "mvp.src.mvp_deployment.load_training_embeddings",
        lambda: pd.DataFrame({"source_id": ["training-1"], "emb_0000": [1.0], "emb_0001": [0.0]}),
    )


@pytest.mark.parametrize(
    "query,intent_updates,candidate_row,expected_ids",
    [
        (
            "I want a French film.",
            {"requested_languages": ["fr"]},
            {"source_id": "1", "tmdb_id": 1, "title": "French Film", "year": 1999, "release_year": 1999, "tmdb_genres": "Drama", "tmdb_original_language": "fr", "tmdb_production_countries": "FR", "tmdb_overview": "A French drama.", "candidate_sources": ["language:fr"], "candidate_source_ranks": ["1"]},
            {"1"},
        ),
        (
            "I want a film from France.",
            {"requested_countries": ["FR"]},
            {"source_id": "2", "tmdb_id": 2, "title": "French Country Film", "year": 2001, "release_year": 2001, "tmdb_genres": "Drama", "tmdb_original_language": "fr", "tmdb_production_countries": "FR|BE", "tmdb_overview": "A French co-production.", "candidate_sources": ["country:FR"], "candidate_source_ranks": ["1"]},
            {"2"},
        ),
        (
            "I want a European film.",
            {"requested_countries": ["FR", "DE", "IT", "ES", "GB"]},
            {"source_id": "6", "tmdb_id": 6, "title": "European Film", "year": 2006, "release_year": 2006, "tmdb_genres": "Drama", "tmdb_original_language": "fr", "tmdb_production_countries": "FR|US", "tmdb_overview": "A European co-production.", "candidate_sources": ["country:FR"], "candidate_source_ranks": ["1"]},
            {"6"},
        ),
        (
            "I want something from the 1990s.",
            {"requested_decades": ["1990s"]},
            {"source_id": "3", "tmdb_id": 3, "title": "Nineties Film", "year": 1994, "release_year": 1994, "tmdb_genres": "Drama", "tmdb_original_language": "en", "tmdb_production_countries": "US", "tmdb_overview": "A 1990s drama.", "candidate_sources": ["decade:1990s"], "candidate_source_ranks": ["1"]},
            {"3"},
        ),
        (
            "I want a drama.",
            {"requested_genres": ["Drama"]},
            {"source_id": "4", "tmdb_id": 4, "title": "Drama Film", "year": 2004, "release_year": 2004, "tmdb_genres": "Drama|Crime", "tmdb_original_language": "en", "tmdb_production_countries": "US", "tmdb_overview": "A drama.", "candidate_sources": ["genre:Drama"], "candidate_source_ranks": ["1"]},
            {"4"},
        ),
        (
            "I do not want drama.",
            {"exclusions": ["drama"]},
            {"source_id": "5", "tmdb_id": 5, "title": "Drama Film", "year": 2005, "release_year": 2005, "tmdb_genres": "Drama", "tmdb_original_language": "en", "tmdb_production_countries": "US", "tmdb_overview": "A drama.", "candidate_sources": ["popular"], "candidate_source_ranks": ["1"]},
            set(),
        ),
    ],
)
def test_hard_constraints_filter_candidates(monkeypatch, query, intent_updates, candidate_row, expected_ids) -> None:
    _patch_embeddings(monkeypatch)
    request = _contextual_request(query, **intent_updates)
    result = discover_catalog_for_request(request, candidate_pool=pd.DataFrame([candidate_row]))
    assert set(result.catalog_frame["source_id"].astype(str)) == expected_ids


def test_impossible_constraint_combination_abstains_without_relaxing(monkeypatch) -> None:
    _patch_embeddings(monkeypatch)
    request = _contextual_request(
        "I want a French Japanese film from the 1990s with no drama.",
        requested_languages=["fr"],
        requested_countries=["JP"],
        requested_decades=["1990s"],
        requested_genres=["Drama"],
        exclusions=["drama"],
    )
    candidate_pool = pd.DataFrame(
        [
            {
                "source_id": "11",
                "tmdb_id": 11,
                "title": "Impossible Candidate",
                "year": 1994,
                "release_year": 1994,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "fr",
                "tmdb_production_countries": "FR",
                "tmdb_overview": "A French drama.",
                "candidate_sources": ["genre:Drama"],
                "candidate_source_ranks": ["1"],
            }
        ]
    )
    result = discover_catalog_for_request(request, candidate_pool=candidate_pool)
    assert result.catalog_frame.empty
    assert result.retrieval_diagnostics.post_constraint_candidate_count == 0


def test_missing_genre_metadata_fails_closed(monkeypatch) -> None:
    _patch_embeddings(monkeypatch)
    request = _contextual_request("I want a drama.", requested_genres=["Drama"])
    request = request.model_copy(
        update={
            "spec": request.spec.model_copy(
                update={
                    "structured_constraints": request.spec.structured_constraints.model_copy(
                        update={"required_genres": ["Drama"]}
                    )
                }
            )
        }
    )
    candidate_pool = pd.DataFrame(
        [
            {
                "source_id": "77",
                "tmdb_id": 77,
                "title": "Sparse Metadata Film",
                "year": 2004,
                "release_year": 2004,
                "tmdb_genres": pd.NA,
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A film with missing genre metadata.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["1"],
            }
        ]
    )

    retrieved = discover_catalog_for_request(request, candidate_pool=candidate_pool)
    assert retrieved.catalog_frame.empty

    qualified, records = qualify_candidates(request, candidate_pool, max_candidates=10)
    assert qualified.empty
    assert records[0].qualification_status == "unsupported"


def test_watched_exclusion_audit_captures_consumed_titles() -> None:
    consumed = pd.DataFrame(
        [
            {"source_id": "101", "title": "Bridesmaids", "year": 2011, "release_year": 2011},
            {"source_id": "102", "title": "The Bling Ring", "year": 2013, "release_year": 2013},
        ]
    )
    enriched = pd.DataFrame(
        [
            {"source_id": "101", "tmdb_id": 101},
            {"source_id": "102", "tmdb_id": 102},
        ]
    )

    exclusions, report = build_watched_exclusions(consumed, enriched, client=None)

    assert set(exclusions["title"].tolist()) == {"Bridesmaids", "The Bling Ring"}
    assert report["tmdb_exclusions"] == 2


def test_coordinated_semantic_exclusions_are_all_captured() -> None:
    request = understand_request("I want a French film tonight, but absolutely no drama, violence, or show business.")

    assert request.spec is not None
    exclusions = {concept.concept for concept in request.spec.semantic_exclusions}
    assert {"drama", "violence", "show_business"}.issubset(exclusions)
