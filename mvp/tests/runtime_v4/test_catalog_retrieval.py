from __future__ import annotations

import numpy as np
import pandas as pd

from mvp.src.generative_v4.intent_chain import understand_request
from mvp.src.retrieval.catalog_retrieval import build_canonical_film_document, build_canonical_film_documents, discover_catalog_for_request


def test_dynamic_catalog_retrieval_accepts_new_candidate_outside_old_universe(monkeypatch, make_embedding_frame) -> None:
    request = understand_request("I want a French drama from the 1980s.").model_copy(
        update={"request_mode": "contextual", "request_relevance_mode": "query_aware"}
    )
    dynamic_candidate = pd.DataFrame(
        [
                {
                    "source_id": "900001",
                    "tmdb_id": 900001,
                    "title": "New Wave",
                    "year": 1986,
                    "release_year": 1986,
                    "tmdb_genres": "Drama",
                    "tmdb_original_language": "fr",
                    "tmdb_production_countries": "FR",
                    "tmdb_overview": "A French drama about memory and identity.",
                    "candidate_sources": ["genre:Drama", "language:fr"],
                    "candidate_source_ranks": ["1", "1"],
                    "predicted_preference": 0.5,
                    "rank": 1,
                }
            ]
        )
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval.generate_candidate_pool",
        lambda candidate_limit=300: (pd.DataFrame(columns=["source_id"]), {"candidate_limit": candidate_limit, "after_deduplication": 0}),
    )
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval._augment_with_constraint_candidates",
        lambda request, client: (dynamic_candidate.copy(), {"constraint_candidates": 1}),
    )
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval.embed_documents",
        lambda documents: (
            make_embedding_frame(documents["source_id"].astype(str).tolist(), [[1.0, 0.0]] * len(documents))
            if not documents.empty and "source_id" in documents.columns
            else pd.DataFrame(columns=["source_id", "embedding_model", "embedding_dim", "status", "document_hash"]),
            pd.DataFrame(),
            {"provider": "local", "model": "test", "coverage": float(len(documents) > 0), "successful": len(documents), "failed": 0, "cache_status": "test", "cache_hits": 0, "cache_misses": len(documents)},
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

    result = discover_catalog_for_request(request, candidate_context_size=50)

    assert set(result.catalog_frame["source_id"].astype(str)) == {"900001"}
    assert result.retrieval_diagnostics.candidate_universe_count == 1
    assert result.retrieval_diagnostics.post_constraint_candidate_count == 1
    assert result.retrieval_diagnostics.query_aware_augmentation_used is True
    assert pd.notna(result.catalog_frame.iloc[0]["b3_applicability_distance"])
    assert "Candidate provenance:" in result.documents.iloc[0]["document_text"]


def test_duplicate_tmdb_candidates_are_deduplicated_and_watched_films_are_excluded(monkeypatch, make_embedding_frame) -> None:
    request = understand_request("Recommend me something to watch.")
    candidate_pool = pd.DataFrame(
        [
            {
                "source_id": "101",
                "tmdb_id": 101,
                "title": "Duplicate One",
                "year": 2001,
                "release_year": 2001,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A watched duplicate candidate.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["1"],
            },
            {
                "source_id": "101",
                "tmdb_id": 101,
                "title": "Duplicate One",
                "year": 2001,
                "release_year": 2001,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "A watched duplicate candidate.",
                "candidate_sources": ["top_rated"],
                "candidate_source_ranks": ["2"],
            },
            {
                "source_id": "202",
                "tmdb_id": 202,
                "title": "Watched Film",
                "year": 2002,
                "release_year": 2002,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": "This one should be excluded.",
                "candidate_sources": ["popular"],
                "candidate_source_ranks": ["3"],
            },
        ]
    )
    monkeypatch.setattr(
        "mvp.src.retrieval.catalog_retrieval.embed_documents",
        lambda documents: (
            make_embedding_frame(documents["source_id"].astype(str).tolist(), [[1.0, 0.0] for _ in range(len(documents))]),
            pd.DataFrame(),
            {"provider": "local", "model": "test", "coverage": 1.0, "successful": len(documents), "failed": 0, "cache_status": "test", "cache_hits": 0, "cache_misses": len(documents)},
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

    result = discover_catalog_for_request(request, candidate_pool=candidate_pool, watched_ids=["202"])

    assert set(result.catalog_frame["source_id"].astype(str)) == {"101"}
    assert len(result.catalog_frame) == 1


def test_canonical_document_retains_provenance_and_omits_missing_overview() -> None:
    row = pd.Series(
        {
            "source_id": "303",
            "tmdb_id": 303,
            "title": "Sparse Film",
            "release_year": 2001,
            "tmdb_genres": "Drama|Mystery",
            "tmdb_original_language": "en",
            "tmdb_production_countries": "US",
            "tmdb_overview": pd.NA,
            "candidate_sources": ["genre:drama", "language:en"],
        }
    )

    document = build_canonical_film_document(row)
    documents = build_canonical_film_documents(pd.DataFrame([row]))

    assert "sparse film" in document
    assert "Overview:" not in document
    assert "Candidate provenance: genre:drama | language:en" in document
    assert documents.iloc[0]["document_version"] == "film_doc_v2"
    assert documents.iloc[0]["title"] == "Sparse Film"
