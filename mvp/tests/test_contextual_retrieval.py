from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mvp.src.generative.contextual_retrieval import build_runtime_v3_recommendation_context
from mvp.src.generative.validators import fallback_recommendations, validate_recommendation_response
from mvp.src.generative.schemas import RecommendationItem, RecommendationResponse


def _taste_profile() -> pd.DataFrame:
    return pd.read_csv("mvp/artifacts/diagnostics/b3_live_behavior/taste_profile.csv")


def _make_universe(size: int = 60) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for index in range(1, size + 1):
        rows.append(
            {
                "source_id": str(index),
                "tmdb_id": index,
                "title": f"Film {index}",
                "year": 2000 + index,
                "tmdb_genres": "Drama",
                "tmdb_original_language": "en",
                "tmdb_production_countries": "US",
                "tmdb_overview": f"Overview {index}",
                "candidate_sources": "['top_rated']",
                "candidate_source_ranks": "['1']",
                "predicted_preference": float(size - index + 1) / size,
                "rank": index,
                "emb_0000": 0.0,
                "emb_0001": 1.0,
                "melancholic": 0.5,
                "atmospheric": 0.5,
            }
        )
    universe = pd.DataFrame(rows)
    semantics = pd.DataFrame(
        {
            "source_id": universe["source_id"].astype(str),
            "melancholic": 0.5,
            "atmospheric": 0.5,
        }
    )
    return universe, semantics, _taste_profile()


def _patch_universe(monkeypatch, universe: pd.DataFrame, semantics: pd.DataFrame, taste_profile: pd.DataFrame) -> None:
    monkeypatch.setattr(
        "mvp.src.generative.contextual_retrieval._prepare_universe",
        lambda **kwargs: (universe.copy(), semantics.copy(), taste_profile.copy()),
    )


def test_full_universe_hard_constraints_precede_truncation(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe()
    universe.loc[54, "tmdb_original_language"] = "fr"
    universe.loc[54, "tmdb_production_countries"] = "FR"
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, diagnostics = build_runtime_v3_recommendation_context(
        "I want a French film tonight.",
        embedding_fn=lambda _: (_ for _ in ()).throw(AssertionError("embedding should not be used")),
    )

    assert diagnostics.contextual_retrieval_used is False
    assert "55" in [candidate.candidate_id for candidate in context.candidate_context]
    assert context.candidate_context[0].candidate_id == "55"


def test_candidate_outside_v2_top50_can_enter_v3_shortlist(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe()
    for index in range(len(universe)):
        universe.loc[index, "emb_0000"] = 0.0
        universe.loc[index, "emb_0001"] = 1.0
    universe.loc[59, "emb_0000"] = 1.0
    universe.loc[59, "emb_0001"] = 0.0
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, diagnostics = build_runtime_v3_recommendation_context(
        "I want something quiet, contemplative and character-driven.",
        embedding_fn=lambda _: np.array([1.0, 0.0]),
    )

    shortlist_ids = [candidate.candidate_id for candidate in context.candidate_context]
    assert diagnostics.contextual_retrieval_used is True
    assert "60" in shortlist_ids
    assert diagnostics.final_shortlist_rank_map["60"] <= 50


def test_b3_still_contributes_to_contextual_shortlist(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe(2)
    universe.loc[0, ["emb_0000", "emb_0001"]] = [1.0, 0.0]
    universe.loc[1, ["emb_0000", "emb_0001"]] = [1.0, 0.0]
    universe.loc[0, "rank"] = 5
    universe.loc[1, "rank"] = 1
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, diagnostics = build_runtime_v3_recommendation_context(
        "I want something quiet, contemplative and character-driven.",
        embedding_fn=lambda _: np.array([1.0, 0.0]),
    )

    shortlist_ids = [candidate.candidate_id for candidate in context.candidate_context[:2]]
    assert diagnostics.contextual_retrieval_used is True
    assert shortlist_ids[0] == "2"
    assert shortlist_ids[1] == "1"


def test_general_discovery_without_context_preserves_b3_order(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe(6)
    universe.loc[:, "rank"] = [6, 5, 4, 3, 2, 1]
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, diagnostics = build_runtime_v3_recommendation_context(
        "Recommend me something to watch.",
        embedding_fn=lambda _: (_ for _ in ()).throw(AssertionError("embedding should not be used")),
    )

    assert diagnostics.contextual_retrieval_used is False
    assert [candidate.candidate_id for candidate in context.candidate_context[:3]] == ["6", "5", "4"]


def test_goodreads_does_not_affect_retrieval(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe(4)
    universe["goodreads_score"] = [99, 1, 50, 25]
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, _ = build_runtime_v3_recommendation_context(
        "Recommend me something to watch.",
        embedding_fn=lambda _: (_ for _ in ()).throw(AssertionError("embedding should not be used")),
    )

    assert [candidate.candidate_id for candidate in context.candidate_context] == ["1", "2", "3", "4"]


def test_semantic_vectors_do_not_directly_alter_b3(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe(4)
    semantics.loc[:, "melancholic"] = [0.9, 0.1, 0.4, 0.2]
    semantics.loc[:, "atmospheric"] = [0.1, 0.8, 0.2, 0.3]
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, _ = build_runtime_v3_recommendation_context(
        "Recommend me something to watch.",
        embedding_fn=lambda _: (_ for _ in ()).throw(AssertionError("embedding should not be used")),
    )

    assert [candidate.candidate_id for candidate in context.candidate_context] == ["1", "2", "3", "4"]


def test_fallback_selects_from_runtime_v3_shortlist(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe(6)
    universe.loc[:, "rank"] = [6, 5, 4, 3, 2, 1]
    universe.loc[5, ["emb_0000", "emb_0001"]] = [1.0, 0.0]
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, _ = build_runtime_v3_recommendation_context(
        "I want something quiet, contemplative and character-driven.",
        embedding_fn=lambda _: np.array([1.0, 0.0]),
    )
    fallback = fallback_recommendations(context.candidate_context, context.intent, 5)
    shortlist_ids = {candidate.candidate_id for candidate in context.candidate_context}

    assert fallback
    assert {item["candidate_id"] for item in fallback}.issubset(shortlist_ids)


def test_no_candidate_outside_runtime_v3_shortlist_can_be_generated(monkeypatch) -> None:
    universe, semantics, taste_profile = _make_universe(6)
    universe.loc[:, "rank"] = [6, 5, 4, 3, 2, 1]
    _patch_universe(monkeypatch, universe, semantics, taste_profile)

    context, _ = build_runtime_v3_recommendation_context(
        "Recommend me something to watch.",
        embedding_fn=lambda _: (_ for _ in ()).throw(AssertionError("embedding should not be used")),
    )
    response = RecommendationResponse(
        intent=context.intent,
        recommendations=[
            RecommendationItem(
                candidate_id="999",
                title="Outside",
                year=2000,
                why_it_may_fit="Grounded explanation.",
                taste_signals=[],
                request_match=None,
                caveat=None,
            )
        ],
        response_summary="Summary",
        methodology_note="Note",
    )

    validation = validate_recommendation_response(response, context.candidate_context, context.intent, 5)
    assert not validation.passed
    assert any("unsupported candidate_id" in error for error in validation.errors)
