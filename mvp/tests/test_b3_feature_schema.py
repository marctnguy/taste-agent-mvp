from __future__ import annotations

from mvp.src.mvp_deployment import load_model_bundle
from mvp.src.semantics import SEMANTIC_COLUMNS


def test_b3_feature_matrix_excludes_candidate_source_and_tmdb_popularity_fields() -> None:
    bundle, _, _ = load_model_bundle("mvp/artifacts/models/b3_mvp")
    forbidden_substrings = ("candidate_", "popular", "vote", "tmdb_popularity")
    assert len(bundle.feature_columns) == 114
    assert not any(
        any(token in column for token in forbidden_substrings)
        for column in bundle.feature_columns
    )


def test_semantic_taxonomy_is_canonical_62_dimensions() -> None:
    assert len(SEMANTIC_COLUMNS) == 62
    assert len(set(SEMANTIC_COLUMNS)) == 62
