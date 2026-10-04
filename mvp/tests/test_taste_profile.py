from __future__ import annotations

import pandas as pd

from mvp.src.taste_profile import build_taste_profile


def test_build_taste_profile_counts_only_strong_semantic_evidence() -> None:
    observations = pd.DataFrame(
        {
            "canonical_id": ["1", "2", "3"],
            "preference_weight": [1.0, 0.5, 0.0],
        }
    )
    semantic_vectors = pd.DataFrame(
        {
            "canonical_id": ["1", "2", "3"],
            "historical": [0.10, 0.25, 0.90],
        }
    )

    profile = build_taste_profile(observations, semantic_vectors)
    historical = profile.loc[profile["dimension"] == "historical"].iloc[0]

    assert int(historical["evidence_count"]) == 2
