from __future__ import annotations

from collections import Counter

from mvp.src.discovery_slate import _delta_hhi, _family_mass


def test_family_mass_splits_multilabel_items_equally() -> None:
    mass = _family_mass(["Drama", "Romance"])

    assert mass == {"Drama": 0.5, "Romance": 0.5}


def test_delta_hhi_does_not_reward_deconcentration() -> None:
    concentrated = Counter({"Drama": 1.0, "Comedy": 1.0})
    diversified_addition = {"Action": 1.0}
    concentrated_addition = {"Drama": 1.0}

    assert _delta_hhi(concentrated, 2.0, diversified_addition) < 0
    assert _delta_hhi(concentrated, 2.0, concentrated_addition) > 0
