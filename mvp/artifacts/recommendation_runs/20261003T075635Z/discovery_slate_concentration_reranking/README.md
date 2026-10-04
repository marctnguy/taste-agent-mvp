# Deterministic Discovery Slate

Frozen input: `mvp/artifacts/recommendation_runs/20261003T075635Z/`.

This run constructs four deterministic top-20 slates: `RAW_B3`, `CONCENTRATION_005`, `CONCENTRATION_010`, and `CONCENTRATION_020`.

Penalty behavior:
- only positive incremental concentration increases are penalized
- multi-label families split mass equally across labels
- no latent-space or semantic dimensions affect ranking
- source is treated as retrieval provenance, not taste
