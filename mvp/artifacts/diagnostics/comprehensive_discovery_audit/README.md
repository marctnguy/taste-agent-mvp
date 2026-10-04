# Comprehensive Discovery Audit

Frozen inputs: `mvp/artifacts/recommendation_runs/20261003T075635Z/`, the corrected B3 contribution export, the 393-film development history, the cached 62-dim semantic profile, and existing embedding caches.

Minimum support rule: categorical amplification ratios are highlighted only when the candidate-pool count is at least 5. Genre combinations use the same support floor and are limited to combinations with at least 5 candidate films.

Taste coverage rule: eligible positive taste dimensions require evidence_count >= 3, evidence_confidence >= 0.4, and pearson_preference_association >= 0.05. Meaningful expression uses semantic_score >= 0.25.
