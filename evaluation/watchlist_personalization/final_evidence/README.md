# Final evaluation evidence

This pack restores the final supplied hosted comparison and completed human review. It replaces the earlier 60-row, unscored experiment as the primary evaluator-facing evidence. It does not change runtime code or constitute a new run.

## Evidence and version boundaries

- `hosted/`: four experiments, full/empty watchlist × A/B, 12 prompts each (48 rows). Original suite summary and manifest contain experiment IDs and URLs. Historical absolute paths in these original files are provenance, not current executable paths.
- `human_review/completed_human_ratings.csv`: original completed 84-row review, byte-identical. 61 yes / 11 no / 12 unsure on would-watch, counting unique prompt–film pairs.
- `human_review/selection_rating_join.csv`: derived mapping to the 194 selected appearances across the four experiments, using actual output order. Original unblinding rank_memberships is not a reliable final slate position.
- `runtime_freeze/`: original later freeze manifest (including source hashes), contract, reported tests, and targeted H12 result. These belong to the three subsequent fixes, not to the hosted experiment's code version. Current cleaned source was not supplied or rehashed in this review.
- `provenance.json`: sources, hashes and the exact compact-results transformation.

## Human comparison

| Scenario | Variant | Would watch: yes | Selected appearances |
|---|---|---:|---:|
| Full watchlist | A | 39 | 51 |
| Full watchlist | B | 39 | 51 |
| Empty watchlist | A | 27 | 46 |
| Empty watchlist | B | 25 | 46 |

This is one person's review with shared candidates across conditions, not independent trials or a statistically established winner. Variant A is the accepted practical default. The results do not prove that historical taste has no value. Three numeric rating entries are outside the numeric 1–5 contract; original values are preserved and identified in review_summary.json. No mean score is silently repaired.

## Limitations to report

The hosted run's H12 returned no match in all four conditions; benchmark_authorized remained false. The 48 rows include partial slates. Stored requested_slate_completion feedback cannot be interpreted as actual five-film completion: retain it as original feedback and use selected lengths for factual slate counts. Dataset ID is null in original metadata although experiment URLs exist. Hosted links were not independently opened during this review.

The later targeted H12 preparation reports 48 eligible / 5 selected with the full watchlist and 0 / 0 with an empty watchlist. That small artifact has no film-level selections and cannot independently demonstrate their quality. The final fixes have reported local verification; no new hosted verification is claimed.

results_compact.json preserves all 48 results' recommendations, order, metadata, feedback and errors while omitting bulky debug candidate/embedding payloads. Nonfinite numbers become null for standards-compliant JSON. Keep the full raw export in the external research archive for deeper forensic reproduction; it is not duplicated here.
