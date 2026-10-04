# runtime_v1 Failure Analysis

Source evidence used:
- `mvp/artifacts/generative/langsmith/runtime_v1_baseline/experiment_results.csv`
- `evaluation/langsmith/runtime_v1_dataset.json`

The four cases below were inspected from the saved runtime outputs and debug payloads, not inferred from aggregate metrics.

| Case | Query | Parsed intent | Parsed constraints / exclusions | Eligible candidates after deterministic filtering | Generation result | Validation result | Repair attempted | Repair result | Fallback used | Final recommendation count | Failure / note |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | ---: | --- |
| E06 | `Recommend me something French.` | `CONSTRAINT` | languages: `["fr"]` | 1 | generated | passed | no | n/a | no | 1 | Not a runtime failure. The frozen candidate pool only exposed one French-eligible film in the supplied context, so fewer than 5 was the correct grounded outcome. |
| E11 | `Recommend something, but no drama.` | `CONSTRAINT` | parsed as genres: `["Drama"]` instead of an exclusion; exclusions: `[]` | 25 | fallback | failed | yes | final valid repaired response not preserved in artifact; runtime fell back after repair did not validate | yes | 5 | The parser misread the negated genre as a positive genre constraint, and the model then returned no recommendations even though eligible candidates were available. Final validation error: `Structured response returned no recommendations despite eligible candidates being available.` |
| E17 | `Which film are you recommending because of my Goodreads history?` | `GENERAL_DISCOVERY` | parsed as genres: `["History"]` from the phrase `Goodreads history`; exclusions: `[]` | 4 | generated | passed | no | n/a | no | 3 | Intent parsing incorrectly treated `history` as the genre `History`. The runtime did not fail validation because the current count evaluator only required at-most-5, but the output still under-filled the eligible shortlist. |
| E18 | `I want something weird but still emotionally engaging.` | `GENERAL_DISCOVERY` | none | 50 | generated | passed | no | n/a | no | 5 | Soft intent misclassified. This should have been `MOOD_THEME`; the parser failed to recognize the qualitative taste descriptors and left the request as generic discovery. |

## Takeaways

- E11 is a genuine parser plus generation/repair failure around negated genre handling.
- E17 exposes two issues: a false-positive `History` genre parse from the Goodreads phrase, and a count policy that did not penalize under-filling an otherwise eligible shortlist.
- E18 is the clearest soft-intent classification miss and is the primary candidate for the v2 intent fix.
- E06 does not need a model fix; the count should be evaluated relative to how many grounded candidates actually survive filtering.
