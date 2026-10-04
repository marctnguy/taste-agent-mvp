# Runtime V4 Pre-Freeze Forensic Audit

Diagnostic only. No code, prompts, models, or manifests were modified.

## Scope
- Reviewed the frozen H01-H12 HITL run in `evaluation/hitl/runtime_v4_final/`.
- Reviewed the runtime_v4 implementation in `mvp/src/generative_v4/`.
- Checked the frozen manifest in `mvp/artifacts/generative/runtime_v4_manifest.json`.
- Checked the frozen candidate pool header in `mvp/artifacts/recommendation_runs/20261003T075635Z/candidate_pool.csv`.

## Repo state
- Git commit hash: unavailable.
- Working tree status: unavailable, because this path is not a git repository.

## Executive readout
- The system is not failing from one bug.
- It has a small number of coupled failures in request parsing, reference extraction, qualification semantics, and B3 feature propagation.
- The positive control cases still work:
  - H04 reference routing is intact.
  - H06 novelty routing is intact.
  - H08 structured country/language constraint routing is intact.

## Direct answers

### 1) Does B3 actually contribute?
Not materially in the frozen generic path.

- The frozen candidate pool artifact does not contain `predicted_preference`, `b3_applicability`, or `b3_applicability_distance`.
- `selection_chain.py` backfills those missing fields with defaults, including `predicted_preference = 0.0`.
- The generic selector then sorts mostly on defaulted values plus stable tie-breakers.
- Result: B3 is named in the reasoning text, but the ranking signal is effectively absent or placeholder-level in the frozen pool.

### 2) Why do the same generic five recur?
Because the generic path is almost fully deterministic after the defaults kick in.

- Generic requests do not enter contextual qualification.
- `predicted_preference` is missing in the frozen pool and becomes `0.0`.
- `request_relevance` is also often flat in broad mode.
- The remaining ordering relies on `b3_applicability_distance` and stable `source_id` tie-breaks.
- That produces the same top five across H01, H03, H05, and H07.

### 3) Was H02 an abstention because nothing matched both semantics?
Yes.

- H02 entered contextual mode with `request_mode = contextual`.
- The runtime saw 279 candidates.
- `qualified_candidate_count = 0`.
- `generation_status = no_match` and `abstention_used = true`.
- The request was parsed as two grounded mood concepts, and the qualifier found no candidate that satisfied the combined requirement set.

### 4) Why did H03 fail on "not cheesy"?
The negation was not cleanly promoted to a hard semantic exclusion.

- The raw H03 request shows `requested_taste_dimensions = ["tone"]`.
- The semantic group contains `comforting` and `not cheesy`, but the request is still routed as `generic`.
- The runtime then returns the generic five and marks the run as `fallback` with `validation_passed = false`.
- So the issue is not just the words "not cheesy"; it is the conversion from negation into a reliable exclusion / contextual request.

### 5) Why did H05 stay generic?
The semantic concepts collapsed into ordinal labels.

- H05 shows `requested_taste_dimensions = ["1", "2", "3"]`.
- The semantic groups also carry aspect IDs like `1`, `2`, `3` instead of the intended concepts.
- That leaves the request effectively ungrounded for contextual routing.

### 6) Why did H07 not become novelty?
The novelty detector is intentionally too narrow for that phrasing.

- `_contains_explicit_novelty_request()` only checks a short list of explicit novelty phrases.
- The H07 wording "mind-blown" / "out there" is not in that list.
- So the request falls back to generic discovery.

### 7) Why did H09 invent a Freedom reference?
The reference extractor is too permissive.

- `_extract_reference_title()` matches the `like` pattern.
- In H09, it captures `like freedom` from "feels like freedom".
- That produces `reference_title = freedom`, which then resolves to a TMDB entity.
- The runtime therefore treats a mood phrase as a reference title.

### 8) Why did H10 not become a hard year threshold?
Because "old/classic" is represented as soft semantics, not a hard temporal constraint.

- The runtime has historical / classic semantic labels.
- It does not automatically convert "old classic" into a year cutoff.
- So older candidates are preferred only if their content/metadata still matches the soft semantics.

### 9) Why was Behold a Pale Horse accepted in H11?
The qualification heuristic treats lexical overlap as subject-matter evidence.

- H11 resolves `La Bola Negra` as a reference.
- `qualification_chain.py` then checks candidates with `_supported_by_synonym()`.
- `Behold a Pale Horse` contains "famous" in its overview, so it is treated as supporting `fame`.
- That is a false-positive subject match, not a real fame/show-business grounding.

### 10) Why did H12 turn into a reference-like query?
Because `based on` is also a reference trigger.

- `_extract_reference_title()` matches the `based on` pattern.
- In H12 it captures `based on what you know about my taste`.
- That is not a film reference, but it is still treated as reference-like input.

### 11) What is the main exclusion bug?
Semantic exclusions are not reliably converted into a hard, end-to-end blocker.

- H03 shows that the negated idea is not consistently preserved as a true exclusion.
- The case-D exclusion failure indicates the same broader issue: exclusion semantics are not consistently enforced through parsing, qualification, and selection as a single invariant.

## Root-cause matrix

| ID | Root cause | Mechanism | Affected cases | Confidence |
|---|---|---|---|---|
| RC1 | Request parsing / normalization collapse | `intent_chain.py` can misclassify novelty, turn mood phrases into generic mode, and collapse concepts into weak or malformed semantic labels | H03, H05, H07, H10, H12 | High |
| RC2 | Reference extraction is overbroad | `_extract_reference_title()` matches ordinary text like "like freedom" and "based on what you know about my taste" | H09, H12 | High |
| RC3 | Qualification uses lexical synonym matching as proof | `_fallback_qualification_output()` admits candidates when overview text contains a synonym, even when the subject match is shallow | H11, partially H04 as the working contrast | High |
| RC4 | B3 features are not actually wired through the frozen pool | The frozen candidate pool lacks B3 columns; selection backfills zeros and NaNs, so generic ordering becomes mostly deterministic tie-breaking | H01, H03, H05, H07 | High |
| RC5 | Exclusion semantics are not end-to-end hard blockers | Negated concepts are not consistently preserved as `semantic_exclusions`, and the qualification/selection chain does not enforce them as a single invariant | H03, case D | Medium-High |

## Case evidence

- H01, H03, H05, H07 all collapse to the same generic slate: `Mother's Day`, `Toy Story 5`, `Blaze of Love`, `Cinema Paradiso`, `The Love Hypothesis`.
- H02 abstains cleanly: 279 candidates, 0 qualified.
- H04 is the protected success class: reference routing works and returns a distinct slate.
- H06 is the protected novelty class: novelty routing works.
- H08 is the protected structured-constraint class: French constraint filtering works.
- H09 resolves a non-film word as a reference title.
- H11 proves the false-positive grounding bug.
- H12 proves the reference-trigger overreach.

## Telemetry contradictions worth preserving
- H03 reports `generation_status = fallback` while `fallback_used = false`.
- H03 also reports `validation_passed = false` after producing a full slate.
- Those contradictions should be treated as part of the bug surface, not ignored as formatting noise.

## Conclusion
The runtime_v4 freeze is close enough to be useful, but it is not yet diagnostically clean.

The most likely repair surface is:
- request parsing / normalization,
- reference extraction,
- qualification grounding,
- and frozen B3 feature propagation.

That is a focused fix set, not a redesign.
