# runtime_v4 Semantic Correction Audit

## Baseline Context

- Preserved hosted baseline: `taste-agent-runtime-v4-hosted-1ef4feed`
- Experiment ID: `469c8480-531b-4ba1-8293-62d85c4a9a7d`
- Dataset: `taste-agent-runtime-v4-eval`
- Goal: correct the architecture, not tune for the 11-case set.

## Component Audit

### 1. Request parser
- Current state: lexical parsing in `mvp/src/generative/context_builder.py::parse_intent`, plus reference-title regex heuristics in `mvp/src/generative_v4/intent_chain.py`.
- Problem: stopword/token matching leaks filler words into film requirements and misses semantic exclusions.
- Classification: `REPLACE`
- Required change: structured semantic request interpretation with deterministic normalization.

### 2. Intent schema
- Current state: `RecommendationIntent` only captures coarse legacy fields.
- Problem: no explicit separation between structured metadata constraints, semantic requirements, semantic exclusions, reference context, and novelty intent.
- Classification: `MODIFY`
- Required change: add a structured request model while preserving legacy compatibility.

### 3. Stopword / token extraction
- Current state: used as the basic semantic decomposition mechanism.
- Problem: produces incorrect aspects like `want`, `actually`, `normally`, `something`.
- Classification: `REMOVE`
- Required change: keep only as low-level normalization helper, not as meaning extraction.

### 4. Exclusion parsing
- Current state: regex-backed lexical exclusions, with partial post-processing.
- Problem: collapses multi-concept exclusions into a single malformed tail.
- Classification: `REPLACE`
- Required change: semantic exclusion extraction as structured concepts.

### 5. Qualification layer
- Current state: deterministic text matching in `mvp/src/generative_v4/qualification_chain.py`, with `llm_used=False`.
- Problem: request-fit is computed from overlapping words, not grounded entailment.
- Classification: `REPLACE`
- Required change: grounded semantic qualification with structured output and conservative rejection.

### 6. `llm_used` reporting
- Current state: hardcoded false in qualification records and zeroed runtime LLM counts.
- Problem: does not reflect actual model use once semantic parsing/qualification are introduced.
- Classification: `MODIFY`
- Required change: track runtime LLM calls and token usage separately from judge usage.

### 7. Dynamic retrieval triggers
- Current state: contextual augmentation exists, but some requests still degrade to the old broad seed behavior.
- Problem: query-aware expansion is not guaranteed before personalization.
- Classification: `MODIFY`
- Required change: build an explicit retrieval plan and expand dynamically from request semantics.

### 8. TMDB retrieval endpoints
- Current state: `discover/movie` and reference recommendations are used; keyword search is not yet used.
- Problem: semantic concepts are not resolved to TMDB keyword IDs.
- Classification: `MODIFY`
- Required change: add keyword resolution and keyword-based discover paths where supported.

### 9. Old 259 dependency
- Current state: still used as the historical candidate seed in broad retrieval.
- Problem: runtime_v4 should not require the old 259 to function.
- Classification: `MODIFY`
- Required change: keep as optional supplemental recall only.

### 10. Reference resolution
- Current state: title regex + TMDB movie search, but resolved references do not reliably influence qualification semantics.
- Problem: reference context is not translated into grounded similarity criteria.
- Classification: `MODIFY`
- Required change: use grounded reference metadata in retrieval and qualification.

### 11. Novelty logic
- Current state: novelty selection is embedding-distance based, but the intent path still conflates novelty with generic lexical cues.
- Problem: novelty should be explicit and independent from B3.
- Classification: `KEEP` with targeted `MODIFY`
- Required change: preserve the distance diagnostic, expose it as a dedicated stage, and stop lexical novelty inference.

### 12. B3 scoring path
- Current state: frozen Ridge/PCA feature path remains isolated.
- Problem: none in this pass.
- Classification: `KEEP`

### 13. Selection path
- Current state: selection is deterministic and B3-aware.
- Problem: should consume semantically qualified candidates only and preserve novelty / reference diagnostics.
- Classification: `MODIFY`

### 14. Explanation path
- Current state: grounded explanations are already separated from taste evidence.
- Problem: request-fit, taste-fit, novelty, and reference-fit claims must remain explicitly separated.
- Classification: `MODIFY`

### 15. Validators
- Current state: validators cover basic grounding, Goodreads, sensitive inference, and unsupported dimensions.
- Problem: some semantic contracts need stronger checks and some non-applicable checks should be skipped rather than scored `0`.
- Classification: `MODIFY`

### 16. LangSmith instrumentation
- Current state: stage hierarchy exists but omits a dedicated retrieval-plan and semantic-qualification framing; runtime token accounting is incomplete.
- Problem: observability does not reflect actual semantic LLM usage.
- Classification: `MODIFY`

### 17. Evaluator field mappings
- Current state: `intent_match` reads the wrong field path and evaluates as empty.
- Problem: evaluator is miswired, not the runtime intent.
- Classification: `MODIFY`

## Architecture Decision

- Keep frozen B3 / PCA / taxonomy / Goodreads behavior unchanged.
- Replace lexical semantic interpretation and qualification with structured, conservative semantic components.
- Preserve deterministic structured filters, but do not use them as a proxy for meaning.
