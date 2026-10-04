# Runtime V4 Architecture

## Goal

Runtime_v4 is a first-principles rebuild of the recommendation runtime.
It separates:

1. request relevance
2. historical preference compatibility
3. interpretable taste evidence
4. generative explanation

## Stage Flow

`USER REQUEST`
→ `REQUEST UNDERSTANDING`
→ `QUERY-AWARE CATALOG RETRIEVAL`
→ `HARD CONSTRAINTS`
→ `SEMANTIC CANDIDATE QUALIFICATION`
→ `FROZEN B3 PERSONALIZATION`
→ `SELECTION`
→ `GROUNDED EXPLANATION`
→ `VALIDATION / ABSTENTION`

## Package Layout

`mvp/src/retrieval/catalog_retrieval.py`

- query-aware TMDB candidate retrieval
- canonical film document construction
- embedding generation and caching
- request relevance scoring
- B3 applicability diagnostics

`mvp/src/generative_v4/schemas.py`

- request understanding
- retrieved candidate structure
- qualification records
- selection records
- runtime metadata/result payloads

`mvp/src/generative_v4/intent_chain.py`

- request parsing
- reference-film resolution
- novelty detection

`mvp/src/generative_v4/qualification_chain.py`

- conservative request-fit qualification
- supported/unsupported request aspect tracking

`mvp/src/generative_v4/selection_chain.py`

- request-first ranking among qualified candidates
- novelty-aware selection
- no-match / abstention support

`mvp/src/generative_v4/explanation_chain.py`

- grounded explanation generation
- 62-d taste evidence attachment
- caveat preservation

`mvp/src/generative_v4/validators.py`

- request-grounding checks
- unsupported-candidate rejection
- caveat preservation
- Goodreads / sensitive inference guardrails

`mvp/src/generative_v4/runtime.py`

- orchestration
- LangSmith tracing
- validation
- final response assembly

## Principles

- B3 must never define the candidate universe.
- Request relevance must be established before B3 personalization.
- Unsupported candidates never proceed to the final selection layer.
- Abstention is valid.
- Fewer than five recommendations is valid.
- The explanation layer may only verbalize evidence already attached to the selected candidate.

