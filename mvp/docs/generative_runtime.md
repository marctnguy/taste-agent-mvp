# Generative Runtime

## Purpose
The generative runtime is a grounded LangChain recommendation interface layered over the frozen B3 shortlist.

It does not:
- retrain B3
- rerank embeddings
- invent candidate films
- use Goodreads as ranking evidence

## Input Contract
- user query
- frozen top-50 B3 candidate context
- candidate metadata from the frozen recommendation run
- eligible 62-dimensional semantic evidence
- compact Taste Profile associations

## Output Contract
Each response includes:
- parsed intent
- up to `recommendation_count` recommendation items
- response summary
- methodology note

## Intent Taxonomy
- `GENERAL_DISCOVERY`
- `MOOD_THEME`
- `NOVELTY`
- `CONSTRAINT`
- `EXPLAIN`

## Grounding Rules
- Only supplied candidate IDs may be returned.
- Explanations may use only candidate metadata, eligible semantic evidence, the Taste Profile, and the explicit query.
- Goodreads is descriptive only.

## Hard vs Soft Constraints
- Hard constraints are filtered deterministically before generation when supported.
- Soft thematic requests may be interpreted through eligible semantic evidence and contextual selection.

## Validation
After generation, the runtime validates:
- candidate IDs exist in context
- titles and years match grounded metadata
- recommendation count is within bounds
- no duplicates appear
- hard filters remain satisfied
- taste signals are grounded
- Goodreads is not used as ranking evidence

## Fallback
If generation or validation fails, the runtime falls back to the highest-ranked eligible B3 candidates.

## Prompt Versions
- `intent_v1`
- `recommendation_v1`

## Limitations
- This runtime is not a validated improvement over B3.
- It is a grounded product interface, not a new predictive model.
- LangSmith evaluation and Streamlit UI are deferred to a later phase.
