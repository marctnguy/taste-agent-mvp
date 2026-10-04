# Runtime v3 Retrieval Audit

## Current Path

- `mvp/src/generative/context_builder.py` loads `mvp/artifacts/recommendation_runs/20261003T075635Z/ranked_candidates.csv`, merges `candidate_pool.csv`, then immediately truncates with `.head(candidate_context_size)`.
- `candidate_context_size` is 50 in `mvp/src/generative/runtime_version.py` and the runtime v2 manifest.
- `mvp/src/generative/recommendation_chain.py` then applies `hard_filter_candidates()` after that truncation.
- So runtime v2 effectively behaves as: B3 top-50 first, hard filters second, LLM last.

## Full Candidate Universe

The frozen live universe is 259 films and the same 259 IDs are present across the frozen B3 ranking, candidate pool, candidate embedding cache, and 62-d semantic vector export.

Available for all 259 before truncation:

- `source_id`
- `title`
- `year`
- `tmdb_genres`
- `tmdb_original_language`
- `tmdb_production_countries`
- `tmdb_overview`
- `predicted_preference` / `rank` from B3
- `candidate_sources` / `candidate_source_ranks`
- 1536-d candidate embeddings from `mvp/artifacts/models/b3_mvp/candidate_embedding_cache/candidate_embeddings.csv`
- 62-d semantic vectors from `mvp/artifacts/diagnostics/comprehensive_discovery_audit/candidate_semantic_vectors.csv`

## Reusable Embedding Space

- Candidate embeddings already use `text-embedding-3-small`.
- The cached vectors are already aligned to the same OpenAI embedding space and can be cosine-compared directly after L2 normalization.
- No query-embedding path exists yet in the runtime codebase; runtime v3 will add one using the same embedding model.

## Hard Constraints

The current hard-constraint semantics are already defined in `mvp/src/generative/context_builder.py` and `mvp/src/generative/validators.py`:

- requested language
- requested country
- requested decade
- requested genre
- explicit exclusions

These are currently applied after the top-50 B3 truncation.

## Runtime v3 Decision

Implement a new retrieval stage that:

1. starts from the full 259-film universe,
2. applies the existing hard-constraint semantics before shortlist construction,
3. uses query-to-film cosine similarity in the existing OpenAI embedding space when the request contains meaningful contextual content,
4. fuses contextual rank and B3 rank with RRF,
5. keeps the LLM shortlist compact at 50 or fewer candidates,
6. leaves runtime v2 unchanged.
