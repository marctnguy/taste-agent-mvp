# Taste Agent MVP Architecture

## Goal
The MVP ships a single deterministic ranking model:

- `B3` = frozen TMDB metadata + latent film embeddings
- Goodreads data stays descriptive only
- No LangChain, no Streamlit, no adaptive retraining in the product path

## Model Bundle
The deployable bundle lives in `mvp/artifacts/models/b3_mvp/` and contains:

- `model.joblib`
- `pca.joblib`
- `metadata_preprocessor.joblib`
- `config.json`
- `feature_schema.json`
- `training_manifest.json`
- `cv_results.json`

## Training Flow
`python -m mvp.app fit-mvp-model`:

1. Loads the frozen 393-film training population.
2. Reuses the locked PCA grid `[16, 32, 64, 128]` and Ridge alpha grid.
3. Selects hyperparameters by CV with Spearman primary, NDCG@10 tie-breaker.
4. Fits the final metadata preprocessor on all 393 training films.
5. Fits PCA on all 393 training film embeddings.
6. Fits the final Ridge model on all 393 training films.

## Recommendation Flow
`python -m mvp.app recommend`:

1. Loads the frozen B3 MVP bundle.
2. Loads candidate films.
3. Optionally removes watched `source_id` values.
4. Scores candidates deterministically.
5. Returns ranked records with:
   - `source_id`
   - `tmdb_id` when present
   - `title`
   - `year` when present
   - `predicted_preference`
   - `rank`
   - `model_status`
   - `architecture`
   - `taste_evidence`

## Candidate Contract
Candidate rows should support:

- canonical ID / TMDB ID
- title
- release year
- TMDB genres
- original language
- production countries
- overview
- optional semantic vector columns for explanation

## Design Boundary
Ranking uses only the frozen B3 model. Descriptive taste evidence can be attached, but it does not change numeric scores.

## Generative Runtime
The product now adds a separate LangChain contextual layer that does not alter B3 scoring:

- `B3` still produces the frozen shortlist and relevance order
- the deterministic runtime supplies a top-50 candidate context from the frozen recommendation run
- the LangChain layer selects and explains only from those supplied candidates
- the 62-dimensional Taste Profile is used for interpretation and explanation only
- Goodreads remains descriptive cross-media context only

This layer lives under `mvp/src/generative/` and is exposed through `python -m mvp.app recommend-agent`.
