# Runtime v3 Contextual Retrieval

## What Changed

Runtime v3 keeps the frozen B3 candidate universe and the same 50-candidate shortlist size, but it changes how that shortlist is selected.

## Retrieval Flow

1. Load the full frozen 259-film universe.
2. Apply the existing hard constraints to the full universe first.
3. If the request contains meaningful contextual content, embed the request with `text-embedding-3-small` and compare it to the cached film embeddings in the same space.
4. Fuse contextual rank and B3 rank with reciprocal rank fusion (`k=60`).
5. Pass only the compact shortlist to the LLM.
6. If generation fails, fallback also stays inside that shortlist.

## Guardrails

- B3 is still used.
- No new ranking taxonomy was introduced.
- Goodreads remains descriptive only.
- Runtime v2 remains unchanged.

## Diagnostics

The retrieval diagnostic reports:

- full-universe size
- post-constraint size
- whether contextual retrieval was used
- runtime v2 top-50 overlap
- newly admitted candidates from outside the old B3 top-50
- top-10 shortlist rankings

