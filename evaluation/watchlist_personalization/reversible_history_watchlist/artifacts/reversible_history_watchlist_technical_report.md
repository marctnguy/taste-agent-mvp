# Reversible History + Watchlist Variant

## Status
- `status`: `completed`
- prompts evaluated: `12`
- LLM calls: `67`
- token usage: `{'input': 141349, 'output': 42349, 'total': 183698}`

## Verification
- Exact H01-H12 prompts were loaded from the frozen runtime-v4 manifest source prompt set.
- The H11 wording retained the La Bola Negra / Penelope Cruz reference.
- The frozen 393/99 split and aligned training embeddings were verified before execution.
- Request-driven external discovery, history-driven external discovery, and watchlist retrieval were all evaluated in isolation and in combination.
- Qualification judgments were shared across scenarios for identical film candidates.

## Competing Choices
- Full watchlist prompts with competing A/B outputs: `9` / `12`
- Empty watchlist prompts with competing A/B outputs: `8` / `12`

## Human Sheet
- rows: `60`
- unblinding rows: `60`

## Limitations
- History improves candidate generation only when TMDB returns viable seed expansions.
- Some prompts may still collapse to the same slate after qualification; that is recorded as a result, not tuned away.
- The new variant is isolated from the frozen corrected run and does not change the production default.