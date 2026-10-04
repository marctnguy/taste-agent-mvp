# Runtime V4 Final Structural Audit

## Freeze Checks
- B3 unchanged: `True`
- PCA unchanged: `True`
- Ridge unchanged: `True`
- 62-d taxonomy unchanged: `True`
- Goodreads status: `descriptive_only`
- Interaction routing enabled: `True`
- Deterministic RequestSpec backfill enabled: `True`
- Missing genre metadata fails closed for genre constraints: `True`

## Case Audit
| H | Intent | Mode | Gen | Fallback | Abstain | LLM | Selected | Trace | Notes |
|---|---|---|---:|---:|---:|---:|---|---|---|
| H01 | GENERAL_DISCOVERY | generic | generated | False | False | 1 | Mother's Day, Toy Story 5, Blaze of Love, Cinema Paradiso, The Love Hypothesis | `01a1075d-189a-7150-846c-7894a529034c` | ok |
| H02 | MOOD_THEME | contextual | no_match | False | True | 2 | ∅ | `01a1075d-dba2-7a93-8c10-a836a5d53758` | ok |
| H03 | MOOD_THEME | generic | fallback | False | False | 1 | Mother's Day, Toy Story 5, Blaze of Love, Cinema Paradiso, The Love Hypothesis | `01a1075e-8553-7550-945f-7edbdf1a352b` | ok |
| H04 | GENERAL_DISCOVERY | reference | generated | False | False | 1 | The Last Woman, The Swimming Pool, Mothers' Instinct, The Thing with Feathers, Vivre Sa Vie | `01a1075f-237f-7141-bc37-78dace5256bb` | ok |
| H05 | MOOD_THEME | generic | generated | False | False | 1 | Mother's Day, Toy Story 5, Blaze of Love, Cinema Paradiso, The Love Hypothesis | `01a10760-3b41-7bd2-ab72-b7f04a46d74d` | ok |
| H06 | NOVELTY | novelty | generated | False | False | 1 | As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty, Regular Show: The Movie, Ocean with David Attenborough, Southbound, Remarkably Bright Creatures | `01a10761-73f7-70b1-8d23-e517907ee97b` | ok |
| H07 | GENERAL_DISCOVERY | generic | generated | False | False | 1 | Mother's Day, Toy Story 5, Blaze of Love, Cinema Paradiso, The Love Hypothesis | `01a10763-2b0d-7d12-84b9-0d589fb4a78a` | ok |
| H08 | CONSTRAINT | contextual | generated | False | False | 1 | Another Day, The Intouchables, De Gaulle: Résistance, Dreamchild, Amélie | `01a10764-4be8-7862-a23b-c3e7f92fe42f` | ok |
| H09 | GENERAL_DISCOVERY | reference | generated | False | False | 1 | I Believe in Unicorns, The Pervert's Guide to Cinema, The Pervert's Guide to Ideology | `01a10769-7d49-7913-bc3f-21f7f16ce09d` | ok |
| H10 | CONSTRAINT | contextual | generated | False | False | 1 | Strung, The Good, the Bad and the Ugly, Practical Magic, Psycho, Cavegirl | `01a1076a-6927-7633-a83b-e594a8c25463` | ok |
| H11 | GENERAL_DISCOVERY | reference | generated | False | False | 1 | Behold a Pale Horse, Once Upon a Time... in Hollywood, The Bling Ring | `01a1076b-a0e3-7db2-a83b-0052edbb05b3` | ok |
| H12 | GENERAL_DISCOVERY | reference | generated | False | False | 1 | As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty, Anna Nicole Smith: You Don't Know Me, A Good Woman Is Hard to Find, Ex Libris: The New York Public Library, OLIVIA RODRIGO: driving home 2 u (a SOUR film) | `01a1076c-d218-7052-8245-f38cd085a741` | ok |

## Holistic Audit
- Semantic request preservation: `True`
- Reference + semantic coexistence: `True`
- Novelty routing: `True`
- Hard constraints: `True`
- Semantic exclusions: `False`
- Candidate identity integrity: `True`
- Grounded request relevance: `observed in raw outputs`
- No post-hoc rationalization: `observed in raw outputs`
- Goodreads provenance truthfulness: `no Goodreads ranking claims observed`
- Profile/personality boundary: `no sensitive inference observed`
- Explanation-target routing: `observed in raw outputs`
- Abstention when appropriate: `1` cases abstained
- No fallback leakage: `True`
- No recommendation-core mutation: `true`

## Issues
- Critical contract failures: `0`
- Non-blocking quality issues: `0`
- Evaluator artifacts: `0`