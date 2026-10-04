# Runtime V4 Corrected Hosted Experiment

- Experiment: `taste-agent-runtime-v4-corrected-bcf79a1f`
- Experiment ID: `85f458c3-eaa6-42c6-a5b8-54ebe239ca33`
- Dataset: `taste-agent-runtime-v4-eval`
- Dataset ID: `86d55dca-57b7-4069-9df8-9319afb925f4`
- Cases: `11`
- Trace coverage: `11/11`
- Validation pass rate: `1.00`

# CASE V05

## User request
I want a French film tonight, but absolutely no drama.

## Parsed request
- Intent: `CONSTRAINT`
- Generic/contextual: `contextual`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `None`
- After hard constraints: `None`
- Outside old 259: `None`
- Dynamic retrieval used: `None`
  - `None` De Gaulle: Résistance (relevance=0.3686564132881101, status=None)
  - `None` The Three Musketeers: D'Artagnan (relevance=0.3621133213654691, status=None)
  - `None` The Intouchables (relevance=0.3590524600571216, status=None)
  - `None` Amélie (relevance=0.3558525101510504, status=None)
  - `None` Good Vibes Only (relevance=0.3514763631130544, status=None)

## Qualification
- Strong: `0`
- Partial: `0`
- Unsupported: `0`

## Final response
```json
{
  "intent": {
    "intent_type": "CONSTRAINT",
    "requested_taste_dimensions": [],
    "requested_genres": [],
    "requested_languages": [
      "fr"
    ],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [
      "drama"
    ],
    "free_text_context": "Request for a French film excluding the drama genre."
  },
  "recommendations": [],
  "response_summary": "No grounded recommendation survived request qualification, so the system is abstaining.",
  "methodology_note": "Request relevance must be grounded before selection. Abstention is valid when no candidate survives qualification."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`0.0` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`3.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`0.0` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`1.0` comment=`None`
- `response_clarity` score=`4.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T165337Z`
- Trace ID: `01a102ae-77c5-75a1-837c-02f28e5b0dd0`

# CASE V06

## User request
I want a film about performers, fame, and show business, preferably with a period setting.

## Parsed request
- Intent: `GENERAL_DISCOVERY`
- Generic/contextual: `contextual`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{"intent_type": "GENERAL_DISCOVERY", "mode": "contextual", "structured_constraints": {"required_languages": [], "excluded_languages": [], "required_countries": [], "excluded_countries": [], "required_genres": [], "excluded_genres": [], "required_decades": ["1920s", "1930s", "1940s", "1950s", "1960s"], "excluded_decades": [], "min_year": 1920, "max_year": 1969}, "semantic_requirements": [{"aspect_id": "performers", "concept": "performers", "importance": "required", "source_span": "film about performers"}, {"aspect_id": "fame", "concept": "fame", "importance": "required", "source_span": "film about fame"}, {"aspect_id": "show_business", "concept": "show_business", "importance": "required", "source_span": "film about show business"}, {"aspect_id": "period_setting", "concept": "period_setting", "importance": "preferred", "source_span": "preferably with a period setting"}], "semantic_exclusions": [], "reference": {"title": null, "relation": null, "resolved_tmdb_id": null, "resolved_tmdb_title": null, "resolution_status": "absent"}, "novelty_goal": {"enabled": false, "priority": null}, "personalization_instruction": {"preference": null, "priority": null}, "request_relevance_mode": "query_aware", "semantic_query_text": "film about performers, fame, and show business, preferably with a period setting", "request_summary": "Request for a film that explores themes of performers, fame, and show business, with a preference for a period setting.", "llm_used": false, "llm_call_count": 0, "runtime_input_tokens": null, "runtime_output_tokens": null, "runtime_total_tokens": null}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `400`
- After hard constraints: `112`
- Outside old 259: `None`
- Dynamic retrieval used: `True`
  - `111642` St. Martin's Lane (relevance=0.3402440683830205, status=strong)
  - `13368` White Christmas (relevance=0.3373965547584739, status=strong)

## Qualification
- Strong: `2`
- Partial: `0`
- Unsupported: `18`

## Final response
```json
{
  "intent": {
    "intent_type": "GENERAL_DISCOVERY",
    "requested_taste_dimensions": [
      "performers",
      "fame",
      "show_business",
      "period_setting"
    ],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [
      "1920s",
      "1930s",
      "1940s",
      "1950s",
      "1960s"
    ],
    "exclusions": [],
    "free_text_context": "Request for a film that explores themes of performers, fame, and show business, with a preference for a period setting."
  },
  "recommendations": [
    {
      "candidate_id": "111642",
      "title": "St. Martin's Lane",
      "year": 1938,
      "why_it_may_fit": "Supported request aspects: performers Taste evidence: historical, moderate, comforting, melancholic, visual.",
      "taste_signals": [
        "historical",
        "moderate",
        "comforting",
        "melancholic",
        "visual"
      ],
      "request_match": "Supported request aspects: performers",
      "caveat": null
    },
    {
      "candidate_id": "13368",
      "title": "White Christmas",
      "year": 1954,
      "why_it_may_fit": "Supported request aspects: show_business Taste evidence: moderate, atmospheric, comforting, visual, morality.",
      "taste_signals": [
        "moderate",
        "atmospheric",
        "comforting",
        "visual",
        "morality"
      ],
      "request_match": "Supported request aspects: show_business",
      "caveat": null
    }
  ],
  "response_summary": "Returned 2 grounded recommendations from the qualified catalog.",
  "methodology_note": "Request relevance was checked first, then frozen B3 compatibility was used to order only the qualified candidates, and the explanation only verbalizes grounded evidence already attached to the selected films."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`5.0` comment=`None`
- `response_clarity` score=`5.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T165505Z`
- Trace ID: `01a102af-84d3-7fd1-9903-691d93dd4c5d`

# CASE V03

## User request
I want something so specific that the catalog probably has no grounded match.

## Parsed request
- Intent: `GENERAL_DISCOVERY`
- Generic/contextual: `generic`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `None`
- After hard constraints: `None`
- Outside old 259: `None`
- Dynamic retrieval used: `None`
  - `None` Renegade Immortal: Battle of the Immortal Slayer (relevance=0.0, status=None)
  - `None` Resident Evil (relevance=0.0, status=None)
  - `None` Coyote vs. Acme (relevance=0.0, status=None)
  - `None` The Fix (relevance=0.0, status=None)
  - `None` Runner (relevance=0.0, status=None)

## Qualification
- Strong: `0`
- Partial: `0`
- Unsupported: `0`

## Final response
```json
{
  "intent": {
    "intent_type": "GENERAL_DISCOVERY",
    "requested_taste_dimensions": [],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "Request for a highly specific film that likely has no existing match in the catalog."
  },
  "recommendations": [],
  "response_summary": "No grounded recommendation survived request qualification, so the system is abstaining.",
  "methodology_note": "Request relevance must be grounded before selection. Abstention is valid when no candidate survives qualification."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`0.0` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`5.0` comment=`None`
- `response_clarity` score=`5.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T165611Z`
- Trace ID: `01a102b0-d8a1-7d81-bf85-1457b1fbb7e8`

# CASE V11

## User request
I just watched La Bola Negra and loved the Penelope Cruz character in it, I want something about performers, fame and/or show business preferably with a period setting.

## Parsed request
- Intent: `GENERAL_DISCOVERY`
- Generic/contextual: `reference`
- Reference film: `La Bola Negra`
- Novelty request: `False`
- Hard constraints: `{"intent_type": "GENERAL_DISCOVERY", "mode": "contextual", "structured_constraints": {"required_languages": [], "excluded_languages": [], "required_countries": [], "excluded_countries": [], "required_genres": [], "excluded_genres": [], "required_decades": [], "excluded_decades": [], "min_year": null, "max_year": null}, "semantic_requirements": [{"aspect_id": "performers", "concept": "performers", "importance": "required", "source_span": "performers, fame and/or show business"}, {"aspect_id": "fame", "concept": "fame", "importance": "required", "source_span": "performers, fame and/or show business"}, {"aspect_id": "show_business", "concept": "show_business", "importance": "required", "source_span": "performers, fame and/or show business"}, {"aspect_id": "period_setting", "concept": "period_setting", "importance": "preferred", "source_span": "preferably with a period setting"}], "semantic_exclusions": [], "reference": {"title": "La Bola Negra", "relation": "watched", "resolved_tmdb_id": 1422041, "resolved_tmdb_title": "La bola negra", "resolution_status": "resolved"}, "novelty_goal": {"enabled": false, "priority": null}, "personalization_instruction": {"preference": null, "priority": null}, "request_relevance_mode": "query_aware", "semantic_query_text": "performers, fame and/or show business with a period setting", "request_summary": "Looking for films about performers, fame, and show business, preferably set in a historical context.", "llm_used": false, "llm_call_count": 0, "runtime_input_tokens": null, "runtime_output_tokens": null, "runtime_total_tokens": null}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `300`
- After hard constraints: `300`
- Outside old 259: `None`
- Dynamic retrieval used: `True`
  - `466272` Once Upon a Time... in Hollywood (relevance=0.40587756436528744, status=strong)
  - `517088` Being the Ricardos (relevance=0.3997599213641534, status=strong)
  - `523607` Maestro (relevance=0.38175972497185784, status=strong)
  - `523172` Late Night (relevance=0.36546201596064487, status=strong)

## Qualification
- Strong: `4`
- Partial: `0`
- Unsupported: `16`

## Final response
```json
{
  "intent": {
    "intent_type": "GENERAL_DISCOVERY",
    "requested_taste_dimensions": [
      "performers",
      "fame",
      "show_business",
      "period_setting"
    ],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "Looking for films about performers, fame, and show business, preferably set in a historical context."
  },
  "recommendations": [
    {
      "candidate_id": "466272",
      "title": "Once Upon a Time... in Hollywood",
      "year": 2019,
      "why_it_may_fit": "Supported request aspects: performers, fame, show_business Taste evidence: historical, moderate, atmospheric, melancholic, visual.",
      "taste_signals": [
        "historical",
        "moderate",
        "atmospheric",
        "melancholic",
        "visual"
      ],
      "request_match": "Supported request aspects: performers, fame, show_business",
      "caveat": null
    },
    {
      "candidate_id": "517088",
      "title": "Being the Ricardos",
      "year": 2021,
      "why_it_may_fit": "Supported request aspects: performers, fame, show_business Taste evidence: moderate, atmospheric, comforting, visual, morality.",
      "taste_signals": [
        "moderate",
        "atmospheric",
        "comforting",
        "visual",
        "morality"
      ],
      "request_match": "Supported request aspects: performers, fame, show_business",
      "caveat": null
    },
    {
      "candidate_id": "523607",
      "title": "Maestro",
      "year": 2023,
      "why_it_may_fit": "Supported request aspects: performers, fame, show_business Taste evidence: historical, moderate, atmospheric, comforting, melancholic.",
      "taste_signals": [
        "historical",
        "moderate",
        "atmospheric",
        "comforting",
        "melancholic"
      ],
      "request_match": "Supported request aspects: performers, fame, show_business",
      "caveat": null
    },
    {
      "candidate_id": "523172",
      "title": "Late Night",
      "year": 2019,
      "why_it_may_fit": "Supported request aspects: performers, fame, show_business Taste evidence: moderate, atmospheric, comforting, visual, morality.",
      "taste_signals": [
        "moderate",
        "atmospheric",
        "comforting",
        "visual",
        "morality"
      ],
      "request_match": "Supported request aspects: performers, fame, show_business",
      "caveat": null
    }
  ],
  "response_summary": "Returned 4 grounded recommendations from the qualified catalog.",
  "methodology_note": "Request relevance was checked first, then frozen B3 compatibility was used to order only the qualified candidates, and the explanation only verbalizes grounded evidence already attached to the selected films."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`5.0` comment=`None`
- `response_clarity` score=`5.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T165800Z`
- Trace ID: `01a102b1-e686-74b3-93ed-7782ce4cb542`

# CASE V09

## User request
Show me something different from what I normally watch.

## Parsed request
- Intent: `NOVELTY`
- Generic/contextual: `novelty`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `None`
- After hard constraints: `None`
- Outside old 259: `None`
- Dynamic retrieval used: `None`
  - `None` As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (relevance=0.33257092221848616, status=None)
  - `None` Ocean with David Attenborough (relevance=0.3064326427834084, status=None)
  - `None` Regular Show: The Movie (relevance=0.30643134557626306, status=None)
  - `None` Southbound (relevance=0.2976731558945618, status=None)
  - `None` Remarkably Bright Creatures (relevance=0.2953396057380255, status=None)

## Qualification
- Strong: `0`
- Partial: `0`
- Unsupported: `0`

## Final response
```json
{
  "intent": {
    "intent_type": "NOVELTY",
    "requested_taste_dimensions": [],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "Request for films that are different from the user's usual preferences."
  },
  "recommendations": [],
  "response_summary": "No grounded recommendation survived request qualification, so the system is abstaining.",
  "methodology_note": "Request relevance must be grounded before selection. Abstention is valid when no candidate survives qualification."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`3.0` comment=`None`
- `explanation_usefulness` score=`2.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`1.0` comment=`None`
- `response_clarity` score=`4.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T165906Z`
- Trace ID: `01a102b3-8512-7ce0-ba3b-e90855640f79`

# CASE V10

## User request
Recommend me something to watch.

## Parsed request
- Intent: `GENERAL_DISCOVERY`
- Generic/contextual: `generic`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{"intent_type": "GENERAL_DISCOVERY", "mode": "generic", "structured_constraints": {"required_languages": [], "excluded_languages": [], "required_countries": [], "excluded_countries": [], "required_genres": [], "excluded_genres": [], "required_decades": [], "excluded_decades": [], "min_year": null, "max_year": null}, "semantic_requirements": [], "semantic_exclusions": [], "reference": {"title": null, "relation": null, "resolved_tmdb_id": null, "resolved_tmdb_title": null, "resolution_status": "absent"}, "novelty_goal": {"enabled": false, "priority": null}, "personalization_instruction": {"preference": null, "priority": null}, "request_relevance_mode": "broad", "semantic_query_text": "recommend something to watch", "request_summary": "User is looking for a film recommendation.", "llm_used": false, "llm_call_count": 0, "runtime_input_tokens": null, "runtime_output_tokens": null, "runtime_total_tokens": null}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `259`
- After hard constraints: `259`
- Outside old 259: `None`
- Dynamic retrieval used: `False`
  - `1003596` Avengers: Doomsday (relevance=0.0, status=strong)
  - `1007757` Swapped (relevance=0.0, status=strong)
  - `101669` Mother's Day (relevance=0.0, status=strong)
  - `1022256` Selena Gomez: My Mind & Me (relevance=0.0, status=strong)
  - `1032863` The Love Hypothesis (relevance=0.0, status=strong)

## Qualification
- Strong: `15`
- Partial: `0`
- Unsupported: `5`

## Final response
```json
{
  "intent": {
    "intent_type": "GENERAL_DISCOVERY",
    "requested_taste_dimensions": [],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "User is looking for a film recommendation."
  },
  "recommendations": [
    {
      "candidate_id": "101669",
      "title": "Mother's Day",
      "year": 2010,
      "why_it_may_fit": "Generic discovery match. Taste evidence: atmospheric, visual, plot_driven, emotional_intensity, alienation.",
      "taste_signals": [
        "atmospheric",
        "visual",
        "plot_driven",
        "emotional_intensity",
        "alienation"
      ],
      "request_match": "Generic discovery match.",
      "caveat": null
    },
    {
      "candidate_id": "1084244",
      "title": "Toy Story 5",
      "year": 2026,
      "why_it_may_fit": "Generic discovery match. Taste evidence: moderate, comforting, visual, plot_driven, ensemble.",
      "taste_signals": [
        "moderate",
        "comforting",
        "visual",
        "plot_driven",
        "ensemble"
      ],
      "request_match": "Generic discovery match.",
      "caveat": null
    },
    {
      "candidate_id": "1032863",
      "title": "The Love Hypothesis",
      "year": 2026,
      "why_it_may_fit": "Generic discovery match. Taste evidence: moderate, comforting, emotional_intensity, joyful, conventional.",
      "taste_signals": [
        "moderate",
        "comforting",
        "emotional_intensity",
        "joyful",
        "conventional"
      ],
      "request_match": "Generic discovery match.",
      "caveat": null
    },
    {
      "candidate_id": "1127384",
      "title": "Deep Water",
      "year": 2026,
      "why_it_may_fit": "Generic discovery match. Taste evidence: atmospheric, visual, plot_driven, emotional_intensity, alienation.",
      "taste_signals": [
        "atmospheric",
        "visual",
        "plot_driven",
        "emotional_intensity",
        "alienation"
      ],
      "request_match": "Generic discovery match.",
      "caveat": null
    },
    {
      "candidate_id": "1101412",
      "title": "Fall 2: Deadpoint",
      "year": 2026,
      "why_it_may_fit": "Generic discovery match. Taste evidence: moderate, visual, plot_driven, emotional_intensity, psychological_intensity.",
      "taste_signals": [
        "moderate",
        "visual",
        "plot_driven",
        "emotional_intensity",
        "psychological_intensity"
      ],
      "request_match": "Generic discovery match.",
      "caveat": null
    }
  ],
  "response_summary": "Returned 5 grounded recommendations from the qualified catalog.",
  "methodology_note": "Request relevance was checked first, then frozen B3 compatibility was used to order only the qualified candidates, and the explanation only verbalizes grounded evidence already attached to the selected films."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`5.0` comment=`None`
- `response_clarity` score=`5.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T170106Z`
- Trace ID: `01a102b4-9820-7142-9111-e1d1ec7519ed`

# CASE V04

## User request
I want something melancholic but still a little intimate.

## Parsed request
- Intent: `MOOD_THEME`
- Generic/contextual: `contextual`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{"intent_type": "MOOD_THEME", "mode": "contextual", "structured_constraints": {"required_languages": [], "excluded_languages": [], "required_countries": [], "excluded_countries": [], "required_genres": [], "excluded_genres": [], "required_decades": [], "excluded_decades": [], "min_year": null, "max_year": null}, "semantic_requirements": [{"aspect_id": "melancholic", "concept": "melancholic", "importance": "required", "source_span": "something melancholic"}, {"aspect_id": "intimate", "concept": "intimate", "importance": "preferred", "source_span": "still a little intimate"}], "semantic_exclusions": [], "reference": {"title": null, "relation": null, "resolved_tmdb_id": null, "resolved_tmdb_title": null, "resolution_status": "absent"}, "novelty_goal": {"enabled": false, "priority": null}, "personalization_instruction": {"preference": null, "priority": null}, "request_relevance_mode": "query_aware", "semantic_query_text": "I want something melancholic but still a little intimate.", "request_summary": "Looking for films that are melancholic and have an intimate feel.", "llm_used": false, "llm_call_count": 0, "runtime_input_tokens": null, "runtime_output_tokens": null, "runtime_total_tokens": null}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `279`
- After hard constraints: `279`
- Outside old 259: `None`
- Dynamic retrieval used: `True`
  - `152601` Her (relevance=0.2928642272801919, status=partial)

## Qualification
- Strong: `0`
- Partial: `1`
- Unsupported: `19`

## Final response
```json
{
  "intent": {
    "intent_type": "MOOD_THEME",
    "requested_taste_dimensions": [
      "melancholic",
      "intimate"
    ],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "Looking for films that are melancholic and have an intimate feel."
  },
  "recommendations": [
    {
      "candidate_id": "152601",
      "title": "Her",
      "year": 2013,
      "why_it_may_fit": "Supported request aspects: intimate Taste evidence: moderate, atmospheric, niche, surreal, ethereal. Caveat: Unsupported aspects preserved as caveats: melancholic.",
      "taste_signals": [
        "moderate",
        "atmospheric",
        "niche",
        "surreal",
        "ethereal"
      ],
      "request_match": "Supported request aspects: intimate",
      "caveat": "Unsupported aspects preserved as caveats: melancholic."
    }
  ],
  "response_summary": "Returned 1 grounded recommendation from the qualified catalog.",
  "methodology_note": "Request relevance was checked first, then frozen B3 compatibility was used to order only the qualified candidates, and the explanation only verbalizes grounded evidence already attached to the selected films."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`3.0` comment=`None`
- `response_clarity` score=`4.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T170221Z`
- Trace ID: `01a102b6-5e74-72c1-b4b6-0e3d6f3fb0fe`

# CASE V08

## User request
Recommend something like an imaginary film title that TMDB cannot resolve.

## Parsed request
- Intent: `GENERAL_DISCOVERY`
- Generic/contextual: `reference`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `None`
- After hard constraints: `None`
- Outside old 259: `None`
- Dynamic retrieval used: `None`
  - `None` Impossible Things (relevance=0.46900672578935454, status=None)
  - `None` The Transformers: The Movie (relevance=0.45636259627639497, status=None)
  - `None` Dead End (relevance=0.45115723444146083, status=None)
  - `None` Crash (relevance=0.45094830493400995, status=None)
  - `None` The Mitchells vs. the Machines (relevance=0.45093409726760936, status=None)

## Qualification
- Strong: `0`
- Partial: `0`
- Unsupported: `0`

## Final response
```json
{
  "intent": {
    "intent_type": "GENERAL_DISCOVERY",
    "requested_taste_dimensions": [],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "Request for a film recommendation similar to an imaginary title that TMDB cannot resolve."
  },
  "recommendations": [],
  "response_summary": "No grounded recommendation survived request qualification, so the system is abstaining.",
  "methodology_note": "Request relevance must be grounded before selection. Abstention is valid when no candidate survives qualification."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`5.0` comment=`None`
- `response_clarity` score=`5.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T170332Z`
- Trace ID: `01a102b7-809c-7c60-b6a3-44a10c93987a`

# CASE V01

## User request
Recommend something that is very high-B3 compatible, but it must not actually be about performers, fame, or show business.

## Parsed request
- Intent: `GENERAL_DISCOVERY`
- Generic/contextual: `contextual`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `None`
- After hard constraints: `None`
- Outside old 259: `None`
- Dynamic retrieval used: `None`
  - `None` Chris Brown: Welcome to My Life (relevance=0.33017277832414005, status=None)
  - `None` Beastie Boys Story (relevance=0.3279442593849743, status=None)
  - `None` Best of the Best (relevance=0.3239861340794363, status=None)
  - `None` Billie Eilish - Hit Me Hard and Soft: The Tour (Live in 3D) (relevance=0.3191522084525843, status=None)
  - `None` Strung (relevance=0.31567787192544305, status=None)

## Qualification
- Strong: `0`
- Partial: `0`
- Unsupported: `0`

## Final response
```json
{
  "intent": {
    "intent_type": "GENERAL_DISCOVERY",
    "requested_taste_dimensions": [
      "high_b3_compatibility"
    ],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [
      "performers"
    ],
    "free_text_context": "Request for films that are high-B3 compatible but exclude themes of performers, fame, or show business."
  },
  "recommendations": [],
  "response_summary": "No grounded recommendation survived request qualification, so the system is abstaining.",
  "methodology_note": "Request relevance must be grounded before selection. Abstention is valid when no candidate survives qualification."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`0.0` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`5.0` comment=`None`
- `response_clarity` score=`5.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T170443Z`
- Trace ID: `01a102b8-b0b0-7a01-9a02-925cbf76ad01`

# CASE V02

## User request
I want a strong request match even if it is only modestly compatible with my historical taste.

## Parsed request
- Intent: `MOOD_THEME`
- Generic/contextual: `novelty`
- Reference film: `None`
- Novelty request: `True`
- Hard constraints: `{"intent_type": "MOOD_THEME", "mode": "contextual", "structured_constraints": {"required_languages": [], "excluded_languages": [], "required_countries": [], "excluded_countries": [], "required_genres": [], "excluded_genres": [], "required_decades": [], "excluded_decades": [], "min_year": null, "max_year": null}, "semantic_requirements": [], "semantic_exclusions": [], "reference": {"title": null, "relation": null, "resolved_tmdb_id": null, "resolved_tmdb_title": null, "resolution_status": "absent"}, "novelty_goal": {"enabled": true, "priority": "modest"}, "personalization_instruction": {"preference": null, "priority": "historical"}, "request_relevance_mode": "query_aware", "semantic_query_text": "I want a strong request match even if it is only modestly compatible with my historical taste.", "request_summary": "Request for films that match even modestly with historical taste.", "llm_used": false, "llm_call_count": 0, "runtime_input_tokens": null, "runtime_output_tokens": null, "runtime_total_tokens": null}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `259`
- After hard constraints: `259`
- Outside old 259: `None`
- Dynamic retrieval used: `False`
  - `505707` Waiting for the Barbarians (relevance=0.2734214983425046, status=strong)
  - `197` Braveheart (relevance=0.2374532413033701, status=strong)
  - `848439` Firebrand (relevance=0.2359473683032405, status=strong)
  - `216015` Fifty Shades of Grey (relevance=0.23427397286398188, status=strong)
  - `326382` Zama (relevance=0.2282532132782182, status=strong)

## Qualification
- Strong: `15`
- Partial: `0`
- Unsupported: `5`

## Final response
```json
{
  "intent": {
    "intent_type": "MOOD_THEME",
    "requested_taste_dimensions": [],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "Request for films that match even modestly with historical taste."
  },
  "recommendations": [
    {
      "candidate_id": "505707",
      "title": "Waiting for the Barbarians",
      "year": 2020,
      "why_it_may_fit": "Taste evidence: historical, atmospheric, melancholic, visual, morality.",
      "taste_signals": [
        "historical",
        "atmospheric",
        "melancholic",
        "visual",
        "morality"
      ],
      "request_match": null,
      "caveat": null
    },
    {
      "candidate_id": "197",
      "title": "Braveheart",
      "year": 1995,
      "why_it_may_fit": "Taste evidence: historical, moderate, atmospheric, visual, morality.",
      "taste_signals": [
        "historical",
        "moderate",
        "atmospheric",
        "visual",
        "morality"
      ],
      "request_match": null,
      "caveat": null
    },
    {
      "candidate_id": "848439",
      "title": "Firebrand",
      "year": 2024,
      "why_it_may_fit": "Taste evidence: historical, moderate, atmospheric, visual, morality.",
      "taste_signals": [
        "historical",
        "moderate",
        "atmospheric",
        "visual",
        "morality"
      ],
      "request_match": null,
      "caveat": null
    },
    {
      "candidate_id": "216015",
      "title": "Fifty Shades of Grey",
      "year": 2015,
      "why_it_may_fit": "Taste evidence: historical, moderate, atmospheric, visual, morality.",
      "taste_signals": [
        "historical",
        "moderate",
        "atmospheric",
        "visual",
        "morality"
      ],
      "request_match": null,
      "caveat": null
    },
    {
      "candidate_id": "326382",
      "title": "Zama",
      "year": 2017,
      "why_it_may_fit": "Taste evidence: historical, atmospheric, melancholic, visual, morality.",
      "taste_signals": [
        "historical",
        "atmospheric",
        "melancholic",
        "visual",
        "morality"
      ],
      "request_match": null,
      "caveat": null
    }
  ],
  "response_summary": "Returned 5 grounded recommendations from the qualified catalog.",
  "methodology_note": "Request relevance was checked first, then frozen B3 compatibility was used to order only the qualified candidates, and the explanation only verbalizes grounded evidence already attached to the selected films."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`5.0` comment=`None`
- `explanation_usefulness` score=`4.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`3.0` comment=`None`
- `response_clarity` score=`4.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T170645Z`
- Trace ID: `01a102b9-bc26-7fc1-a8d9-e1d919f66059`

# CASE V07

## User request
Recommend something like Jeanne Dielman.

## Parsed request
- Intent: `GENERAL_DISCOVERY`
- Generic/contextual: `reference`
- Reference film: `None`
- Novelty request: `False`
- Hard constraints: `{}`

## Retrieval
- Candidate universe after dedup/watch exclusion: `None`
- After hard constraints: `None`
- Outside old 259: `None`
- Dynamic retrieval used: `None`
  - `None` Her Way (relevance=0.430380338786096, status=None)
  - `None` Vivre Sa Vie (relevance=0.425588966627881, status=None)
  - `None` A Place for Her (relevance=0.4230684596399995, status=None)
  - `None` The Last Woman (relevance=0.4105767817030246, status=None)
  - `None` Another Day (relevance=0.3973460485543107, status=None)

## Qualification
- Strong: `0`
- Partial: `0`
- Unsupported: `0`

## Final response
```json
{
  "intent": {
    "intent_type": "GENERAL_DISCOVERY",
    "requested_taste_dimensions": [
      "jeanne_dielman"
    ],
    "requested_genres": [],
    "requested_languages": [],
    "requested_countries": [],
    "requested_decades": [],
    "exclusions": [],
    "free_text_context": "Request for film recommendations similar to Jeanne Dielman."
  },
  "recommendations": [],
  "response_summary": "No grounded recommendation survived request qualification, so the system is abstaining.",
  "methodology_note": "Request relevance must be grounded before selection. Abstention is valid when no candidate survives qualification."
}
```

## Validation
- Selection validation: `True`
- Repair: `False`
- Fallback: `False`
- Abstention: `False`

## LangSmith evaluator results
- `candidate_compliance` score=`1.0` comment=`None`
- `candidate_identity_integrity` score=`1.0` comment=`None`
- `duplicate_candidates` score=`1.0` comment=`None`
- `exclusion_satisfaction` score=`None` comment=`None`
- `explanation_groundedness` score=`3.0` comment=`None`
- `explanation_usefulness` score=`2.0` comment=`None`
- `goodreads_ranking_guardrail` score=`1.0` comment=`None`
- `hard_constraint_satisfaction` score=`None` comment=`None`
- `intent_match` score=`1.0` comment=`None`
- `request_relevance` score=`1.0` comment=`None`
- `response_clarity` score=`4.0` comment=`None`
- `sensitive_inference_guardrail` score=`1.0` comment=`None`
- `taste_signal_grounding` score=`1.0` comment=`None`
- `unsupported_taste_dimension` score=`1.0` comment=`None`

## Trace IDs
- Run ID: `20261003T170803Z`
- Trace ID: `01a102bb-8759-7463-8136-c5f056c99355`
