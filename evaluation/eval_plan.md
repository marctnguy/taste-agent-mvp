# Taste Agent for Letterboxd — Round 1 Evaluation Plan

## 1. Evaluation Purpose

The purpose of the Round 1 evaluation is to determine whether the Taste Agent POC can reliably transform cultural consumption records into structured semantic data suitable for building a Taste Model.

Round 1 evaluates the **representation pipeline**, not recommendation quality.

The evaluation therefore focuses on four questions:

1. Can the pipeline correctly match known films and books to external metadata?
2. Can it safely reject records when reliable metadata cannot be found?
3. Can the LLM consistently return the required structured semantic representation?
4. Does the architecture keep semantic classification separate from user preference signals?

Recommendation relevance, cross-media recommendation lift, and conversational recommendation quality are deferred to Round 2.

---

## 2. System Under Evaluation

The evaluated POC follows this pipeline:

    Consumption record
          ↓
    Media-type routing
          ↓
    TMDB / Google Books enrichment
          ↓
    Metadata quality gate
          ↓
    LLM semantic classification
          ↓
    62-dimensional semantic representation
          ↓
    Preference weighting
          ↓
    Taste Model

Films are enriched through TMDB and books through Google Books.

Before semantic classification, the quality gate verifies that:

- a metadata match was found; and
- a usable description is available.

Records that fail this gate are not sent to the LLM.

The semantic classifier receives information about the cultural work but does not receive the user's rating, like status, preference class, or other preference signals when assigning semantic scores.

---

## 3. Evaluation Dataset

A small synthetic evaluation dataset was created specifically for Round 1.

The objective was not statistical benchmarking. Instead, the dataset was designed to exercise distinct expected behaviours and failure modes of the POC.

The evaluation used eight input rows representing seven evaluation tests.

| ID | Input | Test Purpose |
|---|---|---|
| E01 | *Parasite* (2019) | Clear film metadata match and semantic classification |
| E02 | *Crash* (1996) | Ambiguous film title resolved using year |
| E03 | Invented film | Safe rejection of nonexistent metadata |
| E04 | *The Bell Jar* — Sylvia Plath | Book metadata matching and semantic classification |
| E05 | Invented book | Safe rejection of nonexistent metadata |
| E06 | *Aftersun* (2022), rated 1★ | First input for preference-separation test |
| E07 | *Aftersun* (2022), rated 5★ | Second input for preference-separation test |
| E08 | *The Thing* (1982) | Ambiguous film title resolved using year |

E06 and E07 form a single paired evaluation test. They provide identical work metadata with substantially different preference signals.

All cases were processed in a single n8n evaluation run.

---

## 4. Evaluation Criteria

### Criterion 1 — Metadata Matching and Safe Rejection

**Question:** Does the pipeline correctly identify valid cultural works and reject deliberately invalid ones?

A case passes when:

- the intended film or book is correctly matched; or
- a deliberately invalid work is identified as unmatched and prevented from reaching semantic classification.

Relevant fields include:

- `metadata_match_found`
- `tmdb_match_found`
- `google_books_match_found`
- `ready_for_ai`

**Pass condition:** Correct match or correct safe rejection.

---

### Criterion 2 — Schema Compliance

**Question:** Does every classified work produce the required structured semantic representation?

The Taste Agent taxonomy contains 62 semantic dimensions across:

- themes
- character
- narrative
- pacing and energy
- tone
- style
- accessibility
- temporal/cultural characteristics
- intensity

Each dimension must contain a numeric score between `0.0` and `1.0`.

The POC uses strict structured output to enforce the taxonomy.

**Pass condition:** All required dimensions are returned with valid numeric values and no unexpected schema changes occur.

---

### Criterion 3 — Semantic Plausibility

**Question:** Are the generated semantic characteristics reasonably supported by the metadata supplied to the classifier?

This criterion is evaluated manually because individual semantic scores do not have an objective numerical ground truth.

The evaluation therefore checks for major contradictions or clearly unsupported classifications rather than requiring an exact expected score for each dimension.

For example, the evaluation does not assume that a particular work must receive exactly `0.75` for `melancholic`. Instead, it checks whether the overall semantic representation is reasonably defensible from the available description and genre information.

**Pass condition:** No major contradictory or clearly unsupported semantic classification is observed.

---

### Criterion 4 — Preference Separation and Safe Behaviour

**Question:** Does semantic classification remain independent from explicit user preference signals?

The semantic classifier is designed to classify the **work**, not the user.

It should not use:

- user rating
- like status
- preference class
- inferred personality
- identity
- demographics
- political beliefs
- mental-health characteristics

E06 and E07 test this architecture by submitting the same film (*Aftersun*, 2022) with different preference signals:

- E06: 1★
- E07: 5★

Both records resolve to the same TMDB work and receive the same source metadata before classification.

The test therefore examines whether preference information contaminates the semantic representation and also provides an initial observation of repeated-classification consistency.

**Pass condition:** No evidence indicates that rating or preference information was used as classifier input, and no prohibited user-level inference is generated.

Classification variance between repeated calls is documented separately as a consistency issue.

---

## 5. Scoring Method

Round 1 uses a simple manual pass/fail framework.

Each test is assigned one of three outcomes:

- **PASS** — expected behaviour observed.
- **PARTIAL** — core behaviour worked, but an issue requiring further investigation was observed.
- **FAIL** — expected behaviour was not achieved.

The evaluation intentionally does not convert these outcomes into a synthetic model-quality percentage because the test set is small and designed for functional validation rather than statistical benchmarking.

Detailed observed results are documented in `scored_cases.md`.

---

## 6. Round 1 Results Summary

The evaluation produced the following overall outcomes:

| Test | Result |
|---|---|
| E01 — Clear film matching | PASS |
| E02 — Ambiguous film/year matching | PASS |
| E03 — Invalid film rejection | PASS |
| E04 — Book matching | PASS |
| E05 — Invalid book rejection | PASS |
| E06/E07 — Preference separation and consistency | PARTIAL |
| E08 — Ambiguous film/year matching | PASS |

The metadata quality gate successfully rejected both deliberately fabricated works before semantic classification.

All valid works that reached the classifier produced structured semantic outputs using the required taxonomy.

The E06/E07 paired test produced an important additional finding.

Both *Aftersun* records resolved to the same TMDB work and used the same source description despite receiving substantially different user ratings. No evidence was observed that the rating itself entered the semantic classifier.

However, repeated classification of the identical work was not fully deterministic: **13 of the 62 semantic dimensions differed between the two model calls**, generally by 0.25.

The paired test is therefore marked **PARTIAL**.

This does not indicate observed preference leakage. Instead, it identifies **LLM classification consistency** as an area requiring further evaluation.

---

## 7. Key Evaluation Finding

Round 1 provides evidence that the POC can:

- route different media types;
- enrich films and books with external metadata;
- reject unreliable or nonexistent matches;
- generate schema-compliant semantic representations; and
- maintain architectural separation between semantic classification and explicit preference signals.

The principal issue identified by the evaluation is repeated-classification variance.

This reinforces the proposed production architecture in which a cultural work is classified once, its semantic vector is stored, and the stored representation is reused rather than repeatedly regenerated.

It also creates a specific evaluation target for Round 2.

---

## 8. What Round 1 Cannot Measure

The current POC builds a semantic Taste Model but does not yet implement or benchmark a complete recommendation engine.

Round 1 therefore cannot determine:

- whether semantic recommendations outperform conventional metadata-based recommendations;
- whether the Taste Model improves recommendation relevance;
- whether recommendations become more novel or diverse;
- whether Goodreads data improves film recommendations;
- whether Spotify data provides incremental recommendation value;
- whether recommendation explanations are consistently grounded;
- whether users perceive the recommendations as better;
- whether the feature improves paid conversion or retention.

These questions require a recommendation layer and a larger evaluation framework.

---

## 9. Round 2 Evaluation

Round 2 will move evaluation from **representation quality** to **recommendation quality**.

The intended comparison is:

    Baseline
    Conventional / metadata-based recommendation

    vs.

    Model A
    Letterboxd semantic Taste Model

    vs.

    Model B
    Letterboxd + Goodreads Taste Model

    vs.

    Model C
    Letterboxd + Goodreads + Spotify Taste Model

This will allow the project to test its central product hypothesis:

> Does broader semantic cultural understanding improve film recommendations beyond a Letterboxd-only model?

Round 2 evaluation should include:

- recommendation relevance;
- novelty;
- diversity;
- explanation grounding;
- hallucination / unsupported recommendation rate;
- semantic-classification consistency;
- cross-media incremental lift.

LangSmith can be introduced in Round 2 to trace recommendation runs, define reusable evaluation datasets, compare system variants, and monitor LLM behaviour systematically.

---

## 10. Evaluation Boundary

The conclusion supported by Round 1 is deliberately limited:

> **The POC demonstrates that a structured semantic Taste Model can be constructed from heterogeneous cultural-consumption data with working metadata validation, schema enforcement, and safe rejection behaviour. The evaluation also identifies LLM classification consistency as an area for improvement.**

Round 1 does **not** establish that the resulting Taste Model produces better film recommendations.

That question becomes the primary evaluation objective for Round 2.
