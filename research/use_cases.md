# AI Use Case Proposals — Letterboxd

**Company:** Letterboxd  
**Sector:** Social film discovery / entertainment technology  
**Project:** Taste Agent for Letterboxd  
**Stage:** Capstone Round 1

---

## 1. Purpose

This document evaluates three potential AI use cases for Letterboxd and explains the selection of **Semantic Taste Intelligence / Taste Agent** for the Round 1 Proof of Concept.

The use cases were selected based on the sector research and opportunity analysis documented in:

- `sector_research.md`
- `opportunities_risks.md`

The comparison considers:

- strategic fit with Letterboxd
- relevance to company scale
- availability of suitable data
- suitability for AI
- potential user value
- potential business value
- implementation complexity
- ability to validate the concept through a limited Proof of Concept

The objective is not to identify every possible application of AI at Letterboxd.

It is to select a use case that addresses a real product opportunity, can be meaningfully tested within the capstone, and could justify further development if the initial hypothesis is supported.

---

## 2. Company Context

Letterboxd reported **30.7 million members in Q2 2026**, representing 43% year-over-year growth and 185% growth since Tiny's acquisition.

The platform also generates substantial behavioural data through:

- film ratings
- viewing history
- diary entries
- reviews
- lists
- likes
- watchlists
- social activity

Letterboxd's 2025 Year in Review reported approximately:

- 898.5 million films marked watched
- 672.5 million ratings
- 332.8 million diary entries
- 143.6 million reviews

This scale influences the type of AI use case that is appropriate.

For a platform with tens of millions of members, a useful AI capability should ideally:

1. build on existing platform data
2. support a core Letterboxd behaviour
3. be reusable across a large user base
4. avoid requiring expensive generative processing for every interaction
5. have a credible route to measurable product or commercial value

Three use cases were considered against these requirements.

---

# 3. Use Case 1 — Semantic Taste Intelligence & Personalized Discovery

## Problem

Letterboxd records substantial information about what users watch and how they rate it.

However, conventional film metadata primarily describes attributes such as:

- genre
- director
- cast
- year
- country
- runtime

These attributes do not necessarily capture the characteristics that explain why an individual responds positively to one film and negatively to another.

For example, two films categorized as dramas may differ significantly in:

- pacing
- emotional tone
- narrative structure
- stylistic approach
- thematic focus

This limits how precisely historical ratings can be translated into explainable preference patterns.

---

## Proposed AI Solution

Create a semantic layer that represents films through structured characteristics such as:

- contemplative
- character-driven
- melancholic
- atmospheric
- experimental
- intimate
- sentimental
- emotionally intense

These semantic representations would then be combined with user ratings to create a structured **Taste Model**.

A simplified architecture is:

    FILM HISTORY
         ↓
    FILM METADATA
         ↓
    AI SEMANTIC CLASSIFICATION
         ↓
    SEMANTIC FILM VECTORS
         +
    USER RATINGS
         ↓
    TASTE MODEL
         ↓
    PERSONALIZED DISCOVERY

The AI component describes the cultural work.

User preference is calculated separately using behavioural or explicit preference signals.

---

## Potential Product Applications

The same Taste Model could support:

- personalized film recommendations
- explainable recommendations
- a "My Taste" profile
- Taste Evolution
- semantic search
- conversational discovery through a Taste Agent
- advanced paid-member personalization

The chatbot is therefore not the core product.

The reusable semantic Taste Model is the core AI asset.

---

## Fit With Letterboxd's Scale

**Strong**

At Letterboxd's scale, semantic film classification could potentially be performed once and reused across many users.

Instead of repeatedly asking an LLM to analyze the same film:

    FILM
      ↓
    CLASSIFY ONCE
      ↓
    STORE SEMANTIC VECTOR
      ↓
    REUSE ACROSS USERS

User-specific preference calculations could then operate primarily on stored semantic vectors.

This makes the architecture more suitable for a platform with tens of millions of members than an approach requiring full LLM analysis for every recommendation request.

---

## Available Data

Potential inputs already available through normal Letterboxd activity include:

- ratings
- watched films
- diary history
- likes
- reviews
- watchlists

External film metadata can provide information required for semantic classification.

For the Round 1 POC:

- film metadata is enriched using TMDB
- explicit ratings provide the preference signal
- Google Books is used to test whether the semantic architecture can accept a second cultural medium

---

## Potential Business Value

The use case could potentially contribute to:

- increased discovery engagement
- greater value from existing user data
- Free → Pro conversion
- paid-member retention
- Pro → Patron upsell
- differentiation through advanced personalization

These remain commercial hypotheses rather than demonstrated outcomes.

---

## Key Validation Question

> Can heterogeneous cultural consumption data be transformed into a reliable structured semantic Taste Model?

If technically feasible, a later stage can test the more important product question:

> Does that Taste Model improve film recommendations?

---

# 4. Use Case 2 — AI-Powered Semantic Search & Film Discovery

## Problem

Film discovery often depends on structured filters such as:

- genre
- decade
- runtime
- rating
- country
- streaming availability

These filters are useful when users know the category they want.

However, cultural discovery is often expressed in less structured language.

A user might want:

> "A melancholic, visually atmospheric film that is emotionally intense but not too plot-driven."

These characteristics are difficult to express using conventional filters alone.

---

## Proposed AI Solution

Create a semantic search layer allowing users to discover films through natural-language descriptions.

For example:

    USER QUERY

    "Something intimate and melancholic,
    slow but not depressing"

              ↓

    NATURAL-LANGUAGE INTERPRETATION

              ↓

    SEMANTIC FILM RETRIEVAL

              ↓

    RANKED RESULTS

The system could translate the request into semantic dimensions and retrieve films with matching characteristics.

---

## Potential Product Applications

Possible interfaces include:

- natural-language film search
- advanced semantic filters
- "more like this, but..." discovery
- semantic similarity
- AI-assisted list creation

For example:

> "Films like *Portrait of a Lady on Fire*, but less romantic and more unsettling."

or:

> "Give me visually experimental films under two hours."

---

## Fit With Letterboxd's Scale

**Strong**

A semantic film index could be generated centrally and reused across the entire user base.

The architecture could therefore separate:

    OFFLINE / PERIODIC PROCESSING
    Semantic film classification

                ↓

    STORED SEMANTIC INDEX

                ↓

    USER QUERY
    Semantic retrieval

This avoids performing expensive full-content analysis for each user request.

---

## Available Data

This use case requires:

- film metadata
- semantic film representations
- a searchable film catalogue

It requires less personal data than the Taste Model use case because useful semantic search could operate without analysing an individual's complete viewing history.

---

## Potential Business Value

Potential value includes:

- improved discovery
- increased search engagement
- differentiation from conventional film databases
- increased watchlist activity
- possible premium discovery functionality

However, the feature would make less use of one of Letterboxd's strongest assets: its accumulated individual rating and viewing histories.

---

## Key Validation Question

> Does semantic search help users find relevant films that are difficult to discover through conventional metadata and filters?

---

# 5. Use Case 3 — AI Cultural Trend & Audience Intelligence

## Problem

Letterboxd generates large volumes of ratings, viewing activity, reviews and lists.

Traditional analytics can identify:

- popular films
- highly rated films
- viewing volume
- engagement changes

However, these metrics do not necessarily reveal the semantic characteristics underlying changes in cultural interest.

For example, conventional analytics might identify which films are becoming popular without identifying whether users are increasingly engaging with works that are:

- nostalgic
- experimental
- melancholic
- maximalist
- intimate
- unsettling

---

## Proposed AI Solution

Aggregate semantic characteristics across platform activity to identify broader cultural patterns.

A simplified architecture could be:

    PLATFORM ACTIVITY
           +
    SEMANTIC FILM VECTORS
           ↓
    AGGREGATED ANALYSIS
           ↓
    CULTURAL TREND INTELLIGENCE

Potential outputs could include:

- rising semantic characteristics
- differences between highly watched and highly rated content
- changing preference clusters
- thematic trends
- stylistic trends
- differences between periods or aggregated audiences

---

## Potential Product Applications

This intelligence could support internal teams working on:

- editorial strategy
- product research
- discovery
- content curation
- cultural trend reporting

For example, Letterboxd could investigate whether increased engagement with a group of films reflects a broader rise in interest in a particular style or emotional tone.

---

## Fit With Letterboxd's Scale

**Strong for internal analytics**

This use case becomes more valuable as the volume of platform activity increases.

With tens of millions of members and hundreds of millions of annual film interactions, aggregate semantic analysis could identify patterns that would be difficult to detect manually.

The analysis could also reuse semantic film vectors generated for other use cases.

---

## Available Data

Potential inputs include aggregated:

- ratings
- viewing activity
- diary activity
- lists
- reviews
- semantic film vectors

The system would focus on aggregate patterns rather than individual recommendation.

---

## Potential Business Value

Potential value is primarily indirect.

It could improve:

- editorial decision-making
- product research
- understanding of platform trends
- discovery strategy
- cultural reporting

However, the connection to direct user value or subscription revenue is less immediate than for personalized discovery.

---

## Key Validation Question

> Can semantic aggregation reveal useful cultural patterns that conventional engagement metrics do not?

---

# 6. Use Case Comparison

The three use cases were compared against the needs of Letterboxd and the constraints of the capstone.

| Criterion | Semantic Taste Intelligence | Semantic Search | Cultural Trend Intelligence |
|---|---|---|---|
| Core discovery alignment | High | High | Medium |
| Uses existing user preference data | High | Low | High, aggregated |
| Direct user value | High | High | Low–Medium |
| Potential subscription value | High | Medium | Low |
| AI suitability | High | High | Medium–High |
| Reusable semantic infrastructure | High | High | High |
| POC feasibility | High | High | Medium |
| Explainability potential | High | High | Medium |
| Personalization potential | High | Low | Low |
| Cross-media extension potential | High | Medium | Medium |
| Direct commercial hypothesis | High | Medium | Low |

---

# 7. Selected Use Case

## Semantic Taste Intelligence / Taste Agent

The selected use case for the capstone is:

> **Transform Letterboxd's existing behavioral data into a structured semantic Taste Model that can support more personalized and explainable film discovery.**

This use case was selected because it combines:

- strong alignment with Letterboxd's discovery function
- direct use of existing ratings and viewing data
- a clear role for AI
- a reusable technical asset
- potential user-facing value
- potential subscription value
- measurable technical feasibility
- a path toward future recommendation evaluation

It also provides a foundation for elements of the other two use cases.

The semantic film representations required for the Taste Model could later support:

- semantic search
- similarity retrieval
- aggregate cultural analysis

The selected use case therefore creates infrastructure that could potentially support several downstream applications.

---

# 8. Why Not Build Only a Chatbot?

An alternative interpretation of Taste Agent would be to build a conversational film recommendation chatbot.

This was not selected as the core use case.

A generic chatbot could accept a prompt such as:

> "Recommend me a sad film."

and ask an LLM to generate suggestions.

However, this would make limited use of Letterboxd's existing product and data advantages.

The stronger architecture is:

    LETTERBOXD HISTORY
            ↓
    SEMANTIC TASTE MODEL
            ↓
    RECOMMENDATION SYSTEM
            ↓
    MULTIPLE INTERFACES
            │
            ├── Taste Agent
            ├── My Taste
            ├── Taste Evolution
            └── Personalized Discovery

Under this model, conversation is one interface to a broader intelligence layer.

The value comes from understanding the user's Letterboxd history, not simply from providing access to a general-purpose LLM.

---

# 9. Why Include Cross-Media Data?

The core use case can operate using Letterboxd data alone.

Cross-media data is therefore treated as an **extension hypothesis**, not a requirement.

The hypothesis is:

> Preferences expressed through books or music may provide additional evidence that improves understanding of film taste.

For example, characteristics such as:

- melancholic
- atmospheric
- experimental
- nostalgic
- intimate

may appear across multiple cultural media.

The Round 1 POC includes Goodreads data to test whether film and book records can enter the same semantic modelling architecture.

This demonstrates technical compatibility.

It does **not** establish that book data improves film recommendations.

That distinction is important.

Cross-media value must be tested independently.

---

# 10. Round 1 POC Scope

The Round 1 POC therefore focuses on the foundational capability required by the selected use case.

It tests whether the following pipeline is technically feasible:

    CONSUMPTION DATA
            ↓
    MEDIA ROUTING
            ↓
    METADATA ENRICHMENT
            ↓
    METADATA QUALITY GATE
            ↓
    LLM SEMANTIC CLASSIFICATION
            ↓
    STRUCTURED SEMANTIC VECTOR
            ↓
    PREFERENCE WEIGHTING
            ↓
    TASTE MODEL

The POC processes:

- films
- books

through a common 62-dimensional semantic model.

Preference weighting is calculated separately from semantic classification.

The LLM receives information about the cultural work rather than the user's rating, reducing target leakage between classification and preference analysis.

---

# 11. What the Round 1 POC Tests

The POC is intended to provide evidence for:

### Metadata Enrichment

Can film and book records be enriched with sufficient external metadata?

### Safe Matching

Can uncertain or insufficient metadata be rejected rather than automatically classified?

### Structured AI Output

Can an LLM consistently classify works into a predefined semantic schema?

### Preference Separation

Can semantic characteristics be generated independently from the user's rating?

### Taste Aggregation

Can semantic vectors and preference weights be combined into interpretable preference associations?

### Cross-Media Technical Compatibility

Can more than one cultural medium enter the same semantic modelling architecture?

These questions are appropriate for a feasibility-stage POC.

---

# 12. What the Round 1 POC Does Not Prove

The POC does not establish that:

- Taste Agent recommendations outperform existing Letterboxd discovery
- cross-media data improves film recommendations
- users would pay for Taste Agent
- the system is ready for production deployment
- the current 62-dimensional taxonomy is the final production taxonomy
- the architecture is optimized for Letterboxd's full production scale

These are later validation questions.

The distinction can be summarized as:

    ROUND 1

    Can we build a structured semantic representation
    of cultural preference?

                         ↓

                    FEASIBILITY

    ------------------------------------------------

    NEXT STAGE

    Does that representation create better
    recommendations and measurable user value?

                         ↓

                       VALUE

---

# 13. Initial Success Criteria

The selected use case should progress beyond the POC only if the foundational semantic pipeline demonstrates acceptable reliability.

Round 1 therefore evaluates areas such as:

- correct metadata matching
- safe rejection of unreliable metadata
- structured schema compliance
- semantic plausibility
- separation between semantic classification and preference signals
- safe handling of user-related inference

The detailed Round 1 evaluation methodology is documented separately in:

`evaluation/eval_plan.md`

Recommendation quality is intentionally reserved for the next stage because the Round 1 POC does not yet implement a complete recommendation engine.

---

# 14. Future Development Direction

If Round 1 supports the technical feasibility of Semantic Taste Intelligence, the next stage should move from **representation** to **recommendation**.

The key research question becomes:

> **Does a semantic Taste Model produce more relevant and explainable film recommendations than a simpler Letterboxd-only baseline?**

A further experiment can then investigate:

> **Does adding independently observed book and music preference improve film recommendations beyond the Letterboxd-only semantic Taste Model?**

This creates a clear sequence:

    SEMANTIC REPRESENTATION
             ↓
    FILM RECOMMENDATION
             ↓
    RECOMMENDATION EVALUATION
             ↓
    CROSS-MEDIA COMPARISON
             ↓
    PRODUCT VALIDATION

The project therefore avoids assuming that additional AI complexity automatically creates additional value.

---

# 15. Decision

The selected Round 1 use case is:

## **Semantic Taste Intelligence / Taste Agent**

The core proposition is:

> **Turn Letterboxd's existing behavioral data into a monetizable semantic taste-intelligence layer.**

The user-facing concept is:

> **Letterboxd knows your taste in film. Taste Agent asks whether understanding the characteristics behind that taste — and eventually broader cultural taste — can make Letterboxd better at discovering your next film.**

The Round 1 POC tests the technical foundation required to make that proposition possible.

The subsequent MVP should test whether the resulting intelligence actually improves film recommendations.

---

## Related Documentation

- `sector_research.md` — Sector, company and data context
- `opportunities_risks.md` — AI opportunity and risk landscape
- `../docs/01-use-case-business-case.md` — Detailed business case
- `../docs/02-poc-feasibility.md` — POC architecture and technical findings
- `../docs/03-roi-risk-assessment.md` — Cost, ROI and quantified risk analysis
- `../docs/04-compliance.md` — EU AI Act and GDPR assessment
- `../evaluation/eval_plan.md` — Round 1 POC evaluation methodology

---

## Sources

- Tiny Ltd. — *Tiny Announces Majority Acquisition of Letterboxd* (2023). Used for company scale, acquisition context and Letterboxd positioning.
- Tiny Ltd. — *Tiny Reports Q2 2026 Results* (2026). Used for current reported membership and growth.
- Letterboxd — *2025 Year in Review*. Used for platform activity and engagement volumes.
- Letterboxd — *Paid subscriptions* and *Upgrade to Letterboxd Pro*. Used for membership and product context.
- Letterboxd — *Frequently Asked Questions*. Used for business model context.
- TMDB — Film metadata source used by the Round 1 POC.
- Google Books — Book metadata source used by the Round 1 POC.
