# Taste Agent — Use Case Definition

**Client Case:** Letterboxd  
**Industry:** Digital Media & Entertainment  
**Project:** AI-Powered Taste Intelligence for Film Discovery  
**Stage:** Working MVP → Proposed Pilot

---

## 1. Business Problem

Letterboxd already captures rich first-party signals about cultural preference through watched films, ratings, likes, diary activity, lists and watchlists.

The opportunity is not simply to add an AI chatbot. It is to use those signals more effectively for **personalized, contextual and explainable film discovery**.

The business problem is:

> **How can Letterboxd create additional user and commercial value from the preference data it already collects without reducing taste to genres, popularity, or opaque recommendations?**

This matters because discovery is core to the product and paid membership is strategically important to Letterboxd's business model.

---

## 2. Company Profile and Strategic Context

Letterboxd is a global social platform for film discovery, logging, rating, reviewing and discussion.

Its product already creates a strong preference-data asset:

- watched films
- ratings
- likes
- diary activity
- reviews
- lists
- watchlists
- social and discovery interactions

Taste Agent is designed as an extension of that existing product model rather than as a separate standalone service.

### Current scale and growth

Tiny reported **30.7 million Letterboxd members at the end of Q2 2026**, representing **43% year-over-year growth** and **185% growth since Tiny Fund I acquired its interest**.

This follows approximately 10 million members at the time of the 2023 investment, 26.1 million by Q4 2025 and 29 million by Q1 2026.

The strategic implication is important: Taste Agent is being proposed for a product that has already demonstrated very strong audience growth. The question is therefore not whether Letterboxd can attract a film audience, but how it can turn a growing first-party preference-data asset into stronger discovery, engagement and monetisation.

### Business model

Letterboxd states that **membership fees are its chief source of income**.

Its current public annual pricing is:

- **Pro — $19/year**
- **Patron — $49/year**

Letterboxd also states that it measures success through:

1. the size of its community
2. the level of activity in that community
3. the number of members who choose to support the service financially

This makes paid feature differentiation directly relevant to the company's own success model.

### Product direction

Letterboxd's published product direction includes improving its native mobile apps and continuing to add features for Pro and Patron members.

The platform has also expanded beyond pure social logging into a broader film-discovery and media role, including transactional film rentals and editorial/video activity.

Taste Agent fits this direction as a **native premium discovery capability**, not as a separate AI product.

Its strategic fit is threefold:

- **Paid differentiation:** create a stronger reason to upgrade without changing Letterboxd's core identity
- **Engagement and retention:** turn existing ratings, diary activity and watchlist behaviour into more useful recurring discovery
- **First-party advantage:** create additional value from data Letterboxd already owns rather than depending on a new social graph or external identity profile

### Public sources

- Tiny Q2 2026 results: https://investors.tiny.com/news/news-details/2026/Tiny-Reports-Q2-2026-Results/default.aspx
- Tiny 2025 Annual Shareholder Letter: https://s202.q4cdn.com/676416790/files/doc_financials/2025/sr/Tiny-Ltd-2025-Annual-Shareholder-Letter.pdf
- Letterboxd Purpose: https://letterboxd.com/purpose/
- Letterboxd paid subscriptions: https://letterboxd.com/about/pro/
- Letterboxd Pro / Patron pricing: https://letterboxd.com/pro/
- Tiny Q2 2025 update, including transactional-video plans: https://s202.q4cdn.com/676416790/files/doc_financials/2025/q2/Tiny-Ltd-Q2-2025-Earnings-Call-Presentation-vF-Read-Only.pdf

---

## 3. Proposed AI Solution

Taste Agent is a **contextual film-discovery system** built around four distinct signals:

```text
Current Request
      +
Watchlist / Expressed Intent
      +
Historical Preference Compatibility
      +
Interpretable Taste Evidence
      ↓
Contextual Film Discovery
```

The working MVP uses a request-aware recommendation pipeline:

```text
USER REQUEST
→ REQUEST UNDERSTANDING
→ QUERY-AWARE CATALOG RETRIEVAL
→ HARD CONSTRAINTS
→ SEMANTIC CANDIDATE QUALIFICATION
→ HISTORICAL PREFERENCE COMPATIBILITY
→ SELECTION
→ GROUNDED EXPLANATION
→ VALIDATION / ABSTENTION
```

The product also retains the 62-dimensional Taste Profile developed in Round 1 as an **interpretability and explanation layer**.

The semantic profile is not treated as a complete definition of the user and does not independently determine the candidate universe.

---

## 4. System Type

Taste Agent combines:

- deterministic preprocessing and ranking components
- metadata and semantic embeddings
- LLM-based request understanding and candidate qualification
- a frozen historical-preference compatibility model
- grounded natural-language explanations
- Streamlit as the MVP user interface
- LangSmith tracing and evaluation

The system is designed to recommend films, not to infer personality, identity or sensitive personal characteristics.

---

## 5. Key Stakeholders

| Stakeholder | Primary Interest |
|---|---|
| Letterboxd Product | User value, roadmap fit, adoption |
| Engineering / Data | Reliability, latency, cost, maintainability |
| Design / UX | Explainability, trust, user control |
| Growth / Commercial | Conversion, retention, tier differentiation |
| Legal / Privacy | GDPR, AI Act, profiling, vendor governance |
| Customer Support | Clear explanations and user complaint handling |
| Letterboxd Members | Better discovery, transparency, privacy and control |
| Tiny / Ownership | Sustainable growth and return on investment |

---

## 6. Success Criteria

Success is not defined as simply producing recommendations.

### Recommendation quality

A Pilot should show that users perceive recommendations as relevant, useful and sufficiently novel compared with the selected baseline.

Indicative measures:

- recommendation relevance
- intention to watch
- watchlist additions
- novelty / discovery value
- already-consumed recommendation rate
- explanation usefulness

### Product adoption

- activation rate
- repeat usage
- recommendation interactions
- conversations per active user
- feature retention

### Technical performance

- successful end-to-end recommendation completion
- acceptable latency
- acceptable cost per active user
- hard-constraint compliance
- grounded explanation rate
- abstention correctness

### Commercial value

- Free → paid conversion uplift
- paid renewal / retention uplift
- Pro → Patron upgrade uplift
- incremental revenue per exposed user
- contribution margin after operating costs

---

## 7. Out of Scope

The current MVP does not attempt to provide:

- production Letterboxd authentication or account integration
- production-scale multi-user infrastructure
- payment integration
- final subscription packaging
- automated retraining from live user feedback
- real-time ingestion from third-party cultural platforms
- validated cross-media recommendation uplift
- sensitive personal-attribute inference
- consequential decision-making outside entertainment discovery

Spotify and other external cultural sources remain future hypotheses rather than requirements for the current product.

---

## 8. Evolution from Round 1

Round 1 asked:

> **Can heterogeneous cultural-consumption data be transformed into a structured semantic Taste Model?**

The answer was **yes, technically**.

Peer feedback then challenged the project to prove that the additional modelling actually improved recommendations, to narrow the MVP to movies, to compare against a baseline, and to avoid over-personal profiling.

Round 2 responded directly.

### What was retained

- Letterboxd as the client
- film discovery as the product outcome
- semantic taste intelligence as an interpretability asset
- explainability and user control as product requirements
- optional cross-media enrichment as a longer-term hypothesis

### What changed

The 62-dimensional handcrafted semantic representation **did not outperform the metadata baseline as a preference predictor**.

The project therefore did not use the Taste Profile as the main ranking engine.

Instead, the MVP evolved toward a request-aware architecture in which:

- current intent defines relevance
- watchlist membership represents expressed future interest
- historical preference acts as compatibility evidence rather than a hard filter
- semantic information supports qualification and explanation
- unsupported candidates can be rejected
- abstention is valid

Goodreads data remains descriptive only in the production path because it did not demonstrate sufficient incremental predictive value to justify making cross-media data a dependency.

---

## 9. Current Recommendation

The project has progressed beyond POC feasibility and now has a functional MVP.

The next stage is therefore a **controlled Pilot**, not another architecture rewrite.

The Pilot should answer the remaining product question:

> **Does Taste Agent create measurably better and more useful discovery for real users than Letterboxd's current or selected baseline discovery experience?**

Only if that is validated should Letterboxd proceed to a commercial experiment and broader deployment.