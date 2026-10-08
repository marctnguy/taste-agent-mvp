# Taste Agent — No-Code / Low-Code POC Documentation

**Stage:** Round 1 POC retained as Round 2 evidence  
**Implementation:** n8n + JavaScript + TMDB + Google Books + OpenAI structured output

---

## 1. POC Question

The POC tested one foundational question:

> **Can fragmented cultural-consumption data be transformed into a structured, explainable semantic Taste Model?**

It was intentionally narrower than the final product. It did not attempt to validate production-scale recommendation quality, commercial impact or a conversational user experience.

---

## 2. Workflow

The n8n workflow processes film and book consumption records through a shared semantic pipeline:

```text
Consumption Records
      ↓
Media Routing
      ↓
TMDB / Google Books Enrichment
      ↓
Metadata Matching
      ↓
Quality Gate
      ↓
Structured LLM Classification
      ↓
62-Dimensional Semantic Representation
      ↓
Preference Weighting
      ↓
Taste Model
```

The exported workflow is included in:

`poc/taste-agent-poc-v03.json`

---

## 3. AI Capability Demonstrated

The POC demonstrates that an LLM can act as a constrained semantic classifier when its output is controlled through a strict schema.

The final taxonomy contains **62 dimensions across nine groups**:

- themes
- character
- narrative
- pacing and energy
- tone
- style
- accessibility
- temporal / cultural characteristics
- intensity

The classifier describes the cultural work rather than the user.

It does not receive the user's rating, like status, preference class or preference weight when assigning semantic scores.

---

## 4. Quality Controls

The POC introduced several controls that remained important in Round 2:

### Metadata matching

Film and book records are enriched before AI classification.

### Safe rejection

Records with unreliable matches or insufficient descriptions can fail the quality gate instead of generating confident AI output from weak inputs.

### Strict structured output

Prompt-only classification initially produced schema drift. The final POC uses JSON Schema-constrained output so downstream analysis receives the same semantic structure for every classified work.

### Content / preference separation

The LLM classifies the work independently of the user's explicit preference signal, reducing target leakage and unnecessary behavioural-data exposure.

---

## 5. POC Result

The final v0.3 POC processed:

| Metric | Result |
|---|---:|
| Successfully analyzed works | 107 |
| Films | 96 |
| Books | 11 |
| Semantic dimensions | 62 |

The POC demonstrated technical compatibility across film and literature, but the book sample was too small to establish that cross-media data improves film recommendations.

---

## 6. Limitations vs Production

The POC does **not** provide:

- production Letterboxd integration
- real-time multi-user processing
- final recommendation ranking
- production authentication
- payment integration
- scalable persistence
- commercial validation
- evidence that cross-media data improves recommendation quality

It also performs semantic enrichment inside the workflow. At production scale, reusable content representations should be generated once and persisted rather than repeatedly classified.

---

## 7. How to Reproduce

1. Import `poc/taste-agent-poc-v03.json` into n8n.
2. Configure TMDB, Google Books and OpenAI credentials.
3. Reconnect the n8n data-table node to the intended input table after import.
4. Provide records following the POC schema for film and/or book consumption.
5. Execute the workflow manually.
6. Inspect metadata-match status, `ready_for_ai`, structured semantic output and the final Taste Model.

The original Round 1 repository also contains the POC screenshots, sample data and example output used during development.

---

## 8. POC → MVP Evolution

Round 1 proved that the semantic representation could be constructed.

Round 2 then tested whether that representation should actually drive recommendation.

The answer was more nuanced:

- the handcrafted 62D profile remained useful for interpretation
- it did **not** outperform the metadata baseline as a preference predictor
- Goodreads did not demonstrate enough incremental predictive value to become a ranking dependency
- recommendation therefore evolved toward request-aware retrieval, watchlist intent, historical preference compatibility and grounded explanation

The POC remains the **no-code feasibility artifact**. The Python/Streamlit application is the working MVP.

---

## 9. Demo Recording

A **2–5 minute n8n walkthrough** should show:

1. input records
2. film/book routing
3. metadata enrichment
4. the quality gate
5. structured semantic classification
6. the final Taste Model output

The recording is intentionally focused on the POC workflow; the MVP is demonstrated separately in the final presentation.