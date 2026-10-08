# Taste Agent — GDPR Documentation

**Client Case:** Letterboxd  
**Project:** AI-Powered Taste Intelligence for Film Discovery  
**Scope:** Working MVP → Proposed Pilot  
**Jurisdiction:** European Union

---

## 1. GDPR Position

Taste Agent would process personal data relating to identifiable Letterboxd users, including behavioural history, ratings, watchlists, prompts, inferred taste signals and recommendation interactions.

The system should therefore be treated as **profiling for GDPR compliance purposes**.

The intended recommendation use case is unlikely to produce legal or similarly significant effects under Article 22 because it recommends entertainment content rather than making consequential decisions.

This does not remove the need for transparency, lawful basis, purpose limitation, minimization, data-subject rights and appropriate vendor governance.

---

## 2. Data Flow Map

```text
LETTERBOXD USER DATA
watched films / ratings / likes / watchlist
              │
              ▼
      Normalization / Matching
              │
              ▼
      Content Representations
              │
              +
      User Preference Signals
              │
              ▼
        Taste / History Evidence
              │
              +
         Current User Prompt
              │
              ▼
      Recommendation Runtime
              │
              ▼
      Ranked Film Suggestions
              │
              ▼
      Grounded Explanation
```

Optional future external cultural data should enter through a separate opt-in path and must not be required for the core Letterboxd experience.

---

## 3. Personal Data Inventory

| Data | Category | Source | Purpose |
|---|---|---|---|
| Watched films | Behavioural personal data | Letterboxd | Historical preference evidence |
| Ratings / likes | Preference data | Letterboxd | Preference modelling |
| Watchlist | Intent data | Letterboxd | Candidate and future-interest signal |
| User prompt | User-provided personal data | Taste Agent | Current recommendation intent |
| Taste Profile | Inferred personal data | Taste Agent | Interpretation / explanation |
| Historical compatibility score | Inferred personal data | Taste Agent | Personalization |
| Recommendation history | Interaction data | Taste Agent | Product function / evaluation |
| Optional book/music history | Behavioural / preference data | External user import | Future cross-media enrichment |

A production implementation should maintain a formal Record of Processing Activities where required.

---

## 4. Processing Activities Register

| Processing Activity | Purpose | Main Data | Lawful-Basis Candidate | Retention Approach | Recipients |
|---|---|---|---|---|---|
| Core Taste Agent personalization | Personalized film discovery | ratings, watched history, watchlist, inferred preference signals | Contract and/or legitimate interests subject to final assessment | While feature is active and necessary | Letterboxd + approved processors |
| Current-request processing | Respond to user intent | prompt, request context | Contract / requested service | Minimize; do not retain longer than necessary by default | AI / infrastructure processors |
| Recommendation evaluation | Quality and safety monitoring | prompts, outputs, evaluator results, operational metadata | Legitimate interests subject to balancing test | Defined evaluation window | Internal teams / evaluation processors |
| Optional cross-media enrichment | Improve personalization using external cultural history | imported book/music history | Explicit opt-in; consent may be appropriate depending on design | Delete raw import after normalization where possible | Approved processors |
| Product analytics | Adoption and feature improvement | interaction events | Legitimate interests / consent depending on implementation | Defined analytics retention period | Analytics processors |

The final lawful basis must be selected against the production design, not chosen solely for convenience.

---

## 5. Purpose Limitation

Taste Agent data should remain limited to cultural discovery and clearly compatible product purposes.

Acceptable intended flow:

```text
Film Activity
      ↓
Taste / Preference Evidence
      ↓
Film Discovery
```

The project explicitly rejects function creep such as using cultural histories to infer suitability for employment, insurance, credit, political targeting or other unrelated consequential decisions.

---

## 6. Data Minimization

Production design should:

- use only fields necessary for recommendation
- separate reusable film/content representations from user preference data
- avoid repeatedly sending full user histories to external AI providers
- avoid retaining raw external account exports where normalized data is sufficient
- limit logs containing prompts or behavioural histories
- keep optional external data separate from the core Letterboxd-only experience

---

## 7. Sensitive-Inference Risk

Cultural histories can sometimes reveal or suggest special-category data.

Taste Agent should therefore avoid attempting to infer:

- political opinions
- religion or philosophical beliefs
- health or mental-health status
- sexual orientation
- racial or ethnic origin
- personality diagnoses
- other sensitive personal attributes

The product should model **cultural characteristics**, not identity.

Controls include:

- restricted taxonomy
- sensitive-inference prohibitions
- grounded explanations
- evaluator checks
- review of generated outputs
- purpose limitation

---

## 8. Data-Subject Rights

The production design should support applicable rights including:

- right to information
- right of access
- right to rectification
- right to erasure
- right to restriction
- right to object where applicable
- right to portability where applicable

Derived Taste Model data should remain traceable enough to support access, correction and deletion workflows.

Useful product controls include:

```text
VIEW
What does Taste Agent think I respond to?

EXPLAIN
Why is this recommendation being shown?

CORRECT
This does not represent my taste.

RESET
Delete and rebuild my Taste Model.

DISABLE
Stop using Taste Agent personalization.

DISCONNECT
Remove optional external cultural data.
```

---

## 9. Retention

Retention should be defined by data type rather than left indefinite.

### Raw external import

Delete after successful normalization unless continued retention is genuinely required.

### Normalized consumption history

Retain only while necessary to provide the requested personalization service and subject to account deletion / feature controls.

### Derived Taste Model

Retain while personalization remains active and useful.

### Prompt / conversation history

Retain only where necessary and according to a clearly communicated period.

### Evaluation and operational logs

Use defined retention periods and minimize direct personal content.

---

## 10. Third Parties and International Transfers

Potential processors may include:

- cloud infrastructure providers
- AI / LLM API providers
- metadata providers where personal context is transmitted
- analytics services
- monitoring / evaluation providers

Before Pilot with real personal data, Letterboxd should document:

- processor location
- sub-processors
- storage region
- inference region
- support-access locations
- training-on-customer-data policy

Where personal data is transferred outside the EEA, an appropriate transfer mechanism must be established, such as an adequacy decision or Standard Contractual Clauses with supplementary measures where required.

---

# Short DPIA

## 11. Processing Assessed

Taste Agent systematically analyzes cultural-consumption history and current intent to personalize film discovery.

The DPIA scope includes:

- behavioural history
- inferred preference data
- profiling
- AI-assisted recommendation
- optional future cross-platform data combination
- third-party AI processing

---

## 12. Necessity and Proportionality

The system supports a legitimate product objective: better film discovery.

Proportionality depends on maintaining the following boundaries:

- core value must work from Letterboxd data alone
- external cultural data remains optional
- sensitive inference is prohibited
- recommendation logic is purpose-specific
- users receive transparency and controls
- only necessary fields are processed
- reusable content representations are separated from user preference data

---

## 13. DPIA Risk Snapshot

| Risk to User | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Incorrect Taste Model | Medium | Low–Medium | Correction, reset and explanation controls |
| Unexpected profiling | Medium | Medium | Clear activation and layered transparency |
| Sensitive inference | Low–Medium | High | Restricted taxonomy and explicit guardrails |
| Cross-media function creep | Medium | High | Purpose limitation and optional opt-in |
| External-data overcollection | Medium | Medium | Field-level minimization and raw-data deletion |
| Data breach | Low–Medium | High | Encryption, least privilege, secrets management |
| Misleading recommendation explanation | Medium | Medium | Grounded evidence and evaluator checks |
| Loss of control over connected data | Medium | Medium | Disconnect and deletion controls |
| Third-party processor exposure | Medium | Medium–High | Vendor assessment, DPA and transfer review |

---

## 14. DPIA Conclusion

A formal DPIA should be completed **before a real-user Pilot at meaningful scale**, particularly if cross-media data is introduced.

The current MVP reduces risk through purpose limitation, restricted inference, content/user separation, evaluation and grounded explanations, but those controls do not by themselves establish production GDPR compliance.

If high residual risk remains after mitigation, further legal assessment and, where required, supervisory-authority consultation should occur before deployment.

---

## 15. Production Checklist

Before real-user Pilot:

1. finalize processing purposes
2. select and document lawful bases
3. complete the DPIA
4. publish AI/profiling transparency
5. implement deletion, correction, reset and disable controls
6. define retention periods
7. review processors and DPAs
8. assess international transfers
9. verify sensitive-inference guardrails
10. document data lineage and provenance
11. define breach / incident handling
12. reassess if cross-media data is added

---

## 16. Compliance Principle

> **Taste Agent should understand what characteristics a user responds to in culture without attempting to determine who that user is as a person.**

This remains the central privacy boundary for the product.