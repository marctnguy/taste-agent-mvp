# Taste Agent — Strategic Deployment and Commercialisation Plan

**Client Case:** Letterboxd  
**Project:** AI-Powered Taste Intelligence for Film Discovery  
**Current Stage:** Working MVP  
**Recommended Path:** POC → MVP → Pilot → Commercial Experiment → Full Deployment

---

## 1. Deployment Strategy

Round 1 validated that a semantic Taste Model could be constructed.

Round 2 built and evaluated a functional recommendation MVP.

The next step is therefore **not another MVP cycle and not full deployment**. It is a controlled Pilot designed to test whether the working product creates enough incremental discovery value for real users.

```text
POC — COMPLETE
Technical feasibility
      ↓
MVP — COMPLETE
Functional recommendation product
      ↓
PILOT — NEXT
Real-user product validation
      ↓
COMMERCIAL EXPERIMENT
Conversion / retention / upsell
      ↓
FULL DEPLOYMENT
Conditional scale
```

Investment should increase only when the previous stage produces sufficient evidence.

---

## 2. Roadmap

| Phase | Objective | Main Activities | Decision Gate |
|---|---|---|---|
| POC — Complete | Validate semantic representation | n8n workflow, metadata enrichment, 62D classification, preference modelling | Can a structured Taste Model be constructed? **Yes** |
| MVP — Complete | Build a working recommendation product | Python recommendation runtime, Streamlit UI, request-aware retrieval, watchlist intent, historical compatibility, LangSmith evaluation | Can the capability run end to end? **Yes** |
| Pilot — Next | Validate product value with real users | Controlled opt-in cohort, baseline comparison, usage and quality measurement, latency/cost monitoring | Do users receive measurably better discovery? |
| Commercial Experiment | Validate economic value | Paid-tier experiment, conversion/retention/upsell measurement | Does incremental value exceed incremental cost? |
| Full Deployment | Scale validated product | Native Letterboxd integration, production infrastructure, monitoring, subscription integration | Are product, economic, technical and compliance thresholds met? |

---

## 3. Pilot Design

The Pilot should use a limited opt-in cohort and compare Taste Agent against a selected Letterboxd discovery baseline.

Primary questions:

1. Are recommendations perceived as more relevant or useful?
2. Does Taste Agent create discovery beyond obvious historical preferences?
3. Do users add recommended films to their watchlists or return to the feature?
4. Are explanations useful and trustworthy?
5. Is latency acceptable for repeated consumer use?
6. Can the feature operate at an acceptable cost per active user?

Cross-media data should **not** be required for the Pilot. It should return only as a separate experiment if there is evidence that it can produce incremental value.

---

## 4. Pilot KPIs and Greenlight Criteria

| Area | KPI | Greenlight Logic |
|---|---|---|
| Recommendation Quality | relevance / usefulness vs baseline | Clear positive user preference over baseline |
| Discovery | intention to watch / watchlist additions | Meaningful incremental action rate |
| Novelty | useful recommendations outside obvious history | Positive qualitative and behavioural evidence |
| Adoption | activation and repeat usage | More than one-off novelty use |
| Trust | explanation usefulness / complaint rate | Explanations understood; no material trust issue |
| Technical | completion, errors, hard-constraint compliance | Stable enough for repeated use |
| Performance | median and p95 latency | Meet pre-agreed consumer SLO before scale |
| Cost | cost per active user | Compatible with expected contribution margin |
| Privacy | opt-out / complaint / deletion handling | Controls work and no unresolved high-risk issue |

Pilot thresholds should be fixed before unblinding commercial decisions.

---

## 5. Go-to-Market

Taste Agent is proposed as a native Letterboxd capability, so the primary distribution channel is Letterboxd itself.

A suitable rollout sequence is:

```text
Internal / Selected Tester Cohort
        ↓
Small Opt-In Beta
        ↓
Paid-Member Pilot
        ↓
Commercial Experiment
        ↓
Broader Pro / Patron Rollout
```

Messaging should focus on the user benefit rather than the AI technology.

Example positioning:

> **Discover films through the patterns behind what you already love — without being boxed into the same genres.**

---

## 6. Commercialisation Model

Taste Agent should be tested as a premium capability within Letterboxd's existing subscription model rather than as a separate subscription.

One packaging hypothesis remains:

| Capability | Free | Pro | Patron |
|---|:---:|:---:|:---:|
| Basic Taste preview | ✓ | ✓ | ✓ |
| Full Taste Profile | — | ✓ | ✓ |
| Contextual recommendations | Limited | ✓ | ✓ |
| Taste Agent | Limited | ✓ | ✓ |
| Taste Evolution | — | Limited | ✓ |
| Optional external cultural connections | — | — | Experimental |

This is a testable commercial hypothesis, not a recommendation to change Letterboxd's current plans without evidence.

---

## 7. Stakeholder Communication Plan

| Stakeholder | Primary Question | Evidence Needed | Communication |
|---|---|---|---|
| Product / Leadership | Does this improve Letterboxd enough to fund? | Pilot quality, adoption, ROI | Pilot readout and decision memo |
| Engineering / Data | Can this scale reliably? | latency, cost, failure modes | Architecture and operations review |
| Legal / Privacy | Is profiling proportionate and controllable? | DPIA, lawful basis, processor review | Pre-Pilot compliance gate |
| Design / UX | Do users understand and trust it? | usability and explanation feedback | User research readout |
| Growth / Commercial | Does it change subscription behaviour? | conversion, retention, upsell | Commercial experiment report |
| Support | What can fail and how do we explain it? | known limits, user controls | support playbook |
| Members | Why use it and what data is used? | clear value and transparency | in-product onboarding and settings |

---

## 8. Production Architecture Direction

The production system should preserve the Round 2 separation of concerns:

```text
REQUEST LAYER
Current intent and hard constraints
        ↓
RETRIEVAL LAYER
Request-aware candidate universe
        ↓
PERSONALIZATION LAYER
Historical compatibility + watchlist intent
        ↓
EXPLANATION LAYER
Grounded taste evidence
```

Reusable film representations should be precomputed and cached where possible.

The production system should not repeatedly classify the same film or resend unnecessary user history to an LLM.

---

## 9. Full Deployment Conditions

Full rollout should occur only if:

1. users demonstrably prefer the discovery experience over the selected baseline
2. repeat usage is established
3. recommendation explanations are useful and trustworthy
4. latency meets a production consumer target
5. cost per active user remains economically sustainable
6. commercial impact meets predefined thresholds
7. GDPR / AI Act controls are production-ready
8. operational monitoring and incident handling are in place

Cross-media functionality requires an additional gate:

> **External cultural data should only progress if it produces measurable incremental recommendation value over the simpler Letterboxd-only architecture.**

---

## 10. Strategic Recommendation

The MVP has answered the engineering question: the product can be built and run.

The next investment should answer the product question:

> **Does Taste Agent create enough incremental discovery value to deserve a permanent place in Letterboxd?**

A controlled Pilot is the correct next step because it tests that question before Letterboxd assumes either commercial value or production readiness.