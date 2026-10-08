# Taste Agent — ROI & Risk Assessment

**Client Case:** Letterboxd  
**Project:** AI-Powered Taste Intelligence for Film Discovery  
**Assessment Type:** Scenario-Based Business Case  
**Current Stage:** Working MVP → Proposed Pilot

---

## 1. Purpose

This document evaluates whether Taste Agent presents a sufficiently credible financial opportunity to justify a controlled Pilot and identifies the main risks that must be reduced before full deployment.

The MVP demonstrates that the product can run end to end. It does **not** establish that Taste Agent will increase paid conversion, retention or upsell.

The ROI model therefore remains a **scenario analysis rather than a forecast**.

---

## 2. Business Inputs and Assumptions

The model uses public Letterboxd subscription pricing and a market-benchmarked delivery estimate.

Letterboxd currently lists:

- Pro at **$19/year**
- Patron at **$49/year**

To keep costs and benefits in one currency, the ROI model converts those values using a fixed planning rate observed on 8 October 2026:

```text
1 USD = €0.8925
```

This gives approximate planning values of:

- Pro: **€16.96/year**
- Patron: **€43.73/year**
- Pro → Patron annual difference: **€26.78**

| Input | Value / Assumption |
|---|---:|
| Letterboxd members, Q2 2026 | 30.7M |
| Pro annual price | $19 / ≈ €16.96 |
| Patron annual price | $49 / ≈ €43.73 |
| Pro → Patron annual difference | $30 / ≈ €26.78 |
| Illustrative relevant Free cohort | 5,000,000 |
| Illustrative relevant paid cohort | 500,000 |
| One-time implementation estimate | **€102,000** |
| Annual operating estimate | **€34,000** |
| Year 1 total cost | **€136,000** |
| 36-month total cost | **€204,000** |

The cohort sizes remain **project assumptions**, not Letterboxd internal figures.

Important unknowns include paid-member count, churn, conversion rates, ARPU after discounts/tax, actual internal engineering capacity, feature usage frequency and production AI infrastructure cost.

---

## 3. Cost Estimate — Bottom-Up Market Benchmark

The original Round 1 business case used a top-down cost estimate. Teacher feedback correctly identified that it was too vague to defend as an implementation budget.

The Round 2 estimate is therefore built from:

> **role × daily rate × expected person-days**

using public Malt category rates for experienced freelancers.

### Market rates used

| Delivery capability | Malt benchmark | Planning rate |
|---|---:|---:|
| AI consulting / architecture | €690/day | €690/day |
| AI engineering | €492/day | €502/day blended AI/ML rate |
| Machine learning engineering | €512/day | included in €502/day blended rate |
| Full-stack development | €412/day | €412/day |
| MLOps | €609/day | €609/day |
| UX design | €428/day | €428/day |
| Project management | €585/day | €585/day |
| Cybersecurity | €615/day | €615/day |
| QA / technical testing proxy | €399/day Developer & IT benchmark | €399/day |

The €502 AI/ML rate is the midpoint of the Malt AI Engineer and Machine Learning Engineer category averages.

### Implementation work estimate

| Workstream | Person-days | Planning day rate | Estimated cost |
|---|---:|---:|---:|
| AI solution architecture / consulting | 12 | €690 | €8,280 |
| AI / ML engineering | 55 | €502 | €27,610 |
| Full-stack product engineering | 40 | €412 | €16,480 |
| MLOps / productionization | 15 | €609 | €9,135 |
| UX / product design | 12 | €428 | €5,136 |
| QA / regression / integration testing | 18 | €399 | €7,182 |
| Cybersecurity review | 5 | €615 | €3,075 |
| Project management / delivery coordination | 15 | €585 | €8,775 |
| Privacy / legal review allowance | — | project assumption | €3,000 |
| **Labour / review subtotal** |  |  | **€88,673** |
| 15% delivery contingency |  |  | **€13,301** |
| **Central one-time estimate** |  |  | **€101,974 ≈ €102,000** |

The resulting planning range is:

> **€95,000–€110,000 one-time implementation**

This estimate represents the work required to move from the current MVP to a **pilot-ready, production-oriented implementation**. It is not the cost of the student MVP itself.

The full benchmark and source list are documented in:

`research/malt_cost_benchmark.md`

---

## 4. Annual Operating Cost

The operating estimate separates specialist maintenance work from infrastructure assumptions that cannot yet be calculated from production traffic.

| Operating item | Assumption | Annual estimate |
|---|---|---:|
| LLM / embedding APIs, cloud and observability | Usage-dependent project assumption | €10,000 |
| AI / ML maintenance | 18 days × €502 | €9,036 |
| Full-stack maintenance | 12 days × €412 | €4,944 |
| QA / evaluation refresh | 8 days × €399 | €3,192 |
| MLOps / monitoring | 6 days × €609 | €3,654 |
| Security review | 2 days × €615 | €1,230 |
| Privacy / legal review | Project assumption | €1,500 |
| **Estimated annual operating cost** |  | **€33,556 ≈ €34,000** |

The €10,000 API/cloud line remains deliberately labeled as an assumption. The MVP does not yet provide production traffic volumes from which to derive a reliable usage bill.

---

## 5. Value Creation Model

Taste Agent is proposed as a native premium capability within Letterboxd's existing subscription model.

Potential value comes from three mechanisms:

1. **Free → Pro conversion** through differentiated personalized discovery
2. **Paid retention** through increased recurring discovery value
3. **Pro → Patron upsell** through advanced taste and cross-media capabilities

Hard-to-attribute benefits such as brand differentiation, engagement and partnership opportunities are excluded from the financial model.

---

## 6. 12-Month ROI Scenarios

The commercial assumptions remain unchanged from the Round 1 sensitivity model. What changes is the cost basis: it is now market-anchored and materially higher.

| Scenario | Assumed Free → Pro uplift | Paid churn reduction | Pro → Patron upgrades | Incremental annual value | Year 1 cost | ROI |
|---|---:|---:|---:|---:|---:|---:|
| Conservative | +0.05 pp | 0.25 pp | 1,000 | €90,366 | €136,000 | **-34%** |
| Base | +0.10 pp | 0.50 pp | 2,500 | €194,119 | €136,000 | **43%** |
| Upside | +0.25 pp | 1.00 pp | 5,000 | €430,631 | €136,000 | **217%** |

The calculation follows:

```text
ROI = (Net Benefit / Total Cost) × 100
```

This revision makes an important point visible: under a realistic market-based delivery cost, the conservative scenario **does not break even in Year 1**.

That is not a weakness in the model. It is precisely the reason a controlled Pilot is required before full investment.

---

## 7. 36-Month ROI Scenarios

The simplified 36-month model assumes:

- €102,000 initial implementation cost
- €34,000 annual operating cost
- constant annual incremental benefit
- no major additional development investment
- no discount rate

| Scenario | 36-Month Benefit | 36-Month Cost | Illustrative ROI |
|---|---:|---:|---:|
| Conservative | €271,097 | €204,000 | **33%** |
| Base | €582,356 | €204,000 | **185%** |
| Upside | €1,291,894 | €204,000 | **533%** |

This is a simplified sensitivity model, not a discounted cash-flow valuation.

---

## 8. Break-Even

If conversion were the only value mechanism:

```text
€136,000 / €16.96 ≈ 8,020 additional Pro subscriptions
```

Against the illustrative 5 million relevant Free-member cohort:

```text
8,020 / 5,000,000 ≈ 0.160%
```

The Year 1 break-even threshold is therefore approximately:

> **0.16 percentage points of incremental Free → Pro conversion**

This threshold is roughly twice the original top-down estimate and is a more useful Pilot decision gate because it is grounded in a visible delivery budget.

---

## 9. Cost Sources and Limitations

The implementation benchmark uses current public Malt category rates for experienced freelancers rather than individual profile quotes.

Primary sources:

- AI Consultant — Malt
- AI Engineer — Malt
- Machine Learning Engineer — Malt
- Full-Stack Developer — Malt
- MLOps Engineer — Malt
- UX Designer — Malt
- Project Manager — Malt
- Cybersecurity Expert — Malt
- Developer & IT Specialist — Malt

Source URLs and methodology are preserved in `research/malt_cost_benchmark.md`.

This does **not** imply that Letterboxd would staff the project entirely with freelancers.

A real delivery model could be cheaper if internal specialists are available, or more expensive if work is procured through an agency, requires deeper native-platform integration, or includes enterprise support commitments.

The purpose is not false precision. It is to make the implementation estimate **auditable and decision-useful**.

---

# Risk Assessment

## 10. Scoring Method

```text
Risk Score = Likelihood × Impact
```

Both likelihood and impact are scored from 1 to 5.

| Score | Priority |
|---:|---|
| 1–5 | Low |
| 6–10 | Medium |
| 11–15 | High |
| 16–25 | Critical |

---

## 11. Risk Matrix

| Risk | Category | Likelihood | Impact | Score | Priority | Mitigation |
|---|---|---:|---:|---:|---|---|
| Taste Agent does not outperform Letterboxd's existing discovery experience | Product | 4 | 5 | 20 | Critical | Controlled Pilot against a defined baseline with user-facing recommendation metrics |
| Handcrafted semantic taste signals are over-weighted despite weak predictive value | Technical / Product | 3 | 4 | 12 | High | Keep 62D profile explanatory; use request-aware retrieval and historical compatibility for ranking |
| Cross-media data adds little incremental recommendation value | Product / Privacy | 4 | 3 | 12 | High | Keep Goodreads descriptive only; require measurable lift before reintroducing cross-media ranking |
| Profiles become overly personal or imply unsupported traits | Ethical / Privacy | 2 | 5 | 10 | Medium | Restricted cultural taxonomy, sensitive-inference guardrails, evidence-grounded explanations |
| Recommendation latency is unacceptable for a consumer product | Technical | 3 | 4 | 12 | High | Continue caching/precomputation and production profiling; set Pilot latency SLO before scale |
| AI / infrastructure cost exceeds incremental value | Financial | 3 | 4 | 12 | High | Precompute reusable content representations, monitor cost per active user and contribution margin |
| Commercial uplift is insufficient to recover investment | Financial | 4 | 5 | 20 | Critical | Predefine break-even and ROI thresholds; stop or modify if Pilot/commercial experiment misses them |
| Users do not repeatedly engage with Taste Agent | Adoption | 3 | 4 | 12 | High | Measure repeat usage and watchlist actions, not one-off novelty engagement |
| Incorrect metadata or retrieval contaminates recommendations | Data / Technical | 3 | 4 | 12 | High | Hard constraints, metadata validation, safe rejection and abstention |
| LLM qualification or explanation is inconsistent | AI | 3 | 3 | 9 | Medium | Structured outputs, regression datasets, grounded evidence, LangSmith monitoring |
| GDPR profiling or external-data processing creates excessive compliance burden | Regulatory | 3 | 5 | 15 | High | DPIA, purpose limitation, lawful-basis review, optional external data, user controls |
| Third-party providers create transfer or availability risk | Operational / Regulatory | 2 | 4 | 8 | Medium | Vendor review, DPAs, transfer mechanisms, provider abstraction and fallbacks |

---

## 12. What Round 2 Changed in the Risk Position

Round 2 reduced several uncertainties but also falsified part of the original product assumption.

### Semantic profile value

The 62-dimensional semantic representation is useful for interpretation, but it **did not outperform the metadata baseline as a preference predictor**.

This reduces the risk of overcommitting to the original architecture because the MVP now separates profile explanation from ranking.

### Cross-media value

Goodreads did not demonstrate sufficient incremental predictive value to justify making cross-media data part of the primary recommendation path.

Cross-media expansion should therefore remain behind an explicit future evidence gate.

### Runtime feasibility

Performance work reduced measured recommendation latency from **160.72 seconds to a 48.35-second median**, approximately **69.9% lower / 3.3× faster**.

This is a material improvement, but the remaining latency is still too high to assume production readiness. Latency remains a Pilot engineering KPI.

### Cost credibility

The implementation budget is now built from current Malt market rates and an explicit work breakdown rather than a single top-down estimate.

The revised Year 1 cost of approximately **€136,000** materially changes the ROI profile: the conservative scenario is negative in Year 1, while the base and upside cases remain positive.

This makes the Pilot decision gate more meaningful rather than less attractive.

### Recommendation quality

The final system added hosted evaluation, human review and regression evidence, but the project still does not claim a statistically established commercial or recommendation winner.

The final evidence supports continuing to a controlled Pilot rather than full deployment.

---

## 13. Decision Gates

### MVP → Pilot

Proceed because:

- the MVP runs end to end
- recommendation architecture has been evaluated and hardened
- major technical assumptions have been tested rather than assumed
- user-facing recommendations are sufficiently credible for controlled testing
- the investment case now has a transparent market-based cost model

### Pilot → Commercial Experiment

Proceed only if the Pilot demonstrates:

- recommendation usefulness above the selected baseline
- repeated usage
- acceptable trust and explanation feedback
- acceptable latency and cost
- no material compliance blocker

### Commercial Experiment → Full Deployment

Proceed only if:

- incremental conversion / retention / upsell exceeds predefined thresholds
- contribution margin remains positive after AI operating costs
- privacy and compliance controls are production-ready
- reliability and latency meet production targets

---

## 14. Recommendation

The business case supports **controlled Pilot investment**, not full rollout.

The revised cost model strengthens that recommendation.

At the market-based implementation cost, Taste Agent does not need an unrealistic commercial effect to justify itself, but it also no longer appears profitable under every plausible scenario.

The highest-value unresolved question remains:

> **Does the working Taste Agent experience create enough incremental discovery value for users to change engagement or subscription behaviour?**

The Pilot should be designed to answer that question before Letterboxd commits the full implementation budget.