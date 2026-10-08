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

The model retains the Round 1 assumptions because Letterboxd does not publicly disclose the internal variables required for a bottom-up forecast.

| Input | Value / Assumption |
|---|---:|
| Letterboxd members, Q2 2026 | 30.7M |
| Pro annual price | $19 |
| Patron annual price | $49 |
| Pro → Patron annual difference | $30 |
| Illustrative relevant Free cohort | 5,000,000 |
| Illustrative relevant paid cohort | 500,000 |
| One-time implementation / Pilot cost | $45,000 |
| Annual operating cost | $30,000 |
| Year 1 total cost | $75,000 |
| 36-month total cost | $135,000 |

The cohort sizes and cost figures are **project assumptions**, not Letterboxd internal figures.

Important unknowns include paid-member count, churn, conversion rates, ARPU after discounts/tax, internal engineering cost, feature usage frequency and production AI infrastructure cost.

---

## 3. Value Creation Model

Taste Agent is proposed as a native premium capability within Letterboxd's existing subscription model.

Potential value comes from three mechanisms:

1. **Free → Pro conversion** through differentiated personalized discovery
2. **Paid retention** through increased recurring discovery value
3. **Pro → Patron upsell** through advanced taste and cross-media capabilities

Hard-to-attribute benefits such as brand differentiation, engagement and partnership opportunities are excluded from the financial model.

---

## 4. 12-Month ROI Scenarios

| Scenario | Assumed Free → Pro uplift | Paid churn reduction | Pro → Patron upgrades | Incremental annual revenue | Year 1 cost | ROI |
|---|---:|---:|---:|---:|---:|---:|
| Conservative | +0.05 pp | 0.25 pp | 1,000 | $101,250 | $75,000 | 35% |
| Base | +0.10 pp | 0.50 pp | 2,500 | $217,500 | $75,000 | 190% |
| Upside | +0.25 pp | 1.00 pp | 5,000 | $482,500 | $75,000 | 543% |

The calculation follows:

```text
ROI = (Net Benefit / Total Cost) × 100
```

These scenarios test sensitivity to relatively small subscription-behaviour changes. They do not predict that Taste Agent will achieve them.

---

## 5. 36-Month ROI Scenarios

The simplified 36-month model assumes:

- $45,000 initial implementation cost
- $30,000 annual operating cost
- constant annual incremental benefit
- no major additional development investment
- no discount rate

| Scenario | 36-Month Benefit | 36-Month Cost | Illustrative ROI |
|---|---:|---:|---:|
| Conservative | $303,750 | $135,000 | 125% |
| Base | $652,500 | $135,000 | 383% |
| Upside | $1,447,500 | $135,000 | 972% |

This is a simplified sensitivity model, not a discounted cash-flow valuation.

---

## 6. Break-Even

If conversion were the only value mechanism:

```text
$75,000 / $19 ≈ 3,948 additional Pro subscriptions
```

Against the illustrative 5 million relevant Free-member cohort:

```text
3,948 / 5,000,000 ≈ 0.079%
```

The Year 1 break-even threshold is therefore approximately:

> **0.08 percentage points of incremental Free → Pro conversion**

This is a useful Pilot decision threshold because it states what commercial uplift would need to be demonstrated rather than assuming the feature will pay for itself.

---

# Risk Assessment

## 7. Scoring Method

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

## 8. Risk Matrix

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

## 9. What Round 2 Changed in the Risk Position

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

### Recommendation quality

The final system added hosted evaluation, human review and regression evidence, but the project still does not claim a statistically established commercial or recommendation winner.

The final evidence supports continuing to a controlled Pilot rather than full deployment.

---

## 10. Decision Gates

### MVP → Pilot

Proceed because:

- the MVP runs end to end
- recommendation architecture has been evaluated and hardened
- major technical assumptions have been tested rather than assumed
- user-facing recommendations are sufficiently credible for controlled testing

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

## 11. Recommendation

The business case supports **controlled Pilot investment**, not full rollout.

The highest-value unresolved question is now:

> **Does the working Taste Agent experience create enough incremental discovery value for users to change engagement or subscription behaviour?**

The Pilot should be designed to answer that question with the smallest possible additional investment.