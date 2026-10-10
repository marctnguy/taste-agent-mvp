# Taste Agent — Malt Implementation Cost Benchmark

**Client Case:** Letterboxd  
**Purpose:** Bottom-up implementation and operating-cost benchmark  
**Market Source:** Malt public category rate pages  
**Research Date:** 8 October 2026

---

## 1. Why This Benchmark Was Added

The original business case used a high-level implementation estimate.

That was sufficient for an early scenario model, but it did not make the underlying delivery assumptions visible enough to defend the number in front of a business audience.

This benchmark therefore replaces the lump-sum estimate with a **role × daily rate × person-day model** anchored in current public Malt rate data.

It remains a planning estimate rather than a supplier quotation.

---

## 2. Method

The model uses Malt's published **average daily rate for experienced freelancers** in the categories most relevant to moving Taste Agent from the current MVP to a pilot-ready implementation.

No individual freelancer is used as the pricing basis.

Malt states that rates vary by experience, location and specialization, so the figures below should be read as market anchors rather than guaranteed procurement prices.

---

## 3. Malt Market Benchmarks

| Delivery capability | Malt category benchmark | Planning rate used |
|---|---:|---:|
| AI consulting / architecture | €690/day | €690/day |
| AI engineering | €492/day | €502/day blended AI/ML rate |
| Machine learning engineering | €512/day | €502/day blended AI/ML rate |
| Full-stack development | €412/day | €412/day |
| MLOps | €609/day | €609/day |
| UX design | €428/day | €428/day |
| Project management | €585/day | €585/day |
| Cybersecurity | €615/day | €615/day |
| QA / technical testing proxy | €399/day Developer & IT specialist benchmark | €399/day |

The €502 AI/ML planning rate is the midpoint of the Malt AI Engineer (€492) and Machine Learning Engineer (€512) category averages.

### Malt sources

- AI Consultant: https://www.malt.com/en-gb/a/freelance/business-consulting/ai-consultant
- AI Engineer: https://www.malt.com/a/freelance/tech/ai-engineer
- Machine Learning Engineer: https://www.malt.com/a/freelance/data/machine-learning-engineer
- Full-Stack Developer: https://www.malt.com/a/freelance/tech/backend-developer/fullstack-developer
- MLOps Engineer: https://www.malt.com/a/freelance/data/mlops-engineer
- UX Designer: https://www.malt.com/a/freelance/web-graphic-design/ux-designer
- Project Manager: https://www.malt.com/a/freelance/project-manager-coach/project-manager
- Cybersecurity Expert: https://www.malt.com/a/freelance/tech/cybersecurity-expert
- Developer & IT Specialist benchmark: https://www.malt.com/a/freelance/tech

---

## 4. Bottom-Up Implementation Estimate

This estimate is for the work required to move from the current MVP to a **pilot-ready, production-oriented implementation**. It is not the cost of the student MVP already built.

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
| **Central one-time implementation estimate** |  |  | **€101,974 ≈ €102,000** |

A reasonable planning range is therefore:

> **€95,000–€110,000 one-time implementation**

The range allows for supplier mix, integration complexity and the extent to which Letterboxd could substitute internal staff for external specialists.

---

## 5. Annual Operating Envelope

The annual model separates market-benchmarked specialist labour from infrastructure assumptions that cannot yet be calculated from production traffic.

| Operating item | Assumption | Annual estimate |
|---|---|---:|
| LLM / embedding APIs, cloud and observability | Project assumption; usage-dependent | €10,000 |
| AI / ML maintenance | 18 days × €502 | €9,036 |
| Full-stack maintenance | 12 days × €412 | €4,944 |
| QA / evaluation refresh | 8 days × €399 | €3,192 |
| MLOps / monitoring | 6 days × €609 | €3,654 |
| Security review | 2 days × €615 | €1,230 |
| Privacy / legal review | Project assumption | €1,500 |
| **Estimated annual operating cost** |  | **€33,556 ≈ €34,000** |

The €10,000 API/cloud line remains a scenario assumption because the MVP does not provide real production traffic volumes from which to derive a defensible usage bill.

---

## 6. Revised Cost Baseline

The business case should therefore use:

| Cost metric | Revised planning value |
|---|---:|
| One-time implementation | **€102,000** |
| Annual operating cost | **€34,000** |
| Year 1 total | **€136,000** |
| 36-month total | **€204,000** |

The 36-month figure uses rounded planning values: one €102,000 implementation plus three years at €34,000 operating cost.

---

## 7. Currency Treatment

Letterboxd currently lists Pro at **$19/year** and Patron at **$49/year** on its public upgrade page.

For the revised ROI model, subscription values are converted into euros using a fixed planning rate observed on 8 October 2026:

```text
1 USD = €0.8925
```

This gives approximate planning values of:

- Pro: **€16.96/year**
- Patron: **€43.73/year**
- Pro → Patron difference: **€26.78/year**

Letterboxd pricing source:

- https://letterboxd.com/pro/

The fixed exchange rate is used only so costs and benefits can be compared in one currency. It is not an FX forecast.

---

## 8. Limitations

This benchmark does not imply that Letterboxd would staff the project entirely with freelancers.

A real delivery model could be cheaper if internal engineering, product, legal and security capacity is available, or more expensive if the work is procured through an agency, requires deeper native-platform integration, or includes enterprise support commitments.

The purpose of the benchmark is therefore not precision to the euro. It is to make the implementation estimate **auditable, market-anchored and decision-useful**.