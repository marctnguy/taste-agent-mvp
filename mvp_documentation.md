# Taste Agent — MVP Documentation

**Client Case:** Letterboxd  
**Implementation:** Python + Streamlit + OpenAI / embeddings + TMDB + LangSmith  
**Current Stage:** Working MVP

---

## 1. MVP Purpose

The Round 2 MVP answers a different question from the Round 1 POC.

Round 1 asked:

> **Can a structured semantic Taste Model be constructed?**

Round 2 asks:

> **Can that knowledge be turned into a working, contextual film-discovery product?**

The MVP therefore focuses on one end-to-end capability: **personalized film recommendation from a natural-language request**.

---

## 2. User Experience

The Streamlit product contains two connected surfaces:

### Taste Profile

An interpretable view of long-term preference signals using the 62-dimensional semantic framework developed in Round 1.

### Taste Agent

A conversational discovery interface that accepts requests such as:

> "I want something English from the 80s."

and returns a grounded five-film recommendation slate where enough suitable candidates exist.

The profile supports interpretation; it does not independently define the recommendation universe.

---

## 3. Active Recommendation Architecture

```text
USER REQUEST
→ REQUEST UNDERSTANDING
→ QUERY-AWARE CATALOG RETRIEVAL
→ HARD CONSTRAINTS
→ SEMANTIC CANDIDATE QUALIFICATION
→ HISTORICAL PREFERENCE COMPATIBILITY
→ WATCHLIST / INTENT SIGNAL
→ SELECTION
→ GROUNDED EXPLANATION
→ VALIDATION / ABSTENTION
```

The central design principle is:

> **Current intent establishes relevance before historical taste is allowed to personalize.**

This prevents the system from recommending only films similar to what the user already watches.

---

## 4. Important Round 2 Learning

The original handcrafted semantic representation did **not** outperform the metadata baseline as a preference predictor.

The MVP therefore separates:

- **interpretability** — the 62D Taste Profile
- **historical compatibility** — frozen preference model
- **current relevance** — request-aware retrieval and qualification
- **known future interest** — watchlist signal
- **explanation** — grounded verbalization of attached evidence

Goodreads remains descriptive context only and is not required by the active recommendation path.

---

## 5. Main Components

| Component | Role |
|---|---|
| `mvp/streamlit_app.py` | User-facing MVP |
| `mvp/src/watchlist_personalization/reversible_history_watchlist_service.py` | Active recommendation service facade |
| `mvp/src/retrieval/catalog_retrieval.py` | Query-aware candidate retrieval |
| `mvp/src/generative_v4/` | Request understanding, qualification, selection, explanation, validation |
| `mvp/artifacts/models/b3_mvp/` | Frozen historical-preference model bundle |
| `evaluation/langsmith/` | Evaluation datasets and evaluators |
| `evaluation/watchlist_personalization/final_evidence/` | Canonical hosted/human evaluation evidence |

---

## 6. Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Create local environment variables from:

```text
.env.example
```

The live recommendation path requires the configured external API credentials and the Letterboxd watchlist export used by the accepted service.

---

## 7. Run the MVP

From the repository root:

```bash
PYTHONPATH=. streamlit run mvp/streamlit_app.py
```

The dashboard opens locally in the browser.

### Demo Recording

A recorded walkthrough of the working Streamlit MVP is available here:

[Watch the MVP demo](https://drive.google.com/file/d/1zjmzWOsguTgZxdxXsI7zidOOflcFF3MJ/view?usp=sharing)

This recording is also suitable as the backup demo for the final Round 2 presentation.

---

## 8. Core AI Capability

The MVP's core AI capability actually runs end to end:

1. interpret the user's request
2. retrieve relevant film candidates
3. apply explicit hard constraints
4. semantically qualify the candidate pool
5. apply historical preference compatibility
6. incorporate watchlist intent where appropriate
7. select recommendations
8. generate evidence-grounded explanations
9. validate outputs and abstain when necessary

---

## 9. Error Handling / Safe Behaviour

The MVP includes several safe-failure behaviours:

- hard constraints can reject unsuitable candidates
- unsupported candidates do not proceed to final selection
- watched films are excluded from recommendation
- fewer than five recommendations is valid when the evidence does not support five
- no-match / abstention is valid
- recommendation explanations must be grounded in attached evidence
- sensitive personal inference is prohibited

---

## 10. Evaluation and Testing

Focused checks:

```bash
PYTHONPATH=. pytest -q \
  mvp/tests/runtime_v4/test_hard_constraints.py \
  mvp/tests/watchlist_personalization/test_reversible_history_watchlist.py
```

The final performance-phase regression gate produced:

```text
22 passed
1 known H12 parser-label failure
```

The H12 failure is a pre-existing LLM request-label variation (`based on taste` vs `personalized discovery`); its actual content-exclusion invariant still passes.

---

## 11. Performance

A dedicated latency optimization phase reduced recommendation runtime from:

```text
160.72s baseline
→
48.35s median across three isolated runs
```

This is approximately:

```text
69.9% lower latency
~3.3× faster
```

The result is sufficient for MVP demonstration but should **not** be interpreted as a production latency target. Consumer-grade latency remains a Pilot engineering requirement.

---

## 12. Known Limitations

- current latency remains high for a production consumer interaction
- the MVP uses one user's historical/export data rather than production multi-user infrastructure
- human evaluation is limited and does not establish statistical superiority
- cross-media recommendation lift has not been demonstrated
- business impact has not been tested
- third-party APIs remain external dependencies
- the recommendation runtime contains LLM-backed components and is therefore not perfectly deterministic

---

## 13. Production Boundary

The MVP is intentionally small.

It proves one capability end to end rather than simulating a full Letterboxd production system.

Production work would still require:

- native authentication and data integration
- multi-user storage and isolation
- production observability
- stricter latency SLOs
- cost controls
- user correction / reset / deletion controls
- processor and security review
- controlled Pilot infrastructure

---

## 14. MVP Conclusion

The MVP demonstrates that Taste Agent is no longer only a conceptual semantic profile.

It is a functioning film-discovery product that combines present intent, known interest, historical compatibility and interpretable evidence.

The next question is not whether the capability can run.

It is whether real Letterboxd users value it enough to justify Pilot and commercial investment.