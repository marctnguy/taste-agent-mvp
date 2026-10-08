# Taste Agent — LangSmith Evaluation Summary

**Stage:** Round 2 MVP  
**Purpose:** Recommendation-quality, grounding and runtime validation

---

## 1. What Was Evaluated

Round 1 evaluated whether the semantic representation pipeline could reliably construct a Taste Model.

Round 2 moved the evaluation target to the **recommendation product**:

> **Can Taste Agent interpret a request, retrieve suitable candidates, personalize them without overfitting to history, and explain the result using grounded evidence?**

The evaluation therefore covers:

- request understanding
- hard-constraint compliance
- recommendation relevance
- recommendation grounding
- usefulness / clarity
- watchlist-aware personalization
- historical preference compatibility
- safe abstention
- regression behaviour

---

## 2. Dataset

The final runtime evaluation uses a reusable prompt set covering:

- direct constraints
- contextual discovery
- personalized discovery
- novelty requests
- watchlist-aware cases
- negative / exclusion cases
- known edge cases

The repository retains the Round 2 LangSmith datasets under:

`evaluation/langsmith/`

and the final watchlist-personalization evidence under:

`evaluation/watchlist_personalization/final_evidence/`

---

## 3. Tracing

The MVP runtime is instrumented for LangSmith tracing.

Tracing captures the multi-stage recommendation flow rather than only the final text response, allowing failures to be localized to request interpretation, retrieval, qualification, personalization, selection or explanation.

This directly addresses the Round 1 transparency gap: the evaluation can inspect **why** a recommendation was produced, not only whether a final answer appeared plausible.

---

## 4. Evaluators

The project uses a combination of:

- rule-based validation
- structured runtime checks
- LLM-assisted evaluation
- human review

Evaluator code is retained under:

`evaluation/langsmith/evaluators.py`

and the watchlist-personalization evaluation harness is retained under:

`evaluation/watchlist_personalization/`.

At least one experiment uses automated evaluators against traced MVP outputs, satisfying the Round 2 requirement for a LangSmith experiment with an evaluator.

---

## 5. Final Hosted Evidence

The canonical final evidence pack contains:

- **four hosted experiments**
- full-watchlist and empty-watchlist scenarios
- Variant A and Variant B
- **12 prompts per experiment / 48 hosted rows total**
- completed human review
- runtime-freeze evidence

Variant A remains the accepted practical default.

The human-review artifact contains **84 unique prompt–film ratings**:

```text
61 yes
11 no
12 unsure
```

These results are useful product evidence but are not independent statistical trials and do not establish a universal winning recommendation policy.

---

## 6. What Failed / What Changed

The evaluation process produced several material findings.

### The 62D semantic profile was not enough

The handcrafted semantic representation did not outperform the metadata baseline as a preference predictor.

**Decision:** keep it for interpretation and explanation, not as the sole ranking engine.

### Historical taste could dominate discovery

A preference model trained on already-consumed films could favor familiar historical patterns and create selection bias.

**Decision:** request relevance and candidate retrieval happen before historical personalization.

### Watchlist data represented a missing intent signal

Watchlist membership expresses future interest and is not equivalent to either watched history or positive rating.

**Decision:** model watchlist as its own signal / route rather than treating it as liked content.

### Cross-media lift was not established

Goodreads did not demonstrate enough incremental predictive value to justify becoming a recommendation dependency.

**Decision:** keep cross-media descriptive and optional until a future Pilot proves incremental value.

### H12 exposed parser nondeterminism

One regression test can produce equivalent personalization intent with different wording (`personalized discovery` vs `based on taste`).

The actual content-exclusion invariant still passes.

**Decision:** document the known nondeterministic label failure rather than changing ranking logic on the performance branch.

---

## 7. Human Evaluation Boundary

Human review is intentionally treated as decision support rather than statistical proof.

The final evidence pack notes that:

- the review represents one person's judgments
- candidates overlap across conditions
- some hosted slates are partial
- H12 was problematic in the hosted comparison
- later targeted fixes were verified locally rather than rerun as a fresh hosted experiment

These limitations are retained rather than silently repaired.

---

## 8. What a Pilot Should Monitor

A production Pilot should continue evaluation across three layers.

### Recommendation quality

- relevance
- intention to watch
- watchlist additions
- novelty / discovery value
- already-consumed rate
- explanation usefulness

### Runtime quality

- hard-constraint violations
- unsupported recommendations
- abstention rate
- failure rate
- latency
- cost per recommendation

### Product impact

- activation
- repeat usage
- recommendation interactions
- paid conversion / retention experiment outcomes

---

## 9. Evidence Locations

Primary evaluator-facing evidence:

`evaluation/watchlist_personalization/final_evidence/README.md`

Hosted results:

`evaluation/watchlist_personalization/final_evidence/hosted/`

Completed human review:

`evaluation/watchlist_personalization/final_evidence/human_review/`

Runtime freeze evidence:

`evaluation/watchlist_personalization/final_evidence/runtime_freeze/`

LangSmith datasets / evaluators:

`evaluation/langsmith/`

---

## 10. Conclusion

Round 2 did not use LangSmith to manufacture a single quality score.

It used tracing and evaluators to expose where the recommendation system failed, compare architecture variants, preserve regression evidence and document uncertainty.

The main evaluation outcome is therefore architectural as well as numerical:

> **Current intent must establish relevance first; history personalizes after that; explanations must remain grounded; and cross-media value must be proven before it adds complexity.**