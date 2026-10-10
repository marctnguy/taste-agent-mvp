# Taste Agent — Round 1 Decision

**Client Case:** Letterboxd  
**Industry:** Digital Media & Entertainment  
**Decision:** Keep the use case; narrow and validate the recommendation problem before expanding cross-media scope

---

## 1. Decision After Round 1

The Round 1 recommendation was to **continue with Taste Agent and keep the Letterboxd use case**, while narrowing the MVP to film recommendations and using Round 2 to test whether the semantic approach creates measurable recommendation value.

The business problem did not change:

> **How can Letterboxd create additional user and commercial value from the preference data it already collects without reducing taste to genres, popularity, or opaque recommendations?**

The main change was technical and evidential rather than strategic.

---

## 2. Feedback Received

Peer feedback was consistent on four points:

1. The clearest differentiator was the attempt to model **deeper taste characteristics** rather than genres alone.
2. The project had not yet proved that the Taste Model produced **better recommendations than simpler or existing approaches**.
3. The largest risk was building an inaccurate or overly personal profile without proving that the additional profiling created enough value.
4. Round 2 should prioritize **movie recommendations first**, compare them against a baseline, test recommendation quality with users, and explain how the system differs from existing recommendation systems.

Additional feedback highlighted novelty and exploration: recommendations should not become a closed loop that only repeats the user's existing preferences.

---

## 3. Round 2 Response

Round 2 directly addressed that feedback.

### Film-first MVP

The recommendation product was narrowed to **film discovery**. Goodreads remains descriptive cross-media context rather than a production ranking dependency, and broader cross-media expansion remains a future hypothesis.

### Baseline comparison

The 62-dimensional handcrafted semantic representation was tested against a metadata baseline. It **did not outperform the metadata baseline as a preference predictor**.

The project therefore did not force the original hypothesis to survive. The semantic Taste Profile was retained for interpretation and explanation, while recommendation architecture moved toward request-aware retrieval and contextual personalization.

### Request, history and intent separated

The final runtime distinguishes:

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

This avoids treating a static profile as the sole definition of what a user may want next.

### Novelty preserved

The runtime supports personalized discovery outside the user's obvious history rather than converting historical preference into a hard content exclusion.

### Evaluation expanded

Round 2 added LangSmith tracing, reusable datasets, automated evaluators, human review, hosted experiments, regression checks and runtime profiling.

---

## 4. Final Round 2 Position

The project therefore **kept the same client, business problem and product direction**, but changed the technical assumptions based on evidence.

The current position is:

> **Taste Agent is not a claim that one semantic profile can predict a person's next film. It is a contextual discovery system that combines present intent, known interest, historical compatibility and interpretable taste evidence.**

Cross-media personalization remains optional and should only progress if a future Pilot proves measurable incremental value.