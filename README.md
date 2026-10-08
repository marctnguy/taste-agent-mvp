# Taste Agent

Taste Agent is a Letterboxd capstone that turns film-history signals, expressed watchlist intent and current user requests into contextual, explainable film discovery.

Round 1 validated semantic taste modelling through an n8n POC. Round 2 built and evaluated a functional Python/Streamlit MVP and packages the result as a client-facing consulting deliverable.

---

## Submission Structure

```text
taste-agent-mvp/
├── README.md
├── feedback/
│   └── round1_decision.md
├── research/
│   ├── sector_research.md
│   ├── opportunities_risks.md
│   ├── use_cases.md
│   └── malt_cost_benchmark.md
├── charts/
│   ├── 01-letterboxd-growth.svg
│   ├── 02-preference-associations.svg
│   ├── 03-prevalence-vs-preference.svg
│   ├── 04-roi-scenarios-round1.svg
│   └── 05-roi-scenarios-round2.svg
├── evaluation/
│   ├── eval_plan.md
│   ├── langsmith.md
│   └── ...supporting evaluation evidence
├── use_case_definition.md
├── poc/
│   └── poc_documentation.md
├── roi_risk_assessment.md
├── compliance/
│   ├── eu_ai_act_compliance.md
│   └── gdpr_documentation.md
├── strategic_plan.md
├── presentation/
│   └── README.md
└── mvp/
    ├── mvp_documentation.md
    ├── streamlit_app.py
    └── ...working MVP code, tests and artifacts
```

The final presentation will be added to `presentation/` as `presentation.pdf` or `presentation.pptx`.

---

## Consulting Package

- [`use_case_definition.md`](use_case_definition.md) — business problem, Letterboxd context, solution, stakeholders, success criteria, scope and Round 1 → Round 2 evolution
- [`roi_risk_assessment.md`](roi_risk_assessment.md) — market-benchmarked implementation cost, 12/36-month ROI, break-even and risk matrix
- [`compliance/eu_ai_act_compliance.md`](compliance/eu_ai_act_compliance.md) — AI Act classification, conformity summary and technical-documentation outline
- [`compliance/gdpr_documentation.md`](compliance/gdpr_documentation.md) — data flows, processing register, lawful-basis candidates, short DPIA, rights and transfers
- [`strategic_plan.md`](strategic_plan.md) — POC → MVP → Pilot → commercial experiment → full deployment
- [`feedback/round1_decision.md`](feedback/round1_decision.md) — peer feedback and how Round 2 responded

### Research and charts

The relevant Round 1 research is now retained locally under [`research/`](research/) rather than only being referenced externally. The [`charts/`](charts/) folder retains the Round 1 evidence in presentation-ready SVG form and adds the revised Round 2 ROI chart. Chart provenance is documented in [`charts/README.md`](charts/README.md).

The full original Round 1 repository remains available for historical traceability:

`https://github.com/marctnguy/taste-agent-round1`

---

## POC

- [`poc/poc_documentation.md`](poc/poc_documentation.md) — no-code POC architecture, reproduction steps, limits and POC → MVP evolution
- Original n8n workflow export: `https://github.com/marctnguy/taste-agent-round1/blob/main/poc/workflow/taste-agent-poc-v03.json`
- POC demo recording: [Watch the narrated n8n walkthrough](https://drive.google.com/file/d/1xgIm0cDU0K1vI1VvIf0-W8gcnUDRVAK8/view?usp=sharing)

The workflow export remains linked to its canonical Round 1 source rather than maintaining two divergent copies.

---

## Working MVP

- [`mvp/mvp_documentation.md`](mvp/mvp_documentation.md) — setup, architecture, error handling, testing, performance and limitations
- MVP demo recording: [Watch the working Streamlit MVP](https://drive.google.com/file/d/1zjmzWOsguTgZxdxXsI7zidOOflcFF3MJ/view?usp=sharing)
- UI entrypoint: `mvp/streamlit_app.py`
- Requirements: `requirements.txt`
- Environment template: `.env.example`

Run locally:

```bash
PYTHONPATH=. streamlit run mvp/streamlit_app.py
```

---

## LangSmith / Evaluation

- [`evaluation/eval_plan.md`](evaluation/eval_plan.md) — historical Round 1 POC evaluation plan
- [`evaluation/langsmith.md`](evaluation/langsmith.md) — grader-facing Round 2 evaluation summary
- `evaluation/langsmith/` — datasets, evaluators and evaluation plans
- `evaluation/watchlist_personalization/final_evidence/` — canonical hosted comparison, completed human review and runtime-freeze evidence

---

## Product Evolution

Round 1 proposed a semantic cross-media Taste Model.

Round 2 tested that assumption rather than carrying it forward unchanged.

The handcrafted 62-dimensional semantic representation remained useful for interpretation, but it did **not** outperform the metadata baseline as a preference predictor. Goodreads also did not demonstrate sufficient incremental predictive value to justify becoming a recommendation dependency.

The active recommendation architecture therefore separates:

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

The 62D Taste Profile remains an explanation and interpretation layer rather than the sole ranking engine.

---

## Active Architecture

- Service entrypoint: `mvp.src.watchlist_personalization.reversible_history_watchlist_service.load_reversible_history_watchlist_service()`
- UI entrypoint: `mvp/streamlit_app.py`
- LangSmith / evaluation runner: `python -m mvp.src.watchlist_personalization.langsmith_evaluation`
- Frozen offline runner: `python evaluation/watchlist_personalization/run_reversible_history_watchlist_experiment.py`
- The Streamlit UI calls the service facade and its `recommend(...)` method; recommendation logic stays outside the presentation layer.

Request flow:

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

---

## Required Inputs

- `mvp/data/processed/condition_a_enriched.csv`
- `mvp/data/processed/primary_holdout_split.csv`
- `mvp/data/processed/watched_override_confirmations.csv`
- `mvp/artifacts/experiments/exploratory_latent_semantics/embedding_cache/film_embeddings.csv`
- `evaluation/hitl/runtime_v4_final/05_run_manifest.json`
- Letterboxd watchlist ZIP in the repository parent directory when running the live watchlist service

---

## Setup and Test

```bash
pip install -r requirements.txt
```

Focused checks:

```bash
PYTHONPATH=. pytest -q \
  mvp/tests/runtime_v4/test_hard_constraints.py \
  mvp/tests/watchlist_personalization/test_reversible_history_watchlist.py
```

The final performance regression gate produced 22 passing tests and one known pre-existing H12 parser-label variation; its substantive exclusion invariant still passes.

---

## Performance

Dedicated runtime optimization reduced measured recommendation latency from:

```text
160.72s baseline
→
48.35s median across three isolated runs
```

approximately **69.9% lower / 3.3× faster**.

This is sufficient for MVP demonstration but not presented as a final production latency target.

---

## Evidence

- Canonical evidence pack: `evaluation/watchlist_personalization/final_evidence/README.md`
- Hosted comparison: `evaluation/watchlist_personalization/final_evidence/hosted/`
- Completed human review: `evaluation/watchlist_personalization/final_evidence/human_review/`
- Runtime freeze evidence: `evaluation/watchlist_personalization/final_evidence/runtime_freeze/`

---

## Known Limits

- human evaluation is limited and does not establish statistical superiority
- recommendation latency remains too high to assume production readiness
- the live service depends on external APIs and the Letterboxd watchlist export
- cross-media recommendation lift has not been demonstrated
- business impact has not yet been tested with real users
- LLM-backed stages introduce normal nondeterminism

The next recommended stage is a **controlled Pilot**, not full deployment.