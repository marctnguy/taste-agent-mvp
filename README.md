# Taste Agent

Taste Agent is a Letterboxd AI Consulting capstone that turns film-history signals, expressed watchlist intent and current user requests into contextual, explainable film discovery.

Round 1 validated semantic taste modelling through an n8n POC. Round 2 tested the original assumptions, built a working Python/Streamlit MVP, evaluated it with LangSmith and human review, and packaged the result as a client-facing consulting recommendation.

---

## Final Submission Structure

```text
taste-agent-mvp/
├── README.md
├── use_case_definition.md
├── roi_risk_assessment.md
├── strategic_plan.md
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
│   ├── 05-roi-scenarios-round2.svg
│   └── README.md
├── compliance/
│   ├── eu_ai_act_compliance.md
│   └── gdpr_documentation.md
├── poc/
│   └── poc_documentation.md
├── presentation/
│   └── presentation.pdf
├── evaluation/
│   ├── eval_plan.md
│   ├── langsmith.md
│   └── supporting evaluation evidence
└── mvp/
    ├── mvp_documentation.md
    ├── streamlit_app.py
    ├── src/
    ├── tests/
    ├── data/
    ├── artifacts/
    └── supporting MVP documentation
```

Repository configuration remains at the root through `.env.example`, `.gitignore`, `.streamlit/config.toml`, `pytest.ini` and `requirements.txt`.

---

## Consulting Deliverables

- [`use_case_definition.md`](use_case_definition.md) — business problem, Letterboxd context, solution, stakeholders, success criteria, scope and Round 1 → Round 2 evolution
- [`roi_risk_assessment.md`](roi_risk_assessment.md) — market-benchmarked implementation cost, 12/36-month ROI, break-even and risk matrix
- [`compliance/eu_ai_act_compliance.md`](compliance/eu_ai_act_compliance.md) — AI Act classification, conformity summary and technical-documentation outline
- [`compliance/gdpr_documentation.md`](compliance/gdpr_documentation.md) — data flows, processing register, lawful-basis candidates, short DPIA, rights and transfers
- [`strategic_plan.md`](strategic_plan.md) — POC → MVP → Pilot → commercial experiment → full deployment
- [`feedback/round1_decision.md`](feedback/round1_decision.md) — Round 1 peer feedback and how Round 2 responded
- [`presentation/presentation.pdf`](presentation/presentation.pdf) — final Round 2 decision-maker presentation

---

## Research and Charts

The relevant Round 1 research is retained locally under [`research/`](research/). The [`charts/`](charts/) folder keeps the original evidence in portable presentation form and includes the revised Round 2 ROI view.

The original Round 1 repository remains available for historical traceability:

`https://github.com/marctnguy/taste-agent-round1`

---

## POC

- [`poc/poc_documentation.md`](poc/poc_documentation.md) — no-code POC architecture, reproduction steps, limits and POC → MVP evolution
- Canonical n8n workflow export: `https://github.com/marctnguy/taste-agent-round1/blob/main/poc/workflow/taste-agent-poc-v03.json`
- POC demo recording: [Watch the narrated n8n walkthrough](https://drive.google.com/file/d/1xgIm0cDU0K1vI1VvIf0-W8gcnUDRVAK8/view?usp=sharing)

The POC demonstrated that film and book histories can enter the same semantic architecture and contribute to an interpretable cross-media Taste Profile. It did not establish incremental predictive lift from book data for film ranking.

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

The active service entrypoint is:

```text
mvp.src.watchlist_personalization.reversible_history_watchlist_service.load_reversible_history_watchlist_service()
```

Request flow:

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

---

## Evaluation

- [`evaluation/eval_plan.md`](evaluation/eval_plan.md) — historical Round 1 POC evaluation plan
- [`evaluation/langsmith.md`](evaluation/langsmith.md) — Round 2 evaluation summary
- `evaluation/langsmith/` — datasets, evaluators and evaluation plans
- `evaluation/watchlist_personalization/final_evidence/` — canonical hosted comparison, completed human review and runtime-freeze evidence

The 62-dimensional handcrafted semantic representation did **not** outperform the metadata baseline as a film-preference predictor. The semantic Taste Profile is therefore retained for interpretation, aggregation and explanation rather than used as the sole ranking engine.

---

## Performance

Dedicated runtime optimization reduced measured recommendation latency from:

```text
160.72s baseline
→
48.35s median across three isolated runs
```

approximately **69.9% lower / 3.3× faster**.

This is sufficient for MVP demonstration but is not presented as a final production latency target.

---

## Known Limits and Recommendation

- human evaluation is limited and does not establish statistical superiority
- recommendation latency remains too high to assume production readiness
- the live service depends on external APIs and the Letterboxd watchlist export
- cross-media aggregation is demonstrated, but incremental cross-media predictive lift is not
- business impact has not yet been tested with real Letterboxd users
- LLM-backed stages introduce normal nondeterminism

The recommended next step is a **60–90 day controlled Pilot**, not full deployment.