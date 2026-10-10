# Taste Agent — EU AI Act Compliance Documentation

**Client Case:** Letterboxd  
**Project:** AI-Powered Taste Intelligence for Film Discovery  
**Scope:** Working MVP → Proposed Pilot  
**Jurisdiction:** European Union

---

## 1. Intended Purpose

Taste Agent is intended to provide **personalized entertainment discovery**.

It interprets a user's film request, combines that request with non-consequential preference signals such as watch history and watchlist intent, retrieves candidate films, ranks suitable options and generates grounded recommendation explanations.

It is not intended to determine access to employment, education, credit, insurance, public services, law enforcement, migration, justice or other consequential opportunities.

---

## 2. Risk Classification

### Step 1 — Prohibited practices

The intended system does not require:

- social scoring
- prohibited manipulation
- biometric categorization
- prohibited biometric identification
- predictive policing
- emotion recognition in prohibited contexts

**Assessment:** not a prohibited AI practice under the proposed design.

### Step 2 — High-risk assessment

Taste Agent is an entertainment recommendation system.

Its recommendations do not determine legal rights or access to essential opportunities, and the intended purpose does not fall within the high-risk domains relevant to employment, education, essential services, law enforcement, migration, justice or democratic participation.

**Assessment:** not classified as a high-risk AI system under the current intended purpose.

### Step 3 — Transparency obligations

Taste Agent includes a user-facing conversational AI interaction.

Users should therefore be clearly informed that they are interacting with an AI system and that recommendations are AI-assisted and may be imperfect.

**Assessment:** transparency obligations are relevant even though the system is not high-risk.

---

## 3. Classification Conclusion

```text
Prohibited AI Practice
      NO

High-Risk AI System
      NO under the current intended purpose

User-Facing AI Transparency
      YES

GPAI Model Provider Obligations
      Primarily the underlying model provider,
      subject to final architecture and contractual roles
```

The classification must be reassessed if Taste Agent or its underlying profile is later reused for materially different or consequential purposes.

---

## 4. Applicable Controls

Although the system is not classified as high-risk, the project applies governance controls that are appropriate to the product risk:

- clear AI disclosure
- restricted intended purpose
- no sensitive personal-attribute inference
- grounded recommendation explanations
- structured outputs where AI output enters deterministic processing
- hard constraints and safe rejection
- abstention when the request cannot be supported
- evaluation datasets and regression testing
- LangSmith tracing and evaluator evidence
- user-facing distinction between observed behaviour, inferred taste evidence and generated explanation

---

## 5. Transparency Design

A production interface should disclose, in plain language, that:

- Taste Agent is AI-powered
- recommendations use the user's request and preference signals
- AI-generated outputs may be imperfect
- the system is designed to model cultural preference rather than personality or identity

A suitable interaction-level notice would be:

> **Taste Agent is an AI-powered discovery assistant. Its recommendations use your film activity and current request and may not always be accurate.**

The product should avoid unsupported claims such as:

> "You are an introspective person."

and prefer evidence-based explanations such as:

> "You have historically rated contemplative, character-driven films more highly."

---

## 6. GPAI / Third-Party Model Responsibilities

Taste Agent uses third-party AI models through APIs.

The provider of the underlying general-purpose AI model has its own obligations. Letterboxd remains responsible for the application it deploys, its intended purpose, user experience, data processing, monitoring and provider governance.

Before production Pilot, vendor review should cover:

- AI Act documentation and contractual role
- data-processing terms
- security controls
- data retention
- training-on-customer-data policy
- model-change policy
- geographic processing and sub-processors

---

# Conformity Assessment Summary

## 7. Summary

**System:** Taste Agent  
**Intended purpose:** personalized film discovery and explanation  
**Current classification:** non-high-risk entertainment recommendation system with user-facing AI transparency requirements  
**High-risk conformity assessment required:** no, under the present intended purpose

The project nevertheless maintains a proportional governance package because the system performs preference profiling and uses generative AI.

### Evidence available

- documented intended purpose and scope boundaries
- fixed or controlled semantic representations
- architecture separating request relevance, historical compatibility and explanation
- evaluation datasets and automated evaluators
- hosted experiment evidence and human review
- runtime regression checks
- privacy and sensitive-inference guardrails
- documented limitations and known failure modes

### Reassessment triggers

A new AI Act assessment is required if:

- the system is reused for consequential decisions
- sensitive personal-attribute inference becomes an intended function
- biometric or emotion-recognition components are introduced
- the deployment role changes materially
- the underlying model is substantially modified in a way that changes provider obligations

---

# Technical Documentation Outline

## 8. Production Technical Documentation Skeleton

1. **System identification**
   - product name
   - version
   - owner
   - intended purpose

2. **System architecture**
   - request understanding
   - candidate retrieval
   - hard constraints
   - semantic qualification
   - historical compatibility
   - selection
   - explanation
   - validation / abstention

3. **Models and third-party services**
   - model names and versions
   - embedding providers
   - metadata providers
   - processor roles

4. **Data sources**
   - Letterboxd first-party data
   - optional external data
   - metadata sources
   - retention boundaries

5. **Taste and semantic representations**
   - 62D taxonomy
   - historical compatibility model
   - limits of interpretation

6. **Evaluation**
   - datasets
   - LangSmith traces
   - evaluators
   - human review
   - regression tests
   - known limitations

7. **User transparency and controls**
   - AI disclosure
   - explanation design
   - correction / reset / disable controls

8. **Risk controls**
   - sensitive-inference guardrails
   - hard constraints
   - abstention
   - hallucination / grounding controls

9. **Performance and monitoring**
   - latency
   - cost
   - failure rates
   - recommendation-quality KPIs
   - drift / regression monitoring

10. **Security and vendor governance**
    - access controls
    - secrets
    - processor review
    - incident handling

11. **Change management**
    - model changes
    - architecture changes
    - re-evaluation triggers

12. **Compliance reassessment**
    - AI Act classification triggers
    - GDPR / DPIA review triggers

---

## 9. Compliance Position

Taste Agent can progress to a controlled Pilot under the current intended purpose, provided that transparency, privacy, evaluation and vendor-governance controls remain part of the product architecture.

This assessment is a project-level compliance analysis and should be validated by qualified legal/privacy teams before real production deployment.