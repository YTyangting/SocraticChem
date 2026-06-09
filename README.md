# SocraticChem: Physics-Grounded Socratic Inquiry for Safety-Critical Experimental Science

<p align="center">
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License: CC BY 4.0"></a>
</p>


---

## Abstract

Large language models (LLMs) have emerged as a foundational technology for intelligent education. However, in safety-critical domains like chemistry experiments, current general models face a fundamental **pedagogical paradox**: effective inquiry requires students to learn from errors, but the physical world imposes strict safety constraints where certain errors are impermissible (e.g., mixing reagents incorrectly could cause explosions). Moreover, existing LLM-based tutors typically prioritize textual plausibility over physical reality, leading to **"Hallucinated Pedagogy"** when applied to real-world experiments.

To address this paradox, we propose **SocraticChem**, a physics-grounded framework that formalizes tutoring not as open-ended generation, but as a verifiable, safety-aware decision policy. SocraticChem guides students toward learning objectives while strictly intercepting potentially dangerous actions before they manifest physically, enabling error-driven learning without physical risk.

We construct **SoChemDataset** (15.2K physically grounded teaching turns across 119 middle school chemistry experiments), fine-tune **SoChem-LLM** on this dataset, and establish a comprehensive evaluation suite spanning physics-grounded verification, LLM-based assessment, and standard NLP benchmarks.

**Key Results**: SoChem-LLM achieves 100% safety interception accuracy, State Awareness of 70.32% (vs. GPT-4o's 49.22%), and Safety Score of 9.00 (vs. best baseline of 8.00).

---

## Framework Overview

SocraticChem formalizes the tutoring process as a **safety-aware decision policy** governed by the Turn-Level Contract:

```text
Student Proposal (a_stu)
        │
        ▼
┌─────────────────────────┐
│   Physics Oracle        │
│   Dry-run Simulation    │
│   (S_t, a_stu → S'_t+1) │
│                         │
│   Oracle Report:        │
│   ⟨Flag, Causal Trace⟩  │
└────────┬────────────────┘
         │
    ┌────▼────┐
    │ Decision│
    │  Gate   │
    └─┬────┬──┘
      │    │
  HAZARDOUS  SAFE
      │    │
      ▼    ▼
┌─────────┐  ┌──────────────┐
│INTERCEPT│  │   PROCEED    │
│ S_hold  │  │  S_advance   │
│Interven-│  │ Scaffolding  │
│  tion   │  │              │
└─────────┘  └──────────────┘
