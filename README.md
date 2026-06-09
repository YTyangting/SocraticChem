
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

```

Each tutoring turn produces an output tuple **⟨Decision, Strategy, Response⟩**:

* **Decision (D)**: INTERCEPT or PROCEED — determined by physics simulation
* **Strategy (S)**: INTERVENTION (for hazards) or SCAFFOLDING (for safe actions)
* **Response (R)**: Natural language guidance grounded in causal traces

### Core Innovation: Oracle-Grounded Tutoring

Unlike text-only tutors that may endorse student hallucinations, SocraticChem:

1. **Executes dry-run simulations** via a Physics Engine before responding
2. **Detects hazards** (pressure buildup, toxic reactions, burns) before they manifest
3. **Generates counterfactual questions** based on causal traces, not just text patterns
4. **Maintains state awareness** through a full physical snapshot (topology, variables, reactions)

---

## SoChemDataset

| Property | Value |
| --- | --- |
| Total Teaching Turns | 15,200+ |
| Chemistry Experiments | 119 |
| Student Personas | 10 |
| Avg. Turns per Session | 12.8 |
| Train Experiments | 108 (1,080 sessions) |
| Test Experiments | 11 (110 sessions) |
| Teacher Source | Multi-Agent LLM + Physics Sim |

### Experiment Coverage

Experiments span 119 middle school chemistry topics defined in XDL (eXperiment Description Language), covering:

* Acid-base indicators and neutralization reactions
* Precipitation and dissolution equilibria
* Combustion and thermal decomposition
* Electrochemistry and redox reactions
* Organic chemistry (esterification, substitution, addition)
* Gas preparation and collection

### Student Personas

Each experiment is simulated with 10 distinct student archetypes to ensure diverse coverage:

| Persona | Traits | Clumsiness | Knowledge |
| --- | --- | --- | --- |
| Standard Novice | Nervous, hesitant | High | Novice |
| Standard Average | Methodical, obedient | Average | Average |
| Standard Expert | Confident, professional | Low | Expert |
| Overconfident-Clumsy | Strong theory, poor hands-on | High | Expert |
| Blindly Confident | Reckless, ignores instructions | Average | Novice |
| Silent | Extremely introverted, passive | Average | Average |
| Curious | Eager but easily distracted | High | Novice |
| Rebellious | Contrarian, challenges authority | Average | Average |
| Careless | Forgetful, ignores details | Average | Average |
| Extremely Anxious | Overly cautious, shaky hands | High | Novice |

### Train/Test Split

The dataset is split at the **experiment topic level** (not by random sample), ensuring that all chemical reactions in the test set are entirely unseen during training. 108 experiments for training, 11 held-out experiments for testing.

---

## SoChem-LLM: Training Details

### Base Model & Fine-tuning

| Hyperparameter | Value |
| --- | --- |
| Base Model | Qwen-2.5-7B-Instruct |
| Fine-tuning Method | LoRA |
| LoRA Rank (r) | 32 |
| LoRA Alpha (α) | 64 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Precision | BF16 (Bfloat16) |
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁴ |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.03 |
| Global Batch Size | 128 |
| Per Device Batch Size | 8 |
| Gradient Accumulation Steps | 4 |
| Max Sequence Length | 4096 |
| Epochs | 3 |
| Weight Decay | 0.01 |
| Max Gradient Norm | 1.0 |
| Hardware | 4 × NVIDIA 4090 GPUs |
| Training Time | ~3.5 hours |

### Structured XML Chain-of-Thought Format

SoChem-LLM is trained with a strict XML-based CoT format that forces explicit reasoning before responding:

```xml
<Input>
  <State> {Liquid: Water (100°C), Container: TestTube (Sealed), Tool: BunsenBurner (On)} </State>
  <LogicDAG> Current_Goal: "Boil Water"; Status: Active </LogicDAG>
  <Student_Action> "I will heat the bottom of the tube now." </Student_Action>
</Input>

<Think>
  <FindNode> Target: Heating_Phase; Status: Matches_Goal </FindNode>
  <Verify>
    <Condition> Heating a sealed container </Condition>
    <Prediction> Pressure will rise > Glass limit </Prediction>
    <Flag> HAZARDOUS </Flag>
    <Trace> Heat → Gas Expansion → Explosion </Trace>
    <Result> INTERCEPT </Result>
  </Verify>
  <Diagnosis> Student ignores pressure constraints. </Diagnosis>
  <Strategy mode="Intervention"> PredictiveQuestioning </Strategy>
</Think>

<Response>
  Stop! Before you apply heat, look at the stopper. What happens to the air pressure inside a sealed tube if you heat it?
</Response>

```

The four reasoning stages in `<Think>`:

1. **`<FindNode>`**: Identify the current task node in the Experimental Logic DAG
2. **`<Verify>`**: Perform "mental simulation" — predict physical consequences and safety flags
3. **`<Diagnosis>`**: Diagnose the student's cognitive error
4. **`<Strategy>`**: Select the teaching strategy (Intervention or Scaffolding)

---

## Project Structure

```text
socChem_final/
├── config.py                              # Global configuration (API keys, model assignments)
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitattributes
│
├── # === Core Simulation Engine ===
├── soc_chem_dia_refactored.py             # Physics engine + multi-agent dialogue (data generation)
├── soc_chem_dia_Interactive_Platform.py   # Interactive platform engine (3D API)
│
├── # === Data Generation ===
├── batch_runner_v2.py                     # Batch data generation runner
├── generate_sft_data.py                   # Raw data → SFT training format
├── convert_script.py                      # Raw data → XML structured format
├── split_dataset_v1.0.py                  # Experiment-level train/test split
├── extract_safety_golden.py               # Extract safety-critical golden test set
├── dataset_validator.py                   # Human annotation quality checker
│
├── # === Model Training ===
├── socratic_finetune_preprocessor.py      # Dialogue data preprocessor
├── finetune_qwen_socratic.py              # Qwen-2.5-7B LoRA fine-tuning
├── run_socratic_finetune.sh               # End-to-end training pipeline
├── example_usage.py                       # Usage examples
│
├── # === Model Inference ===
├── run_vllm_inference_json.py             # vLLM batch inference (fine-tuned model)
├── inference_gpt4.py                      # GPT-4o API inference (baseline)
├── run_ablation_inference.py              # Ablation study inference
│
├── # === Evaluation ===
├── eval_sochem_v25_production.py          # Main evaluation (Node Acc, Safety, ROUGE, BLEU, LLM-judge)
├── eval_full_metrics.py                   # NLP metrics (ROUGE, BLEU, BERTScore)
├── evaluate_safe.py                       # Safety-specific evaluation
├── eval_soclm_v5.py                       # SoChem-LLM evaluation
├── eval_vanilla_sft.py                    # Vanilla SFT evaluation
├── eval_real_data.py                      # Real student deployment evaluation
├── batch_gpt4o_score.py                   # GPT-4o batch scoring
├── calculate_strict_pass.py               # Strict pass rate calculation
├── calculate_safety_precision_matrix.py   # Safety precision/recall/F1
├── correlation_analysis.py                # Human-GPT correlation (Cohen's κ)
├── human_eval_collector.py                # Human evaluation framework
│
├── # === API Server ===
├── api_server.py                          # FastAPI server (Unity 3D integration)
│
├── # === XDL Tools ===
├── xdl_utils.py                           # XDL parser
├── xdl_validator.py                       # XDL validator
├── xdl_pipeline.py                        # XDL processing pipeline
│
├── # === Data Directories ===
├── experiments/                           # 119 XDL experiment definitions
├── database/                              # Chemical databases (substances, reactions, equipment)
├── tools/                                 # XDL processing utilities
├── finetune_data_dataset/                 # 1,190 generated training JSONL files
├── dataset_experiment_split/              # Train/test split (108 train / 11 test)
├── reference/                             # Reference dialogue samples
├── new_test/                              # Model predictions + ablation results
├── final_results/                         # Final evaluation results
├── distribution_shift_test/               # Distribution shift test XDLs
├── distribution_shift_test_result/        # Distribution shift predictions
├── dis_test_result/                       # Distribution shift evaluation
├── session_records/                       # Interactive session logs
│
├── # === Core Data Files ===
├── sft_finetune_chemlab_train_1.json      # SFT training set (144MB)
├── sft_finetune_chemlab_test_noisy.json   # SFT test set (13MB)
├── eval_raw_samples.jsonl                 # Evaluation samples
├── gpt4o_scores_v3.json                   # GPT-4o evaluation scores
├── distribution_shift.json                # Distribution shift test data
└── real_student_session.jsonl             # Real student interaction logs

```

---

## Quick Start

### 1. Environment Setup

```bash
# Python 3.10+
pip install -r requirements.txt

```

Key dependencies:

* `torch >= 2.4.0`
* `transformers >= 4.57.1`
* `peft >= 0.17.1` (for LoRA fine-tuning)
* `vllm >= 0.6.3` (for efficient inference)
* `openai >= 2.14.0` (for GPT-4o baseline)
* `rouge-score`, `nltk`, `bert-score` (for evaluation)
* `fastapi`, `uvicorn` (for API server)

### 2. Data Generation

The data generation pipeline uses a multi-agent LLM framework with physics simulation:

```bash
# Step 1: Generate Socratic dialogue data (requires OpenAI API key)
python batch_runner_v2.py

# Step 2: Convert to SFT training format
python generate_sft_data.py

# Step 3: Split into train/test by experiment type
python split_dataset_v1.0.py

```

### 3. Fine-tuning SoChem-LLM

```bash
# Option A: Run the end-to-end pipeline
bash run_socratic_finetune.sh

# Option B: Run step by step
# 3a. Preprocess dialogue data
python socratic_finetune_preprocessor.py \
  --data_dir ./finetune_data_dataset \
  --output_dir ./processed_data \
  --output_format qwen \
  --analyze

# 3b. Fine-tune Qwen-2.5-7B with LoRA
python finetune_qwen_socratic.py \
  --train_data ./processed_data/socratic_dialogue_qwen_train.jsonl \
  --val_data ./processed_data/socratic_dialogue_qwen_val.jsonl \
  --output_dir ./socratic_model \
  --model_name Qwen/Qwen-2.5-7B-Instruct \
  --epochs 3 \
  --batch_size 8 \
  --use_lora \
  --quantization

```

### 4. Inference

```bash
# Fine-tuned model inference (vLLM, 4-GPU)
python run_vllm_inference_json.py

# GPT-4o baseline inference
python inference_gpt4.py

# Ablation study (remove specific context components)
python run_ablation_inference.py

```

### 5. Evaluation

```bash
# Main evaluation pipeline (Node Acc, Safety, ROUGE, BLEU, LLM-judge)
python eval_sochem_v25_production.py

# NLP metrics (ROUGE, BLEU, BERTScore)
python eval_full_metrics.py

# Safety-specific evaluation
python evaluate_safe.py

# GPT-4o batch scoring
python batch_gpt4o_score.py

```

---

## Evaluation Results

### Main Results

| Model | Node Acc | State Awareness | Intercept Acc | Safety Score | Fidelity | Socratic Score | BLEU-4 | ROUGE-L | BERTScore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SoChem-LLM** | **74.96%** | **70.32%** | **100.00%** | **9.00** | **0.30** | **8.51** | **39.57** | **45.18** | **81.73** |
| GPT-4o | 79.26% | 49.22% | 75.00% | 8.00 | 0.09 | 7.15 | 9.23 | 24.47 | 71.71 |
| Qwen2.5-7B | 74.58% | 41.71% | 66.67% | 7.75 | 0.13 | 6.88 | 7.62 | 23.37 | 71.03 |
| GLM-4-9B | 74.73% | 38.03% | 33.33% | 7.25 | 0.07 | 6.92 | 7.18 | 23.32 | 70.41 |
| Llama-3-8B | 73.66% | 31.80% | 0.00% | 0.00 | 0.00 | 6.50 | 9.48 | 23.40 | 70.43 |
| EduChat-7B | 68.05% | 29.68% | 27.88% | 7.39 | 0.06 | 6.12 | 11.67 | 20.48 | 69.63 |
| ChatGLM3-6B | 48.85% | 31.76% | 8.33% | 8.00 | 0.00 | 5.89 | 3.59 | 18.71 | 68.16 |
| MathDial-1.5B | 13.52% | 24.43% | 0.00% | 0.00 | 0.00 | 2.10 | 2.10 | 5.93 | 60.45 |
| SocraticLM-7B | 0.46% | 0.00% | 0.00% | 0.00 | 0.00 | 1.48 | 1.48 | 5.23 | 56.39 |

### Metrics Explained

| Metric | Type | Description |
| --- | --- | --- |
| **Node Accuracy** | Hard Logic | Whether the predicted DAG node matches ground truth |
| **State Awareness** | Hard Logic | Whether the predicted experiment state (PASS/FAIL) is correct |
| **Intercept Accuracy** | Hard Logic | Whether dangerous actions are correctly intercepted |
| **Safety Score** | LLM-judge (0-10) | Quality of safety intervention (GPT-4o as judge) |
| **Fidelity** | LLM-judge (0-1) | Error attribution accuracy for student mistakes |
| **Socratic Score** | LLM-judge (0-10) | Quality of Socratic pedagogy (guides vs. tells) |
| **BLEU-4** | NLP | Character-level BLEU for Chinese text |
| **ROUGE-L** | NLP | Longest common subsequence recall |
| **BERTScore** | NLP | Semantic similarity (bert-base-chinese) |

### Human-GPT Agreement

GPT-4o as evaluator correlates well with human teachers (8 raters, 100 samples):

* **Safety Score**: Cohen's κ = 0.82 (almost perfect agreement)
* **Socratic Score**: Cohen's κ = 0.69 (substantial agreement)

---

## Ablation Study

To understand the contribution of each input component:

| Variant | Node Acc | State Awareness | Intercept Acc | Safety Score | Socratic Score | BERTScore |
| --- | --- | --- | --- | --- | --- | --- |
| **Full Context** | **74.96%** | **70.32%** | **100.00%** | **9.00** | **8.51** | **81.73** |
| w/o Environment | 70.66% | 58.80% | 91.67% | 9.00 | 8.41 | 80.35 |
| w/o Profile | 70.43% | 66.19% | 75.00% | 9.00 | 8.35 | 81.43 |
| w/o Observation | 78.73% | 69.07% | 66.67% | 9.00 | 8.48 | 81.98 |
| w/o History | 55.84% | 70.84% | 91.67% | 8.27 | 8.30 | 80.73 |
| w/o Logic Chain | 68.13% | 62.45% | 83.33% | 8.50 | 8.22 | 80.01 |

Key findings:

* **Dialogue history** is critical for node tracking (Node Acc drops from 74.96% → 55.84%)
* **Observation logs** are essential for hazard detection (Intercept drops from 100% → 66.67%)
* **Environment state** provides significant grounding for safety (Intercept drops to 91.67%)
* **Student profile** helps with pedagogical personalization

---

## Real-Student Deployment

SoChem-LLM was deployed in a live teaching environment with 10 students across 93 dialogue turns:

| Metric | Score |
| --- | --- |
| Socratic Score (0-10) | 8.9 |
| Safety Score (0-10) | 9.0 |
| Safety Interception (0-5 Likert) | 5.0 |
| Physics Realism (0-5 Likert) | 4.5 |
| Socratic Pedagogy (0-5 Likert) | 4.6 |
| Learning Outcomes (0-5 Likert) | 4.8 |

---

## XDL Experiment Format

Experiments are defined in XDL (eXperiment Description Language), an XML-based format:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<XDL>
  <Metadata title="..." goal="..." difficulty="Easy|Normal|Hard"/>
  <Synthesis>
    <Hardware>
      <Component id="beaker_1" type="beaker" capacity="50ml"/>
      <Component id="test_tube_1" type="test_tube" capacity="50ml"/>
    </Hardware>
    <Reagents>
      <Reagent name="dilute_HCl" state="liquid"/>
      <Reagent name="NaOH_solution" state="liquid"/>
    </Reagents>
    <Procedure>
      <Add vessel="test_tube_1" reagent="dilute_HCl" volume="5 mL"/>
      <Heat vessel="test_tube_1" temperature="100" time="5 min"/>
      <Stir vessel="beaker_1"/>
    </Procedure>
  </Synthesis>
</XDL>

```

Supported procedure actions: `Add`, `Transfer`, `Insert`, `Stir`, `Heat`, `Wait`, `Drain`, `Evaporate`, `Filter`, `Dry`.

---

## Physics Engine

The simulation engine (`soc_chem_dia_refactored.py`) provides:

* **180 chemical reactions** across 19 reaction types (redox, decomposition, neutralization, combustion, etc.)
* **Topology-aware simulation**: tracks fluid paths, detects disconnected equipment
* **Thermodynamic modeling**: heat transfer, boiling, pressure buildup (Ideal Gas Law)
* **Visual simulation**: solution color mixing via Beer-Lambert law (RGB interpolation)
* **Safety checks**: sealed container heating, incompatible reagent mixing, burn risks

---
This project is licensed under [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

```

```
