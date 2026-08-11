# 🎙️ Loan Collection Conversation Quality & Compliance Analysis

An enterprise-grade AI & Rule-Based Quality Assurance and Compliance Audit System for financial loan collection calls. Built with **Streamlit**, **Scikit-Learn**, and **Hugging Face / llama-cpp-python** (Qwen2.5 LLMs).

- **🌐 Live Interactive Web Application:** [https://loan-call-assignment-6erfrwgzqnzzgf9v7pswpl.streamlit.app/](https://loan-call-assignment-6erfrwgzqnzzgf9v7pswpl.streamlit.app/)

---

## 📌 Deliverables & Executive Summary

This repository evaluates three computational paradigms for real-time compliance and quality assurance in financial loan collection calls:
1. **Rule-Based Engine (Regex + Temporal State Machine)**
2. **Machine Learning Pipeline (TF-IDF + Class-Weighted Logistic Regression Multi-Model Orchestration)**
3. **Large Language Model (Qwen2.5 Serverless Inference API)**

Based on empirical performance, compute cost, latency requirements, and privacy compliance, we recommend a **Hybrid Engine (Rule-Based + ML)** for live production deployment, backed by an **Offline LLM Active Learning Pipeline** for continuous rule discovery.

- **Technical Report (Markdown):** `TECHNICAL_REPORT.md`
- **Technical Report (PDF):** `TECHNICAL_REPORT.pdf`

---

## 📌 Data Preparation & Preprocessing Summary

* **Dataset Scale:** 100 Loan Collection Audio Call Transcripts (790 total dialogue utterances).
* **Structural Validation (`data_loader.py`):** Strictly enforces JSON schema compliance, speaker validation (`Agent`/`Customer`), timestamp ordering (`stime <= etime`), and UTF-8-sig encoding safeguards.
* **Template Grouping:** 11 distinct conversation template families anonymized via entity masking (numbers, names, cities, companies) and SHA-1 hashing to prevent data leakage across splits.
* **Multi-ML Dataset Architecture:** Extracted 3 dedicated binary dataset artifacts (`profanity_training.csv`, `verification_training.csv`, `compliance_events_training.csv`) to train specialized model artifacts (`profanity.joblib`, `verification.joblib`, `compliance_events.joblib`).

| Task / Dataset | Target Model Artifact | Total Turns | Positive Samples | Negative Samples | Class Ratio |
|---|---|---:|---:|---:|---:|
| **Profanity Detection** | `profanity.joblib` | 790 | 62 | 728 | ~1 : 11.7 |
| **Identity Verification** | `verification.joblib` | 790 | 90 | 700 | ~1 : 7.8 |
| **Financial Disclosure** | `compliance_events.joblib` | 790 | 25 | 765 | ~1 : 30.6 |

---

## 📊 Performance Benchmark Matrix (100 Calls / 790 Dialogue Utterances)

### Task 1: Profanity & Abusive Language Detection (Utterance-Level)

| Approach | Accuracy | Precision | Recall | F1 Score | TP | FP | FN | TN | Avg Latency / Call | Compute Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Rule-Based (Regex)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 62 | 0 | 0 | 728 | **< 0.001s** | **$0.00** |
| **Machine Learning (TF-IDF + LogReg)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 62 | 0 | 0 | 728 | **0.002s** | **$0.00** |
| **LLM (Qwen2.5 7B API)** | **0.9342** | **1.0000** | 0.1613 | **0.2778** | 10 | 0 | 52 | 728 | **1.080s** | ~$0.001 / call |
| **LLM (Qwen2.5 0.5B Calibrated)** | **0.9582** | **1.0000** | 0.4677 | **0.6374** | 29 | 0 | 33 | 728 | **0.230s** | ~$0.0002 / call |

### Task 2: Compliance Violation Detection (Identity Verification vs. Financial Disclosure)

| Approach | Accuracy | Precision | Recall | F1 Score | TP | FP | FN | TN | Key Operational Characteristic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Rule-Based (State Machine)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 25 | 0 | 0 | 765 | **Deterministic temporal order enforcement** |
| **Machine Learning (TF-IDF + LogReg)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 25 | 0 | 0 | 765 | **High-speed pattern recognition** |
| **LLM (Qwen2.5 7B API)** | **0.9709** | **1.0000** | 0.0800 | **0.1481** | 2 | 0 | 23 | 765 | Strict audit prompt (Zero False Positives) |

---

## 🛠️ Key Findings & System Analysis

### 1. Strengths & Limits of Rule-Based Regex (State Machine)
* **Strengths:** Lightning-fast (< 1ms per call), zero operational cost, 100% deterministic. The temporal state machine tracks exact call timelines (`verification_time <= disclosure_time`), guaranteeing zero false positives for identity compliance.
* **Limits:** Rigid dictionary matching; misses novel profane slang, typos, or obfuscated swearing.

### 2. Strengths of Machine Learning (TF-IDF + Logistic Regression)
* **Strengths:** Extremely lightweight model size (~60 KB), sub-millisecond execution, zero API token costs. 
* **Generalization Power:** Automatically generalizes to out-of-vocabulary variations, phonetic rephrasings, and typos that bypass exact regex string matching.

### 3. Trade-offs of LLMs (Large Language Models)
* **Overhead in Live Production:** Running 100 calls through an LLM adds **100+ seconds of network latency** (~1s per call) and ongoing token billing.
* **Recall & Calibration Drift:** Zero-shot/few-shot LLMs either suffer from over-triggering (high False Positives in uncalibrated setups) or high conservatism (low Recall in strict setups).

---

## 💡 Production Deployment Recommendation: Hybrid Architecture

The optimal production strategy combines **Live Hybrid Inference** (Regex + ML) with **Offline LLM Active Learning**:

```
 [Live Call Transcript]
          │
          ├──► [1. Rule-Based Regex Engine] ──► Immediate Match (Known Profanity / Direct Disclosure)
          │
          ├──► [2. ML TF-IDF Classifier]   ──► Fuzzy / Novel Profanity Match
          │
          └──► Combined Live Decision (Latency < 5ms | Cost = $0.00)
                   │
                   ▼
 [Offline Asynchronous Batch Job]
          │
          └──► [3. LLM Audit & Mining (Qwen2.5)]
                   │
                   ├──► Flag ambiguous edge cases for human QA review
                   └──► Mine emerging profane slang & non-standard phrases
                         │
                         ▼
             [Update Regex & Retrain ML Models]
```

---

## 📁 Repository Structure

```text
loan-call-assignment/
├── README.md                  # Project overview & quickstart guide
├── TECHNICAL_REPORT.md        # Comprehensive technical report (Markdown)
├── TECHNICAL_REPORT.pdf        # Formatted PDF technical report
└── streamlit/
    ├── app.py                 # Main Streamlit application entry point
    ├── compliance.py          # Temporal state-machine for privacy violation detection
    ├── rules.py               # Regex patterns & rule-based detection logic
    ├── ml_pipeline.py         # Joblib classifier loading & prediction helpers
    ├── llm_pipeline.py        # LLM prompt engineering & structured JSON generation
    ├── load_llm.py            # Hugging Face Serverless API & GGUF model loader
    ├── data_loader.py         # JSON transcript parsing & schema validation
    ├── utils.py               # Robust model path resolution & evidence formatters
    ├── requirements.txt       # Dependencies for Streamlit Cloud & Linux deployment
    ├── pyproject.toml         # Project metadata & dependency definitions
    └── models/                # Pre-trained ML artifacts (.joblib)
        ├── profanity.joblib
        ├── compliance_events.joblib
        └── verification.joblib
```

---

## 🚀 Quick Start & Local Setup

```bash
git clone <repository_url>
cd loan-call-assignment/streamlit
pip install -r requirements.txt
streamlit run app.py
```