# 🎙️ Loan Collection Conversation Analysis

An AI & Rule-based Quality Assurance and Compliance Audit System for financial loan collection calls. Built with **Streamlit**, **Scikit-Learn**, and **llama-cpp-python** (Local GGUF LLMs).

---

## 📌 Overview

During debt collection calls, regulatory frameworks require strict adherence to customer privacy standards and call conduct rules. This application analyzes JSON call transcripts to evaluate compliance across two core detection tasks using **three distinct technological approaches**.

### Core Detection Tasks

1. **🤬 Profanity & Abuse Detection:** Identifies vulgar, offensive, or abusive language used by either the **Agent** or the **Customer**, returning speaker identification and exact evidence snippets.
2. **⚖️ Privacy & Compliance Violation:** Detects temporal privacy breaches—specifically flagging instances where an Agent discloses sensitive financial details (e.g., loan balance, EMI dues) **before** verifying the customer's identity (e.g., DOB, account digits).

---

## 🛠️ Detection Approaches

| Approach | Engine | Characteristics & Performance |
| :--- | :--- | :--- |
| **Rule-Based (Regex)** | Pattern matching & State Machine | **Instant (< 10ms)**, 100% deterministic, zero memory overhead. Tracks temporal turn state (`UNVERIFIED` ➔ `VERIFICATION_REQUESTED` ➔ `VERIFIED`). |
| **Machine Learning** | TF-IDF + Scikit-Learn Classifiers | **Ultra-fast (< 50ms)**, probabilistic scoring per utterance using pre-trained `.joblib` model artifacts. |
| **Local LLM (GGUF)** | `llama-cpp-python` (Qwen2.5-1.5B / 3B GGUF) | **Context-aware (3–8s)**, zero-shot structured JSON inference running 100% locally on CPU without external API keys. |

---

## 📥 Expected Input JSON Schema

The application accepts call transcript `.json` files structured as an array of chronological utterances:

```json
[
  {
    "speaker": "Agent",
    "text": "Hello, am I speaking with Mr. Sharma?",
    "stime": 0.0,
    "etime": 2.5
  },
  {
    "speaker": "Customer",
    "text": "Yes, this is he.",
    "stime": 2.8,
    "etime": 4.1
  },
  {
    "speaker": "Agent",
    "text": "Could you please confirm your date of birth for verification?",
    "stime": 4.5,
    "etime": 7.0
  }
]
```

---

## 📁 Repository Structure

```text
loan-call-assignment/
├── README.md                  # Project overview & documentation
└── streamlit/
    ├── app.py                 # Main Streamlit web application entry point
    ├── compliance.py          # Temporal state-machine for privacy violation detection
    ├── rules.py               # Regex patterns & rule-based detection logic
    ├── ml_pipeline.py         # Joblib classifier loading & prediction helpers
    ├── llm_pipeline.py        # LLM prompt templates & structured JSON generation
    ├── load_llm.py            # Local GGUF model downloader & memory loader
    ├── data_loader.py         # JSON transcript parsing & schema validation
    ├── utils.py               # Result formatting & utility functions
    ├── requirements.txt       # Dependencies for Streamlit Cloud & Linux deployment
    ├── pyproject.toml         # Project metadata & dependency definitions
    └── models/                # ML artifacts (.joblib) & GGUF model binaries (.gguf)
```

---

## 🚀 Quick Start & Local Setup

### Prerequisites
* **Python 3.12** installed
* Optional: [`uv`](https://github.com/astral-sh/uv) (fast package manager) or standard `pip`

### 1. Clone & Navigate
```bash
cd loan-call-assignment/streamlit
```

### 2. Install Dependencies

Using `pip`:
```bash
pip install -r requirements.txt
```

Or using `uv`:
```bash
uv sync
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
*(Or via uv: `uv run app.py`)*

The web app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. **Push Code to GitHub**: Ensure `streamlit/requirements.txt` is present in your repo.
2. **Deploy on Streamlit**: Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. **Repository Settings**:
   * **Repository:** `YourUsername/your-repo-name`
   * **Main file path:** `streamlit/app.py`
4. Click **Deploy!**

> [!NOTE]
> On initial boot, if no GGUF model file is present in `streamlit/models/`, the application will automatically download **`Qwen2.5-1.5B-Instruct-GGUF`** from Hugging Face into memory.

---

## 💻 Hardware & Memory Footprint

* **RAM Requirement:** 8 GB RAM (Runs comfortably using ~1.0–1.5 GB RAM for local LLM inference).
* **LLM Inference Speed on CPU:** ~20–40 tokens/sec (~3–8 seconds per transcript JSON).
* **GPU Requirement:** None (Uses `llama.cpp` CPU thread optimization).