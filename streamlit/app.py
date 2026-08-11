"""Streamlit entry point for the assignment demo."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # must run before any module reads os.environ

import streamlit as st

from compliance import detect_compliance
from data_loader import load_conversation
from rules import detect_profanity
from utils import format_evidence, model_path


def main():
    st.set_page_config(page_title="Loan Collection Analysis", layout="wide")
    st.title("Loan Collection Conversation Analysis")

    uploaded = st.file_uploader("Upload a conversation JSON file", type=["json"])
    task = st.selectbox("Detection task", ["Profanity Detection", "Privacy & Compliance Violation"])
    approach = st.selectbox("Detection approach", ["Rule-Based (Regex)", "Machine Learning", "LLM"])

    if uploaded and st.button("Analyze"):
        try:
            conversation = load_conversation(uploaded.getvalue())
            if approach == "Rule-Based (Regex)":
                result = detect_profanity(conversation) if task.startswith("Profanity") else detect_compliance(conversation)
            elif approach == "Machine Learning":
                from ml_pipeline import load_artifact, predict_conversation
                from utils import model_path

                task_key = "profanity" if task.startswith("Profanity") else "compliance_events"
                artifact = load_artifact(model_path(task_key))
                preds = predict_conversation(conversation, artifact)

                if task.startswith("Profanity"):
                    evidence = [p for p in preds if p["label"]]
                    result = {
                        "present": bool(evidence),
                        "agent_detected": any(p["speaker"] == "Agent" for p in evidence),
                        "customer_detected": any(p["speaker"] == "Customer" for p in evidence),
                        "evidence": evidence,
                    }
                else:
                    evidence = [p for p in preds if p["label"]]
                    result = {
                        "present": bool(evidence),
                        "violation": bool(evidence),
                        "evidence": evidence,
                        "confidence": max((p["confidence"] for p in evidence), default=0.0),
                    }
            elif approach == "LLM":
                from llm_pipeline import classify_with_llm

                llm_task = "profanity" if task.startswith("Profanity") else "compliance"
                result = classify_with_llm(conversation, task=llm_task)
            else:
                result = None

            if result is not None:
                st.metric("Detection result", "Present" if result.get("present") else "Not Present")
                if "agent_detected" in result:
                    st.write({"Agent": result["agent_detected"], "Customer": result["customer_detected"]})
                st.subheader("Supporting snippets")
                st.code(format_evidence(result.get("evidence", [])))
                with st.expander("Structured result"):
                    st.json(result)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    import sys
    if st.runtime.exists():
        main()
    else:
        try:
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
            sys.exit(stcli.main())
        except Exception:
            print("Please run this app using: streamlit run app.py")
            sys.exit(1)
else:
    if st.runtime.exists():
        main()

