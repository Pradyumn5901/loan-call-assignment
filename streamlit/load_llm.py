"""Hugging Face Inference API client wrapper.

Switched to Hugging Face Serverless Inference API for cloud stability,
zero local RAM consumption, and fast response times.

(All local llama-cpp-python GGUF code is commented out below for reference).
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# Default model hosted on Hugging Face Serverless API
DEFAULT_HF_MODEL: str = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")


def _get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
                token = st.secrets["HF_TOKEN"]
            elif hasattr(st, "secrets") and "HUGGINGFACE_TOKEN" in st.secrets:
                token = st.secrets["HUGGINGFACE_TOKEN"]
        except Exception:
            pass
    return token


def get_hf_client(model: str | None = None) -> InferenceClient:
    """Return a Hugging Face InferenceClient instance."""
    target_model = model or DEFAULT_HF_MODEL
    token = _get_hf_token()
    return InferenceClient(model=target_model, token=token)

