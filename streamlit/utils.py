"""Shared formatting and environment helpers."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_ML_MODEL_DIR: str = os.environ.get("ML_MODEL_DIR", "models")


def model_path(task: str) -> str:
    name = "profanity" if task == "profanity" else "compliance_events"
    return os.path.join(_ML_MODEL_DIR, f"{name}.joblib")


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "No supporting snippet found."
    return "\n".join(
        f"[{item.get('stime', 0):.1f}s] {item.get('speaker', '')}: {item.get('text', '')}"
        for item in evidence
    )
