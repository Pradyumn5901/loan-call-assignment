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


def format_evidence(evidence: list[Any]) -> str:
    if not evidence:
        return "No supporting snippet found."
    formatted = []
    for item in evidence:
        if isinstance(item, dict):
            stime = item.get("stime", 0.0)
            speaker = item.get("speaker", "")
            text = item.get("text", "")
            prefix = f"[{stime:.1f}s] " if stime else ""
            prefix += f"{speaker}: " if speaker else ""
            formatted.append(f"{prefix}{text}")
        elif isinstance(item, str):
            formatted.append(item)
    return "\n".join(formatted) if formatted else "No supporting snippet found."
