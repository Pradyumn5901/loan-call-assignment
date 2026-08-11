"""Shared formatting and environment helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def model_path(task: str) -> str:
    name = "profanity" if task == "profanity" else "compliance_events"
    filename = f"{name}.joblib"

    streamlit_dir = Path(__file__).parent.resolve()
    module_models_dir = streamlit_dir / "models"

    candidates: list[Path] = []

    env_dir = os.environ.get("ML_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
        candidates.append(streamlit_dir / env_dir)
        candidates.append(Path.cwd() / env_dir)

    candidates.append(module_models_dir)
    candidates.append(Path.cwd() / "streamlit" / "models")
    candidates.append(Path.cwd() / "models")

    for candidate in candidates:
        filepath = candidate / filename
        if filepath.is_file():
            return str(filepath.resolve())

    return str((module_models_dir / filename).resolve())


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
