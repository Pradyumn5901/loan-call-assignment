"""Input loading and validation for loan-collection conversations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"speaker", "text", "stime", "etime"}
VALID_SPEAKERS = {"Agent", "Customer"}


def normalize_text(text: str) -> str:
    """Return a simple normalized representation for matching."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def validate_conversation(conversation: Any) -> list[dict[str, Any]]:
    if not isinstance(conversation, list):
        raise ValueError("Conversation must be a JSON array.")

    cleaned: list[dict[str, Any]] = []
    for index, utterance in enumerate(conversation):
        if not isinstance(utterance, dict) or not REQUIRED_FIELDS.issubset(utterance):
            raise ValueError(f"Utterance {index} is missing required fields.")
        if utterance["speaker"] not in VALID_SPEAKERS:
            raise ValueError(f"Utterance {index} has an invalid speaker.")
        if not isinstance(utterance["text"], str) or not utterance["text"].strip():
            raise ValueError(f"Utterance {index} has empty text.")
        try:
            stime, etime = float(utterance["stime"]), float(utterance["etime"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Utterance {index} has invalid timestamps.") from exc
        if stime > etime:
            raise ValueError(f"Utterance {index} has stime after etime.")
        cleaned.append({**utterance, "stime": stime, "etime": etime})
    return cleaned


def load_conversation(source: str | Path | bytes) -> list[dict[str, Any]]:
    if isinstance(source, bytes):
        payload = json.loads(source.decode("utf-8-sig"))
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    return validate_conversation(payload)


def call_id_from_source(source: str | Path) -> str:
    return Path(source).stem
