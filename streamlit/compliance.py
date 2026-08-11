"""Temporal identity-verification and disclosure logic."""

from __future__ import annotations

from typing import Any

from data_loader import normalize_text
from rules import is_sensitive_disclosure, is_verification_event


def _looks_like_verification_answer(text: str) -> bool:
    text = normalize_text(text)
    refusal_markers = ("don't know", "do not know", "refuse", "won't", "will not", "can't", "cannot")
    if text.startswith(refusal_markers):
        return False
    # The supplied calls use short direct answers such as a city name, a date,
    # an ID confirmation, or "Yes" after the verification request.
    return bool(text)


def detect_compliance(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    state = "UNVERIFIED"
    verification_index = None
    disclosure_events = []
    violation_events = []

    for index, utterance in enumerate(conversation):
        text = utterance["text"]
        if utterance["speaker"] == "Agent" and is_verification_event(text):
            state = "VERIFICATION_REQUESTED"

        if (
            state == "VERIFICATION_REQUESTED"
            and utterance["speaker"] == "Customer"
            and _looks_like_verification_answer(text)
        ):
            state = "VERIFIED"
            verification_index = index

        if is_sensitive_disclosure(text, utterance["speaker"]):
            event = {
                "utterance_index": index,
                "speaker": utterance["speaker"],
                "text": text,
                "stime": utterance["stime"],
                "etime": utterance["etime"],
                "confidence": 1.0,
            }
            disclosure_events.append(event)
            if state != "VERIFIED":
                violation_events.append(event)

    return {
        "present": bool(violation_events),
        "violation": bool(violation_events),
        "verification_index": verification_index,
        "disclosures": disclosure_events,
        "evidence": violation_events,
    }
