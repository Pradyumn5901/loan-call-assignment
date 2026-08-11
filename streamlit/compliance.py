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
    verification_time = None
    disclosure_events = []
    violation_events = []

    for index, utterance in enumerate(conversation):
        text = utterance["text"]
        stime = float(utterance.get("stime", 0.0))

        if utterance["speaker"] == "Agent" and is_verification_event(text):
            state = "VERIFICATION_REQUESTED"

        if (
            state == "VERIFICATION_REQUESTED"
            and utterance["speaker"] == "Customer"
            and _looks_like_verification_answer(text)
        ):
            state = "VERIFIED"
            verification_index = index
            verification_time = stime

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

    is_violation = len(violation_events) > 0
    return {
        "present": is_violation,
        "violation": is_violation,
        "verification_index": verification_index,
        "verification_time": verification_time,
        "disclosures": disclosure_events,
        "evidence": violation_events if is_violation else [],
    }

