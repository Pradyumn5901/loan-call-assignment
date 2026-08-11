"""Transparent regex/pattern detectors."""

from __future__ import annotations

import re
from typing import Any

from data_loader import normalize_text


PROFANITY_PATTERNS = {
    # ── existing ──────────────────────────────────────────────────────────────
    "damn": re.compile(r"\bdamn\b", re.I),
    "freaking": re.compile(r"\bfreaking\b", re.I),
    "bullshit": re.compile(r"\bbullshit\b", re.I),
    "asshole": re.compile(r"\bassholes?\b", re.I),
    "idiot": re.compile(r"\bidiots?\b", re.I),
    "bloody idiot": re.compile(r"\bbloody\s+idiot\b", re.I),
    "shut up": re.compile(r"\bshut\s+up\b", re.I),
    "go to hell": re.compile(r"\bgo\s+to\s+hell\b", re.I),
    "what the hell": re.compile(r"\bwhat\s+the\s+hell\b", re.I),
    "whatever the hell": re.compile(r"\bwhatever\s+the\s+hell\b", re.I),
    # ── added from ground-truth review of all 100 calls ───────────────────────
    "fuck": re.compile(r"\bf+[u*]+ck\w*\b", re.I),
    "for fuck's sake": re.compile(r"for\s+fuck'?s?\s+sake", re.I),
    "piece of shit": re.compile(r"piece\s+of\s+shit", re.I),
    "shit": re.compile(r"\bsh[i*!1]+t\b", re.I),
    "schedule my ass": re.compile(r"schedule\s+my\s+ass", re.I),
    "my ass": re.compile(r"\bmy\s+ass\b", re.I),
    "you assholes": re.compile(r"you\s+assholes?", re.I),
    "you idiots": re.compile(r"you\s+idiots?", re.I),
    "you damn fools": re.compile(r"you\s+damn\s+fools?", re.I),
    "why the hell": re.compile(r"why\s+the\s+(hell|heck)\b", re.I),
    "hell alone": re.compile(r"hell\s+alone\b", re.I),
    "bloody number": re.compile(r"bloody\s+number\b", re.I),
    "bloody": re.compile(r"\bbloody\b", re.I),
    "hell": re.compile(r"\bhell\b", re.I),
}

VERIFICATION_PATTERNS = [
    re.compile(r"\bdate of birth\b|\bdob\b", re.I),
    re.compile(r"\b(last four|four digits)\b.*\b(account|aadhaar|aadhar|id)\b", re.I),
    re.compile(r"\b(customer id|account number|aadhaar|aadhar|ssn)\b", re.I),
    re.compile(r"\bverify your identity\b|\bconfirm your identity\b", re.I),
]

DISCLOSURE_PATTERNS = [
    re.compile(r"\b(owe|owes|owed|dues?|outstanding|balance|unpaid amount)\b.*\brupees?\b", re.I),
    re.compile(r"\b(emi|payment|amount)\b.*\brupees?\b", re.I),
    re.compile(r"\b(account|loan|emi|payment)\b.*\b(default|overdue|bounced|failed|pending|unpaid)\b", re.I),
]


def detect_profanity(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = []
    for index, utterance in enumerate(conversation):
        text = normalize_text(utterance["text"])
        matches = [name for name, pattern in PROFANITY_PATTERNS.items() if pattern.search(text)]
        if matches:
            evidence.append({
                "utterance_index": index,
                "speaker": utterance["speaker"],
                "text": utterance["text"],
                "stime": utterance["stime"],
                "etime": utterance["etime"],
                "matches": matches,
                "confidence": 1.0,
            })
    return {
        "present": bool(evidence),
        "agent_detected": any(x["speaker"] == "Agent" for x in evidence),
        "customer_detected": any(x["speaker"] == "Customer" for x in evidence),
        "evidence": evidence,
    }


def is_verification_event(text: str) -> bool:
    return any(pattern.search(text) for pattern in VERIFICATION_PATTERNS)


def is_sensitive_disclosure(text: str, speaker: str) -> bool:
    return speaker == "Agent" and any(pattern.search(text) for pattern in DISCLOSURE_PATTERNS)
