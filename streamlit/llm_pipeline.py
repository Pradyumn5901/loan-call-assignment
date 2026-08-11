"""LLM inference pipeline — optimized for Qwen2.5 and all GGUF models via llama-cpp-python.

Supports any model via llama-cpp-python:
  Qwen2.5-3B-Instruct, TinyLlama, Phi-3, Mistral-7B, etc.

Uses llama_cpp.create_chat_completion for native ChatML/Instruct prompt formatting,
returning structured JSON for profanity and compliance analysis.
"""

from __future__ import annotations

import json
import re
from typing import Any

from load_llm import DEFAULT_MAX_TOKENS, get_llm


# ── System and User Prompt Templates ──────────────────────────────────────────

_PROFANITY_SYSTEM = """\
You are an expert AI quality assurance auditor for financial loan collection calls.

Task:
Analyze the provided transcript to detect any profanity, vulgarity, insults, or abusive language used by either the Agent or the Customer.

Instructions:
1. Examine every utterance in the transcript.
2. Determine if profane or abusive language is present.
3. Identify whether the Agent, the Customer, or both used profane/abusive language.
4. Extract the exact text of any abusive utterances as evidence.
5. Return ONLY a valid JSON object matching the JSON schema below. Do not include markdown codeblocks, commentary, or extra text.
"""

_PROFANITY_USER = """\
Output JSON Schema:
{{
  "present": boolean,             // true if any profanity/abuse was detected, else false
  "agent_detected": boolean,      // true if the Agent used profanity/abuse, else false
  "customer_detected": boolean,   // true if the Customer used profanity/abuse, else false
  "evidence": [                   // array of offending utterances (empty if none)
    {{
      "speaker": string,          // "Agent" or "Customer"
      "text": string              // exact text of the utterance
    }}
  ]
}}

Transcript to analyze:
{transcript}
"""

_COMPLIANCE_SYSTEM = """\
You are an expert financial compliance auditor evaluating debt collection call transcripts under privacy guidelines.

Task:
Determine whether the Agent violated customer privacy by disclosing sensitive financial information (such as EMI amount, outstanding balance, or default status) BEFORE verifying the customer's identity (such as asking for Date of Birth, last 4 digits of Account/Aadhaar number, or Customer ID).

Instructions:
1. Review the chronological order of utterances and their start times (stime).
2. Identify the timestamp (stime) of Identity Verification (Agent asking for ID/DOB and Customer confirming).
3. Identify the timestamp (stime) of Financial Disclosure (Agent revealing EMI or dues amount).
4. If Financial Disclosure occurred BEFORE Identity Verification (disclosure_time < verification_time) or if verification never occurred, flag it as a PRIVACY VIOLATION.
5. Return ONLY a valid JSON object matching the JSON schema below. Do not include markdown codeblocks or commentary.
"""

_COMPLIANCE_USER = """\
Output JSON Schema:
{{
  "present": boolean,             // true if a privacy violation occurred, else false
  "violation": boolean,           // true if financial disclosure occurred before identity verification, else false
  "verification_time": float,     // stime of identity verification turn (null if not verified)
  "disclosure_time": float,       // stime of financial disclosure turn (null if no disclosure)
  "evidence": [                   // array of premature disclosure events (empty if compliant)
    {{
      "event": "unverified_disclosure",
      "speaker": "Agent",
      "text": string,             // exact text of the premature disclosure
      "stime": float,             // start timestamp in seconds
      "etime": float              // end timestamp in seconds
    }}
  ]
}}

Transcript to analyze:
{transcript}
"""



# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_transcript(conversation: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{u['stime']:.1f}s] {u['speaker']}: {u['text']}"
        for u in conversation
    )


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from LLM response text."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM response:\n{raw}")
    return json.loads(match.group())


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_with_llm(
    conversation: list[dict[str, Any]],
    task: str,
    model_path: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run *task* against *conversation* using Qwen2.5 / GGUF model via llama-cpp-python.

    Parameters
    ----------
    conversation : list[dict]
        Validated utterance list from data_loader.
    task : str
        ``"profanity"`` or ``"compliance"``.
    model_path : str | None
        Path to a .gguf file. Falls back to load_llm.DEFAULT_MODEL_PATH.
    max_tokens : int | None
        Max tokens for the model reply. Falls back to LLM_MAX_TOKENS in .env.
    temperature : float
        Sampling temperature (0 = deterministic).
    """
    llm = get_llm(model_path)
    transcript = _format_transcript(conversation)
    tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

    if task == "profanity":
        system_prompt = _PROFANITY_SYSTEM
        user_prompt = _PROFANITY_USER.format(transcript=transcript)
    else:
        system_prompt = _COMPLIANCE_SYSTEM
        user_prompt = _COMPLIANCE_USER.format(transcript=transcript)

    # Use llama_cpp chat completion (supports ChatML / Qwen2.5 templates natively)
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=tokens,
            temperature=temperature,
        )
        text = response["choices"][0]["message"]["content"]
    except Exception:
        # Fallback to direct prompt completion
        prompt = f"{system_prompt}\n\n{user_prompt}"
        raw = llm(prompt, max_tokens=tokens, temperature=temperature)
        text = raw["choices"][0]["text"] if isinstance(raw, dict) else str(raw)

    return _extract_json(text)
