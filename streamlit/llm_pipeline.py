"""LLM inference pipeline using Hugging Face Serverless Inference API.

Uses Hugging Face's hosted high-speed cloud inference (e.g. Qwen2.5-72B-Instruct)
for zero local memory consumption, fast response times (~1-2s), and high accuracy.

(All local llama-cpp-python code is commented out below).
"""

from __future__ import annotations

import json
import re
from typing import Any

from load_llm import get_hf_client


# ── System and User Prompt Templates ──────────────────────────────────────────

_PROFANITY_SYSTEM = """\
You are an expert AI quality assurance auditor for financial loan collection calls.

Task:
Analyze the provided transcript to detect ONLY EXPLICIT profanity, curse words, vulgarities, or abusive insults used by either the Agent or the Customer.

STRICT PROFANITY RULES:
1. ONLY flag PROFANITY ("present": true) if EXPLICIT curse words, swear words, vulgarities, or abusive insults are present (e.g., "damn", "bloody", "bitch", "bastard", "idiot", "fool", "fuck", "shit", "hell").
2. Sarcasm, jokes, dismissive phrases, or idioms (e.g., "go fly a kite", "paperwork is a mess", "mind your own business", "whatever", "take a hike") are NOT profanity ("present": false, "evidence": []).
3. Normal collection terms (e.g., "overdue", "pay your dues", "default", "call back", "legal notice", "EMI") are NOT profanity ("present": false, "evidence": []).
4. Extract exact offending utterances into the "evidence" array ONLY if explicit profane or abusive words exist.

Return ONLY a valid JSON object. Do not include markdown codeblocks or extra text.
"""

_PROFANITY_USER = """\
Output JSON Schema:
{{
  "present": <boolean>,
  "agent_detected": <boolean>,
  "customer_detected": <boolean>,
  "evidence": [
    {{
      "speaker": "<Agent_or_Customer>",
      "text": "<exact_abusive_text>"
    }}
  ]
}}

Transcript to analyze:
{transcript}
"""

_COMPLIANCE_SYSTEM = """\
You are an expert financial compliance auditor evaluating debt collection call transcripts under privacy guidelines.

Task:
Determine whether the Agent violated customer privacy by disclosing sensitive financial information (such as EMI amount, outstanding balance, or default status) BEFORE verifying the customer's identity (asking DOB, last 4 digits of Account/Aadhaar number, or Customer ID).

STRICT CLASSIFICATION RULES:
1. If identity verification occurs BEFORE financial disclosure (verification_time <= disclosure_time), NO VIOLATION occurred.
2. If financial disclosure NEVER occurred, NO VIOLATION occurred.
3. Only flag a violation if financial details (e.g., specific dues amount/balance) were revealed BEFORE identity was verified.

Return ONLY a valid JSON object. Do not include markdown codeblocks.
"""

_COMPLIANCE_USER = """\
Output JSON Schema:
{{
  "present": <boolean>,
  "violation": <boolean>,
  "verification_time": <number_or_null>,
  "disclosure_time": <number_or_null>,
  "evidence": [
    {{
      "speaker": "Agent",
      "text": "<exact_premature_disclosure_text>"
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


def _enrich_evidence(
    evidence_list: list[Any],
    conversation: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, bool]:
    """Match LLM evidence items against transcript to resolve speaker and timestamps."""
    enriched: list[dict[str, Any]] = []
    agent_detected = False
    customer_detected = False

    for item in evidence_list:
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            speaker = str(item.get("speaker", ""))
        elif isinstance(item, str):
            text = item
            speaker = ""
        else:
            continue

        if not text.strip():
            continue

        # Match text against conversation transcript to find speaker & timestamps
        match_u = None
        norm_target = text.casefold().strip()
        for u in conversation:
            u_text = str(u.get("text", "")).casefold().strip()
            if norm_target in u_text or u_text in norm_target:
                match_u = u
                break

        if match_u:
            resolved_speaker = match_u.get("speaker", speaker)
            stime = float(match_u.get("stime", 0.0))
            etime = float(match_u.get("etime", 0.0))
            full_text = str(match_u.get("text", text))
        else:
            resolved_speaker = speaker or "Agent"
            stime = 0.0
            etime = 0.0
            full_text = text

        if resolved_speaker == "Agent":
            agent_detected = True
        elif resolved_speaker == "Customer":
            customer_detected = True

        enriched.append({
            "speaker": resolved_speaker,
            "text": full_text,
            "stime": stime,
            "etime": etime,
        })

    return enriched, agent_detected, customer_detected


def _extract_json(
    raw: str,
    conversation: list[dict[str, Any]],
    task: str = "profanity",
) -> dict[str, Any]:
    """Extract and parse the JSON object from LLM response text with transcript grounding."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    data = None
    if match:
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    if not isinstance(data, dict):
        match_greedy = re.search(r"\{.*\}", raw, re.DOTALL)
        if match_greedy:
            try:
                data = json.loads(match_greedy.group())
            except json.JSONDecodeError:
                pass

    if not isinstance(data, dict):
        if task == "profanity":
            return {"present": False, "agent_detected": False, "customer_detected": False, "evidence": []}
        return {"present": False, "violation": False, "evidence": [], "verification_time": None, "disclosure_time": None}


    raw_evidence = data.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []

    # Enrich evidence items with original transcript speakers & timestamps
    enriched_evidence, match_agent, match_customer = _enrich_evidence(raw_evidence, conversation)
    
    # Determine presence logic safely
    llm_present = bool(data.get("present")) or bool(data.get("violation"))
    # For compliance tasks, deterministically enforce temporal order:
    # If verification_time <= disclosure_time, NO violation occurred!
    if task == "compliance":
        ver_time = data.get("verification_time")
        disc_time = data.get("disclosure_time")
        if ver_time is not None and disc_time is not None:
            try:
                v_t = float(ver_time)
                d_t = float(disc_time)
                if v_t <= d_t:
                    data["present"] = False
                    data["violation"] = False
                    data["evidence"] = []
                    return data
            except (TypeError, ValueError):
                pass

    has_evidence = len(enriched_evidence) > 0
    final_present = llm_present or has_evidence

    data["evidence"] = enriched_evidence
    data["present"] = final_present

    if task == "profanity":
        data["agent_detected"] = bool(data.get("agent_detected")) or match_agent
        data["customer_detected"] = bool(data.get("customer_detected")) or match_customer
    else:
        data["violation"] = final_present

    return data




# ── Public API ─────────────────────────────────────────────────────────────────

def classify_with_llm(
    conversation: list[dict[str, Any]],
    task: str,
    model_path: str | None = None,
    max_tokens: int | None = 512,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run *task* against *conversation* using Hugging Face Inference API."""
    transcript = _format_transcript(conversation)
    tokens = max_tokens or 512

    if task == "profanity":
        system_prompt = _PROFANITY_SYSTEM
        user_prompt = _PROFANITY_USER.format(transcript=transcript)
    else:
        system_prompt = _COMPLIANCE_SYSTEM
        user_prompt = _COMPLIANCE_USER.format(transcript=transcript)

    client = get_hf_client()

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content
    except Exception as exc:
        # Fallback to secondary model if primary HuggingFace model endpoint is busy
        try:
            fallback_client = get_hf_client(model="Qwen/Qwen2.5-0.5B-Instruct")
            response = fallback_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
        except Exception:
            raise RuntimeError(f"Hugging Face Inference API error: {exc}") from exc

    return _extract_json(text, conversation=conversation, task=task)

