"""Training-independent ML inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_texts(texts: list[str], artifact: dict[str, Any]) -> list[dict[str, Any]]:
    model = artifact["model"]
    # train_models.py stores the vectorizer and classifier together in a
    # sklearn Pipeline. Keep compatibility with older artifacts that stored
    # them as separate ``model`` and ``vectorizer`` objects.
    if "vectorizer" in artifact:
        features = artifact["vectorizer"].transform(texts)
        probabilities = model.predict_proba(features)[:, 1]
    else:
        probabilities = model.predict_proba(texts)[:, 1]
    threshold = float(artifact.get("threshold", 0.5))
    return [
        {"label": bool(probability >= threshold), "confidence": float(probability)}
        for probability in probabilities
    ]


def predict_conversation(conversation: list[dict[str, Any]], artifact: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = predict_texts([u["text"] for u in conversation], artifact)
    return [{
        **prediction,
        "utterance_index": i,
        "speaker": u["speaker"],
        "text": u["text"],
        "stime": u["stime"],
        "etime": u["etime"],
    }
            for i, (u, prediction) in enumerate(zip(conversation, predictions))]
