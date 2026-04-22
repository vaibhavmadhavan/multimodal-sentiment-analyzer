from transformers import pipeline
from deepface import DeepFace
from PIL import Image
import numpy as np
import tempfile
import os

# ── Model Initialisation ──────────────────────────────────────────
_text_model = None

def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
            device="cpu"
        )
    return _text_model

# ── Emotion → Valence Mapping ─────────────────────────────────────
# Maps DeepFace emotion labels to a -1 to +1 sentiment scale
EMOTION_VALENCE = {
    "happy":    1.0,
    "surprise": 0.3,
    "neutral":  0.0,
    "fear":    -0.6,
    "sad":     -0.7,
    "disgust": -0.8,
    "angry":   -1.0
}

# ── Text Analysis ─────────────────────────────────────────────────
def analyse_text(text: str) -> dict:
    """
    Returns a sentiment score from -1.0 (very negative) to +1.0 (very positive).
    Uses DistilBERT fine-tuned on SST-2 dataset.
    """
    model = get_text_model()
    result = model(text)[0]
    label = result["label"]       # "POSITIVE" or "NEGATIVE"
    confidence = result["score"]  # 0.5 to ~1.0

    # Positive = +score, Negative = -score
    score = confidence if label == "POSITIVE" else -confidence

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "score": round(score, 4)
    }

# ── Vision Analysis ───────────────────────────────────────────────
def analyse_face(image_input) -> dict:
    """
    Accepts a PIL Image or file path.
    Returns dominant emotion + full emotion probability distribution.
    """
    # Handle PIL Image input by saving to a temp file
    if isinstance(image_input, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image_input.save(tmp.name)
            path = tmp.name
    else:
        path = image_input

    try:
        result = DeepFace.analyze(
            img_path=path,
            actions=["emotion"],
            enforce_detection=False,  # Don't crash if no face found
            silent=True
        )
        emotions = result[0]["emotion"]
        dominant = result[0]["dominant_emotion"]

        return {
            "dominant_emotion": dominant,
            "valence_score": round(EMOTION_VALENCE.get(dominant, 0.0), 4),
            "emotion_distribution": {k: round(v, 2) for k, v in emotions.items()},
            "face_detected": True
        }
    except Exception as e:
        return {
            "dominant_emotion": "neutral",
            "valence_score": 0.0,
            "emotion_distribution": {k: 0.0 for k in EMOTION_VALENCE},
            "face_detected": False,
            "error": str(e)
        }
    finally:
        # Clean up the temp file
        if isinstance(image_input, Image.Image) and os.path.exists(path):
            os.unlink(path)

# ── Fusion Engine ─────────────────────────────────────────────────
def fuse(text_score: float, face_score: float) -> dict:
    """
    Combines text and facial sentiment scores.

    Alignment Score: 0 = complete conflict, 1 = perfect alignment
    Conflict Threshold: >0.8 absolute difference = significant conflict
    """
    difference = abs(text_score - face_score)
    alignment  = round(1 - (difference / 2), 4)  # Normalise to 0–1

    # Weighted combined sentiment (60% text, 40% face)
    combined = round((text_score * 0.6) + (face_score * 0.4), 4)

    conflict = difference > 0.8

    if alignment >= 0.75:
        alignment_label = "Strong Alignment"
    elif alignment >= 0.5:
        alignment_label = "Moderate Alignment"
    else:
        alignment_label = "Significant Conflict"

    return {
        "alignment_score":  alignment,
        "alignment_label":  alignment_label,
        "combined_score":   combined,
        "difference":       round(difference, 4),
        "conflict_detected": conflict
    }

# ── Main Analysis Function ────────────────────────────────────────
def analyse(text: str, image_input) -> dict:
    """
    Full multi-modal analysis pipeline.
    Returns all intermediate and final results.
    """
    text_result   = analyse_text(text)
    face_result   = analyse_face(image_input)
    fusion_result = fuse(text_result["score"], face_result["valence_score"])

    return {
        "text":   text_result,
        "face":   face_result,
        "fusion": fusion_result
    }