from transformers import pipeline
from deepface import DeepFace
from PIL import Image
import numpy as np
import tempfile
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


_text_model = None
_text_tokenizer = None

def get_text_model():
    global _text_model, _text_tokenizer
    if _text_model is None:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        _text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _text_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _text_model.eval()
    return _text_model, _text_tokenizer


# Maps emotion labels to a -1 to +1 sentiment scale
EMOTION_VALENCE = {
    "happy":    1.0,
    "surprise": 0.3,
    "neutral":  0.0,
    "fear":    -0.6,
    "sad":     -0.7,
    "disgust": -0.8,
    "angry":   -1.0
}


def analyse_text(text: str) -> dict:
    model, tokenizer = get_text_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.softmax(outputs.logits, dim=1).squeeze()
    # Labels order: 0=Negative, 1=Neutral, 2=Positive
    neg, neu, pos = scores[0].item(), scores[1].item(), scores[2].item()


    if pos >= neu and pos >= neg:
        label = "POSITIVE"
        score = round((pos - neg) * (1 - neu) ** 7, 4)
    elif neg >= neu and neg >= pos:
        label = "NEGATIVE"
        score = round(-(neg - pos), 4)
    else:
        label = "NEUTRAL"
        score = round(pos - neg, 4)

    return {
        "label":      label,
        "confidence": round(max(pos, neu, neg), 4),
        "score":      round(score, 4)
    }


# ── Vision Analysis ───────────────────────────────────────────────────────────
def analyse_face(image_input) -> dict:
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
            detector_backend="retinaface",
            enforce_detection=False,
            silent=True
        )
        emotions = result[0]["emotion"]
        dominant = result[0]["dominant_emotion"]

        top_confidence = max(emotions.values())
        return {
            "dominant_emotion": dominant,
            "valence_score": round(EMOTION_VALENCE.get(dominant, 0.0), 4),
            "emotion_distribution": {k: round(v, 2) for k, v in emotions.items()},
            "face_detected": True,
            "low_confidence": top_confidence < 30
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
        if isinstance(image_input, Image.Image) and os.path.exists(path):
            os.unlink(path)


# ── Fusion Engine ─────────────────────────────────────────────────────────────
def fuse(text_score: float, face_score: float) -> dict:
    difference = abs(text_score - face_score)
    alignment = round(1 - (difference / 2), 4)
    combined = round((text_score * 0.6) + (face_score * 0.4), 4)
    conflict = difference > 1.2

    if alignment >= 0.75:
        alignment_label = "Strong Alignment"
    elif alignment >= 0.55:
        alignment_label = "Moderate Alignment"
    elif alignment >= 0.40:
        alignment_label = "Slight Misalignment" 
    else:
        alignment_label = "Significant Conflict"

    return {
        "alignment_score":   alignment,
        "alignment_label":   alignment_label,
        "combined_score":    combined,
        "difference":        round(difference, 4),
        "conflict_detected": conflict
    }


# ── Main Analysis Function ────────────────────────────────────────────────────
def analyse(text: str, image_input) -> dict:
    text_result   = analyse_text(text)
    face_result   = analyse_face(image_input)
    fusion_result = fuse(text_result["score"], face_result["valence_score"])

    return {
        "text":   text_result,
        "face":   face_result,
        "fusion": fusion_result
    }