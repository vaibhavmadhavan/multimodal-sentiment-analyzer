from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PIL import Image
import torch

_text_model = None
_text_tokenizer = None
_face_model = None

def get_text_model():
    global _text_model, _text_tokenizer
    if _text_model is None:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        _text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _text_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _text_model.eval()
    return _text_model, _text_tokenizer

def get_face_model():
    global _face_model
    if _face_model is None:
        _face_model = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            device="cpu"
        )
    return _face_model


EMOTION_VALENCE = {
    "happy":    1.0,
    "surprise": 0.3,
    "neutral":  0.0,
    "fear":    -0.6,
    "sad":     -0.7,
    "disgust": -0.8,
    "angry":   -1.0
}

# Model label → our label
LABEL_MAP = {
    "happy":    "happy",
    "happiness": "happy",
    "sad":      "sad",
    "sadness":  "sad",
    "angry":    "angry",
    "anger":    "angry",
    "surprise": "surprise",
    "surprised": "surprise",
    "fear":     "fear",
    "disgust":  "disgust",
    "neutral":  "neutral",
}


def analyse_text(text: str) -> dict:
    model, tokenizer = get_text_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.softmax(outputs.logits, dim=1).squeeze()
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


def analyse_face(image_input) -> dict:
    try:
        if not isinstance(image_input, Image.Image):
            image_input = Image.open(image_input)
        image_input = image_input.convert("RGB")

        detector = get_face_model()
        results = detector(image_input)  # list of {"label": ..., "score": ...}

        if not results:
            raise ValueError("No results from model")

        # Normalise labels and build distribution
        emotion_dist = {k: 0.0 for k in EMOTION_VALENCE}
        for r in results:
            mapped = LABEL_MAP.get(r["label"].lower(), "neutral")
            emotion_dist[mapped] = round(emotion_dist.get(mapped, 0.0) + r["score"], 4)

        dominant = max(emotion_dist, key=emotion_dist.get)
        top_confidence = emotion_dist[dominant]

        return {
            "dominant_emotion":    dominant,
            "valence_score":       round(EMOTION_VALENCE.get(dominant, 0.0), 4),
            "emotion_distribution": emotion_dist,
            "face_detected":       True,
            "low_confidence":      top_confidence < 0.30
        }

    except Exception as e:
        return {
            "dominant_emotion":    "neutral",
            "valence_score":       0.0,
            "emotion_distribution": {k: 0.0 for k in EMOTION_VALENCE},
            "face_detected":       False,
            "error":               str(e)
        }


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


def analyse(text: str, image_input) -> dict:
    text_result   = analyse_text(text)
    face_result   = analyse_face(image_input)
    fusion_result = fuse(text_result["score"], face_result["valence_score"])

    return {
        "text":   text_result,
        "face":   face_result,
        "fusion": fusion_result
    }