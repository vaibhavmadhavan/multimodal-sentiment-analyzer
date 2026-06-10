import os
import threading
import requests
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")
HF_TOKEN           = os.environ.get("HF_TOKEN", "")

HF_HEADERS   = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
TEXT_API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
FACE_API_URL = "https://router.huggingface.co/hf-inference/models/trpakov/vit-face-expression"

EMOTION_VALENCE = {
    "happy": 1.0, "surprise": 0.3, "neutral": 0.0,
    "fear": -0.6, "sad": -0.7, "disgust": -0.8, "angry": -1.0
}
LABEL_MAP = {
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "angry": "angry", "anger": "angry",
    "surprise": "surprise", "surprised": "surprise",
    "fear": "fear", "disgust": "disgust", "neutral": "neutral",
}


def analyse_text(text: str) -> float:
    resp = requests.post(TEXT_API_URL, headers=HF_HEADERS, json={"inputs": text}, timeout=30)
    results = resp.json()
    if isinstance(results, list) and isinstance(results[0], list):
        results = results[0]
    scores = {r["label"].lower(): r["score"] for r in results}
    pos, neg, neu = scores.get("positive", 0), scores.get("negative", 0), scores.get("neutral", 0)
    if pos >= neu and pos >= neg:
        return round((pos - neg) * (1 - neu) ** 7, 4)
    elif neg >= neu and neg >= pos:
        return round(-(neg - pos), 4)
    return round(pos - neg, 4)


def analyse_face(image_bytes: bytes) -> float:
    resp = requests.post(FACE_API_URL, headers=HF_HEADERS, data=image_bytes, timeout=30)
    results = resp.json()
    if not isinstance(results, list) or not results:
        return 0.0
    emotion_dist = {k: 0.0 for k in EMOTION_VALENCE}
    for r in results:
        mapped = LABEL_MAP.get(r["label"].lower(), "neutral")
        emotion_dist[mapped] = round(emotion_dist.get(mapped, 0.0) + r["score"], 4)
    dominant = max(emotion_dist, key=emotion_dist.get)
    return round(EMOTION_VALENCE.get(dominant, 0.0), 4)


def fuse(text_score: float, face_score: float) -> dict:
    difference = abs(text_score - face_score)
    combined = round((text_score * 0.6) + (face_score * 0.4), 4)
    return {"combined_score": combined, "conflict_detected": difference > 1.2}


def generate_reply(combined_score: float, conflict: bool) -> str:
    if conflict:
        return "Thank you for your message. A member of our team will follow up with you shortly."
    elif combined_score >= 0.3:
        return "Thank you so much! We're thrilled to hear you're enjoying your experience."
    elif combined_score <= -0.3:
        return "We're sorry to hear this. Your feedback is important — our team will be in touch."
    return "Thank you for reaching out! How can we assist you further?"


def process_and_reply(text: str, media_url: str | None, from_number: str, to_number: str):
    try:
        text_score = analyse_text(text)
        face_score = 0.0
        if media_url:
            image_bytes = requests.get(
                media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30
            ).content
            face_score = analyse_face(image_bytes)
        fusion = fuse(text_score, face_score)
        reply = generate_reply(fusion["combined_score"], fusion["conflict_detected"])
    except Exception as e:
        print(f"ERROR in process_and_reply: {e}", flush=True)
        reply = "Sorry, we couldn't process your message. Please try again."

    Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
        body=reply, from_=to_number, to=from_number
    )


@app.route("/")
def health():
    return "OK", 200


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    thread = threading.Thread(
        target=process_and_reply,
        args=(
            request.form.get("Body", ""),
            request.form.get("MediaUrl0", None),
            request.form.get("From", ""),
            request.form.get("To", ""),
        ),
        daemon=True
    )
    thread.start()
    return str(MessagingResponse())


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
