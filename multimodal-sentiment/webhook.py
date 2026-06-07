import os
import requests
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from PIL import Image
from io import BytesIO

from fusion import analyse, get_text_model, get_face_model

app = Flask(__name__)

# Pre-load both models at startup so the first request doesn't time out
get_text_model()
get_face_model()

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN")


def generate_reply(combined_score: float, conflict: bool) -> str:
    if conflict:
        return "Thank you for your message. A member of our team will follow up with you shortly."
    elif combined_score >= 0.3:
        return "Thank you so much! We're thrilled to hear you're enjoying your experience."
    elif combined_score <= -0.3:
        return "We're sorry to hear this. Your feedback is important — our team will be in touch."
    else:
        return "Thank you for reaching out! How can we assist you further?"


@app.route("/")
def health():
    return "OK", 200


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_text = request.form.get("Body", "")
    media_url     = request.form.get("MediaUrl0", None)

    image = None
    if media_url:
        response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        image = Image.open(BytesIO(response.content))

    result   = analyse(incoming_text, image)
    combined = result["fusion"]["combined_score"]
    conflict = result["fusion"]["conflict_detected"]

    resp = MessagingResponse()
    resp.message(generate_reply(combined, conflict))
    return str(resp)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
