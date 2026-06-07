import os
import threading
import requests
from dotenv import load_dotenv

load_dotenv()
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from PIL import Image
from io import BytesIO

from fusion import analyse, get_text_model, get_face_model

app = Flask(__name__)

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


def process_and_reply(text: str, media_url: str | None, from_number: str, to_number: str):
    try:
        image = None
        if media_url:
            r = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            image = Image.open(BytesIO(r.content))

        result   = analyse(text, image)
        combined = result["fusion"]["combined_score"]
        conflict = result["fusion"]["conflict_detected"]
        reply    = generate_reply(combined, conflict)
    except Exception:
        reply = "Sorry, we couldn't process your message. Please try again."

    Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN).messages.create(
        body=reply,
        from_=to_number,
        to=from_number
    )


@app.route("/")
def health():
    return "OK", 200


@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_text = request.form.get("Body", "")
    media_url     = request.form.get("MediaUrl0", None)
    from_number   = request.form.get("From", "")
    to_number     = request.form.get("To", "")

    thread = threading.Thread(
        target=process_and_reply,
        args=(incoming_text, media_url, from_number, to_number),
        daemon=True
    )
    thread.start()

    # Return empty TwiML immediately — reply arrives via REST API
    return str(MessagingResponse())


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
