# WhatsApp Sentiment Agent — Setup Guide

## Step 1: Twilio Account (do this FIRST, before running any code)

1. Go to [twilio.com](https://www.twilio.com) and create a free account
2. In the Twilio Console: **Messaging → Try it Out → WhatsApp**
3. You'll see a sandbox number (e.g. `+14155238886`) and a join code (e.g. `join science-physical`)
4. Send that join code from YOUR WhatsApp to the sandbox number — your phone is now connected
5. Note down from the Console dashboard:
   - **Account SID** (starts with "AC...")
   - **Auth Token**

---

## Step 2: Claude writes the code

Claude creates two things:
- `webhook.py` — the Flask + Twilio webhook
- Updates `requirements.txt` — adds flask, twilio, requests, gunicorn

Nothing to do here — just approve it.

---

## Step 3: Install new dependencies

In your terminal, inside the `multimodal-sentiment` folder:

```bash
pip install -r requirements.txt
```

---

## Step 4: Set your Twilio credentials as environment variables

In PowerShell (repeat each session, or add to your PowerShell profile):

```powershell
$env:TWILIO_ACCOUNT_SID = "ACxxxxxxxxxxxxxxxx"
$env:TWILIO_AUTH_TOKEN  = "your_auth_token_here"
```

Never put these values directly in code.

---

## Step 5: Run the webhook locally

```bash
# Terminal 1 — start the Flask webhook
python webhook.py

# Terminal 2 — expose it to the internet
ngrok http 5000
```

ngrok prints a public URL like `https://abc123.ngrok.io`. Copy it.

---

## Step 6: Connect Twilio to your webhook

1. Go back to Twilio Console → WhatsApp Sandbox settings
2. In the **"When a message comes in"** field, paste:
   ```
   https://abc123.ngrok.io/whatsapp
   ```
3. Save

---

## Step 7: Test it

Send a WhatsApp message (text only first) to your sandbox number — you should get an auto-reply within a few seconds.

Then send a message **with a face photo attached** — the reply will reflect both text and facial sentiment.

To test conflict escalation: send a very negative message with a happy face photo.

---

## Step 8 (optional): Deploy to Render so it stays online permanently

Only do this once local testing works.

1. Push the project to a GitHub repo (make sure `webhook.py` and `requirements.txt` are in the root)
2. Go to [render.com](https://render.com) → **New → Web Service** → connect your GitHub repo
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn webhook:app`
4. Add environment variables in the Render dashboard:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
5. After deploy, go to Twilio Console → update the webhook URL to:
   ```
   https://your-app.onrender.com/whatsapp
   ```

---

## Running both interfaces at the same time

| Interface | Command | URL |
|-----------|---------|-----|
| Streamlit dashboard | `streamlit run app.py` | localhost:8501 |
| WhatsApp webhook | `python webhook.py` | localhost:5000 (+ ngrok) |

Both run independently and share the same `fusion.py` model underneath.
