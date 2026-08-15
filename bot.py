import os
import logging
import re

import requests
from flask import Flask, request, jsonify

# ============================================================
# Abi WhatsApp Bot
# Evolution API -> WhatsApp
#
# Required Railway environment variables:
# EVOLUTION_API_URL
# EVOLUTION_API_KEY
# EVOLUTION_INSTANCE
#
# Optional:
# WEBHOOK_SECRET
# PORT
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "").rstrip("/")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "default")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
PORT = int(os.getenv("PORT", "5000"))


# ============================================================
# WhatsApp helpers
# ============================================================

def evolution_headers():
    return {
        "Content-Type": "application/json",
        "apikey": EVOLUTION_API_KEY
    }


def normalize_number(value):
    if not value:
        return ""

    value = str(value).strip()

    if "@g.us" in value:
        return value

    if "@" in value:
        value = value.split("@", 1)[0]

    for char in ["+", " ", "-", "(", ")"]:
        value = value.replace(char, "")

    return value


def extract_event(data):
    return str(
        data.get("event")
        or data.get("type")
        or data.get("eventType")
        or ""
    ).lower()


def extract_chat_id(data):
    event_data = data.get("data", data)

    if not isinstance(event_data, dict):
        return ""

    key = event_data.get("key", {})

    if isinstance(key, dict):
        remote_jid = key.get("remoteJid")
        if remote_jid:
            return remote_jid

    return (
        event_data.get("remoteJid")
        or data.get("remoteJid")
        or data.get("chatId")
        or ""
    )


def extract_message_text(data):
    event_data = data.get("data", data)

    if not isinstance(event_data, dict):
        return ""

    message = event_data.get("message", {})

    if not isinstance(message, dict):
        return ""

    # Normal text
    value = message.get("conversation")
    if isinstance(value, str) and value.strip():
        return value.strip()

    # Extended text
    extended = message.get("extendedTextMessage")
    if isinstance(extended, dict):
        value = extended.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Ephemeral message
    ephemeral = message.get("ephemeralMessage")
    if isinstance(ephemeral, dict):
        inner = ephemeral.get("message", {})

        if isinstance(inner, dict):
            value = inner.get("conversation")
            if isinstance(value, str) and value.strip():
                return value.strip()

            extended = inner.get("extendedTextMessage")
            if isinstance(extended, dict):
                value = extended.get("text")
                if isinstance(value, str) and value.strip():
                    return value.strip()

    # View-once message
    view_once = message.get("viewOnceMessage")
    if isinstance(view_once, dict):
        inner = view_once.get("message", {})

        if isinstance(inner, dict):
            value = inner.get("conversation")
            if isinstance(value, str) and value.strip():
                return value.strip()

            extended = inner.get("extendedTextMessage")
            if isinstance(extended, dict):
                value = extended.get("text")
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return ""


def is_from_me(data):
    event_data = data.get("data", data)

    if not isinstance(event_data, dict):
        return False

    key = event_data.get("key", {})

    if isinstance(key, dict):
        return bool(key.get("fromMe", False))

    return False


def webhook_secret_valid():
    if not WEBHOOK_SECRET:
        return True

    supplied = (
        request.headers.get("x-webhook-secret")
        or request.args.get("secret")
        or ""
    )

    return supplied == WEBHOOK_SECRET


# ============================================================
# Reply rules
# ============================================================

# These are deliberately handled WITHOUT AI.
# This prevents the model from inventing replies such as
# "Got it, let's see..." when the bot should follow fixed rules.

GREETING_WORDS = {
    "hi",
    "hai",
    "hello",
    "hey",
    "hii",
    "hiii",
    "helo",
    "hola",
    "namaste"
}

GOOD_MORNING_WORDS = {
    "good morning",
    "gm"
}


def clean_text(text):
    text = text.lower().strip()

    # Remove common punctuation around words.
    text = re.sub(r"[.!?,:;]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def make_reply(user_text):
    cleaned = clean_text(user_text)

    # Good morning -> send it back.
    if cleaned in GOOD_MORNING_WORDS:
        return "Good morning"

    # Simple greetings -> ask what they want to ask Abi.
    if cleaned in GREETING_WORDS:
        return "What do you want to ask Abi?"

    # Greeting with Abi's name, e.g. "Hi Abi".
    greeting_pattern = (
        r"^(hi|hai|hello|hey|hii|hiii|helo)\s+"
        r"(abi|abhi)[!?.,]*$"
    )

    if re.match(greeting_pattern, cleaned):
        return "What do you want to ask Abi?"

    # Everything else.
    return "Wait, Abi will reply soon."


# ============================================================
# Send WhatsApp message
# ============================================================

def send_whatsapp(chat_id, text):
    if not EVOLUTION_API_URL:
        raise RuntimeError("EVOLUTION_API_URL is missing.")

    if not EVOLUTION_API_KEY:
        raise RuntimeError("EVOLUTION_API_KEY is missing.")

    if not EVOLUTION_INSTANCE:
        raise RuntimeError("EVOLUTION_INSTANCE is missing.")

    url = (
        f"{EVOLUTION_API_URL}"
        f"/message/sendText/"
        f"{EVOLUTION_INSTANCE}"
    )

    payload = {
        "number": normalize_number(chat_id),
        "text": text
    }

    response = requests.post(
        url,
        headers=evolution_headers(),
        json=payload,
        timeout=30
    )

    if not response.ok:
        logging.error(
            "Evolution send error %s: %s",
            response.status_code,
            response.text[:2000]
        )
        response.raise_for_status()

    logging.info(
        "WhatsApp reply sent to %s: %s",
        chat_id,
        text
    )

    return response.json()


# ============================================================
# Routes
# ============================================================

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "Abi WhatsApp Bot",
        "instance": EVOLUTION_INSTANCE
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "evolution_url": bool(EVOLUTION_API_URL),
        "evolution_key": bool(EVOLUTION_API_KEY),
        "evolution_instance": bool(EVOLUTION_INSTANCE)
    })


@app.post("/webhook")
def webhook():
    if not webhook_secret_valid():
        return jsonify({"error": "Unauthorized"}), 401

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "status": "ignored",
                "reason": "invalid JSON"
            }), 400

        event = extract_event(payload)

        # Only process messages.upsert when Evolution provides
        # an event name.
        if event and event != "messages.upsert":
            return jsonify({
                "status": "ignored",
                "event": event
            })

        # Never reply to the bot's own messages.
        if is_from_me(payload):
            return jsonify({
                "status": "ignored",
                "reason": "fromMe"
            })

        chat_id = extract_chat_id(payload)
        text = extract_message_text(payload)

        if not chat_id:
            return jsonify({
                "status": "ignored",
                "reason": "no chat id"
            })

        if not text:
            return jsonify({
                "status": "ignored",
                "reason": "no text"
            })

        # Groups are disabled.
        if "@g.us" in str(chat_id):
            logging.info(
                "Ignoring WhatsApp group message: %s",
                chat_id
            )

            return jsonify({
                "status": "ignored",
                "reason": "group"
            })

        logging.info(
            "Incoming private message from %s: %s",
            chat_id,
            text
        )

        answer = make_reply(text)

        send_whatsapp(chat_id, answer)

        return jsonify({
            "status": "replied",
            "reply": answer
        })

    except Exception as exc:
        logging.exception("Webhook processing error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# Local/manual reply test
# ============================================================

@app.get("/test-reply")
def test_reply():
    message = request.args.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Use ?message=Hello"
        }), 400

    return jsonify({
        "message": message,
        "reply": make_reply(message)
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("Abi WhatsApp Bot starting...")
    logging.info("Evolution URL: %s", EVOLUTION_API_URL)
    logging.info("Evolution instance: %s", EVOLUTION_INSTANCE)
    logging.info("Webhook: /webhook")
    logging.info("=" * 60)

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
