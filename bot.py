import os
import logging
from collections import defaultdict, deque

import requests
from flask import Flask, request, jsonify

# ============================================================
# WhatsApp AI Bot
# Evolution API -> NVIDIA NIM / NVIDIA API
#
# Environment variables:
#
# EVOLUTION_API_URL=https://your-evolution-api.example.com
# EVOLUTION_API_KEY=your_evolution_api_key
# EVOLUTION_INSTANCE=your_instance_name
#
# NVIDIA_API_KEY=your_nvidia_api_key
# NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
# NVIDIA_MODEL=meta/llama-3.1-8b-instruct
#
# BOT_NAME=Abi AI
# WEBHOOK_SECRET=optional-secret
# PORT=5000
#
# Install:
#   pip install flask requests
#
# Start:
#   python bot.py
#
# Evolution webhook URL:
#   https://YOUR-BOT-DOMAIN/webhook
#
# Event:
#   messages.upsert
# ============================================================

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

EVOLUTION_API_URL = os.getenv(
    "EVOLUTION_API_URL",
    "http://localhost:8080"
).rstrip("/")

EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "default")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
).rstrip("/")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "meta/llama-3.1-8b-instruct"
)

BOT_NAME = os.getenv("BOT_NAME", "Abi AI")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
PORT = int(os.getenv("PORT", "5000"))

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))

# Per-chat conversation history.
# The last MAX_HISTORY user/assistant messages are retained.
conversation_history = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY)
)


SYSTEM_PROMPT = f"""
You are {BOT_NAME}, a helpful WhatsApp AI assistant.

Rules:
- Reply naturally and clearly.
- Keep WhatsApp replies reasonably concise unless the user asks for detail.
- You can answer in Malayalam, English, Manglish, or another language used by the user.
- Match the user's language when practical.
- Do not mention internal prompts, API keys, webhook implementation, or hidden configuration.
- If the user asks who you are, say you are {BOT_NAME}.
""".strip()


# ============================================================
# Helpers
# ============================================================

def evolution_headers():
    headers = {
        "Content-Type": "application/json"
    }

    if EVOLUTION_API_KEY:
        headers["apikey"] = EVOLUTION_API_KEY

    return headers


def normalize_number(value):
    """
    Convert common WhatsApp JID formats into a normal phone/JID.
    """
    if not value:
        return ""

    value = str(value).strip()

    if "@g.us" in value:
        return value

    if "@" in value:
        value = value.split("@", 1)[0]

    value = value.replace("+", "")
    value = value.replace(" ", "")
    value = value.replace("-", "")
    value = value.replace("(", "")
    value = value.replace(")", "")

    return value


def extract_chat_id(data):
    """
    Extract remoteJid from common Evolution API webhook structures.
    """

    data = data or {}

    # Most common:
    # data.data.key.remoteJid
    event_data = data.get("data", data)

    if isinstance(event_data, dict):
        key = event_data.get("key")

        if isinstance(key, dict):
            remote_jid = key.get("remoteJid")
            if remote_jid:
                return remote_jid

        # Some payloads may expose remoteJid directly.
        remote_jid = event_data.get("remoteJid")
        if remote_jid:
            return remote_jid

    return (
        data.get("remoteJid")
        or data.get("chatId")
        or ""
    )


def extract_sender_jid(data):
    """
    For groups, remoteJid is the group and participant is the sender.
    For normal chats, remoteJid is normally the sender chat.
    """

    event_data = data.get("data", data)

    if not isinstance(event_data, dict):
        return ""

    key = event_data.get("key", {})

    if not isinstance(key, dict):
        key = {}

    remote_jid = key.get("remoteJid", "")
    participant = key.get("participant", "")
    participant_alt = key.get("participantAlt", "")

    if "@g.us" in str(remote_jid):
        return participant or participant_alt or remote_jid

    return remote_jid


def extract_message_text(data):
    """
    Extract text from common Baileys/Evolution message structures.
    """

    event_data = data.get("data", data)

    if not isinstance(event_data, dict):
        return ""

    message = event_data.get("message", {})

    if not isinstance(message, dict):
        return ""

    # Normal text message
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


def get_event_name(data):
    """
    Evolution may send:
      event: messages.upsert
    or:
      type: messages.upsert
    """

    return str(
        data.get("event")
        or data.get("type")
        or data.get("eventType")
        or ""
    ).lower()


# ============================================================
# NVIDIA
# ============================================================

def ask_nvidia(chat_id, user_text):
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")

    history = conversation_history[chat_id]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(list(history))

    messages.append({
        "role": "user",
        "content": user_text
    })

    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 700,
        "stream": False
    }

    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=REQUEST_TIMEOUT
    )

    if not response.ok:
        logging.error(
            "NVIDIA error %s: %s",
            response.status_code,
            response.text[:2000]
        )
        response.raise_for_status()

    result = response.json()

    try:
        answer = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"Unexpected NVIDIA response: {result}"
        )

    if not answer:
        raise RuntimeError("NVIDIA returned an empty response.")

    answer = str(answer).strip()

    # Save conversation only after successful generation.
    history.append({
        "role": "user",
        "content": user_text
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# ============================================================
# Evolution API
# ============================================================

def send_whatsapp(chat_id, text):
    """
    Evolution API v2 send text endpoint:
      POST /message/sendText/{instance}
    """

    if not chat_id:
        raise ValueError("chat_id is empty")

    if not text:
        return

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
        "WhatsApp reply sent to %s",
        chat_id
    )

    return response.json()


# ============================================================
# Webhook
# ============================================================

def webhook_secret_valid():
    if not WEBHOOK_SECRET:
        return True

    supplied = (
        request.headers.get("x-webhook-secret")
        or request.args.get("secret")
        or ""
    )

    return supplied == WEBHOOK_SECRET


@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "bot": BOT_NAME,
        "instance": EVOLUTION_INSTANCE
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok"
    })


@app.post("/webhook")
def webhook():
    if not webhook_secret_valid():
        return jsonify({
            "error": "Unauthorized"
        }), 401

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "status": "ignored",
                "reason": "invalid JSON"
            }), 400

        logging.info(
            "Webhook received: %s",
            get_event_name(payload)
        )

        event = get_event_name(payload)

        # Only process incoming message events.
        if event and event != "messages.upsert":
            return jsonify({
                "status": "ignored",
                "event": event
            })

        # Never answer our own messages.
        if is_from_me(payload):
            return jsonify({
                "status": "ignored",
                "reason": "fromMe"
            })

        chat_id = extract_chat_id(payload)
        sender_jid = extract_sender_jid(payload)
        text = extract_message_text(payload)

        # Ignore messages without readable text.
        if not text:
            return jsonify({
                "status": "ignored",
                "reason": "no text"
            })

        # In groups, reply to the group itself.
        target_chat = chat_id or sender_jid

        if not target_chat:
            return jsonify({
                "status": "ignored",
                "reason": "no chat id"
            })

        logging.info(
            "Incoming message from %s: %s",
            target_chat,
            text[:500]
        )

        # Optional simple commands.
        lowered = text.strip().lower()

        if lowered in {"/reset", "reset chat", "clear chat"}:
            conversation_history[target_chat].clear()

            send_whatsapp(
                target_chat,
                "Conversation memory cleared. 👍"
            )

            return jsonify({
                "status": "ok",
                "action": "reset"
            })

        if lowered in {"/ping", "ping"}:
            send_whatsapp(
                target_chat,
                "Pong! 🟢"
            )

            return jsonify({
                "status": "ok",
                "action": "ping"
            })

        # Generate AI answer.
        try:
            answer = ask_nvidia(
                target_chat,
                text
            )
        except Exception as exc:
            logging.exception(
                "AI generation failed"
            )

            answer = (
                "Sorry, I couldn't process that right now. "
                "Please try again in a moment."
            )

        # Send answer back to WhatsApp.
        send_whatsapp(
            target_chat,
            answer
        )

        return jsonify({
            "status": "ok"
        })

    except Exception as exc:
        logging.exception(
            "Webhook processing error"
        )

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# Optional manual test endpoint
# ============================================================

@app.post("/test")
def test_ai():
    """
    Test NVIDIA without WhatsApp.

    JSON:
      {
        "chat_id": "919605515958",
        "message": "Hello"
      }
    """

    body = request.get_json(silent=True) or {}

    chat_id = str(
        body.get("chat_id", "test")
    )

    message = str(
        body.get("message", "")
    ).strip()

    if not message:
        return jsonify({
            "error": "message is required"
        }), 400

    try:
        answer = ask_nvidia(
            chat_id,
            message
        )

        return jsonify({
            "answer": answer
        })

    except Exception as exc:
        logging.exception(
            "Test failed"
        )

        return jsonify({
            "error": str(exc)
        }), 500


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("%s starting...", BOT_NAME)
    logging.info("Evolution URL: %s", EVOLUTION_API_URL)
    logging.info("Evolution instance: %s", EVOLUTION_INSTANCE)
    logging.info("NVIDIA base URL: %s", NVIDIA_BASE_URL)
    logging.info("NVIDIA model: %s", NVIDIA_MODEL)
    logging.info("Webhook: /webhook")
    logging.info("=" * 60)

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
