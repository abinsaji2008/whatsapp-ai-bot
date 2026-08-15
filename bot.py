import os
import logging
from collections import defaultdict, deque

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ============================================================
# RAILWAY / EVOLUTION / NVIDIA SETTINGS
# ============================================================

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "").strip().rstrip("/")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "").strip()
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "").strip()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
).strip().rstrip("/")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "stepfun-ai/step-3.7-flash"
).strip()

PORT = int(os.getenv("PORT", "8080"))

MAX_HISTORY = 10
MAX_REPLY_CHARS = 30
REQUEST_TIMEOUT = 60

# One history per private WhatsApp chat.
histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

SYSTEM_PROMPT = """
You are Abi's WhatsApp AI assistant.

IMPORTANT:
Every reply must be 30 characters or fewer.

Rules:
1. If the person sends a simple greeting such as
hi, hello, hey, hii, hiii, hai, or similar,
ask them if they would like to ask Abi.

2. If the person says good morning, reply with
a friendly good morning.

3. For anything else, reply:
Wait, Abi will reply soon.

4. Do not answer their actual questions.
5. Do not pretend to be Abi.
6. Do not reveal private information about Abi.
7. Keep replies very short.
8. Match the user's language when practical.
""".strip()


# ============================================================
# HELPERS
# ============================================================

def limit_reply(text):
    """Hard limit every WhatsApp reply to 30 characters."""
    if not text:
        return "Wait, Abi will reply soon."

    text = str(text).strip()
    return text[:MAX_REPLY_CHARS].rstrip()


def extract_event(payload):
    return str(
        payload.get("event")
        or payload.get("type")
        or payload.get("eventType")
        or ""
    ).lower().strip()


def get_event_data(payload):
    data = payload.get("data", payload)
    return data if isinstance(data, dict) else {}


def extract_remote_jid(payload):
    data = get_event_data(payload)
    key = data.get("key", {})

    if isinstance(key, dict):
        jid = key.get("remoteJid")
        if jid:
            return str(jid)

    jid = data.get("remoteJid")
    if jid:
        return str(jid)

    return ""


def extract_text(payload):
    data = get_event_data(payload)
    message = data.get("message", {})

    if not isinstance(message, dict):
        return ""

    # Normal WhatsApp text
    text = message.get("conversation")
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Extended text
    extended = message.get("extendedTextMessage", {})
    if isinstance(extended, dict):
        text = extended.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    # Image caption
    image = message.get("imageMessage", {})
    if isinstance(image, dict):
        text = image.get("caption")
        if isinstance(text, str) and text.strip():
            return text.strip()

    # Ephemeral message
    ephemeral = message.get("ephemeralMessage", {})
    if isinstance(ephemeral, dict):
        inner = ephemeral.get("message", {})
        if isinstance(inner, dict):
            text = inner.get("conversation")
            if isinstance(text, str) and text.strip():
                return text.strip()

            extended = inner.get("extendedTextMessage", {})
            if isinstance(extended, dict):
                text = extended.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

    return ""


def is_from_me(payload):
    data = get_event_data(payload)
    key = data.get("key", {})

    return (
        isinstance(key, dict)
        and bool(key.get("fromMe", False))
    )


def evolution_headers():
    headers = {
        "Content-Type": "application/json"
    }

    if EVOLUTION_API_KEY:
        headers["apikey"] = EVOLUTION_API_KEY

    return headers


def normalize_number(jid):
    """
    Evolution sendText normally accepts the phone number.
    Keep group JIDs unchanged, although groups are ignored below.
    """
    if not jid:
        return ""

    jid = str(jid).strip()

    if "@g.us" in jid:
        return jid

    if "@" in jid:
        jid = jid.split("@", 1)[0]

    return (
        jid.replace("+", "")
           .replace(" ", "")
           .replace("-", "")
           .replace("(", "")
           .replace(")", "")
    )


# ============================================================
# NVIDIA AI
# ============================================================

def ask_nvidia(chat_id, user_text):
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing.")

    history = histories[chat_id]

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

    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 40,
            "stream": False
        },
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

    answer = limit_reply(answer)

    # Store only successful conversation turns.
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
# EVOLUTION API
# ============================================================

def send_whatsapp(chat_id, text):
    if not EVOLUTION_API_URL:
        raise RuntimeError("EVOLUTION_API_URL is missing.")

    if not EVOLUTION_API_KEY:
        raise RuntimeError("EVOLUTION_API_KEY is missing.")

    if not EVOLUTION_INSTANCE:
        raise RuntimeError("EVOLUTION_INSTANCE is missing.")

    text = limit_reply(text)

    url = (
        f"{EVOLUTION_API_URL}"
        f"/message/sendText/"
        f"{EVOLUTION_INSTANCE}"
    )

    response = requests.post(
        url,
        headers=evolution_headers(),
        json={
            "number": normalize_number(chat_id),
            "text": text
        },
        timeout=30
    )

    if not response.ok:
        logging.error(
            "Evolution send error %s: %s",
            response.status_code,
            response.text[:2000]
        )
        response.raise_for_status()

    return response.json()


# ============================================================
# HOME / HEALTH / AI STATUS
# ============================================================

@app.get("/")
def home():
    return jsonify({
        "status": "online",
        "bot": "Abi AI",
        "model": NVIDIA_MODEL
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "nvidia_key": bool(NVIDIA_API_KEY),
        "evolution_url": bool(EVOLUTION_API_URL),
        "evolution_key": bool(EVOLUTION_API_KEY),
        "evolution_instance": bool(EVOLUTION_INSTANCE)
    })


@app.get("/ai-status")
def ai_status():
    if not NVIDIA_API_KEY:
        return jsonify({
            "ai": "error",
            "message": "NVIDIA_API_KEY is missing"
        }), 500

    try:
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": NVIDIA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply only OK"
                    }
                ],
                "temperature": 0,
                "max_tokens": 10,
                "stream": False
            },
            timeout=30
        )

        if not response.ok:
            return jsonify({
                "ai": "error",
                "status_code": response.status_code,
                "message": response.text[:2000]
            }), 500

        result = response.json()

        answer = result["choices"][0]["message"]["content"].strip()

        return jsonify({
            "ai": "ok",
            "model": NVIDIA_MODEL,
            "response": answer
        })

    except Exception as exc:
        logging.exception("AI status check failed")

        return jsonify({
            "ai": "error",
            "message": str(exc)
        }), 500


# ============================================================
# EVOLUTION WEBHOOK
# ============================================================

@app.post("/webhook")
def webhook():
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "status": "ignored",
                "reason": "invalid JSON"
            }), 400

        event = extract_event(payload)

        # Process only messages.upsert when Evolution includes
        # an event name.
        if event and event != "messages.upsert":
            return jsonify({
                "status": "ignored",
                "event": event
            })

        # Never answer our own messages.
        if is_from_me(payload):
            return jsonify({
                "status": "ignored_from_me"
            })

        remote_jid = extract_remote_jid(payload)

        # Groups are intentionally disabled.
        if "@g.us" in remote_jid:
            logging.info(
                "Ignoring WhatsApp group message: %s",
                remote_jid
            )
            return jsonify({
                "status": "ignored_group"
            })

        text = extract_text(payload)

        if not remote_jid or not text:
            return jsonify({
                "status": "ignored_non_text"
            })

        logging.info(
            "Incoming private message from %s: %s",
            remote_jid,
            text[:200]
        )

        try:
            answer = ask_nvidia(
                remote_jid,
                text
            )
        except Exception:
            logging.exception("NVIDIA generation failed")
            answer = "Wait, Abi will reply soon."

        answer = limit_reply(answer)

        send_whatsapp(
            remote_jid,
            answer
        )

        return jsonify({
            "status": "replied",
            "message": answer
        })

    except Exception as exc:
        logging.exception("Webhook processing failed")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# MANUAL AI TEST
# ============================================================

@app.post("/test")
def test_ai():
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
            "answer": limit_reply(answer)
        })

    except Exception as exc:
        logging.exception("AI test failed")

        return jsonify({
            "error": str(exc)
        }), 500


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    logging.info("Abi AI starting...")
    logging.info("NVIDIA model: %s", NVIDIA_MODEL)
    logging.info("Webhook: /webhook")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
