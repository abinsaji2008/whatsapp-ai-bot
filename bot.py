import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "").rstrip("/")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE")

# NVIDIA model
MODEL = "stepfun-ai/step-3.7-flash"

# Maximum reply length
MAX_REPLY_CHARS = 30

# Conversation history
histories = {}
MAX_HISTORY = 20


# =========================================================
# AI
# =========================================================

def ask_nvidia(user_number, text):

    history = histories.setdefault(user_number, [])

    # Add system instructions only once
    if not history:
        history.append({
            "role": "system",
            "content": """
You are Abi's WhatsApp AI assistant.

IMPORTANT RULE:
Every reply MUST be 30 characters
or fewer.

Rules:

1. If the person sends a simple
greeting such as hi, hello, hey,
hii, hiii or hai, ask:
"Would you like to ask Abi?"

2. If they say good morning,
reply with a friendly good morning.

3. For anything else reply:
"Wait, Abi will reply soon."

4. Do not answer their questions.
5. Do not pretend to be Abi.
6. Do not provide private information.
7. Keep replies short.
"""
        })

    history.append({
        "role": "user",
        "content": text
    })

    # Keep system message + latest messages
    if len(history) > MAX_HISTORY + 1:
        history = [history[0]] + history[-MAX_HISTORY:]
        histories[user_number] = history

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": history,
            "temperature": 0.3,
            "max_tokens": 50
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    answer = data["choices"][0]["message"]["content"].strip()

    # =====================================================
    # HARD 30 CHARACTER LIMIT
    # =====================================================

    if len(answer) > MAX_REPLY_CHARS:
        answer = answer[:MAX_REPLY_CHARS].rstrip()

    history.append({
        "role": "assistant",
        "content": answer
    })

    if len(history) > MAX_HISTORY + 1:
        history[:] = [history[0]] + history[-MAX_HISTORY:]

    return answer


# =========================================================
# SEND WHATSAPP MESSAGE
# =========================================================

def send_text(remote_jid, text):

    url = (
        f"{EVOLUTION_API_URL}"
        f"/message/sendText/"
        f"{EVOLUTION_INSTANCE}"
    )

    response = requests.post(
        url,
        headers={
            "apikey": EVOLUTION_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "number": remote_jid,
            "text": text
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# WHATSAPP WEBHOOK
# =========================================================

@app.route("/webhook", methods=["POST"])
@app.route("/webhook/<path:event_path>", methods=["POST"])
def webhook(event_path=None):

    data = request.get_json(silent=True) or {}

    # Evolution API data
    msg_data = data.get("data", {})

    key = msg_data.get("key", {})

    message = msg_data.get("message", {})

    remote_jid = key.get(
        "remoteJid",
        ""
    )

    from_me = key.get(
        "fromMe",
        False
    )

    # =====================================================
    # IGNORE BOT'S OWN MESSAGES
    # =====================================================

    if from_me:
        return jsonify({
            "status": "ignored_from_me"
        })


    # =====================================================
    # IGNORE GROUPS
    # =====================================================

    if "@g.us" in remote_jid:
        return jsonify({
            "status": "ignored_group"
        })


    # =====================================================
    # GET MESSAGE TEXT
    # =====================================================

    text = (
        message.get("conversation")
        or message.get(
            "extendedTextMessage",
            {}
        ).get("text")
        or message.get(
            "imageMessage",
            {}
        ).get("caption")
        or ""
    ).strip()


    # =====================================================
    # IGNORE EMPTY MESSAGES
    # =====================================================

    if not text or not remote_jid:
        return jsonify({
            "status": "ignored_non_text"
        })


    # =====================================================
    # CHECK ENVIRONMENT VARIABLES
    # =====================================================

    if not NVIDIA_API_KEY:
        return jsonify({
            "error": "NVIDIA_API_KEY missing"
        }), 500

    if not EVOLUTION_API_URL:
        return jsonify({
            "error": "EVOLUTION_API_URL missing"
        }), 500

    if not EVOLUTION_API_KEY:
        return jsonify({
            "error": "EVOLUTION_API_KEY missing"
        }), 500

    if not EVOLUTION_INSTANCE:
        return jsonify({
            "error": "EVOLUTION_INSTANCE missing"
        }), 500


    # =====================================================
    # ASK NVIDIA AI
    # =====================================================

    try:

        answer = ask_nvidia(
            remote_jid,
            text
        )

        # Make absolutely sure reply is <= 30 chars
        answer = answer[:MAX_REPLY_CHARS].rstrip()

        send_text(
            remote_jid,
            answer
        )

        return jsonify({
            "status": "replied",
            "message": answer
        })


    except Exception as e:

        app.logger.exception(
            "Webhook processing failed"
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# AI STATUS
# =========================================================

@app.route("/ai-status", methods=["GET"])
def ai_status():

    if not NVIDIA_API_KEY:

        return jsonify({
            "ai": "error",
            "message": "NVIDIA_API_KEY is missing"
        }), 500


    try:

        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",

            headers={
                "Authorization":
                    f"Bearer {NVIDIA_API_KEY}",
                "Content-Type":
                    "application/json"
            },

            json={
                "model": MODEL,

                "messages": [
                    {
                        "role": "user",
                        "content": "Reply only OK"
                    }
                ],

                "max_tokens": 10,
                "temperature": 0
            },

            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        answer = (
            data["choices"][0]
            ["message"]["content"]
            .strip()
        )

        return jsonify({
            "ai": "ok",
            "model": MODEL,
            "response": answer
        })


    except Exception as e:

        return jsonify({
            "ai": "error",
            "message": str(e)
        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "ai_key": bool(NVIDIA_API_KEY),
        "evolution_url": bool(EVOLUTION_API_URL),
        "evolution_key": bool(EVOLUTION_API_KEY),
        "instance": bool(EVOLUTION_INSTANCE)
    })


# =========================================================
# HOME
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "running",
        "service": "WhatsApp AI Bot",
        "ai": MODEL
    })


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "8080"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# WEBHOOK
# ============================================================

@app.route("/webhook", methods=["POST"])
@app.route("/webhook/<path:event_path>", methods=["POST"])
def webhook(event_path=None):

    data = request.get_json(silent=True) or {}

    # Evolution API normally puts the message here.
    message_data = data.get("data", {})

    key = message_data.get("key", {})
    message = message_data.get("message", {})

    remote_jid = key.get("remoteJid", "")
    from_me = key.get("fromMe", False)

    # --------------------------------------------------------
    # Ignore messages sent by the bot itself.
    # --------------------------------------------------------

    if from_me:
        return jsonify({
            "status": "ignored_from_me"
        })

    # --------------------------------------------------------
    # IMPORTANT:
    # Ignore ALL WhatsApp group messages.
    #
    # WhatsApp group JIDs normally end with @g.us
    # --------------------------------------------------------

    if remote_jid.endswith("@g.us"):
        app.logger.info(
            "Ignoring WhatsApp group message: %s",
            remote_jid
        )

        return jsonify({
            "status": "ignored_group"
        })

    # --------------------------------------------------------
    # Get text from common Evolution API message formats.
    # --------------------------------------------------------

    text = ""

    if isinstance(message, dict):

        # Normal text message
        text = message.get("conversation", "") or ""

        # Extended text / reply
        if not text:
            extended = message.get(
                "extendedTextMessage",
                {}
            )

            if isinstance(extended, dict):
                text = extended.get("text", "") or ""

        # Image caption
        if not text:
            image = message.get(
                "imageMessage",
                {}
            )

            if isinstance(image, dict):
                text = image.get("caption", "") or ""

    text = text.strip()

    # Ignore empty/non-text messages.
    if not text:
        return jsonify({
            "status": "ignored_non_text"
        })

    # --------------------------------------------------------
    # Check required configuration.
    # --------------------------------------------------------

    missing = []

    if not NVIDIA_API_KEY:
        missing.append("NVIDIA_API_KEY")

    if not EVOLUTION_API_URL:
        missing.append("EVOLUTION_API_URL")

    if not EVOLUTION_API_KEY:
        missing.append("EVOLUTION_API_KEY")

    if not EVOLUTION_INSTANCE:
        missing.append("EVOLUTION_INSTANCE")

    if missing:
        app.logger.error(
            "Missing environment variables: %s",
            ", ".join(missing)
        )

        return jsonify({
            "error": "Missing environment variables",
            "missing": missing
        }), 500

    # --------------------------------------------------------
    # Generate AI response and send it to WhatsApp.
    # --------------------------------------------------------

    try:

        answer = ask_nvidia(
            remote_jid,
            text
        )

        send_text(
            remote_jid,
            answer
        )

        return jsonify({
            "status": "replied"
        })

    except Exception as e:

        app.logger.exception(
            "Webhook processing failed"
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/ai-status", methods=["GET"])
def ai_status():
    if not NVIDIA_API_KEY:
        return jsonify({
            "ai": "error",
            "message": "NVIDIA_API_KEY is missing"
        }), 500

    try:
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "Reply only with OK"}
                ],
                "max_tokens": 10,
                "temperature": 0
            },
            timeout=30
        )

        response.raise_for_status()

        return jsonify({
            "ai": "ok",
            "model": MODEL,
            "response": response.json()["choices"][0]["message"]["content"]
        })

    except Exception as e:
        return jsonify({
            "ai": "error",
            "message": str(e)
        }), 500


@app.route("/", methods=["GET"])
def home():
    return "WhatsApp AI bot is running."


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "8080")
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
