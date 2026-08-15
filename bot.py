import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
# NVIDIA_API_KEY       = your NVIDIA API key
# EVOLUTION_API_URL    = your Evolution API URL
# EVOLUTION_API_KEY    = your Evolution API key
# EVOLUTION_INSTANCE   = your Evolution instance name
#
# Do NOT put these secrets directly in this file.
# Add them in Railway -> Variables.
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "").strip().rstrip("/")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "").strip()
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "").strip()

MODEL = os.getenv("NVIDIA_MODEL", "stepfun-ai/step-3.7-flash")

# Keep the last few messages for each private chat.
MAX_HISTORY = 10
histories = {}


# ============================================================
# AI
# ============================================================

def ask_nvidia(chat_id, text):
    """Send the user's message to NVIDIA NIM and return the answer."""

    history = histories.setdefault(chat_id, [])

    history.append({
        "role": "user",
        "content": text
    })

    # Keep memory small.
    history[:] = history[-MAX_HISTORY:]

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": history,
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        timeout=90,
    )

    response.raise_for_status()

    result = response.json()

    answer = result["choices"][0]["message"]["content"].strip()

    history.append({
        "role": "assistant",
        "content": answer
    })

    history[:] = history[-MAX_HISTORY:]

    return answer


# ============================================================
# EVOLUTION API
# ============================================================

def send_text(remote_jid, text):
    """Send a WhatsApp text message using Evolution API."""

    url = (
        f"{EVOLUTION_API_URL}/message/sendText/"
        f"{EVOLUTION_INSTANCE}"
    )

    response = requests.post(
        url,
        headers={
            "apikey": EVOLUTION_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "number": remote_jid,
            "text": text,
        },
        timeout=30,
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

@app.route("/", methods=["GET"])
def home():
    return "WhatsApp AI Bot is running."


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
