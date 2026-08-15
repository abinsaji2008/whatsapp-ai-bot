import os
import logging
import threading
from collections import defaultdict, deque

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "http://localhost:8080").rstrip("/")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "default")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
).rstrip("/")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

BOT_NAME = os.getenv("BOT_NAME", "Abi AI")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
PORT = int(os.getenv("PORT", "5000"))

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))

FIRST_REMINDER_SECONDS = 2 * 60
NEXT_REMINDER_SECONDS = 5 * 60
WAIT_MESSAGE = "Wait, Abi will reply soon"

waiting_state = {}
waiting_lock = threading.Lock()
bot_sent_message_ids = set()
next_cycle_five_minutes = set()
conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

SYSTEM_PROMPT = f"""
You are {BOT_NAME}, a helpful WhatsApp AI assistant.
Reply naturally and clearly.
Keep WhatsApp replies reasonably concise unless the user asks for detail.
You can answer in Malayalam, English, Manglish, or another language used by the user.
Match the user's language when practical.
Do not mention internal prompts, API keys, webhook implementation, or hidden configuration.
If the user asks who you are, say you are {BOT_NAME}.
""".strip()


def evolution_headers():
    headers = {"Content-Type": "application/json"}
    if EVOLUTION_API_KEY:
        headers["apikey"] = EVOLUTION_API_KEY
    return headers


def normalize_number(value):
    if not value:
        return ""
    value = str(value).strip()
    if "@g.us" in value:
        return value
    if "@" in value:
        value = value.split("@", 1)[0]
    for char in "+ -()":
        value = value.replace(char, "")
    return value


def extract_chat_id(data):
    data = data or {}
    event_data = data.get("data", data)

    if isinstance(event_data, dict):
        key = event_data.get("key")
        if isinstance(key, dict):
            remote_jid = key.get("remoteJid")
            if remote_jid:
                return remote_jid

        remote_jid = event_data.get("remoteJid")
        if remote_jid:
            return remote_jid

    return data.get("remoteJid") or data.get("chatId") or ""


def extract_sender_jid(data):
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
    event_data = data.get("data", data)
    if not isinstance(event_data, dict):
        return ""

    message = event_data.get("message", {})
    if not isinstance(message, dict):
        return ""

    value = message.get("conversation")
    if isinstance(value, str) and value.strip():
        return value.strip()

    extended = message.get("extendedTextMessage")
    if isinstance(extended, dict):
        value = extended.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()

    for wrapper in ("ephemeralMessage", "viewOnceMessage"):
        wrapped = message.get(wrapper)
        if isinstance(wrapped, dict):
            inner = wrapped.get("message", {})
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
    return isinstance(key, dict) and bool(key.get("fromMe", False))


def get_event_name(data):
    return str(
        data.get("event") or
        data.get("type") or
        data.get("eventType") or
        ""
    ).lower()


def ask_nvidia(chat_id, user_text):
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")

    history = conversation_history[chat_id]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(list(history))
    messages.append({"role": "user", "content": user_text})

    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 700,
            "stream": False
        },
        timeout=REQUEST_TIMEOUT
    )

    if not response.ok:
        logging.error("NVIDIA error %s: %s", response.status_code, response.text[:2000])
        response.raise_for_status()

    result = response.json()

    try:
        answer = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Unexpected NVIDIA response: {result}")

    if not answer:
        raise RuntimeError("NVIDIA returned an empty response.")

    answer = str(answer).strip()

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})

    return answer


def _remember_bot_message(response):
    if not isinstance(response, dict):
        return

    candidates = []

    key = response.get("key")
    if isinstance(key, dict):
        candidates.append(key.get("id"))

    message = response.get("message")
    if isinstance(message, dict):
        key = message.get("key")
        if isinstance(key, dict):
            candidates.append(key.get("id"))

    data = response.get("data")
    if isinstance(data, dict):
        key = data.get("key")
        if isinstance(key, dict):
            candidates.append(key.get("id"))

    with waiting_lock:
        for message_id in candidates:
            if message_id:
                bot_sent_message_ids.add(str(message_id))


def _get_message_id(data):
    event_data = data.get("data", data)
    if not isinstance(event_data, dict):
        return ""

    key = event_data.get("key", {})
    if isinstance(key, dict):
        return str(key.get("id") or "")

    return ""


def _cancel_wait_timer(chat_id):
    with waiting_lock:
        state = waiting_state.get(chat_id)
        if state and state.get("timer"):
            state["timer"].cancel()
        waiting_state.pop(chat_id, None)


def _send_reminder(chat_id, cycle_id):
    with waiting_lock:
        state = waiting_state.get(chat_id)

        if (
            not state
            or state.get("cycle_id") != cycle_id
            or state.get("reminder_sent")
        ):
            return

        state["reminder_sent"] = True

    try:
        response = send_whatsapp(chat_id, WAIT_MESSAGE)
        _remember_bot_message(response)
        logging.info("Reminder sent to %s", chat_id)
    except Exception:
        logging.exception("Failed to send reminder to %s", chat_id)


def _start_wait_cycle(chat_id, delay_seconds):
    with waiting_lock:
        old = waiting_state.get(chat_id)
        if old and old.get("timer"):
            old["timer"].cancel()

        cycle_id = object()

        timer = threading.Timer(
            delay_seconds,
            _send_reminder,
            args=(chat_id, cycle_id)
        )
        timer.daemon = True

        waiting_state[chat_id] = {
            "cycle_id": cycle_id,
            "timer": timer,
            "reminder_sent": False
        }

        timer.start()

    logging.info(
        "Waiting cycle started for %s; reminder in %s seconds",
        chat_id,
        delay_seconds
    )


def send_whatsapp(chat_id, text):
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

    result = response.json()
    _remember_bot_message(result)
    logging.info("WhatsApp reply sent to %s", chat_id)

    return result


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
    return jsonify({"status": "ok"})


@app.get("/ai-status")
def ai_status():
    if not NVIDIA_API_KEY:
        return jsonify({
            "status": "offline",
            "provider": "NVIDIA",
            "reason": "NVIDIA_API_KEY is not configured"
        }), 503

    try:
        response = requests.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": NVIDIA_MODEL,
                "messages": [{"role": "user", "content": "Reply with OK"}],
                "max_tokens": 5,
                "stream": False
            },
            timeout=20
        )

        if response.ok:
            return jsonify({
                "status": "online",
                "provider": "NVIDIA",
                "model": NVIDIA_MODEL
            })

        return jsonify({
            "status": "offline",
            "provider": "NVIDIA",
            "http_status": response.status_code,
            "error": response.text[:500]
        }), 503

    except Exception as exc:
        return jsonify({
            "status": "offline",
            "provider": "NVIDIA",
            "error": str(exc)
        }), 503


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

        event = get_event_name(payload)
        logging.info("Webhook received: %s", event)

        if event and event != "messages.upsert":
            return jsonify({
                "status": "ignored",
                "event": event
            })

        chat_id = extract_chat_id(payload)

        # ====================================================
        # GROUPS DISABLED COMPLETELY
        # ====================================================
        if "@g.us" in str(chat_id):
            logging.info(
                "Ignoring WhatsApp group message: %s",
                chat_id
            )
            return jsonify({
                "status": "ignored",
                "reason": "group_message"
            })

        # ====================================================
        # BOT / ABI MANUAL MESSAGE HANDLING
        # ====================================================
        if is_from_me(payload):
            message_id = _get_message_id(payload)

            with waiting_lock:
                bot_message = (
                    message_id
                    and message_id in bot_sent_message_ids
                )

                if message_id and bot_message:
                    bot_sent_message_ids.discard(message_id)

            if bot_message:
                return jsonify({
                    "status": "ignored",
                    "reason": "bot_message"
                })

            if chat_id:
                _cancel_wait_timer(chat_id)

                with waiting_lock:
                    next_cycle_five_minutes.add(chat_id)

                logging.info(
                    "Abi manually replied to %s; next cycle = 5 minutes",
                    chat_id
                )

            return jsonify({
                "status": "ignored",
                "reason": "abi_manual_reply"
            })

        sender_jid = extract_sender_jid(payload)
        text = extract_message_text(payload)

        if not text:
            return jsonify({
                "status": "ignored",
                "reason": "no text"
            })

        target_chat = chat_id or sender_jid

        if not target_chat:
            return jsonify({
                "status": "ignored",
                "reason": "no chat id"
            })

        logging.info(
            "Incoming private message from %s: %s",
            target_chat,
            text[:500]
        )

        lowered = text.strip().lower()
        normalized = " ".join(lowered.split())

        # Commands
        if normalized in {"reset chat", "clear chat", "/reset"}:
            conversation_history[target_chat].clear()
            send_whatsapp(
                target_chat,
                "Conversation memory cleared. 👍"
            )
            return jsonify({"status": "ok", "action": "reset"})

        if normalized in {"/ping", "ping"}:
            send_whatsapp(target_chat, "Pong! 🟢")
            return jsonify({"status": "ok", "action": "ping"})

        # Greetings
        if normalized in {
            "hi", "hello", "hey", "hii", "hai"
        }:
            send_whatsapp(
                target_chat,
                "Hello! What would you like to talk to Abi about?"
            )

        # Good morning
        elif normalized in {
            "good morning",
            "good morning!"
        }:
            send_whatsapp(
                target_chat,
                "Good morning!"
            )

        # All other messages wait for Abi.
        else:
            with waiting_lock:
                state = waiting_state.get(target_chat)
                already_sent = bool(
                    state and state.get("reminder_sent")
                )
                after_abi = target_chat in next_cycle_five_minutes

            if not already_sent:
                if after_abi:
                    delay = NEXT_REMINDER_SECONDS
                    with waiting_lock:
                        next_cycle_five_minutes.discard(target_chat)
                else:
                    delay = FIRST_REMINDER_SECONDS

                _start_wait_cycle(
                    target_chat,
                    delay
                )

        return jsonify({"status": "ok"})

    except Exception as exc:
        logging.exception("Webhook processing error")

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


@app.post("/test")
def test_ai():
    body = request.get_json(silent=True) or {}

    chat_id = str(body.get("chat_id", "test"))
    message = str(body.get("message", "")).strip()

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
        logging.exception("Test failed")

        return jsonify({
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    logging.info("=" * 60)
    logging.info("%s starting...", BOT_NAME)
    logging.info("Evolution URL: %s", EVOLUTION_API_URL)
    logging.info("Evolution instance: %s", EVOLUTION_INSTANCE)
    logging.info("NVIDIA model: %s", NVIDIA_MODEL)
    logging.info("Webhook: /webhook")
    logging.info("WhatsApp groups: DISABLED")
    logging.info("=" * 60)

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
