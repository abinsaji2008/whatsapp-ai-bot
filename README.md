# WhatsApp AI Bot

Evolution API -> Python Flask webhook -> NVIDIA API -> Evolution API -> WhatsApp.

## Files

- bot.py — bot server
- requirements.txt — Python packages
- Dockerfile — Docker deployment
- .gitignore — keeps .env and temporary files out of Git
- .env.example — configuration template

## Local setup

1. Copy `.env.example` to `.env`.
2. Put your real API values in `.env`.
3. Run `pip install -r requirements.txt`.
4. Run `python bot.py`.

## Evolution webhook

Set the Evolution API webhook URL to:
`https://YOUR-BOT-DOMAIN/webhook`

Enable `messages.upsert`.

## Railway

Connect this GitHub repository to Railway and add the environment variables there. Do not upload `.env` to GitHub.
