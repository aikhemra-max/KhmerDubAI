# KhmerDubAI Lightweight Fixed

This build removes Gemini, Torch, Transformers, and the large NLLB model.

## Required environment variable

- `TELEGRAM_BOT_TOKEN`

## Recommended environment variables

- `WHISPER_MODEL=tiny` (best for low-memory free servers)
- `WHISPER_DEVICE=cpu`
- `WHISPER_COMPUTE_TYPE=int8`
- `MAX_MEDIA_SECONDS=180`
- `TRANSLATION_RETRIES=3`

## Important

Translation uses the free public Google Translate web endpoint through `deep-translator`. It needs internet, has no API key, and may occasionally be rate-limited.
