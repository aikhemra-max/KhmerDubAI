# KhmerDubAI Starter

Telegram Bot starter for:

- Chinese text → natural Khmer
- SRT → Khmer SRT while preserving numbering and timestamps
- Chinese audio/video → Chinese SRT + Khmer SRT
- Khmer text → MP3 using `/voice`

## Security first

The Telegram token and Gemini key previously shown in chat/screenshots must be
revoked and replaced. Never place real secrets in GitHub files.

## Railway variables

Add these in Railway → your Service → Variables:

- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEY`
- `GEMINI_MODEL=gemini-2.5-flash`
- `WHISPER_MODEL=small`
- `WHISPER_COMPUTE_TYPE=int8`
- `MAX_FILE_MB=45`
- `KHMER_VOICE=km-KH-SreymomNeural`

## Deploy

1. Upload all files in this project to the root of the GitHub repository.
2. Railway → New Project → GitHub Repository → select `KhmerDubAI`.
3. Add the variables above.
4. Railway detects the Dockerfile and deploys automatically.
5. Open Deployments/Logs. Wait for `KhmerDubAI is running`.
6. Open Telegram and send `/start`.

## Important limits

- Start with short clips (about 1–3 minutes).
- Speech recognition uses CPU and can be slow on low-memory Railway plans.
- Telegram download limits and Railway trial resources may restrict large files.
- This starter returns SRT files. Automatic per-character speaker detection and
  final MP4 dubbing should be added only after this base version is stable.
