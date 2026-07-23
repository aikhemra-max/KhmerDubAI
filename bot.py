import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import edge_tts
from faster_whisper import WhisperModel
from google import genai
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("khmerdubai")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "45"))
KHMER_VOICE = os.getenv("KHMER_VOICE", "km-KH-SreymomNeural")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
_whisper_model = None
_whisper_lock = asyncio.Lock()


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model: %s", WHISPER_MODEL)
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def clean_gemini_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:srt|text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def translate_to_khmer(text: str, preserve_srt: bool = False) -> str:
    if preserve_srt:
        instruction = """
Translate the supplied subtitle file from Chinese (or its detected language)
into natural spoken Khmer for movie dubbing.

Strict rules:
1. Return ONLY valid SRT content.
2. Preserve every subtitle number and timestamp exactly.
3. Do not omit any dialogue.
4. Remove all Chinese characters from translated dialogue.
5. Use short, fluent, emotionally natural Khmer—not literal word-for-word translation.
6. Do not add Markdown fences, explanations, speaker labels, or notes.
"""
    else:
        instruction = """
Translate the supplied text from Chinese (or its detected language) into
natural, fluent spoken Khmer. Preserve meaning and emotion. Return only the
Khmer translation, with no explanations and no Markdown.
"""
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{instruction}\n\nINPUT:\n{text}",
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return clean_gemini_output(response.text)


def transcribe_to_srt(media_path: Path) -> str:
    model = get_whisper_model()
    segments, info = model.transcribe(
        str(media_path),
        language="zh",
        vad_filter=True,
        beam_size=5,
    )
    blocks = []
    for index, segment in enumerate(segments, start=1):
        line = segment.text.strip()
        if not line:
            continue
        blocks.append(
            f"{index}\n{srt_time(segment.start)} --> {srt_time(segment.end)}\n{line}"
        )
    if not blocks:
        raise RuntimeError("No speech was detected in this media.")
    return "\n\n".join(blocks) + "\n"


def extract_audio(video_path: Path, audio_path: Path) -> None:
    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio_path)
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


async def make_tts(text: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=KHMER_VOICE)
    await communicate.save(str(output_path))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "សួស្តី! ខ្ញុំគឺ KhmerDubAI 🤖\n\n"
        "ផ្ញើមកខ្ញុំ៖\n"
        "• អត្ថបទចិន → បកប្រែជាខ្មែរ\n"
        "• ឯកសារ SRT → បកប្រែរក្សាពេលវេលា\n"
        "• Audio/Video → បង្កើត Chinese SRT និង Khmer SRT\n\n"
        "ពាក្យបញ្ជា៖ /voice សម្រាប់បង្កើត MP3 ពីអត្ថបទខ្មែរ"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def translate_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    source = (update.message.text or "").strip()
    if not source:
        return
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        translated = await asyncio.to_thread(translate_to_khmer, source, False)
        await update.message.reply_text(translated)
    except Exception as exc:
        logger.exception("Text translation failed")
        await update.message.reply_text(f"មានបញ្ហាពេលបកប្រែ៖ {exc}")


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args).strip()
    if not text and update.message.reply_to_message:
        text = (update.message.reply_to_message.text or "").strip()
    if not text:
        await update.message.reply_text(
            "ប្រើ៖ /voice អត្ថបទខ្មែរ\n"
            "ឬ Reply លើសារខ្មែរ ហើយផ្ញើ /voice"
        )
        return

    await update.message.chat.send_action(ChatAction.RECORD_VOICE)
    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        output_path = Path(tmp) / "khmer_voice.mp3"
        try:
            await make_tts(text, output_path)
            with output_path.open("rb") as audio:
                await update.message.reply_audio(
                    audio=audio,
                    filename="khmer_voice.mp3",
                    title="KhmerDubAI Voice",
                )
        except Exception as exc:
            logger.exception("TTS failed")
            await update.message.reply_text(f"មិនអាចបង្កើតសំឡេងបាន៖ {exc}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.message.document
    if not document:
        return
    filename = document.file_name or "file"
    suffix = Path(filename).suffix.lower()

    if document.file_size and document.file_size > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text(f"ឯកសារធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB")
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / filename
        telegram_file = await document.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)

        try:
            if suffix == ".srt":
                await update.message.chat.send_action(ChatAction.TYPING)
                source_text = source_path.read_text(encoding="utf-8-sig", errors="replace")
                khmer_srt = await asyncio.to_thread(translate_to_khmer, source_text, True)
                output_path = tmpdir / f"{source_path.stem}_km.srt"
                output_path.write_text(khmer_srt, encoding="utf-8")
                with output_path.open("rb") as result:
                    await update.message.reply_document(
                        document=result,
                        filename=output_path.name,
                        caption="✅ Khmer SRT រួចរាល់",
                    )
                return

            if suffix in {".mp3", ".wav", ".m4a", ".ogg", ".aac", ".mp4", ".mov", ".mkv", ".avi"}:
                await process_media(update, source_path, tmpdir)
                return

            await update.message.reply_text("សូមផ្ញើ SRT, Audio ឬ Video។")
        except Exception as exc:
            logger.exception("Document processing failed")
            await update.message.reply_text(f"មានបញ្ហាពេលដំណើរការ៖ {exc}")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    video = update.message.video
    if not video:
        return
    if video.file_size and video.file_size > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text(f"វីដេអូធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB")
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / "video.mp4"
        telegram_file = await video.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)
        try:
            await process_media(update, source_path, tmpdir)
        except Exception as exc:
            logger.exception("Video processing failed")
            await update.message.reply_text(f"មានបញ្ហាពេលដំណើរការ៖ {exc}")


async def process_media(update: Update, source_path: Path, tmpdir: Path) -> None:
    status = await update.message.reply_text("⏳ កំពុងដកសំឡេង និងស្គាល់ពាក្យចិន…")
    media_for_whisper = source_path

    if source_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
        wav_path = tmpdir / "audio.wav"
        await asyncio.to_thread(extract_audio, source_path, wav_path)
        media_for_whisper = wav_path

    async with _whisper_lock:
        chinese_srt = await asyncio.to_thread(transcribe_to_srt, media_for_whisper)

    zh_path = tmpdir / "chinese.srt"
    zh_path.write_text(chinese_srt, encoding="utf-8")
    await status.edit_text("✅ ស្គាល់សំឡេងរួច។ កំពុងបកប្រែទៅខ្មែរ…")

    khmer_srt = await asyncio.to_thread(translate_to_khmer, chinese_srt, True)
    km_path = tmpdir / "khmer.srt"
    km_path.write_text(khmer_srt, encoding="utf-8")

    with zh_path.open("rb") as zh_file:
        await update.message.reply_document(
            document=zh_file,
            filename="chinese.srt",
            caption="Chinese SRT",
        )
    with km_path.open("rb") as km_file:
        await update.message.reply_document(
            document=km_file,
            filename="khmer.srt",
            caption="✅ Khmer SRT",
        )
    await status.edit_text("✅ បកប្រែរួចរាល់")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception", exc_info=context.error)


def main() -> None:
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(120)
        .write_timeout(120)
        .connect_timeout(30)
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("voice", voice_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, translate_text_message)
    )
    application.add_error_handler(error_handler)
    logger.info("KhmerDubAI is running")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
