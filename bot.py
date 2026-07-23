import asyncio
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
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

MALE_VOICE = os.getenv("MALE_VOICE", "km-KH-PisethNeural")
FEMALE_VOICE = os.getenv("FEMALE_VOICE", "km-KH-SreymomNeural")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", MALE_VOICE)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
_whisper_model = None
_whisper_lock = asyncio.Lock()


@dataclass
class Cue:
    index: int
    start: float
    end: float
    tag: str
    text: str


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
    ms = max(0, round(seconds * 1000))
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def time_to_seconds(value: str) -> float:
    m = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    h, minute, sec, ms = map(int, m.groups())
    return h * 3600 + minute * 60 + sec + ms / 1000


def clean_gemini_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:srt|text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def translate_to_khmer(text: str, preserve_srt: bool = False) -> str:
    if preserve_srt:
        instruction = """
You are an Expert Khmer Movie Subtitler and AI Dubbing Translator.

Translate the supplied Chinese subtitle into natural spoken Khmer.

STRICT RULES:
1. Return ONLY valid SRT. No Markdown and no explanations.
2. Preserve every subtitle number and timestamp exactly.
3. Do not omit, merge, reorder, or invent dialogue.
4. Remove all Chinese characters from translated dialogue.
5. Use short, fluent, emotionally natural spoken Khmer, not literal translation.
6. Match Khmer pronouns to age, rank, relationship, and scene context.
7. Preserve anger, sadness, fear, love, comedy, sarcasm, and urgency.
8. Start every dialogue line with exactly one tag:
   [M] male spoken dialogue
   [F] female spoken dialogue
   [M_THINK] male inner thought or narration
   [F_THINK] female inner thought or narration
9. Infer the most likely speaker from dialogue and nearby context.
10. If gender truly cannot be inferred, use [M].
11. Never put tags on the timestamp line.
12. Example:

1
00:00:01,000 --> 00:00:03,000
[M] ឯងទៅណាហ្នឹង?
"""
    else:
        instruction = """
Translate the supplied Chinese text into natural, fluent spoken Khmer.
Preserve meaning, emotion, and context. Do not translate word-for-word.
Return only the Khmer translation with no explanation and no Markdown.
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
    segments, _ = model.transcribe(
        str(media_path),
        language="zh",
        vad_filter=True,
        beam_size=5,
    )
    blocks = []
    for index, segment in enumerate(segments, start=1):
        line = segment.text.strip()
        if line:
            blocks.append(
                f"{index}\n{srt_time(segment.start)} --> {srt_time(segment.end)}\n{line}"
            )
    if not blocks:
        raise RuntimeError("No speech was detected in this media.")
    return "\n\n".join(blocks) + "\n"


def parse_tagged_srt(srt_text: str) -> list[Cue]:
    normalized = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cues = []

    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        tm = re.fullmatch(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1],
        )
        if not tm:
            continue

        dialogue = " ".join(lines[2:]).strip()
        tag_match = re.match(r"^\[(M|F|M_THINK|F_THINK)\]\s*(.*)$", dialogue, re.I)

        if tag_match:
            tag = tag_match.group(1).upper()
            spoken_text = tag_match.group(2).strip()
        else:
            tag = "M"
            spoken_text = dialogue

        spoken_text = re.sub(r"<[^>]+>", "", spoken_text).strip()
        if spoken_text:
            cues.append(
                Cue(
                    index=index,
                    start=time_to_seconds(tm.group(1)),
                    end=time_to_seconds(tm.group(2)),
                    tag=tag,
                    text=spoken_text,
                )
            )

    if not cues:
        raise RuntimeError("Could not parse Khmer SRT.")
    return sorted(cues, key=lambda cue: (cue.start, cue.index))


def voice_for_tag(tag: str) -> str:
    return FEMALE_VOICE if tag in {"F", "F_THINK"} else MALE_VOICE


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-2500:]}")


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio_path),
    ])


def ffprobe_duration(media_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return max(0.01, float(result.stdout.strip()))


def atempo_chain(speed: float) -> str:
    speed = max(0.05, speed)
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.6f}")
    return ",".join(parts)


async def synthesize_cue(cue: Cue, output_path: Path) -> None:
    thought = cue.tag.endswith("_THINK")
    communicate = edge_tts.Communicate(
        text=cue.text,
        voice=voice_for_tag(cue.tag),
        rate="-8%" if thought else "+0%",
        volume="-5%" if thought else "+0%",
    )
    await communicate.save(str(output_path))


async def create_timed_dub_mp3(
    cues: list[Cue],
    output_path: Path,
    workdir: Path,
) -> None:
    prepared: list[tuple[Path, int]] = []

    for position, cue in enumerate(cues, start=1):
        raw_path = workdir / f"cue_{position:04d}_raw.mp3"
        fit_path = workdir / f"cue_{position:04d}_fit.wav"

        await synthesize_cue(cue, raw_path)

        raw_duration = await asyncio.to_thread(ffprobe_duration, raw_path)
        target_duration = max(0.20, cue.end - cue.start)
        speed = raw_duration / target_duration

        await asyncio.to_thread(
            run_ffmpeg,
            [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-filter:a",
                (
                    f"{atempo_chain(speed)},"
                    f"apad=pad_dur={target_duration:.3f},"
                    f"atrim=0:{target_duration:.3f},"
                    "aresample=48000"
                ),
                "-ac", "2",
                "-c:a", "pcm_s16le",
                str(fit_path),
            ],
        )
        prepared.append((fit_path, round(cue.start * 1000)))

    command = ["ffmpeg", "-y"]
    for file_path, _ in prepared:
        command += ["-i", str(file_path)]

    filter_parts = []
    labels = []
    for index, (_, delay_ms) in enumerate(prepared):
        label = f"a{index}"
        filter_parts.append(f"[{index}:a]adelay={delay_ms}:all=1[{label}]")
        labels.append(f"[{label}]")

    total_duration = max(cue.end for cue in cues) + 0.30
    filter_parts.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:"
        f"duration=longest:dropout_transition=0,"
        f"alimiter=limit=0.95,"
        f"atrim=0:{total_duration:.3f}[mix]"
    )

    command += [
        "-filter_complex", ";".join(filter_parts),
        "-map", "[mix]",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(output_path),
    ]
    await asyncio.to_thread(run_ffmpeg, command)


async def create_khmer_dub_from_srt(
    khmer_srt: str,
    output_path: Path,
    workdir: Path,
) -> None:
    await create_timed_dub_mp3(parse_tagged_srt(khmer_srt), output_path, workdir)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "សួស្តី! ខ្ញុំគឺ KhmerDubAI 🤖\n\n"
        "ផ្ញើ MP3/M4A/WAV សំឡេងរឿងចិន ឬវីដេអូខ្លី។\n\n"
        "Bot នឹងបង្កើត៖\n"
        "① Chinese SRT\n"
        "② Khmer SRT មាន [M]/[F]/[M_THINK]/[F_THINK]\n"
        "③ Khmer MP3 សំឡេង Piseth និង Sreymom\n\n"
        "សាកល្បងជាមួយឯកសារ 30 វិនាទី–2 នាទីសិន។"
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

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        output_path = Path(tmp) / "khmer_voice.mp3"
        try:
            await edge_tts.Communicate(text=text, voice=DEFAULT_VOICE).save(
                str(output_path)
            )
            with output_path.open("rb") as audio:
                await update.message.reply_audio(
                    audio=audio,
                    filename="khmer_voice.mp3",
                    title="KhmerDubAI Voice",
                )
        except Exception as exc:
            logger.exception("TTS failed")
            await update.message.reply_text(f"មិនអាចបង្កើតសំឡេងបាន៖ {exc}")


async def process_srt_upload(update: Update, source_path: Path, tmpdir: Path) -> None:
    status = await update.message.reply_text("⏳ កំពុងបកប្រែ SRT និងបង្កើត MP3…")
    source_text = source_path.read_text(encoding="utf-8-sig", errors="replace")
    khmer_srt = await asyncio.to_thread(translate_to_khmer, source_text, True)

    km_path = tmpdir / f"{source_path.stem}_km.srt"
    mp3_path = tmpdir / f"{source_path.stem}_khmer_dub.mp3"
    km_path.write_text(khmer_srt, encoding="utf-8")

    await status.edit_text("🎙️ កំពុងបង្កើតសំឡេង Piseth និង Sreymom…")
    await create_khmer_dub_from_srt(khmer_srt, mp3_path, tmpdir)

    with km_path.open("rb") as result:
        await update.message.reply_document(
            document=result,
            filename=km_path.name,
            caption="✅ Khmer Tagged SRT",
        )
    with mp3_path.open("rb") as audio:
        await update.message.reply_audio(
            audio=audio,
            filename=mp3_path.name,
            title="KhmerDubAI Dub",
            caption="✅ Khmer Dub MP3",
        )
    await status.edit_text("✅ SRT និង MP3 រួចរាល់")


async def process_media(update: Update, source_path: Path, tmpdir: Path) -> None:
    status = await update.message.reply_text("⏳ កំពុងស្គាល់សំឡេងចិន…")
    media_for_whisper = source_path

    if source_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
        wav_path = tmpdir / "audio.wav"
        await asyncio.to_thread(extract_audio, source_path, wav_path)
        media_for_whisper = wav_path

    async with _whisper_lock:
        chinese_srt = await asyncio.to_thread(transcribe_to_srt, media_for_whisper)

    zh_path = tmpdir / "chinese.srt"
    zh_path.write_text(chinese_srt, encoding="utf-8")

    await status.edit_text("✅ ស្គាល់សំឡេងរួច។ កំពុងបកប្រែ និងបែងចែកតួអង្គ…")
    khmer_srt = await asyncio.to_thread(translate_to_khmer, chinese_srt, True)

    km_path = tmpdir / "khmer_tagged.srt"
    km_path.write_text(khmer_srt, encoding="utf-8")

    await status.edit_text("🎙️ កំពុងបង្កើតសំឡេង Piseth និង Sreymom…")
    mp3_path = tmpdir / "khmer_dub.mp3"
    await create_khmer_dub_from_srt(khmer_srt, mp3_path, tmpdir)

    with zh_path.open("rb") as zh_file:
        await update.message.reply_document(
            document=zh_file,
            filename="chinese.srt",
            caption="Chinese SRT",
        )
    with km_path.open("rb") as km_file:
        await update.message.reply_document(
            document=km_file,
            filename="khmer_tagged.srt",
            caption="✅ Khmer SRT [M/F/THINK]",
        )
    with mp3_path.open("rb") as audio:
        await update.message.reply_audio(
            audio=audio,
            filename="khmer_dub.mp3",
            title="KhmerDubAI Dub",
            caption="✅ Piseth + Sreymom Khmer MP3",
        )
    await status.edit_text("✅ បកប្រែ និងបង្កើត MP3 រួចរាល់")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.message.document
    if not document:
        return

    filename = document.file_name or "file"
    suffix = Path(filename).suffix.lower()

    if document.file_size and document.file_size > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text(
            f"ឯកសារធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB"
        )
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / filename
        telegram_file = await document.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)

        try:
            if suffix == ".srt":
                await process_srt_upload(update, source_path, tmpdir)
            elif suffix in {
                ".mp3", ".wav", ".m4a", ".ogg", ".aac",
                ".mp4", ".mov", ".mkv", ".avi",
            }:
                await process_media(update, source_path, tmpdir)
            else:
                await update.message.reply_text(
                    "សូមផ្ញើ SRT, MP3, M4A, WAV, Audio ឬ Video។"
                )
        except Exception as exc:
            logger.exception("Document processing failed")
            await update.message.reply_text(f"មានបញ្ហាពេលដំណើរការ៖ {exc}")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    media = update.message.audio
    if not media:
        return

    if media.file_size and media.file_size > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text(
            f"ឯកសារធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB"
        )
        return

    suffix = Path(media.file_name or "audio.mp3").suffix.lower() or ".mp3"

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / f"audio{suffix}"
        telegram_file = await media.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)
        try:
            await process_media(update, source_path, tmpdir)
        except Exception as exc:
            logger.exception("Audio processing failed")
            await update.message.reply_text(f"មានបញ្ហាពេលដំណើរការសំឡេង៖ {exc}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice
    if not voice:
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / "voice.ogg"
        telegram_file = await voice.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)
        try:
            await process_media(update, source_path, tmpdir)
        except Exception as exc:
            logger.exception("Voice processing failed")
            await update.message.reply_text(f"មានបញ្ហាពេលដំណើរការសំឡេង៖ {exc}")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    video = update.message.video
    if not video:
        return

    if video.file_size and video.file_size > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text(
            f"វីដេអូធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB"
        )
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


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception", exc_info=context.error)


def main() -> None:
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(300)
        .write_timeout(300)
        .connect_timeout(60)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("voice", voice_command))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, translate_text_message)
    )
    application.add_error_handler(error_handler)

    logger.info("KhmerDubAI intelligent dubbing bot is running")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
