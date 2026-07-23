import asyncio
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import edge_tts
from faster_whisper import WhisperModel
from google import genai
from telegram import Message, ReplyKeyboardMarkup, Update
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
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "45"))

MALE_VOICE = os.getenv("MALE_VOICE", "km-KH-PisethNeural")
FEMALE_VOICE = os.getenv("FEMALE_VOICE", "km-KH-SreymomNeural")

# លុបសារ/ឯកសារនៅ Telegram ក្រោយពេលកំណត់។
# Bot មិនអាចដឹងថាអ្នកប្រើបានបើក Telegram ឬអត់ទេ។
AUTO_DELETE_MINUTES = int(os.getenv("AUTO_DELETE_MINUTES", "5"))
DELETE_USER_UPLOAD = os.getenv("DELETE_USER_UPLOAD", "true").lower() == "true"
DELETE_OUTPUT_MESSAGES = os.getenv("DELETE_OUTPUT_MESSAGES", "true").lower() == "true"


MAX_MEDIA_SECONDS = int(os.getenv("MAX_MEDIA_SECONDS", "300"))
SESSION_IDLE_SECONDS = int(os.getenv("SESSION_IDLE_SECONDS", "600"))
NEW_PROJECT_BUTTON = "🆕 ធ្វើថ្មី"

NEW_PROJECT_KEYBOARD = ReplyKeyboardMarkup(
    [[NEW_PROJECT_BUTTON]],
    resize_keyboard=True,
    is_persistent=True,
)

sessions: dict[int, dict] = {}

MAX_TTS_RETRIES = int(os.getenv("MAX_TTS_RETRIES", "3"))
TTS_CONCURRENCY = int(os.getenv("TTS_CONCURRENCY", "4"))
MIN_SPEED = float(os.getenv("MIN_SPEED", "0.88"))
MAX_SPEED = float(os.getenv("MAX_SPEED", "1.15"))

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
_whisper_model = None
_whisper_lock = asyncio.Lock()

ProgressCallback = Callable[[int, str], Awaitable[None]]


@dataclass
class SubtitleCue:
    index: int
    start: float
    end: float
    tag: str
    emotion: str
    text: str


VOICE_PROFILES = {
    "M_YOUNG": {"voice": MALE_VOICE, "rate": "+6%", "pitch": "+4Hz", "volume": "+0%"},
    "F_YOUNG": {"voice": FEMALE_VOICE, "rate": "+5%", "pitch": "+5Hz", "volume": "+0%"},
    "M_ADULT": {"voice": MALE_VOICE, "rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
    "F_ADULT": {"voice": FEMALE_VOICE, "rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
    "M_OLD": {"voice": MALE_VOICE, "rate": "-10%", "pitch": "-16Hz", "volume": "+2%"},
    "F_OLD": {"voice": FEMALE_VOICE, "rate": "-10%", "pitch": "-14Hz", "volume": "+2%"},
    "BOY": {"voice": MALE_VOICE, "rate": "+10%", "pitch": "+18Hz", "volume": "+0%"},
    "GIRL": {"voice": FEMALE_VOICE, "rate": "+10%", "pitch": "+18Hz", "volume": "+0%"},
    "M_THINK": {"voice": MALE_VOICE, "rate": "-6%", "pitch": "-3Hz", "volume": "-5%"},
    "F_THINK": {"voice": FEMALE_VOICE, "rate": "-6%", "pitch": "-3Hz", "volume": "-5%"},
    "NARRATOR_M": {"voice": MALE_VOICE, "rate": "-3%", "pitch": "-2Hz", "volume": "+0%"},
    "NARRATOR_F": {"voice": FEMALE_VOICE, "rate": "-3%", "pitch": "-2Hz", "volume": "+0%"},
}
VALID_TAGS = set(VOICE_PROFILES)

EMOTION_ADJUSTMENTS = {
    "NEUTRAL": {"rate": 0, "pitch": 0, "volume": 0},
    "HAPPY": {"rate": 5, "pitch": 3, "volume": 1},
    "SAD": {"rate": -8, "pitch": -3, "volume": -2},
    "ANGRY": {"rate": 7, "pitch": 3, "volume": 3},
    "FEAR": {"rate": 5, "pitch": 4, "volume": 0},
    "LOVE": {"rate": -5, "pitch": 1, "volume": -1},
    "SARCASM": {"rate": -1, "pitch": 2, "volume": 0},
    "CRYING": {"rate": -9, "pitch": -2, "volume": -3},
    "THINKING": {"rate": -6, "pitch": -3, "volume": -5},
}
VALID_EMOTIONS = set(EMOTION_ADJUSTMENTS)



def get_session(chat_id: int) -> dict:
    session = sessions.get(chat_id)
    if session is None:
        session = {
            "active": False,
            "generation": 0,
            "last_activity": 0.0,
            "message_ids": set(),
            "expiry_task": None,
            "processing_task": None,
        }
        sessions[chat_id] = session
    return session


def track_message(message: Optional[Message]) -> None:
    if message is not None:
        get_session(message.chat_id)["message_ids"].add(message.message_id)


async def safe_delete_message(bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as exc:
        logger.debug("Delete ignored: %s", exc)


async def clear_previous_project(
    bot,
    chat_id: int,
    keep_message_id: Optional[int] = None,
) -> None:
    session = get_session(chat_id)
    old_ids = list(session["message_ids"])
    session["message_ids"].clear()

    for message_id in old_ids:
        if keep_message_id is not None and message_id == keep_message_id:
            continue
        await safe_delete_message(bot, chat_id, message_id)


async def expire_after_inactivity(
    bot,
    chat_id: int,
    generation: int,
) -> None:
    while True:
        session = get_session(chat_id)
        if session["generation"] != generation:
            return

        remaining = SESSION_IDLE_SECONDS - (
            time.monotonic() - session["last_activity"]
        )
        if remaining > 0:
            await asyncio.sleep(remaining)
            continue

        session["active"] = False
        session["processing_task"] = None

        notice = await bot.send_message(
            chat_id=chat_id,
            text=(
                "⌛ គម្រោងផុតកំណត់ ព្រោះគ្មានសកម្មភាព 10 នាទី។\n\n"
                "ចុច «🆕 ធ្វើថ្មី» មុនពេលផ្ញើឯកសារថ្មី។"
            ),
            reply_markup=NEW_PROJECT_KEYBOARD,
        )
        track_message(notice)
        return


def touch_session(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    session = get_session(chat_id)
    session["last_activity"] = time.monotonic()

    old_task = session.get("expiry_task")
    if old_task and not old_task.done():
        old_task.cancel()

    session["expiry_task"] = asyncio.create_task(
        expire_after_inactivity(
            context.bot,
            chat_id,
            session["generation"],
        )
    )


async def require_new_project_started(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    session = get_session(update.effective_chat.id)

    if not session["active"]:
        prompt = await update.message.reply_text(
            "សូមចុច «🆕 ធ្វើថ្មី» ជាមុនសិន "
            "ទើបអាចផ្ញើវីដេអូ ឬសំឡេងបាន។",
            reply_markup=NEW_PROJECT_KEYBOARD,
        )
        track_message(prompt)
        return False

    track_message(update.message)
    touch_session(context, update.effective_chat.id)
    return True


async def new_project(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    running_task = session.get("processing_task")
    if (
        running_task
        and not running_task.done()
        and running_task is not asyncio.current_task()
    ):
        running_task.cancel()

    expiry_task = session.get("expiry_task")
    if expiry_task and not expiry_task.done():
        expiry_task.cancel()

    await clear_previous_project(
        context.bot,
        chat_id,
        keep_message_id=update.message.message_id,
    )

    session["generation"] += 1
    session["active"] = True
    session["last_activity"] = time.monotonic()
    session["processing_task"] = None
    session["message_ids"] = {update.message.message_id}

    ready = await update.message.reply_text(
        "✅ បានលុបគម្រោងចាស់ និងបង្កើតគម្រោងថ្មីរួចរាល់។\n\n"
        "ឥឡូវអាចផ្ញើវីដេអូ ឬសំឡេងរឿងចិនបាន។\n"
        "⏱ រយៈពេលត្រូវត្រឹម 5 នាទី ឬតិចជាងនេះ។",
        reply_markup=NEW_PROJECT_KEYBOARD,
    )
    track_message(ready)
    touch_session(context, chat_id)


def validate_media_duration(media_path: Path) -> None:
    duration = ffprobe_duration(media_path)
    if duration > MAX_MEDIA_SECONDS + 0.5:
        raise ValueError(
            "វីដេអូ ឬសំឡេងវែងជាង 5 នាទី។ "
            "សូមកាត់ឱ្យនៅត្រឹម 5 នាទី ឬតិចជាងនេះ។"
        )

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


def progress_bar(percent: int) -> str:
    percent = max(0, min(100, percent))
    filled = round(percent / 10)
    return "█" * filled + "░" * (10 - filled)


async def update_progress(
    message: Message,
    percent: int,
    label: str,
    state: dict,
) -> None:
    percent = max(0, min(100, int(percent)))
    # កុំ Edit ញឹកញាប់ពេក ដើម្បីជៀសវាង Telegram rate limit។
    if percent < 100 and percent - state.get("last_percent", -10) < 2:
        return

    state["last_percent"] = percent
    text = (
        f"⏳ {label}\n"
        f"`{progress_bar(percent)}`  **{percent}%**\n\n"
        f"សូមកុំផ្ញើឯកសារថ្មី រហូតដល់ការងារនេះចប់។"
    )
    try:
        await message.edit_text(text, parse_mode="Markdown")
    except Exception as exc:
        logger.debug("Progress edit ignored: %s", exc)


async def delete_message_later(message: Optional[Message], seconds: int) -> None:
    if message is None or seconds <= 0:
        return
    await asyncio.sleep(seconds)
    try:
        await message.delete()
    except Exception as exc:
        logger.debug("Could not auto-delete message %s: %s", message.message_id, exc)


def schedule_delete(message: Optional[Message], enabled: bool = True) -> None:
    if not enabled or message is None or AUTO_DELETE_MINUTES <= 0:
        return
    asyncio.create_task(
        delete_message_later(message, AUTO_DELETE_MINUTES * 60)
    )


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def time_to_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def clean_gemini_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:srt|text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def translation_prompt() -> str:
    return """
You are an Expert Khmer Movie Subtitler and Emotional AI Dubbing Translator.

Translate the supplied Chinese subtitle into natural spoken Khmer.

STRICT RULES:
1. Return ONLY valid SRT. No Markdown and no explanations.
2. Preserve every subtitle number and timestamp exactly.
3. Do not omit, merge, reorder, or invent dialogue.
4. Remove all Chinese characters from translated dialogue.
5. Use short, fluent, emotionally natural spoken Khmer; never translate word-for-word.
6. Match pronouns to age, rank, relationship, and scene context.
7. Preserve anger, sadness, fear, love, comedy, sarcasm, crying, and urgency.
8. Start every dialogue with exactly one voice tag and one emotion tag.

VOICE TAGS:
[M_YOUNG] [F_YOUNG] [M_ADULT] [F_ADULT]
[M_OLD] [F_OLD] [BOY] [GIRL]
[M_THINK] [F_THINK] [NARRATOR_M] [NARRATOR_F]

EMOTION TAGS:
[NEUTRAL] [HAPPY] [SAD] [ANGRY] [FEAR]
[LOVE] [SARCASM] [CRYING] [THINKING]

9. Infer the most likely speaker from wording and nearby context.
10. If age is unclear, use [M_ADULT] or [F_ADULT].
11. Keep Khmer concise enough to fit the original time window.
12. One block must look exactly like:

1
00:00:01,000 --> 00:00:03,000
[M_ADULT][ANGRY] ឯងហ៊ានក្បត់អញមែនទេ!
"""


def translate_to_khmer_srt(source_srt: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{translation_prompt()}\n\nINPUT:\n{source_srt}",
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return clean_gemini_output(response.text)


def transcribe_to_srt(media_path: Path) -> str:
    model = get_whisper_model()
    segments, _info = model.transcribe(
        str(media_path),
        language="zh",
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )

    blocks = []
    output_index = 1
    for segment in segments:
        line = segment.text.strip()
        if not line:
            continue
        blocks.append(
            f"{output_index}\n"
            f"{srt_time(segment.start)} --> {srt_time(segment.end)}\n"
            f"{line}"
        )
        output_index += 1

    if not blocks:
        raise RuntimeError("No speech was detected in this media.")
    return "\n\n".join(blocks) + "\n"


def parse_tagged_srt(srt_text: str) -> list[SubtitleCue]:
    normalized = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = re.split(r"\n\s*\n", normalized)
    cues: list[SubtitleCue] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        time_match = re.fullmatch(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1],
        )
        if not time_match:
            continue

        dialogue = " ".join(lines[2:]).strip()
        tag_match = re.match(
            r"^\[([A-Z_]+)\]\[([A-Z_]+)\]\s*(.*)$",
            dialogue,
        )

        if tag_match:
            tag = tag_match.group(1)
            emotion = tag_match.group(2)
            text = tag_match.group(3).strip()
        else:
            tag = "M_ADULT"
            emotion = "NEUTRAL"
            text = dialogue

        if tag not in VALID_TAGS:
            tag = "M_ADULT"
        if emotion not in VALID_EMOTIONS:
            emotion = "NEUTRAL"

        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text:
            continue

        cues.append(
            SubtitleCue(
                index=index,
                start=time_to_seconds(time_match.group(1)),
                end=time_to_seconds(time_match.group(2)),
                tag=tag,
                emotion=emotion,
                text=text,
            )
        )

    if not cues:
        raise RuntimeError("Could not parse any subtitle blocks from Khmer SRT.")
    return sorted(cues, key=lambda cue: (cue.start, cue.index))


def parse_signed_number(value: str) -> int:
    match = re.search(r"[-+]?\d+", value)
    return int(match.group()) if match else 0


def signed_percent(value: int) -> str:
    return f"{value:+d}%"


def signed_hz(value: int) -> str:
    return f"{value:+d}Hz"


def combined_voice_settings(cue: SubtitleCue) -> dict:
    profile = VOICE_PROFILES.get(cue.tag, VOICE_PROFILES["M_ADULT"])
    emotion = EMOTION_ADJUSTMENTS.get(
        cue.emotion,
        EMOTION_ADJUSTMENTS["NEUTRAL"],
    )

    rate = parse_signed_number(profile["rate"]) + emotion["rate"]
    pitch = parse_signed_number(profile["pitch"]) + emotion["pitch"]
    volume = parse_signed_number(profile["volume"]) + emotion["volume"]

    rate = max(-25, min(25, rate))
    pitch = max(-25, min(25, pitch))
    volume = max(-15, min(10, volume))

    return {
        "voice": profile["voice"],
        "rate": signed_percent(rate),
        "pitch": signed_hz(pitch),
        "volume": signed_percent(volume),
    }


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-3000:]}")


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


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio_path),
    ])


async def synthesize_with_retry(cue: SubtitleCue, output_path: Path) -> None:
    settings = combined_voice_settings(cue)
    last_error = None

    for attempt in range(1, MAX_TTS_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(
                text=cue.text,
                voice=settings["voice"],
                rate=settings["rate"],
                pitch=settings["pitch"],
                volume=settings["volume"],
            )
            await communicate.save(str(output_path))
            if not output_path.exists() or output_path.stat().st_size < 100:
                raise RuntimeError("Generated audio is empty.")
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "TTS failed cue=%s attempt=%s/%s: %s",
                cue.index, attempt, MAX_TTS_RETRIES, exc,
            )
            await asyncio.sleep(min(2 * attempt, 6))

    raise RuntimeError(
        f"TTS failed after {MAX_TTS_RETRIES} attempts for cue {cue.index}: "
        f"{last_error}"
    )


async def prepare_cue_audio(
    position: int,
    cue: SubtitleCue,
    workdir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[int, Path, int]:
    raw_path = workdir / f"cue_{position:04d}_raw.mp3"
    fit_path = workdir / f"cue_{position:04d}_fit.wav"

    async with semaphore:
        await synthesize_with_retry(cue, raw_path)

    raw_duration = await asyncio.to_thread(ffprobe_duration, raw_path)
    target_duration = max(0.25, cue.end - cue.start)
    desired_speed = raw_duration / target_duration
    safe_speed = max(MIN_SPEED, min(MAX_SPEED, desired_speed))

    await asyncio.to_thread(
        run_ffmpeg,
        [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-filter:a",
            (
                f"atempo={safe_speed:.6f},"
                f"apad=pad_dur={target_duration:.3f},"
                f"atrim=0:{target_duration:.3f},"
                "aresample=48000"
            ),
            "-ac", "2",
            "-c:a", "pcm_s16le",
            str(fit_path),
        ],
    )
    return position, fit_path, round(cue.start * 1000)


async def create_timed_dub_mp3(
    cues: list[SubtitleCue],
    output_path: Path,
    workdir: Path,
    progress: Optional[ProgressCallback] = None,
) -> None:
    semaphore = asyncio.Semaphore(max(1, TTS_CONCURRENCY))
    tasks = [
        asyncio.create_task(
            prepare_cue_audio(i, cue, workdir, semaphore)
        )
        for i, cue in enumerate(cues, start=1)
    ]

    prepared_results = []
    completed = 0
    total = len(tasks)

    for task in asyncio.as_completed(tasks):
        prepared_results.append(await task)
        completed += 1
        if progress:
            percent = 45 + round((completed / total) * 40)
            await progress(
                percent,
                f"កំពុងបង្កើតសំឡេងខ្មែរ ({completed}/{total})",
            )

    prepared_results.sort(key=lambda item: item[0])
    prepared = [(path, delay) for _pos, path, delay in prepared_results]

    if len(prepared) != len(cues):
        raise RuntimeError("Some subtitle voices are missing.")

    if progress:
        await progress(88, "កំពុងបញ្ចូលសំឡេងតាម Timestamp")

    command = ["ffmpeg", "-y"]
    for file_path, _delay_ms in prepared:
        command += ["-i", str(file_path)]

    filter_parts = []
    labels = []
    for index, (_file_path, delay_ms) in enumerate(prepared):
        label = f"a{index}"
        filter_parts.append(
            f"[{index}:a]adelay={delay_ms}:all=1,volume=1[{label}]"
        )
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
        "-b:a", "160k",
        str(output_path),
    ]
    await asyncio.to_thread(run_ffmpeg, command)

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Final Khmer MP3 was not created correctly.")


async def process_srt_upload(
    update: Update,
    source_path: Path,
    tmpdir: Path,
) -> None:
    status = await update.message.reply_text("⏳ កំពុងចាប់ផ្ដើម… 0%")
    track_message(status)
    state = {"last_percent": -10}

    async def progress(percent: int, label: str) -> None:
        get_session(update.effective_chat.id)["last_activity"] = time.monotonic()
        await update_progress(status, percent, label, state)

    try:
        await progress(5, "កំពុងអានឯកសារ SRT")
        source_text = source_path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        await progress(15, "កំពុងបកប្រែជាភាសាខ្មែរ")
        khmer_srt = await asyncio.to_thread(
            translate_to_khmer_srt,
            source_text,
        )

        km_path = tmpdir / "khmer_dub.srt"
        km_path.write_text(khmer_srt, encoding="utf-8")
        cues = parse_tagged_srt(khmer_srt)

        await progress(40, f"បកប្រែរួច {len(cues)} ឃ្លា")
        mp3_path = tmpdir / "khmer_dub.mp3"
        await create_timed_dub_mp3(
            cues,
            mp3_path,
            tmpdir,
            progress=progress,
        )

        await progress(94, "កំពុងផ្ញើឯកសារ")
        with km_path.open("rb") as km_file:
            srt_message = await update.message.reply_document(
                document=km_file,
                filename="khmer_dub.srt",
                caption=(
                    "✅ Khmer SRT\n"
                    f"🗑 នឹងលុបសារនេះក្រោយ {AUTO_DELETE_MINUTES} នាទី"
                ),
            )

        with mp3_path.open("rb") as audio:
            mp3_message = await update.message.reply_audio(
                audio=audio,
                filename="khmer_dub.mp3",
                title="KhmerDubAI Dub",
                caption=(
                    "✅ Khmer Dub MP3\n"
                    f"🗑 នឹងលុបសារនេះក្រោយ {AUTO_DELETE_MINUTES} នាទី"
                ),
            )

        await progress(100, "ការងាររួចរាល់ ✅")
        track_message(srt_message)
        track_message(mp3_message)
        schedule_delete(status)
        schedule_delete(srt_message, DELETE_OUTPUT_MESSAGES)
        schedule_delete(mp3_message, DELETE_OUTPUT_MESSAGES)

    except Exception:
        schedule_delete(status)
        raise


async def process_media(
    update: Update,
    source_path: Path,
    tmpdir: Path,
) -> None:
    status = await update.message.reply_text("⏳ កំពុងចាប់ផ្ដើម… 0%")
    track_message(status)
    state = {"last_percent": -10}

    async def progress(percent: int, label: str) -> None:
        get_session(update.effective_chat.id)["last_activity"] = time.monotonic()
        await update_progress(status, percent, label, state)

    try:
        media_for_whisper = source_path

        await progress(3, "កំពុងទទួលឯកសារ")
        if source_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
            await progress(8, "កំពុងដកសំឡេងចេញពីវីដេអូ")
            wav_path = tmpdir / "audio.wav"
            await asyncio.to_thread(extract_audio, source_path, wav_path)
            media_for_whisper = wav_path

        await progress(12, "កំពុងស្គាល់សំឡេងចិន")
        async with _whisper_lock:
            chinese_srt = await asyncio.to_thread(
                transcribe_to_srt,
                media_for_whisper,
            )

        await progress(30, "ស្គាល់សំឡេងរួច កំពុងបកប្រែ")
        khmer_srt = await asyncio.to_thread(
            translate_to_khmer_srt,
            chinese_srt,
        )

        km_path = tmpdir / "khmer_dub.srt"
        km_path.write_text(khmer_srt, encoding="utf-8")
        cues = parse_tagged_srt(khmer_srt)

        await progress(42, f"បកប្រែរួច {len(cues)} ឃ្លា")
        mp3_path = tmpdir / "khmer_dub.mp3"
        await create_timed_dub_mp3(
            cues,
            mp3_path,
            tmpdir,
            progress=progress,
        )

        await progress(94, "កំពុងផ្ញើឯកសារទៅ Telegram")
        with km_path.open("rb") as km_file:
            srt_message = await update.message.reply_document(
                document=km_file,
                filename="khmer_dub.srt",
                caption=(
                    "✅ Khmer SRT\n"
                    f"🗑 នឹងលុបសារនេះក្រោយ {AUTO_DELETE_MINUTES} នាទី"
                ),
            )

        with mp3_path.open("rb") as audio:
            mp3_message = await update.message.reply_audio(
                audio=audio,
                filename="khmer_dub.mp3",
                title="KhmerDubAI Dub",
                caption=(
                    "✅ Khmer Dub MP3\n"
                    f"🗑 នឹងលុបសារនេះក្រោយ {AUTO_DELETE_MINUTES} នាទី"
                ),
            )

        await progress(100, "បកប្រែ និងបង្កើតសំឡេងរួចរាល់ ✅")
        track_message(srt_message)
        track_message(mp3_message)
        schedule_delete(status)
        schedule_delete(srt_message, DELETE_OUTPUT_MESSAGES)
        schedule_delete(mp3_message, DELETE_OUTPUT_MESSAGES)

    except Exception:
        schedule_delete(status)
        raise


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_session(update.effective_chat.id)
    session["active"] = False

    message = await update.message.reply_text(
        "សួស្តី! ខ្ញុំគឺ KhmerDubAI Turbo 🤖\n\n"
        "មុនផ្ញើឯកសារ សូមចុច «🆕 ធ្វើថ្មី»។\n"
        "ប៊ូតុងនេះនឹងលុបគម្រោងចាស់ "
        "ហើយបង្កើតគម្រោងថ្មី។\n\n"
        "⏱ ទទួលវីដេអូ ឬសំឡេងរហូតដល់ 5 នាទី។\n"
        "⌛ បើគ្មានសកម្មភាព 10 នាទី "
        "គម្រោងនឹងផុតកំណត់ ហើយត្រូវចុចប៊ូតុងម្ដងទៀត។",
        reply_markup=NEW_PROJECT_KEYBOARD,
    )
    track_message(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = " ".join(context.args).strip()
    if not text and update.message.reply_to_message:
        text = (update.message.reply_to_message.text or "").strip()

    if not text:
        message = await update.message.reply_text(
            "ប្រើ៖ /voice អត្ថបទខ្មែរ\n"
            "ឬ Reply លើសារខ្មែរ ហើយផ្ញើ /voice"
        )
        schedule_delete(message)
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        output_path = Path(tmp) / "khmer_voice.mp3"
        try:
            await update.message.chat.send_action(ChatAction.RECORD_VOICE)
            await edge_tts.Communicate(
                text=text,
                voice=MALE_VOICE,
            ).save(str(output_path))
            with output_path.open("rb") as audio:
                result = await update.message.reply_audio(
                    audio=audio,
                    filename="khmer_voice.mp3",
                    title="KhmerDubAI Voice",
                    caption=f"🗑 នឹងលុបក្រោយ {AUTO_DELETE_MINUTES} នាទី",
                )
            schedule_delete(result, DELETE_OUTPUT_MESSAGES)
        except Exception as exc:
            logger.exception("TTS failed")
            error = await update.message.reply_text(
                f"មិនអាចបង្កើតសំឡេងបាន៖ {exc}"
            )
            schedule_delete(error)


async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not await require_new_project_started(update, context):
        return

    session = get_session(update.effective_chat.id)
    session["processing_task"] = asyncio.current_task()

    document = update.message.document
    if not document:
        return

    filename = document.file_name or "file"
    suffix = Path(filename).suffix.lower()

    if document.file_size and document.file_size > MAX_FILE_MB * 1024 * 1024:
        message = await update.message.reply_text(
            f"ឯកសារធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB"
        )
        track_message(message)
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
                await asyncio.to_thread(
                    validate_media_duration,
                    source_path,
                )
                await process_media(update, source_path, tmpdir)
            else:
                message = await update.message.reply_text(
                    "សូមផ្ញើ SRT, MP3, M4A, WAV, Audio ឬ Video។"
                )
                track_message(message)
        except asyncio.CancelledError:
            message = await update.message.reply_text(
                "🛑 ការងារចាស់ត្រូវបានបញ្ឈប់។ "
                "អ្នកអាចចាប់ផ្ដើមគម្រោងថ្មីបាន។"
            )
            track_message(message)
            raise
        except Exception as exc:
            logger.exception("Document processing failed")
            message = await update.message.reply_text(
                f"មានបញ្ហាពេលដំណើរការ៖ {exc}"
            )
            track_message(message)
        finally:
            session["processing_task"] = None


async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not await require_new_project_started(update, context):
        return

    session = get_session(update.effective_chat.id)
    session["processing_task"] = asyncio.current_task()

    media = update.message.audio
    if not media:
        return

    if media.duration and media.duration > MAX_MEDIA_SECONDS:
        message = await update.message.reply_text(
            "សំឡេងវែងជាង 5 នាទី។ សូមកាត់ឱ្យខ្លីជាងនេះ។"
        )
        track_message(message)
        session["processing_task"] = None
        return

    if media.file_size and media.file_size > MAX_FILE_MB * 1024 * 1024:
        message = await update.message.reply_text(
            f"ឯកសារធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB"
        )
        track_message(message)
        session["processing_task"] = None
        return

    suffix = Path(media.file_name or "audio.mp3").suffix.lower() or ".mp3"

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / f"audio{suffix}"

        telegram_file = await media.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)

        try:
            await asyncio.to_thread(validate_media_duration, source_path)
            await process_media(update, source_path, tmpdir)
        except asyncio.CancelledError:
            message = await update.message.reply_text(
                "🛑 ការងារចាស់ត្រូវបានបញ្ឈប់។"
            )
            track_message(message)
            raise
        except Exception as exc:
            logger.exception("Audio processing failed")
            message = await update.message.reply_text(
                f"មានបញ្ហាពេលដំណើរការសំឡេង៖ {exc}"
            )
            track_message(message)
        finally:
            session["processing_task"] = None


async def handle_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not await require_new_project_started(update, context):
        return

    session = get_session(update.effective_chat.id)
    session["processing_task"] = asyncio.current_task()

    voice = update.message.voice
    if not voice:
        return

    if voice.duration and voice.duration > MAX_MEDIA_SECONDS:
        message = await update.message.reply_text(
            "សំឡេងវែងជាង 5 នាទី។ សូមកាត់ឱ្យខ្លីជាងនេះ។"
        )
        track_message(message)
        session["processing_task"] = None
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / "voice.ogg"

        telegram_file = await voice.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)

        try:
            await asyncio.to_thread(validate_media_duration, source_path)
            await process_media(update, source_path, tmpdir)
        except asyncio.CancelledError:
            message = await update.message.reply_text(
                "🛑 ការងារចាស់ត្រូវបានបញ្ឈប់។"
            )
            track_message(message)
            raise
        except Exception as exc:
            logger.exception("Voice processing failed")
            message = await update.message.reply_text(
                f"មានបញ្ហាពេលដំណើរការសំឡេង៖ {exc}"
            )
            track_message(message)
        finally:
            session["processing_task"] = None


async def handle_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not await require_new_project_started(update, context):
        return

    session = get_session(update.effective_chat.id)
    session["processing_task"] = asyncio.current_task()

    video = update.message.video
    if not video:
        return

    if video.duration and video.duration > MAX_MEDIA_SECONDS:
        message = await update.message.reply_text(
            "វីដេអូវែងជាង 5 នាទី។ "
            "សូមកាត់ឱ្យនៅត្រឹម 5 នាទី ឬតិចជាងនេះ។"
        )
        track_message(message)
        session["processing_task"] = None
        return

    if video.file_size and video.file_size > MAX_FILE_MB * 1024 * 1024:
        message = await update.message.reply_text(
            f"វីដេអូធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB"
        )
        track_message(message)
        session["processing_task"] = None
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / "video.mp4"

        telegram_file = await video.get_file()
        await telegram_file.download_to_drive(custom_path=source_path)

        try:
            await asyncio.to_thread(validate_media_duration, source_path)
            await process_media(update, source_path, tmpdir)
        except asyncio.CancelledError:
            message = await update.message.reply_text(
                "🛑 ការងារចាស់ត្រូវបានបញ្ឈប់។"
            )
            track_message(message)
            raise
        except Exception as exc:
            logger.exception("Video processing failed")
            message = await update.message.reply_text(
                f"មានបញ្ហាពេលដំណើរការ៖ {exc}"
            )
            track_message(message)
        finally:
            session["processing_task"] = None


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
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
    application.add_handler(
        MessageHandler(
            filters.Regex(f"^{re.escape(NEW_PROJECT_BUTTON)}$"),
            new_project,
        )
    )
    application.add_handler(
        MessageHandler(filters.Document.ALL, handle_document)
    )
    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_error_handler(error_handler)

    logger.info(
        "KhmerDubAI New Project | max=%ss | idle=%ss | whisper=%s | parallel=%s",
        MAX_MEDIA_SECONDS,
        SESSION_IDLE_SECONDS,
        WHISPER_MODEL,
        TTS_CONCURRENCY,
    )
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
