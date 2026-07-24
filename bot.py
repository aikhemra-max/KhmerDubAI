from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional

import edge_tts
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from telegram import Message, ReplyKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Logging and configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("khmerdubai")


def env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true/false, got {raw!r}")


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TRANSLATION_SOURCE = os.getenv("TRANSLATION_SOURCE", "zh-CN").strip()
TRANSLATION_TARGET = os.getenv("TRANSLATION_TARGET", "km").strip()
TRANSLATION_DELAY_SECONDS = env_float("TRANSLATION_DELAY_SECONDS", 0.15, 0.0)
TRANSLATION_MAX_CHARS = env_int("TRANSLATION_MAX_CHARS", 3000, 100)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip()
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip()

MAX_FILE_MB = env_int("MAX_FILE_MB", 45, 1)
MAX_MEDIA_SECONDS = env_int("MAX_MEDIA_SECONDS", 300, 1)
SESSION_IDLE_SECONDS = env_int("SESSION_IDLE_SECONDS", 600, 30)
AUTO_DELETE_MINUTES = env_int("AUTO_DELETE_MINUTES", 5, 0)

DELETE_USER_UPLOAD = env_bool("DELETE_USER_UPLOAD", True)
DELETE_OUTPUT_MESSAGES = env_bool("DELETE_OUTPUT_MESSAGES", True)

MAX_TTS_RETRIES = env_int("MAX_TTS_RETRIES", 3, 1)
TRANSLATION_RETRIES = env_int("TRANSLATION_RETRIES", 3, 1)
TTS_TIMEOUT_SECONDS = env_int("TTS_TIMEOUT_SECONDS", 90, 5)
TRANSLATION_TIMEOUT_SECONDS = env_int(
    "TRANSLATION_TIMEOUT_SECONDS", 900, 30
)
SUBPROCESS_TIMEOUT_SECONDS = env_int("SUBPROCESS_TIMEOUT_SECONDS", 300, 10)
TTS_CONCURRENCY = env_int("TTS_CONCURRENCY", 4, 1)
UPDATE_CONCURRENCY = env_int("UPDATE_CONCURRENCY", 8, 1)

MIN_SPEED = env_float("MIN_SPEED", 0.88, 0.5)
MAX_SPEED = env_float("MAX_SPEED", 1.85, 0.5)
if MIN_SPEED > MAX_SPEED:
    raise RuntimeError("MIN_SPEED cannot be greater than MAX_SPEED")

MALE_VOICE = os.getenv("MALE_VOICE", "km-KH-PisethNeural").strip()
FEMALE_VOICE = os.getenv("FEMALE_VOICE", "km-KH-SreymomNeural").strip()

NEW_PROJECT_BUTTON = "🆕 ធ្វើថ្មី"
PROJECT_KEYBOARD = ReplyKeyboardMarkup(
    [[NEW_PROJECT_BUTTON]],
    resize_keyboard=True,
    is_persistent=True,
)

SUPPORTED_DOCUMENT_SUFFIXES = {
    ".srt",
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".oga",
    ".opus",
    ".aac",
    ".flac",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

ProgressCallback = Callable[[int, str], Awaitable[None]]


# ---------------------------------------------------------------------------
# Data models and state
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SubtitleCue:
    index: int
    start: float
    end: float
    tag: str
    emotion: str
    text: str


@dataclass
class ChatSession:
    active: bool = False
    generation: int = 0
    last_activity: float = 0.0
    message_ids: set[int] = field(default_factory=set)
    expiry_task: asyncio.Task | None = None
    processing_task: asyncio.Task | None = None
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


sessions: dict[int, ChatSession] = {}

_whisper_model: WhisperModel | None = None
_whisper_lock = asyncio.Lock()


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

# Accept the short labels commonly used in SRT files and map them to the
# exact Khmer Edge-TTS voices.  This prevents every unknown label from
# silently falling back to the male voice.
VOICE_TAG_ALIASES = {
    "M": "M_ADULT",
    "MALE": "M_ADULT",
    "MAN": "M_ADULT",
    "PISETH": "M_ADULT",
    "M_ADULT": "M_ADULT",
    "M_YOUNG": "M_YOUNG",
    "M_OLD": "M_OLD",
    "M_THINK": "M_THINK",
    "NARRATOR_M": "NARRATOR_M",
    "F": "F_ADULT",
    "FEMALE": "F_ADULT",
    "WOMAN": "F_ADULT",
    "SREYMOM": "F_ADULT",
    "F_ADULT": "F_ADULT",
    "F_YOUNG": "F_YOUNG",
    "F_OLD": "F_OLD",
    "F_THINK": "F_THINK",
    "NARRATOR_F": "NARRATOR_F",
    "BOY": "BOY",
    "GIRL": "GIRL",
}

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


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def validate_runtime() -> None:
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    for program in ("ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise RuntimeError(
                f"{program} was not found. Install FFmpeg and ensure "
                f"{program} is available in PATH."
            )


def initialize_clients() -> None:
    # Translation model is loaded lazily on the first job. This keeps startup
    # fast and avoids downloading the model until it is actually needed.
    return None


# ---------------------------------------------------------------------------
# Session and Telegram helpers
# ---------------------------------------------------------------------------

def get_session(chat_id: int) -> ChatSession:
    return sessions.setdefault(chat_id, ChatSession())


def track_message(message: Message | None) -> None:
    if message is not None:
        get_session(message.chat_id).message_ids.add(message.message_id)


async def safe_delete_message(bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except (BadRequest, Forbidden):
        pass
    except (NetworkError, TimedOut) as exc:
        logger.debug("Temporary delete failure: %s", exc)
    except Exception:
        logger.debug("Unexpected delete failure", exc_info=True)


async def delete_message_later(message: Message | None, seconds: int) -> None:
    if message is None or seconds <= 0:
        return
    try:
        await asyncio.sleep(seconds)
        await message.delete()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Auto-delete ignored for %s: %s", message.message_id, exc)


def schedule_delete(message: Message | None, enabled: bool = True) -> None:
    if not enabled or message is None or AUTO_DELETE_MINUTES <= 0:
        return
    asyncio.create_task(
        delete_message_later(message, AUTO_DELETE_MINUTES * 60),
        name=f"delete-message-{message.chat_id}-{message.message_id}",
    )


async def clear_previous_project(
    bot,
    chat_id: int,
    keep_message_id: int | None = None,
) -> None:
    session = get_session(chat_id)
    old_ids = tuple(session.message_ids)
    session.message_ids.clear()

    await asyncio.gather(
        *(
            safe_delete_message(bot, chat_id, message_id)
            for message_id in old_ids
            if message_id != keep_message_id
        ),
        return_exceptions=True,
    )


async def expire_after_inactivity(
    bot,
    chat_id: int,
    generation: int,
) -> None:
    try:
        while True:
            session = get_session(chat_id)
            if session.generation != generation or not session.active:
                return

            remaining = SESSION_IDLE_SECONDS - (
                time.monotonic() - session.last_activity
            )
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue

            if session.processing_task and not session.processing_task.done():
                session.last_activity = time.monotonic()
                await asyncio.sleep(min(60, SESSION_IDLE_SECONDS))
                continue

            session.active = False
            notice = await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⌛ គម្រោងផុតកំណត់ ព្រោះគ្មានសកម្មភាព។\n\n"
                    "ចុច «🆕 ធ្វើថ្មី» មុនពេលផ្ញើឯកសារថ្មី។"
                ),
                reply_markup=PROJECT_KEYBOARD,
            )
            track_message(notice)
            return
    except asyncio.CancelledError:
        return


def touch_session(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    session = get_session(chat_id)
    session.last_activity = time.monotonic()

    old_task = session.expiry_task
    if old_task and not old_task.done():
        old_task.cancel()

    session.expiry_task = asyncio.create_task(
        expire_after_inactivity(
            context.bot,
            chat_id,
            session.generation,
        ),
        name=f"session-expiry-{chat_id}-{session.generation}",
    )


async def require_project(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    if update.effective_chat is None or update.effective_message is None:
        return False

    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if not session.active:
        prompt = await update.effective_message.reply_text(
            "សូមចុច «🆕 ធ្វើថ្មី» ជាមុនសិន "
            "ទើបអាចផ្ញើវីដេអូ សំឡេង ឬ SRT បាន។",
            reply_markup=PROJECT_KEYBOARD,
        )
        track_message(prompt)
        return False

    current = session.processing_task
    if current and not current.done() and current is not asyncio.current_task():
        prompt = await update.effective_message.reply_text(
            "⏳ កំពុងដំណើរការឯកសារមួយរួចហើយ។\n"
            "សូមរង់ចាំឱ្យចប់ ឬចុច «🆕 ធ្វើថ្មី» ដើម្បីបញ្ឈប់ការងារចាស់។",
            reply_markup=PROJECT_KEYBOARD,
        )
        track_message(prompt)
        schedule_delete(prompt)
        return False

    track_message(update.effective_message)
    touch_session(context, chat_id)
    return True


async def set_processing_task(chat_id: int) -> bool:
    session = get_session(chat_id)
    async with session.state_lock:
        current = session.processing_task
        if current and not current.done() and current is not asyncio.current_task():
            return False
        session.processing_task = asyncio.current_task()
        return True


async def clear_processing_task(chat_id: int) -> None:
    session = get_session(chat_id)
    async with session.state_lock:
        if session.processing_task is asyncio.current_task():
            session.processing_task = None


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def time_to_seconds(value: str) -> float:
    match = re.fullmatch(
        r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})",
        value.strip(),
    )
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")

    hours, minutes, seconds, milliseconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def normalize_srt(text: str) -> str:
    return text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def split_srt_blocks(text: str) -> list[str]:
    normalized = normalize_srt(text)
    return [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]


def srt_signature(text: str) -> list[tuple[int, str]]:
    signature: list[tuple[int, str]] = []
    for block in split_srt_blocks(text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError("Invalid SRT block: expected index, timestamp and text")
        try:
            index = int(lines[0])
        except ValueError as exc:
            raise ValueError(f"Invalid SRT index: {lines[0]!r}") from exc

        timestamp = re.sub(r"\s+", " ", lines[1]).replace(".", ",")
        if not re.fullmatch(
            r"\d{1,3}:\d{2}:\d{2},\d{3}\s*-->\s*"
            r"\d{1,3}:\d{2}:\d{2},\d{3}",
            timestamp,
        ):
            raise ValueError(f"Invalid SRT timestamp line: {lines[1]!r}")
        signature.append((index, timestamp))
    if not signature:
        raise ValueError("The SRT file contains no valid subtitle blocks")
    return signature


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", text))


def infer_emotion(source_text: str) -> str:
    text = source_text.strip()
    if any(mark in text for mark in ("!", "！", "滚", "住手", "混蛋", "该死")):
        return "ANGRY"
    if any(mark in text for mark in ("?", "？", "怎么", "为什么", "谁")):
        return "FEAR"
    if any(mark in text for mark in ("哭", "死", "对不起", "不要离开")):
        return "SAD"
    if any(mark in text for mark in ("爱", "喜欢", "想你")):
        return "LOVE"
    if any(mark in text for mark in ("哈哈", "呵呵", "太好了")):
        return "HAPPY"
    return "NEUTRAL"


def infer_voice_tag(source_text: str, previous_tag: str) -> str:
    """Best-effort speaker tag selection from subtitle wording."""
    text = source_text.strip()
    if any(word in text for word in ("旁白", "解说", "画外音")):
        return "NARRATOR_M"
    if any(word in text for word in ("娘", "姐姐", "妹妹", "小姐", "夫人", "奶奶")):
        return "F_ADULT"
    if any(word in text for word in ("爹", "哥哥", "弟弟", "公子", "大人", "爷爷")):
        return "M_ADULT"
    return previous_tag or "M_ADULT"


def parse_plain_srt(source_srt: str) -> list[tuple[int, str, str]]:
    cues: list[tuple[int, str, str]] = []
    for block in split_srt_blocks(source_srt):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"Invalid subtitle block: {block[:120]!r}")
        index = int(lines[0])
        timestamp = re.sub(r"\s+", " ", lines[1]).replace(".", ",")
        dialogue = " ".join(lines[2:]).strip()
        dialogue = re.sub(r"^\[[A-Z_]+\]\[[A-Z_]+\]\s*", "", dialogue)
        if not dialogue:
            raise ValueError(f"Cue {index} has empty dialogue")
        cues.append((index, timestamp, dialogue))
    return cues


def _split_translation_text(text: str, limit: int) -> list[str]:
    """Split long text without breaking ordinary short subtitle lines."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text]
    pieces = re.split(r"(?<=[。！？!?；;，,])", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > limit:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def translate_text_free(text: str) -> str:
    """Translate one Chinese subtitle to Khmer with no API key.

    This uses Google Translate's public web endpoint through deep-translator.
    It is lightweight and avoids loading the multi-gigabyte NLLB/Torch model.
    """
    last_error: Exception | None = None
    chunks = _split_translation_text(text, TRANSLATION_MAX_CHARS)
    translated_chunks: list[str] = []

    for chunk in chunks:
        for attempt in range(1, TRANSLATION_RETRIES + 1):
            try:
                result = GoogleTranslator(
                    source=TRANSLATION_SOURCE,
                    target=TRANSLATION_TARGET,
                ).translate(chunk)
                result = re.sub(r"\s+", " ", result or "").strip()
                if not result:
                    raise RuntimeError("translation service returned empty text")
                translated_chunks.append(result)
                if TRANSLATION_DELAY_SECONDS:
                    time.sleep(TRANSLATION_DELAY_SECONDS)
                break
            except Exception as exc:
                last_error = exc
                if attempt >= TRANSLATION_RETRIES:
                    raise RuntimeError(
                        "Free translation service failed after "
                        f"{TRANSLATION_RETRIES} attempts: {exc}"
                    ) from exc
                time.sleep(min(2 ** attempt, 8))

    result = " ".join(translated_chunks).strip()
    if not result and last_error:
        raise RuntimeError(str(last_error))
    return result


def translate_to_khmer_srt(source_srt: str) -> str:
    """Translate SRT to Khmer without Gemini, Torch, or a paid API key."""
    expected_signature = srt_signature(source_srt)
    source_cues = parse_plain_srt(source_srt)
    blocks: list[str] = []
    previous_tag = "M_ADULT"

    for index, timestamp, source_text in source_cues:
        khmer_text = translate_text_free(source_text)
        khmer_text = re.sub(r"[\u3400-\u4DBF\u4E00-\u9FFF]+", "", khmer_text).strip()
        if not khmer_text:
            raise RuntimeError(f"Cue {index} could not be translated to Khmer")
        tag = infer_voice_tag(source_text, previous_tag)
        emotion = infer_emotion(source_text)
        previous_tag = tag
        blocks.append(f"{index}\n{timestamp}\n[{tag}][{emotion}] {khmer_text}")

    result = "\n\n".join(blocks) + "\n"
    if srt_signature(result) != expected_signature:
        raise RuntimeError("Translation changed subtitle numbering or timestamps")
    if contains_chinese(result):
        raise RuntimeError("Chinese characters remained in Khmer SRT")
    parse_tagged_srt(result)
    return result


def _clean_dialogue_text(text: str) -> str:
    """Remove labels/markup that Edge-TTS might pronounce aloud."""
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _voice_tag_from_dubbing_voice(voice_name: str) -> str:
    name = voice_name.strip().lower()
    if "sreymom" in name:
        return "F_ADULT"
    if "piseth" in name:
        return "M_ADULT"
    raise ValueError(f"Unsupported dubbing voice: {voice_name!r}")


def parse_tagged_srt(srt_text: str) -> list[SubtitleCue]:
    """Parse SRT with either bracket labels or <dubbing voice=...> labels.

    Supported examples:
      [F][SAD] text
      [M_ADULT][NEUTRAL] text
      <dubbing voice="km-KH-SreymomNeural">text</dubbing>
    """
    cues: list[SubtitleCue] = []

    for block in split_srt_blocks(srt_text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"Invalid subtitle block: {block[:120]!r}")

        index = int(lines[0])
        time_match = re.fullmatch(
            r"(\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
            r"(\d{1,3}:\d{2}:\d{2}[,.]\d{3})",
            lines[1],
        )
        if not time_match:
            raise ValueError(f"Invalid timestamp for cue {index}: {lines[1]!r}")

        dialogue = " ".join(lines[2:]).strip()
        tag = ""
        emotion = "NEUTRAL"
        text = ""

        xml_match = re.fullmatch(
            r'<dubbing\s+voice=["\']([^"\']+)["\']\s*>(.*?)</dubbing>',
            dialogue,
            flags=re.I | re.S,
        )
        if xml_match:
            tag = _voice_tag_from_dubbing_voice(xml_match.group(1))
            text = xml_match.group(2)
        else:
            # Emotion is optional. A single [F] or [M] label is valid.
            bracket_match = re.fullmatch(
                r"\[([^\]]+)\](?:\[([^\]]+)\])?\s*(.+)",
                dialogue,
                flags=re.S,
            )
            if not bracket_match:
                raise ValueError(
                    f"Cue {index} needs [M]/[F], [VOICE][EMOTION], "
                    "or a <dubbing voice=...> label"
                )
            raw_tag, raw_emotion, text = bracket_match.groups()
            normalized_tag = re.sub(r"[^A-Z0-9_]", "", raw_tag.upper())
            tag = VOICE_TAG_ALIASES.get(normalized_tag, normalized_tag)
            if raw_emotion:
                emotion = re.sub(r"[^A-Z0-9_]", "", raw_emotion.upper())

        if tag not in VALID_TAGS:
            raise ValueError(f"Unsupported voice tag [{tag}] at cue {index}")
        if emotion not in VALID_EMOTIONS:
            raise ValueError(f"Unsupported emotion [{emotion}] at cue {index}")

        text = _clean_dialogue_text(text)
        if not text:
            raise ValueError(f"Cue {index} has empty dialogue")

        start = time_to_seconds(time_match.group(1))
        end = time_to_seconds(time_match.group(2))
        if end <= start:
            raise ValueError(f"Cue {index} has end time <= start time")

        cues.append(SubtitleCue(index, start, end, tag, emotion, text))

    return sorted(cues, key=lambda cue: (cue.start, cue.index))


# ---------------------------------------------------------------------------
# Whisper, FFmpeg and TTS
# ---------------------------------------------------------------------------

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            WHISPER_MODEL,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def run_command(command: list[str], timeout: int = SUBPROCESS_TIMEOUT_SECONDS) -> None:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout}s: {command[0]}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run {command[0]}: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"{command[0]} failed with exit code {result.returncode}:\n"
            f"{stderr[-2500:]}"
        )


def ffprobe_duration(media_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(media_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        duration = float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        raise RuntimeError(f"Could not read media duration: {exc}") from exc

    if duration <= 0:
        raise RuntimeError("Media duration is zero or invalid")
    return duration


def validate_media_duration(media_path: Path) -> None:
    duration = ffprobe_duration(media_path)
    if duration > MAX_MEDIA_SECONDS + 0.5:
        raise ValueError(
            f"ឯកសារវែងជាង {MAX_MEDIA_SECONDS // 60} នាទី។ "
            f"សូមកាត់ឱ្យខ្លីជាងនេះ។"
        )


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
    )


def transcribe_to_srt(media_path: Path) -> str:
    model = get_whisper_model()
    segments, _info = model.transcribe(
        str(media_path),
        language="zh",
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    blocks: list[str] = []
    for output_index, segment in enumerate(segments, start=1):
        line = segment.text.strip()
        if not line:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{srt_time(segment.start)} --> {srt_time(segment.end)}\n"
            f"{line}"
        )

    if not blocks:
        raise RuntimeError("No Chinese speech was detected in this media")
    return "\n\n".join(blocks) + "\n"


def parse_signed_number(value: str) -> int:
    match = re.search(r"[-+]?\d+", value)
    return int(match.group()) if match else 0


def combined_voice_settings(cue: SubtitleCue) -> dict[str, str]:
    profile = VOICE_PROFILES[cue.tag]
    emotion = EMOTION_ADJUSTMENTS[cue.emotion]

    rate = parse_signed_number(profile["rate"]) + emotion["rate"]
    pitch = parse_signed_number(profile["pitch"]) + emotion["pitch"]
    volume = parse_signed_number(profile["volume"]) + emotion["volume"]

    return {
        "voice": profile["voice"],
        "rate": f"{max(-25, min(25, rate)):+d}%",
        "pitch": f"{max(-25, min(25, pitch)):+d}Hz",
        "volume": f"{max(-15, min(10, volume)):+d}%",
    }


async def synthesize_with_retry(cue: SubtitleCue, output_path: Path) -> None:
    settings = combined_voice_settings(cue)
    last_error: Exception | None = None

    for attempt in range(1, MAX_TTS_RETRIES + 1):
        try:
            output_path.unlink(missing_ok=True)
            communicate = edge_tts.Communicate(
                text=cue.text,
                voice=settings["voice"],
                rate=settings["rate"],
                pitch=settings["pitch"],
                volume=settings["volume"],
            )
            await asyncio.wait_for(
                communicate.save(str(output_path)),
                timeout=TTS_TIMEOUT_SECONDS,
            )
            if not output_path.exists() or output_path.stat().st_size < 100:
                raise RuntimeError("Generated TTS audio is empty")
            return
        except asyncio.CancelledError:
            output_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            last_error = exc
            logger.warning(
                "TTS cue=%s attempt=%s/%s failed: %s",
                cue.index,
                attempt,
                MAX_TTS_RETRIES,
                exc,
            )
            if attempt < MAX_TTS_RETRIES:
                await asyncio.sleep(min(2 * attempt, 6))

    raise RuntimeError(
        f"TTS failed for subtitle {cue.index}: {last_error}"
    )


def build_atempo_filter(speed: float) -> str:
    """Build a legal FFmpeg atempo chain without cutting spoken words."""
    speed = max(0.25, min(4.0, speed))
    factors: list[float] = []
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    factors.append(speed)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


async def prepare_cue_audio(
    position: int,
    cue: SubtitleCue,
    available_duration: float,
    workdir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[int, Path, int, float]:
    raw_path = workdir / f"cue_{position:05d}_raw.mp3"
    fit_path = workdir / f"cue_{position:05d}_fit.wav"

    async with semaphore:
        await synthesize_with_retry(cue, raw_path)

    raw_duration = await asyncio.to_thread(ffprobe_duration, raw_path)
    target_duration = max(0.35, available_duration)
    requested_speed = raw_duration / target_duration
    speed = max(MIN_SPEED, min(MAX_SPEED, requested_speed))

    # Never hard-trim TTS at the subtitle end. Hard trimming was the main cause
    # of broken final syllables and choppy playback. A tiny fade only removes
    # encoder clicks; the complete spoken sentence is preserved.
    audio_filter = (
        f"{build_atempo_filter(speed)},"
        "highpass=f=65,"
        "lowpass=f=13500,"
        "afade=t=in:st=0:d=0.008,"
        "aresample=48000:async=1:first_pts=0"
    )

    await asyncio.to_thread(
        run_command,
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-filter:a",
            audio_filter,
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(fit_path),
        ],
    )
    fitted_duration = await asyncio.to_thread(ffprobe_duration, fit_path)
    return position, fit_path, round(cue.start * 1000), fitted_duration


async def create_timed_dub_mp3(
    cues: list[SubtitleCue],
    output_path: Path,
    workdir: Path,
    progress: ProgressCallback | None = None,
) -> None:
    if not cues:
        raise RuntimeError("No subtitle cues were provided for TTS")

    semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
    tasks = []
    for i, cue in enumerate(cues, start=1):
        # Prefer the real gap before the next subtitle. This keeps complete
        # syllables while reducing overlaps between consecutive speakers.
        next_start = cues[i].start if i < len(cues) else cue.end
        available = max(0.35, max(cue.end, next_start - 0.04) - cue.start)
        tasks.append(
            asyncio.create_task(
                prepare_cue_audio(i, cue, available, workdir, semaphore),
                name=f"tts-cue-{cue.index}",
            )
        )

    results: list[tuple[int, Path, int, float]] = []
    try:
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            results.append(await task)
            if progress:
                await progress(
                    45 + round((completed / len(tasks)) * 40),
                    f"កំពុងបង្កើតសំឡេងខ្មែរ ({completed}/{len(tasks)})",
                )
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    results.sort(key=lambda item: item[0])
    if len(results) != len(cues):
        raise RuntimeError("Some subtitle audio files are missing")

    if progress:
        await progress(88, "កំពុងតម្រៀបសំឡេងតាម Timestamp")

    command = ["ffmpeg", "-y"]
    for _position, path, _delay, _duration in results:
        command.extend(["-i", str(path)])

    filter_parts: list[str] = []
    labels: list[str] = []
    adjusted_results: list[tuple[int, Path, int, float]] = []
    previous_end = 0.0
    minimum_gap = 0.025
    for item, cue in zip(results, cues):
        position, path, original_delay_ms, fitted_duration = item
        original_start = original_delay_ms / 1000.0
        # If TTS is still longer than its subtitle slot, move the next line
        # forward instead of mixing two voices on top of each other.
        effective_start = max(original_start, previous_end + minimum_gap)
        adjusted_results.append(
            (position, path, round(effective_start * 1000), fitted_duration)
        )
        previous_end = effective_start + fitted_duration

    for input_index, (_position, _path, delay_ms, _duration) in enumerate(adjusted_results):
        label = f"a{input_index}"
        filter_parts.append(
            f"[{input_index}:a]adelay={delay_ms}:all=1[{label}]"
        )
        labels.append(f"[{label}]")

    total_duration = max(cue.end for cue in cues)
    for (_position, _path, delay_ms, fitted_duration) in adjusted_results:
        total_duration = max(total_duration, delay_ms / 1000 + fitted_duration)
    total_duration += 0.20

    filter_parts.append(
        f"{''.join(labels)}"
        f"amix=inputs={len(labels)}:duration=longest:dropout_transition=0:normalize=0,"
        "acompressor=threshold=-18dB:ratio=2.2:attack=12:release=180:makeup=1.5dB,"
        "alimiter=limit=0.95:attack=5:release=80,"
        f"apad=pad_dur=0.20,atrim=0:{total_duration:.3f}[mix]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[mix]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(output_path),
        ]
    )
    await asyncio.to_thread(run_command, command)

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError("Final Khmer MP3 was not created correctly")


# ---------------------------------------------------------------------------
# Progress and processing
# ---------------------------------------------------------------------------

def progress_bar(percent: int) -> str:
    percent = max(0, min(100, percent))
    filled = round(percent / 10)
    return "█" * filled + "░" * (10 - filled)


async def update_progress(
    message: Message,
    percent: int,
    label: str,
    state: dict[str, int],
) -> None:
    percent = max(0, min(100, int(percent)))
    last_percent = state.get("last_percent", -10)

    if percent < 100 and percent - last_percent < 5:
        return

    state["last_percent"] = percent
    text = (
        f"⏳ {label}\n"
        f"<code>{progress_bar(percent)}</code> <b>{percent}%</b>\n\n"
        "សូមកុំផ្ញើឯកសារថ្មី រហូតដល់ការងារនេះចប់។"
    )
    try:
        await message.edit_text(text, parse_mode=ParseMode.HTML)
    except BadRequest as exc:
        if "message is not modified" not in str(exc).lower():
            logger.debug("Progress update rejected: %s", exc)
    except RetryAfter as exc:
        await asyncio.sleep(float(exc.retry_after))
    except (NetworkError, TimedOut) as exc:
        logger.debug("Temporary progress update failure: %s", exc)


async def send_outputs(
    update: Update,
    status: Message,
    srt_path: Path,
    mp3_path: Path,
) -> None:
    message = update.effective_message
    if message is None:
        raise RuntimeError("Telegram message is unavailable")

    caption_suffix = (
        f"\n🗑 នឹងលុបក្រោយ {AUTO_DELETE_MINUTES} នាទី"
        if AUTO_DELETE_MINUTES > 0
        else ""
    )

    with srt_path.open("rb") as srt_file:
        srt_message = await message.reply_document(
            document=srt_file,
            filename="khmer_dub.srt",
            caption="✅ Khmer SRT" + caption_suffix,
        )

    with mp3_path.open("rb") as mp3_file:
        mp3_message = await message.reply_audio(
            audio=mp3_file,
            filename="khmer_dub.mp3",
            title="KhmerDubAI Dub",
            caption="✅ Khmer Dub MP3" + caption_suffix,
        )

    track_message(srt_message)
    track_message(mp3_message)
    schedule_delete(status)
    schedule_delete(srt_message, DELETE_OUTPUT_MESSAGES)
    schedule_delete(mp3_message, DELETE_OUTPUT_MESSAGES)


async def process_source(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    source_path: Path,
    tmpdir: Path,
    is_srt: bool,
) -> None:
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None:
        return

    status = await message.reply_text("⏳ កំពុងចាប់ផ្ដើម… 0%")
    track_message(status)
    state = {"last_percent": -10}

    async def progress(percent: int, label: str) -> None:
        session = get_session(chat.id)
        session.last_activity = time.monotonic()
        await update_progress(status, percent, label, state)

    try:
        if is_srt:
            await progress(5, "កំពុងអានឯកសារ SRT")
            source_srt = source_path.read_text(
                encoding="utf-8-sig",
                errors="replace",
            )
            srt_signature(source_srt)
            await progress(15, "កំពុងបកប្រែជាភាសាខ្មែរ")
        else:
            media_for_whisper = source_path
            await progress(3, "កំពុងទទួលឯកសារ")

            if source_path.suffix.lower() in VIDEO_SUFFIXES:
                await progress(8, "កំពុងដកសំឡេងចេញពីវីដេអូ")
                wav_path = tmpdir / "audio.wav"
                await asyncio.to_thread(extract_audio, source_path, wav_path)
                media_for_whisper = wav_path

            await progress(12, "កំពុងស្គាល់សំឡេងចិន")
            async with _whisper_lock:
                source_srt = await asyncio.to_thread(
                    transcribe_to_srt,
                    media_for_whisper,
                )
            await progress(30, "ស្គាល់សំឡេងរួច កំពុងបកប្រែ")

        khmer_srt = await asyncio.wait_for(
            asyncio.to_thread(translate_to_khmer_srt, source_srt),
            timeout=TRANSLATION_TIMEOUT_SECONDS,
        )

        srt_path = tmpdir / "khmer_dub.srt"
        srt_path.write_text(khmer_srt, encoding="utf-8")
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
        await send_outputs(update, status, srt_path, mp3_path)
        await progress(100, "ការងាររួចរាល់ ✅")

    except asyncio.CancelledError:
        schedule_delete(status)
        raise
    except Exception as exc:
        logger.exception("Media processing failed")
        friendly = str(exc).strip() or exc.__class__.__name__
        if len(friendly) > 700:
            friendly = friendly[:700] + "…"
        try:
            await status.edit_text(
                "❌ ការងារមិនបានសម្រេច\n\n"
                f"មូលហេតុ៖ {friendly}\n\n"
                "សូមសាកវីដេអូខ្លី 20–30 វិនាទី ឬផ្ញើ SRT មកសាកជាមុន។"
            )
        except Exception:
            pass
        raise


def safe_filename(original: str | None, fallback: str) -> str:
    name = Path(original or fallback).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name[:120] or fallback


async def run_downloaded_job(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    telegram_media,
    filename: str,
    is_srt: bool = False,
) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return

    if not await require_project(update, context):
        return
    if not await set_processing_task(chat.id):
        return

    try:
        if telegram_media.file_size and telegram_media.file_size > MAX_FILE_MB * 1024 * 1024:
            reply = await message.reply_text(
                f"ឯកសារធំពេក។ កំណត់បច្ចុប្បន្ន៖ {MAX_FILE_MB} MB"
            )
            track_message(reply)
            return

        if (
            not is_srt
            and getattr(telegram_media, "duration", None)
            and telegram_media.duration > MAX_MEDIA_SECONDS
        ):
            reply = await message.reply_text(
                f"ឯកសារវែងជាង {MAX_MEDIA_SECONDS // 60} នាទី។ "
                "សូមកាត់ឱ្យខ្លីជាងនេះ។"
            )
            track_message(reply)
            return

        with tempfile.TemporaryDirectory(prefix="khmerdubai_") as tmp:
            tmpdir = Path(tmp)
            source_path = tmpdir / safe_filename(filename, "upload.bin")

            telegram_file = await telegram_media.get_file()
            await telegram_file.download_to_drive(custom_path=source_path)

            if not is_srt:
                await asyncio.to_thread(validate_media_duration, source_path)

            await process_source(
                update,
                context,
                source_path,
                tmpdir,
                is_srt=is_srt,
            )

    except asyncio.CancelledError:
        notice = await message.reply_text(
            "🛑 ការងារចាស់ត្រូវបានបញ្ឈប់។ "
            "អ្នកអាចចាប់ផ្ដើមគម្រោងថ្មីបាន។"
        )
        track_message(notice)
        raise
    except Exception as exc:
        logger.exception("Processing failed")
        error_text = str(exc)
        if len(error_text) > 900:
            error_text = error_text[:900] + "…"
        reply = await message.reply_text(
            f"❌ មានបញ្ហាពេលដំណើរការ៖\n{error_text}"
        )
        track_message(reply)
    finally:
        await clear_processing_task(chat.id)
        if DELETE_USER_UPLOAD:
            schedule_delete(message)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return

    session = get_session(chat.id)
    if session.expiry_task and not session.expiry_task.done():
        session.expiry_task.cancel()

    session.active = False
    session.processing_task = None
    session.last_activity = time.monotonic()

    reply = await message.reply_text(
        "🤖 KhmerDubAI Free Offline Translator\n\n"
        "ចុច «🆕 ធ្វើថ្មី» រួចផ្ញើវីដេអូ សំឡេង ឬ SRT រឿងចិន។\n\n"
        f"⏱ កំណត់៖ {MAX_MEDIA_SECONDS // 60} នាទី ឬតិចជាងនេះ\n"
        "🎭 បែងចែកប្រុស ស្រី ក្មេង មនុស្សចាស់ និងសំឡេងគិត\n"
        "🆓 មិនត្រូវការ Gemini API Key ឬម៉ូដែល NLLB ធំៗ\n"
        "📦 លទ្ធផល៖ khmer_dub.srt និង khmer_dub.mp3",
        reply_markup=PROJECT_KEYBOARD,
    )
    track_message(reply)


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message is None:
        return
    reply = await message.reply_text(
        "ℹ️ របៀបប្រើ KhmerDubAI\n\n"
        "1. ចុច «🆕 ធ្វើថ្មី»\n"
        f"2. ផ្ញើវីដេអូ សំឡេង ឬ SRT មិនលើស {MAX_MEDIA_SECONDS // 60} នាទី\n"
        "3. រង់ចាំ Progress 0%–100%\n"
        "4. ទទួល khmer_dub.srt និង khmer_dub.mp3",
        reply_markup=PROJECT_KEYBOARD,
    )
    track_message(reply)


async def new_project(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    chat = update.effective_chat
    message = update.effective_message
    if chat is None or message is None:
        return

    session = get_session(chat.id)
    async with session.state_lock:
        running = session.processing_task
        if running and not running.done() and running is not asyncio.current_task():
            running.cancel()

        if session.expiry_task and not session.expiry_task.done():
            session.expiry_task.cancel()

        await clear_previous_project(
            context.bot,
            chat.id,
            keep_message_id=message.message_id,
        )

        session.generation += 1
        session.active = True
        session.last_activity = time.monotonic()
        session.processing_task = None
        session.message_ids = {message.message_id}

    reply = await message.reply_text(
        "✅ Project ថ្មីរួចរាល់។\n\n"
        "📤 ឥឡូវផ្ញើវីដេអូ សំឡេង ឬ SRT រឿងចិនបាន។\n"
        f"⏱ រយៈពេលត្រូវត្រឹម {MAX_MEDIA_SECONDS // 60} នាទី ឬតិចជាងនេះ។\n"
        "📦 Bot នឹងផ្ញើ Khmer SRT និង Khmer MP3។",
        reply_markup=PROJECT_KEYBOARD,
    )
    track_message(reply)
    touch_session(context, chat.id)


async def voice_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message is None:
        return

    text = " ".join(context.args).strip()
    if not text and message.reply_to_message:
        text = (message.reply_to_message.text or "").strip()

    if not text:
        reply = await message.reply_text(
            "ប្រើ៖ /voice អត្ថបទខ្មែរ\n"
            "ឬ Reply លើសារខ្មែរ ហើយផ្ញើ /voice"
        )
        schedule_delete(reply)
        return

    if len(text) > 3000:
        reply = await message.reply_text(
            "អត្ថបទវែងពេក។ /voice អនុញ្ញាតអតិបរមា 3000 តួអក្សរ។"
        )
        schedule_delete(reply)
        return

    with tempfile.TemporaryDirectory(prefix="khmerdubai_voice_") as tmp:
        output_path = Path(tmp) / "khmer_voice.mp3"
        try:
            await message.chat.send_action(ChatAction.RECORD_VOICE)
            await asyncio.wait_for(
                edge_tts.Communicate(
                    text=text,
                    voice=MALE_VOICE,
                ).save(str(output_path)),
                timeout=TTS_TIMEOUT_SECONDS,
            )
            with output_path.open("rb") as audio:
                result = await message.reply_audio(
                    audio=audio,
                    filename="khmer_voice.mp3",
                    title="KhmerDubAI Voice",
                    caption=(
                        f"🗑 នឹងលុបក្រោយ {AUTO_DELETE_MINUTES} នាទី"
                        if AUTO_DELETE_MINUTES > 0
                        else None
                    ),
                )
            schedule_delete(result, DELETE_OUTPUT_MESSAGES)
        except Exception as exc:
            logger.exception("/voice TTS failed")
            reply = await message.reply_text(
                f"❌ មិនអាចបង្កើតសំឡេងបាន៖ {exc}"
            )
            schedule_delete(reply)


async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message is None or message.document is None:
        return

    document = message.document
    filename = safe_filename(document.file_name, "document")
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        reply = await message.reply_text(
            "សូមផ្ញើ SRT, MP3, M4A, WAV, OGG, AAC, FLAC ឬ Video។"
        )
        track_message(reply)
        return

    await run_downloaded_job(
        update,
        context,
        document,
        filename,
        is_srt=(suffix == ".srt"),
    )


async def handle_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message and message.audio:
        filename = safe_filename(message.audio.file_name, "audio.mp3")
        await run_downloaded_job(
            update,
            context,
            message.audio,
            filename,
        )


async def handle_voice(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message and message.voice:
        await run_downloaded_job(
            update,
            context,
            message.voice,
            "voice.ogg",
        )


async def handle_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message and message.video:
        filename = safe_filename(message.video.file_name, "video.mp4")
        await run_downloaded_job(
            update,
            context,
            message.video,
            filename,
        )


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if isinstance(context.error, asyncio.CancelledError):
        return
    logger.error(
        "Unhandled exception while processing update %r",
        update,
        exc_info=context.error,
    )


async def post_shutdown(application: Application) -> None:
    pending: list[asyncio.Task] = []
    for session in sessions.values():
        for task in (session.expiry_task, session.processing_task):
            if task and not task.done():
                task.cancel()
                pending.append(task)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)



def main() -> None:
    validate_runtime()
    initialize_clients()

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(UPDATE_CONCURRENCY)
        .read_timeout(300)
        .write_timeout(300)
        .connect_timeout(60)
        .pool_timeout(60)
        .post_shutdown(post_shutdown)
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
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_error_handler(error_handler)

    logger.info(
        "KhmerDubAI | max=%ss | idle=%ss | whisper=%s | "
        "translator=free-web | tts_parallel=%s | update_parallel=%s",
        MAX_MEDIA_SECONDS,
        SESSION_IDLE_SECONDS,
        WHISPER_MODEL,
        TTS_CONCURRENCY,
        UPDATE_CONCURRENCY,
    )
    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
