from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import edge_tts
from faster_whisper import WhisperModel
from google import genai
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ChatAction
from telegram.error import NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


# =============================================================================
# Configuration
# =============================================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("khmerdubai.v10")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "100"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "1")))
TELEGRAM_SEND_LIMIT_MB = int(os.getenv("TELEGRAM_SEND_LIMIT_MB", "49"))
TELEGRAM_SEND_LIMIT_BYTES = TELEGRAM_SEND_LIMIT_MB * 1024 * 1024

TTS_CONCURRENCY = max(1, int(os.getenv("TTS_CONCURRENCY", "2")))
TTS_RETRIES = max(1, int(os.getenv("TTS_RETRIES", "3")))
GEMINI_RETRIES = max(1, int(os.getenv("GEMINI_RETRIES", "4")))

AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "24000"))
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "128k")
MIX_ORIGINAL_VOLUME = float(os.getenv("MIX_ORIGINAL_VOLUME", "0.18"))

ALLOWED_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi",
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".webm",
}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".oga"}

VALID_TAGS = {
    "M", "F", "BOY", "GIRL", "OLD_M", "OLD_F",
    "M_THINK", "F_THINK", "BOY_THINK", "GIRL_THINK",
    "OLD_M_THINK", "OLD_F_THINK", "NARRATOR", "SYSTEM",
    "CROWD", "UNKNOWN",
}

TAG_PATTERN = re.compile(
    r"^\[(M|F|BOY|GIRL|OLD_M|OLD_F|M_THINK|F_THINK|"
    r"BOY_THINK|GIRL_THINK|OLD_M_THINK|OLD_F_THINK|"
    r"NARRATOR|SYSTEM|CROWD|UNKNOWN)\]\s*",
    re.IGNORECASE,
)
CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MULTISPACE_PATTERN = re.compile(r"[ \t]+")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?។៕,，。！？])\1{1,}")
SRT_TIME_PATTERN = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*$"
)

# Edge-TTS currently exposes two Khmer base voices. Child/elderly/thought
# profiles use gentle rate/volume changes only; pitch stays neutral to reduce
# unstable, robotic sound.
VOICE_PROFILES: dict[str, dict[str, str]] = {
    "M":            {"voice": "km-KH-PisethNeural", "rate": "-5%",  "pitch": "+0Hz", "volume": "+0%"},
    "F":            {"voice": "km-KH-SreymomNeural", "rate": "-5%", "pitch": "+0Hz", "volume": "+0%"},
    "BOY":          {"voice": "km-KH-PisethNeural", "rate": "+2%",  "pitch": "+0Hz", "volume": "+0%"},
    "GIRL":         {"voice": "km-KH-SreymomNeural", "rate": "+2%", "pitch": "+0Hz", "volume": "+0%"},
    "OLD_M":        {"voice": "km-KH-PisethNeural", "rate": "-12%", "pitch": "+0Hz", "volume": "+0%"},
    "OLD_F":        {"voice": "km-KH-SreymomNeural", "rate": "-12%","pitch": "+0Hz", "volume": "+0%"},
    "M_THINK":      {"voice": "km-KH-PisethNeural", "rate": "-9%",  "pitch": "+0Hz", "volume": "-5%"},
    "F_THINK":      {"voice": "km-KH-SreymomNeural", "rate": "-9%", "pitch": "+0Hz", "volume": "-5%"},
    "BOY_THINK":    {"voice": "km-KH-PisethNeural", "rate": "-3%",  "pitch": "+0Hz", "volume": "-5%"},
    "GIRL_THINK":   {"voice": "km-KH-SreymomNeural", "rate": "-3%", "pitch": "+0Hz", "volume": "-5%"},
    "OLD_M_THINK":  {"voice": "km-KH-PisethNeural", "rate": "-15%", "pitch": "+0Hz", "volume": "-6%"},
    "OLD_F_THINK":  {"voice": "km-KH-SreymomNeural", "rate": "-15%","pitch": "+0Hz", "volume": "-6%"},
    "NARRATOR":     {"voice": "km-KH-PisethNeural", "rate": "-8%",  "pitch": "+0Hz", "volume": "+0%"},
    "SYSTEM":       {"voice": "km-KH-SreymomNeural", "rate": "-4%", "pitch": "+0Hz", "volume": "+0%"},
    "CROWD":        {"voice": "km-KH-PisethNeural", "rate": "-2%",  "pitch": "+0Hz", "volume": "-2%"},
    "UNKNOWN":      {"voice": "km-KH-PisethNeural", "rate": "-5%",  "pitch": "+0Hz", "volume": "+0%"},
}

PROGRESS_TEXT = {
    "download": "⬇️ 1/12 កំពុងទាញយកឯកសារ…",
    "extract": "🎵 2/12 កំពុងទាញសំឡេង និងកែសម្រួល…",
    "transcribe": "📝 3/12 កំពុងស្គាល់សំឡេងភាសាចិន…",
    "translate": "🌐 4/12 កំពុងបកប្រែជាភាសាខ្មែរ…",
    "speaker": "👥 5/12 កំពុងកំណត់ប្រភេទតួអង្គ…",
    "srt": "📄 6/12 កំពុងរៀបចំឯកសារ SRT…",
    "review": "✍️ 7/12 សូមពិនិត្យ ឬកែ SRT មុនបង្កើតសំឡេង។",
    "tts": "🎙️ 8/12 កំពុងបង្កើតសំឡេងខ្មែរ…",
    "combine": "🎧 9/12 កំពុងរៀបចំ MP3 ជាប់តាមពេលវេលា…",
    "mp4": "🎬 10/12 កំពុងបង្កើតវីដេអូ MP4…",
    "send": "📤 11/12 កំពុងផ្ញើឯកសារ…",
    "clean": "🧹 12/12 កំពុងសម្អាតឯកសារបណ្ដោះអាសន្ន…",
}

REVIEW_KEYBOARD = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("🎧 Generate MP3", callback_data="v10:mp3"),
            InlineKeyboardButton("🎬 Generate MP4", callback_data="v10:mp4"),
        ],
        [
            InlineKeyboardButton("📄 Upload Edited SRT", callback_data="v10:upload_srt"),
            InlineKeyboardButton("❌ Cancel Project", callback_data="v10:cancel"),
        ],
    ]
)

AUDIO_MODE_KEYBOARD = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton("🔇 Replace Original Audio", callback_data="v10:mp4:replace")],
        [InlineKeyboardButton("🎚 Mix With Original Audio", callback_data="v10:mp4:mix")],
        [InlineKeyboardButton("⬅️ Back", callback_data="v10:back")],
    ]
)


# =============================================================================
# Data models
# =============================================================================

@dataclass(slots=True)
class Subtitle:
    index: int
    start_ms: int
    end_ms: int
    text: str
    tag: str = "M"

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)


@dataclass(slots=True)
class Project:
    user_id: int
    chat_id: int
    project_id: str
    root: Path
    source_path: Optional[Path] = None
    normalized_wav: Optional[Path] = None
    chinese_srt: Optional[Path] = None
    khmer_srt: Optional[Path] = None
    dubbed_mp3: Optional[Path] = None
    dubbed_mp4: Optional[Path] = None
    progress_message_id: Optional[int] = None
    waiting_for_srt: bool = False
    processing: bool = False
    cancelled: bool = False
    created_at: float = field(default_factory=time.time)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@dataclass(slots=True)
class TTSGroup:
    tag: str
    start_ms: int
    end_ms: int
    text: str
    members: list[Subtitle]


# =============================================================================
# Global services
# =============================================================================

JOB_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
USER_LOCKS: dict[int, asyncio.Lock] = {}
PROJECTS: dict[int, Project] = {}

_gemini_client: Optional[genai.Client] = None
_whisper_model: Optional[WhisperModel] = None
_whisper_lock = asyncio.Lock()


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


async def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        async with _whisper_lock:
            if _whisper_model is None:
                logger.info(
                    "Loading Faster-Whisper model=%s device=%s compute_type=%s",
                    WHISPER_MODEL,
                    WHISPER_DEVICE,
                    WHISPER_COMPUTE_TYPE,
                )
                _whisper_model = await asyncio.to_thread(
                    WhisperModel,
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
    return _whisper_model


# =============================================================================
# Utility helpers
# =============================================================================

def sanitize_filename(name: str) -> str:
    name = Path(name or "upload.bin").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return stem[:120] or "upload.bin"


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def require_system_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if not command_exists(tool)]
    if missing:
        raise RuntimeError(
            "Missing system tools: "
            + ", ".join(missing)
            + ". Install FFmpeg before running KhmerDubAI."
        )


async def run_command(
    args: list[str],
    *,
    timeout: int = 3600,
    cwd: Optional[Path] = None,
) -> tuple[str, str]:
    logger.debug("Running command: %s", " ".join(args))
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        raise RuntimeError(f"Command timed out after {timeout}s: {args[0]}") from None

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    if process.returncode != 0:
        logger.error("Command failed (%s): %s", process.returncode, stderr[-4000:])
        raise RuntimeError(f"{args[0]} failed: {stderr[-1200:]}")
    return stdout, stderr


async def update_progress(
    context: ContextTypes.DEFAULT_TYPE,
    project: Project,
    text: str,
) -> None:
    if project.cancelled:
        raise asyncio.CancelledError("Project cancelled by user")

    try:
        if project.progress_message_id:
            await context.bot.edit_message_text(
                chat_id=project.chat_id,
                message_id=project.progress_message_id,
                text=text,
            )
        else:
            msg = await context.bot.send_message(project.chat_id, text)
            project.progress_message_id = msg.message_id
    except TelegramError as exc:
        # "Message is not modified" and temporary edit failures should not abort a job.
        logger.warning("Progress update failed: %s", exc)


def ms_to_srt(ms: int) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt_time(value: str) -> tuple[int, int]:
    match = SRT_TIME_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")

    parts = [int(item) for item in match.groups()]
    sh, sm, ss, sms, eh, em, es, ems = parts
    start = ((sh * 60 + sm) * 60 + ss) * 1000 + min(sms, 999)
    end = ((eh * 60 + em) * 60 + es) * 1000 + min(ems, 999)
    if end <= start:
        end = start + 300
    return start, end


def clean_text(text: str) -> str:
    text = CONTROL_PATTERN.sub("", text or "")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_khmer_tts_text(text: str) -> str:
    text = TAG_PATTERN.sub("", clean_text(text))
    text = CJK_PATTERN.sub("", text)
    replacements = {
        "，": ", ",
        "。": "។ ",
        "！": "! ",
        "？": "? ",
        "：": ": ",
        "；": "; ",
        "…": "… ",
        "\n": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)
    text = MULTISPACE_PATTERN.sub(" ", text)
    return text.strip(" \t\r\n-")


def split_tagged_text(text: str) -> tuple[str, str]:
    text = clean_text(text)
    match = TAG_PATTERN.match(text)
    if not match:
        return "M", text
    tag = match.group(1).upper()
    if tag == "UNKNOWN":
        tag = "M"
    return tag if tag in VALID_TAGS else "M", text[match.end():].strip()


def subtitles_to_srt(subtitles: Iterable[Subtitle]) -> str:
    blocks: list[str] = []
    for number, sub in enumerate(subtitles, start=1):
        tag = sub.tag if sub.tag in VALID_TAGS and sub.tag != "UNKNOWN" else "M"
        text = normalize_khmer_tts_text(sub.text)
        if not text:
            text = "…"
        blocks.append(
            f"{number}\n"
            f"{ms_to_srt(sub.start_ms)} --> {ms_to_srt(sub.end_ms)}\n"
            f"[{tag}] {text}"
        )
    return "\n\n".join(blocks) + "\n"


def parse_srt(content: str) -> list[Subtitle]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    raw_blocks = re.split(r"\n\s*\n", content)
    result: list[Subtitle] = []

    for raw in raw_blocks:
        lines = [clean_text(line) for line in raw.splitlines() if clean_text(line)]
        if len(lines) < 2:
            continue

        timestamp_idx = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if timestamp_idx < 0:
            continue

        try:
            start_ms, end_ms = parse_srt_time(lines[timestamp_idx])
        except ValueError:
            continue

        dialogue = " ".join(lines[timestamp_idx + 1:]).strip()
        if not dialogue:
            continue

        tag, text = split_tagged_text(dialogue)
        text = normalize_khmer_tts_text(text)
        if not text:
            continue

        result.append(
            Subtitle(
                index=len(result) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                tag=tag,
            )
        )

    if not result:
        raise ValueError("No valid subtitle blocks were found.")

    result.sort(key=lambda item: (item.start_ms, item.end_ms))
    for index, item in enumerate(result, start=1):
        item.index = index
    return result


def safe_json_loads(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Gemini did not return a JSON array.")
    return json.loads(text[start:end + 1])


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]


def new_project(user_id: int, chat_id: int) -> Project:
    previous = PROJECTS.pop(user_id, None)
    if previous:
        previous.cancelled = True
        previous.cleanup()

    root = Path(tempfile.mkdtemp(prefix=f"khmerdub_v10_{user_id}_"))
    project = Project(
        user_id=user_id,
        chat_id=chat_id,
        project_id=uuid.uuid4().hex,
        root=root,
    )
    PROJECTS[user_id] = project
    return project


def get_project(user_id: int) -> Project:
    project = PROJECTS.get(user_id)
    if not project:
        raise RuntimeError("No active project. Upload a video or audio file first.")
    return project


# =============================================================================
# Media inspection and conversion
# =============================================================================

async def probe_duration_ms(path: Path) -> int:
    stdout, _ = await run_command(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout=60,
    )
    try:
        return max(1, int(float(stdout.strip()) * 1000))
    except (TypeError, ValueError):
        return 1


async def has_audio_stream(path: Path) -> bool:
    stdout, _ = await run_command(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(path),
        ],
        timeout=60,
    )
    return bool(stdout.strip())


async def extract_normalized_audio(source: Path, output_wav: Path) -> None:
    if not await has_audio_stream(source):
        raise RuntimeError("The uploaded file does not contain a readable audio stream.")

    await run_command(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "-af", "highpass=f=70,lowpass=f=7600,dynaudnorm=f=150:g=7",
            str(output_wav),
        ],
        timeout=3600,
    )


# =============================================================================
# Speech recognition and Gemini translation
# =============================================================================

def transcribe_sync(model: WhisperModel, audio_path: Path) -> list[Subtitle]:
    segments, info = model.transcribe(
        str(audio_path),
        language="zh",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 200,
        },
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    logger.info(
        "Whisper detected language=%s probability=%.3f",
        getattr(info, "language", "unknown"),
        float(getattr(info, "language_probability", 0.0)),
    )

    subtitles: list[Subtitle] = []
    for segment in segments:
        text = clean_text(segment.text)
        if not text:
            continue
        start_ms = max(0, round(float(segment.start) * 1000))
        end_ms = max(start_ms + 250, round(float(segment.end) * 1000))
        subtitles.append(
            Subtitle(
                index=len(subtitles) + 1,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
                tag="M",
            )
        )

    if not subtitles:
        raise RuntimeError("Whisper could not detect any spoken dialogue.")
    return subtitles


async def transcribe_audio(audio_path: Path) -> list[Subtitle]:
    model = await get_whisper_model()
    return await asyncio.to_thread(transcribe_sync, model, audio_path)


TRANSLATION_SYSTEM_PROMPT = """
You are the translation and speaker-classification engine for KhmerDubAI V10.

Translate Chinese drama/movie dialogue into fluent, natural SPOKEN Khmer suitable
for professional dubbing. Never translate mechanically word-for-word.

Speaker tags:
M, F, BOY, GIRL, OLD_M, OLD_F, M_THINK, F_THINK, BOY_THINK,
GIRL_THINK, OLD_M_THINK, OLD_F_THINK, NARRATOR, SYSTEM, CROWD, UNKNOWN.

Rules:
1. Preserve the exact id for every input item.
2. Return exactly one output item for every input item. Never skip reactions,
   short lines, names, questions, insults, or emotional words.
3. Infer age/gender/thought/narration from the nearby dialogue and context.
4. Keep a character type consistent across connected lines. Do not randomly
   switch an apparent character between adult, child, elderly, male, and female.
5. Use M when truly uncertain instead of UNKNOWN.
6. Khmer must sound natural in daily speech and retain emotion, status,
   relationship, politeness, threats, comedy, fear, sadness, or affection.
7. Choose Khmer pronouns according to relationship and rank.
8. Keep each line concise enough for its displayed duration.
9. Do not include Chinese characters in the Khmer output.
10. Do not add explanations, markdown, or code fences.

Return only a valid JSON array:
[{"id": 1, "tag": "M", "text": "ភាសាខ្មែរ"}, ...]
""".strip()


async def gemini_generate(prompt: str) -> str:
    client = get_gemini_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text
        except Exception as exc:  # SDK may expose several transport exception types.
            last_error = exc
            if attempt >= GEMINI_RETRIES:
                break
            delay = min(20, 2 ** (attempt - 1))
            logger.warning(
                "Gemini attempt %s/%s failed: %s; retrying in %ss",
                attempt,
                GEMINI_RETRIES,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(f"Gemini translation failed: {last_error}") from last_error


def make_translation_batches(subtitles: list[Subtitle], batch_size: int = 35) -> list[list[Subtitle]]:
    return [subtitles[i:i + batch_size] for i in range(0, len(subtitles), batch_size)]


async def translate_batch(
    batch: list[Subtitle],
    previous_context: list[Subtitle],
) -> list[Subtitle]:
    context_payload = [
        {"id": item.index, "text": item.text, "tag": item.tag}
        for item in previous_context[-5:]
    ]
    input_payload = [
        {
            "id": item.index,
            "start": ms_to_srt(item.start_ms),
            "end": ms_to_srt(item.end_ms),
            "duration_ms": item.duration_ms,
            "chinese": item.text,
        }
        for item in batch
    ]
    prompt = (
        TRANSLATION_SYSTEM_PROMPT
        + "\n\nPrevious context (do not output these again):\n"
        + json.dumps(context_payload, ensure_ascii=False)
        + "\n\nTranslate these items:\n"
        + json.dumps(input_payload, ensure_ascii=False)
    )

    raw = await gemini_generate(prompt)
    data = safe_json_loads(raw)
    if not isinstance(data, list):
        raise ValueError("Gemini JSON response is not a list.")

    by_id: dict[int, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            item_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        by_id[item_id] = row

    output: list[Subtitle] = []
    missing: list[int] = []
    for original in batch:
        row = by_id.get(original.index)
        if not row:
            missing.append(original.index)
            continue

        tag = str(row.get("tag", "M")).upper().strip("[] ")
        if tag not in VALID_TAGS or tag == "UNKNOWN":
            tag = "M"

        text = normalize_khmer_tts_text(str(row.get("text", "")))
        if not text:
            missing.append(original.index)
            continue

        output.append(
            Subtitle(
                index=original.index,
                start_ms=original.start_ms,
                end_ms=original.end_ms,
                text=text,
                tag=tag,
            )
        )

    if missing:
        raise ValueError(f"Gemini omitted or invalidated subtitle ids: {missing}")

    return output


async def translate_subtitles(subtitles: list[Subtitle]) -> list[Subtitle]:
    translated: list[Subtitle] = []
    batches = make_translation_batches(subtitles)
    for number, batch in enumerate(batches, start=1):
        logger.info("Translating batch %s/%s", number, len(batches))
        translated.extend(await translate_batch(batch, translated))
    translated.sort(key=lambda item: item.index)
    return translated


# =============================================================================
# TTS grouping, synthesis, and timeline construction
# =============================================================================

def build_tts_groups(
    subtitles: list[Subtitle],
    *,
    max_gap_ms: int = 220,
    max_group_ms: int = 11_000,
    max_chars: int = 180,
) -> list[TTSGroup]:
    """
    Join only short, consecutive lines from the same speaker. This reduces tone
    resets while preserving SRT timing. It intentionally does not group different
    speakers or large pauses.
    """
    groups: list[TTSGroup] = []
    current: Optional[TTSGroup] = None

    for sub in subtitles:
        text = normalize_khmer_tts_text(sub.text)
        if not text:
            continue

        can_join = (
            current is not None
            and current.tag == sub.tag
            and 0 <= sub.start_ms - current.end_ms <= max_gap_ms
            and sub.end_ms - current.start_ms <= max_group_ms
            and len(current.text) + len(text) + 2 <= max_chars
        )
        if can_join:
            current.text = f"{current.text}។ {text}"
            current.end_ms = sub.end_ms
            current.members.append(sub)
        else:
            current = TTSGroup(
                tag=sub.tag,
                start_ms=sub.start_ms,
                end_ms=sub.end_ms,
                text=text,
                members=[sub],
            )
            groups.append(current)

    return groups


async def synthesize_tts(group: TTSGroup, output_mp3: Path) -> None:
    profile = VOICE_PROFILES.get(group.tag, VOICE_PROFILES["M"])
    text = normalize_khmer_tts_text(group.text)
    if not text:
        raise ValueError("Cannot synthesize empty TTS text.")

    last_error: Optional[Exception] = None
    for attempt in range(1, TTS_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=profile["voice"],
                rate=profile["rate"],
                volume=profile["volume"],
                pitch=profile["pitch"],
            )
            await asyncio.wait_for(communicate.save(str(output_mp3)), timeout=180)
            if not output_mp3.exists() or output_mp3.stat().st_size < 256:
                raise RuntimeError("Edge-TTS produced an empty audio file.")
            return
        except Exception as exc:
            last_error = exc
            output_mp3.unlink(missing_ok=True)
            if attempt >= TTS_RETRIES:
                break
            await asyncio.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError(f"Edge-TTS failed after {TTS_RETRIES} attempts: {last_error}")


async def convert_clip_to_wav(source_mp3: Path, output_wav: Path) -> None:
    await run_command(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_mp3),
            "-ac", "1",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            "-af", "afade=t=in:st=0:d=0.015,areverse,"
                   "afade=t=in:st=0:d=0.02,areverse,"
                   "loudnorm=I=-18:TP=-2:LRA=8",
            str(output_wav),
        ],
        timeout=180,
    )


async def clip_duration_ms(path: Path) -> int:
    return await probe_duration_ms(path)


def atempo_chain(speed: float) -> str:
    """
    FFmpeg atempo accepts 0.5..100 in recent builds, but chaining near 1.0 is
    safer and easier to reason about. speed > 1 makes speech shorter.
    """
    speed = max(0.85, min(speed, 1.18))
    return f"atempo={speed:.5f}"


async def fit_clip_safely(
    input_wav: Path,
    output_wav: Path,
    available_ms: int,
) -> None:
    duration = await clip_duration_ms(input_wav)
    if duration <= available_ms or available_ms <= 0:
        shutil.copy2(input_wav, output_wav)
        return

    required_speed = duration / max(1, available_ms)
    if required_speed <= 1.18:
        await run_command(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_wav),
                "-filter:a", atempo_chain(required_speed),
                "-ac", "1",
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                str(output_wav),
            ],
            timeout=180,
        )
    else:
        # Do not aggressively accelerate or cut speech. Preserve it and log that
        # it may overlap the following segment.
        logger.warning(
            "TTS clip %s needs unsafe speed %.3f; preserving natural speech.",
            input_wav.name,
            required_speed,
        )
        shutil.copy2(input_wav, output_wav)


async def create_silence_wav(duration_ms: int, output_wav: Path) -> None:
    duration_sec = max(0.001, duration_ms / 1000)
    await run_command(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={AUDIO_SAMPLE_RATE}:cl=mono",
            "-t", f"{duration_sec:.3f}",
            "-c:a", "pcm_s16le",
            str(output_wav),
        ],
        timeout=180,
    )


async def mix_group_clips(
    clips: list[tuple[int, Path]],
    total_duration_ms: int,
    output_mp3: Path,
) -> None:
    if not clips:
        raise RuntimeError("No generated TTS clips to combine.")

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for _, clip in clips:
        args.extend(["-i", str(clip)])

    filters: list[str] = []
    labels: list[str] = []
    for index, (start_ms, _) in enumerate(clips):
        label = f"a{index}"
        filters.append(
            f"[{index}:a]adelay={max(0, start_ms)}|{max(0, start_ms)},"
            f"aresample={AUDIO_SAMPLE_RATE},aformat=sample_fmts=fltp:"
            f"sample_rates={AUDIO_SAMPLE_RATE}:channel_layouts=mono[{label}]"
        )
        labels.append(f"[{label}]")

    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:"
        f"dropout_transition=0,alimiter=limit=0.95,"
        f"apad=whole_dur={max(0.1, total_duration_ms / 1000):.3f},"
        f"atrim=0:{max(0.1, total_duration_ms / 1000):.3f}[out]"
    )

    args.extend(
        [
            "-filter_complex", ";".join(filters),
            "-map", "[out]",
            "-ac", "1",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-c:a", "libmp3lame",
            "-b:a", AUDIO_BITRATE,
            str(output_mp3),
        ]
    )
    await run_command(args, timeout=3600)


async def generate_dubbed_mp3(
    project: Project,
    subtitles: list[Subtitle],
    context: ContextTypes.DEFAULT_TYPE,
) -> Path:
    groups = build_tts_groups(subtitles)
    if not groups:
        raise RuntimeError("The SRT contains no usable Khmer dialogue.")

    tts_dir = project.root / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY)

    async def create_one(index: int, group: TTSGroup) -> tuple[int, Path]:
        if project.cancelled:
            raise asyncio.CancelledError
        raw_mp3 = tts_dir / f"{index:05d}_raw.mp3"
        normalized_wav = tts_dir / f"{index:05d}_normalized.wav"
        fitted_wav = tts_dir / f"{index:05d}_fitted.wav"

        async with tts_semaphore:
            await synthesize_tts(group, raw_mp3)
        await convert_clip_to_wav(raw_mp3, normalized_wav)

        # Give a little room up to the next group's start, but never use a
        # negative window. This helps avoid cutting final syllables.
        next_start = groups[index + 1].start_ms if index + 1 < len(groups) else group.end_ms + 700
        available = max(group.end_ms - group.start_ms, next_start - group.start_ms - 40)
        await fit_clip_safely(normalized_wav, fitted_wav, available)
        return group.start_ms, fitted_wav

    tasks = [asyncio.create_task(create_one(i, group)) for i, group in enumerate(groups)]
    completed: list[tuple[int, Path]] = []
    try:
        for done_number, future in enumerate(asyncio.as_completed(tasks), start=1):
            completed.append(await future)
            if done_number == 1 or done_number == len(tasks) or done_number % 5 == 0:
                percent = round(done_number * 100 / len(tasks))
                await update_progress(
                    context,
                    project,
                    f"{PROGRESS_TEXT['tts']}\n{done_number}/{len(tasks)} ({percent}%)",
                )
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    completed.sort(key=lambda pair: pair[0])
    total_duration = max(
        await probe_duration_ms(project.source_path) if project.source_path else 1,
        max(item.end_ms for item in subtitles) + 700,
    )

    output = project.root / "KhmerDubAI_V10_dubbed.mp3"
    await update_progress(context, project, PROGRESS_TEXT["combine"])
    await mix_group_clips(completed, total_duration, output)
    project.dubbed_mp3 = output
    return output


# =============================================================================
# MP4 creation
# =============================================================================

async def create_dubbed_video(project: Project, mode: str) -> Path:
    if not project.source_path or not is_video(project.source_path):
        raise RuntimeError("MP4 generation requires an uploaded video file.")
    if not project.dubbed_mp3 or not project.dubbed_mp3.exists():
        raise RuntimeError("Generate the Khmer MP3 before creating MP4.")

    output = project.root / f"KhmerDubAI_V10_{mode}.mp4"

    if mode == "replace":
        await run_command(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(project.source_path),
                "-i", str(project.dubbed_mp3),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(output),
            ],
            timeout=7200,
        )
    elif mode == "mix":
        await run_command(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(project.source_path),
                "-i", str(project.dubbed_mp3),
                "-filter_complex",
                f"[0:a]volume={MIX_ORIGINAL_VOLUME}[orig];"
                f"[1:a]volume=1.0[dub];"
                f"[orig][dub]amix=inputs=2:duration=first:dropout_transition=2,"
                f"alimiter=limit=0.95[mix]",
                "-map", "0:v:0",
                "-map", "[mix]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(output),
            ],
            timeout=7200,
        )
    else:
        raise ValueError("Invalid audio mode.")

    project.dubbed_mp4 = output
    return output


# =============================================================================
# Telegram helpers
# =============================================================================

async def send_with_retry(
    callable_obj: Any,
    *args: Any,
    attempts: int = 3,
    **kwargs: Any,
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return await callable_obj(*args, **kwargs)
        except RetryAfter as exc:
            last_error = exc
            await asyncio.sleep(float(exc.retry_after) + 0.5)
        except (TimedOut, NetworkError) as exc:
            last_error = exc
            if attempt < attempts:
                await asyncio.sleep(2 ** (attempt - 1))
        except TelegramError:
            raise
    raise RuntimeError(f"Telegram operation failed: {last_error}") from last_error


async def send_output_file(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    path: Path,
    *,
    caption: str,
    as_video: bool = False,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size > TELEGRAM_SEND_LIMIT_BYTES:
        await context.bot.send_message(
            chat_id,
            f"⚠️ ឯកសារ {path.name} មានទំហំធំពេកសម្រាប់ការកំណត់ផ្ញើរបស់ Bot "
            f"({TELEGRAM_SEND_LIMIT_MB} MB)។ ឯកសារនៅលើម៉ាស៊ីនមេ៖ {path.name}",
        )
        return

    with path.open("rb") as handle:
        if as_video:
            await send_with_retry(
                context.bot.send_video,
                chat_id=chat_id,
                video=handle,
                caption=caption,
                supports_streaming=True,
                read_timeout=300,
                write_timeout=300,
                connect_timeout=60,
                pool_timeout=60,
            )
        else:
            await send_with_retry(
                context.bot.send_document,
                chat_id=chat_id,
                document=handle,
                filename=path.name,
                caption=caption,
                read_timeout=300,
                write_timeout=300,
                connect_timeout=60,
                pool_timeout=60,
            )


async def get_upload_info(message: Message) -> tuple[Any, str, int]:
    if message.video:
        item = message.video
        return item, sanitize_filename(item.file_name or f"video_{item.file_unique_id}.mp4"), item.file_size or 0
    if message.audio:
        item = message.audio
        return item, sanitize_filename(item.file_name or f"audio_{item.file_unique_id}.mp3"), item.file_size or 0
    if message.voice:
        item = message.voice
        return item, sanitize_filename(f"voice_{item.file_unique_id}.ogg"), item.file_size or 0
    if message.document:
        item = message.document
        filename = sanitize_filename(item.file_name or f"file_{item.file_unique_id}")
        return item, filename, item.file_size or 0
    raise ValueError("Unsupported Telegram message type.")


async def download_telegram_file(
    message: Message,
    destination: Path,
) -> None:
    media, _, _ = await get_upload_info(message)
    telegram_file = await media.get_file()
    await telegram_file.download_to_drive(custom_path=str(destination))


def is_srt_message(message: Message) -> bool:
    return bool(
        message.document
        and message.document.file_name
        and Path(message.document.file_name).suffix.lower() == ".srt"
    )


# =============================================================================
# Core workflows
# =============================================================================

async def process_new_upload(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    project: Project,
    filename: str,
) -> None:
    message = update.effective_message
    assert message is not None

    try:
        async with JOB_SEMAPHORE:
            project.processing = True
            await update_progress(context, project, PROGRESS_TEXT["download"])
            source = project.root / filename
            await download_telegram_file(message, source)
            project.source_path = source

            if project.cancelled:
                raise asyncio.CancelledError

            await update_progress(context, project, PROGRESS_TEXT["extract"])
            normalized = project.root / "audio_16k_mono.wav"
            await extract_normalized_audio(source, normalized)
            project.normalized_wav = normalized

            await update_progress(context, project, PROGRESS_TEXT["transcribe"])
            chinese_subtitles = await transcribe_audio(normalized)
            chinese_srt = project.root / "Chinese_Whisper.srt"
            # Chinese draft has no final speaker guarantee, but preserves timing.
            chinese_srt.write_text(
                "\n\n".join(
                    f"{i}\n{ms_to_srt(sub.start_ms)} --> {ms_to_srt(sub.end_ms)}\n{sub.text}"
                    for i, sub in enumerate(chinese_subtitles, start=1)
                ) + "\n",
                encoding="utf-8",
            )
            project.chinese_srt = chinese_srt

            await update_progress(context, project, PROGRESS_TEXT["translate"])
            khmer_subtitles = await translate_subtitles(chinese_subtitles)

            await update_progress(context, project, PROGRESS_TEXT["speaker"])
            # Speaker classification is included in the Gemini translation pass.
            # Validate again locally to guarantee a usable profile.
            for sub in khmer_subtitles:
                if sub.tag not in VALID_TAGS or sub.tag == "UNKNOWN":
                    sub.tag = "M"

            await update_progress(context, project, PROGRESS_TEXT["srt"])
            khmer_srt = project.root / "KhmerDubAI_V10_Khmer.srt"
            khmer_srt.write_text(subtitles_to_srt(khmer_subtitles), encoding="utf-8")
            project.khmer_srt = khmer_srt

            await send_output_file(
                context,
                project.chat_id,
                khmer_srt,
                caption="✅ KhmerDubAI V10 — សូមពិនិត្យ SRT មុនបង្កើតសំឡេង។",
            )
            await update_progress(context, project, PROGRESS_TEXT["review"])
            await context.bot.send_message(
                project.chat_id,
                "ជ្រើសរើសជំហានបន្ទាប់៖",
                reply_markup=REVIEW_KEYBOARD,
            )

    except asyncio.CancelledError:
        await context.bot.send_message(project.chat_id, "❌ គម្រោងត្រូវបានបោះបង់។")
    except Exception as exc:
        logger.exception("Upload processing failed for user %s", project.user_id)
        await context.bot.send_message(
            project.chat_id,
            "❌ ដំណើរការមិនបាន៖\n"
            f"{str(exc)[:1500]}\n\n"
            "សូមពិនិត្យ FFmpeg, API key, ទំហំឯកសារ និងសាកល្បងម្ដងទៀត។",
        )
    finally:
        project.processing = False


async def generate_mp3_workflow(
    context: ContextTypes.DEFAULT_TYPE,
    project: Project,
    *,
    send_file: bool = True,
) -> Path:
    if not project.khmer_srt or not project.khmer_srt.exists():
        raise RuntimeError("Khmer SRT is missing.")

    subtitles = parse_srt(project.khmer_srt.read_text(encoding="utf-8-sig"))
    async with JOB_SEMAPHORE:
        project.processing = True
        try:
            output = await generate_dubbed_mp3(project, subtitles, context)
            if send_file:
                await update_progress(context, project, PROGRESS_TEXT["send"])
                await send_output_file(
                    context,
                    project.chat_id,
                    output,
                    caption="✅ KhmerDubAI V10 — សំឡេងខ្មែរ MP3 រួចរាល់។",
                )
                await context.bot.send_message(
                    project.chat_id,
                    "អ្នកអាចបង្កើត MP4 បន្ត ឬ Upload SRT ដែលបានកែ។",
                    reply_markup=REVIEW_KEYBOARD,
                )
            return output
        finally:
            project.processing = False


async def generate_mp4_workflow(
    context: ContextTypes.DEFAULT_TYPE,
    project: Project,
    mode: str,
) -> None:
    # Generate MP3 before acquiring the MP4 semaphore. Acquiring the same
    # semaphore recursively would deadlock when MAX_CONCURRENT_JOBS=1.
    if not project.dubbed_mp3 or not project.dubbed_mp3.exists():
        await generate_mp3_workflow(context, project, send_file=False)

    async with JOB_SEMAPHORE:
        project.processing = True
        try:
            await update_progress(context, project, PROGRESS_TEXT["mp4"])
            output = await create_dubbed_video(project, mode)
            await update_progress(context, project, PROGRESS_TEXT["send"])
            await send_output_file(
                context,
                project.chat_id,
                output,
                caption=f"✅ KhmerDubAI V10 — MP4 ({mode}) រួចរាល់។",
                as_video=True,
            )
            await context.bot.send_message(
                project.chat_id,
                "✅ គម្រោងរួចរាល់។ អ្នកអាច Upload ឯកសារថ្មីបាន។",
                reply_markup=REVIEW_KEYBOARD,
            )
        finally:
            project.processing = False


# =============================================================================
# Telegram handlers
# =============================================================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    if not message:
        return
    await message.reply_text(
        "🤖 KhmerDubAI V10\n\n"
        "ផ្ញើវីដេអូ ឬសំឡេងភាសាចិនមក Bot៖\n"
        "MP4 • MKV • MOV • AVI • MP3 • WAV • M4A • OGG\n\n"
        "Bot នឹងបង្កើត SRT ខ្មែរ ហើយអនុញ្ញាតឱ្យអ្នកពិនិត្យ "
        "មុនបង្កើត MP3 ឬ MP4។\n\n"
        f"ទំហំអតិបរមាដែលបានកំណត់៖ {MAX_FILE_MB} MB"
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start_handler(update, context)


async def cancel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return

    project = PROJECTS.pop(user.id, None)
    if not project:
        await message.reply_text("មិនមានគម្រោងកំពុងដំណើរការទេ។")
        return

    project.cancelled = True
    project.cleanup()
    await message.reply_text("❌ បានបោះបង់ និងសម្អាតគម្រោងរបស់អ្នក។")


async def upload_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    chat = update.effective_chat
    if not user or not message or not chat:
        return

    # Edited SRT is handled before generic document validation.
    current = PROJECTS.get(user.id)
    if is_srt_message(message) and current and current.waiting_for_srt:
        await edited_srt_handler(update, context)
        return

    try:
        media, filename, file_size = await get_upload_info(message)
        del media
    except ValueError:
        await message.reply_text("សូមផ្ញើវីដេអូ សំឡេង ឬ Telegram Voice។")
        return

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        await message.reply_text(
            "❌ ប្រភេទឯកសារមិនគាំទ្រ។\n"
            "គាំទ្រ៖ MP4, MKV, MOV, AVI, MP3, WAV, M4A, OGG និង Telegram Voice។"
        )
        return

    if file_size and file_size > MAX_FILE_BYTES:
        await message.reply_text(
            f"❌ ឯកសារធំពេក។ កំណត់បច្ចុប្បន្នគឺ {MAX_FILE_MB} MB។"
        )
        return

    try:
        require_system_tools()
    except RuntimeError as exc:
        await message.reply_text(f"❌ {exc}")
        return

    lock = get_user_lock(user.id)
    if lock.locked():
        await message.reply_text("⏳ គម្រោងរបស់អ្នកកំពុងដំណើរការ។ សូមរង់ចាំ ឬប្រើ /cancel។")
        return

    async with lock:
        project = new_project(user.id, chat.id)
        progress = await message.reply_text(PROGRESS_TEXT["download"])
        project.progress_message_id = progress.message_id
        await process_new_upload(update, context, project, filename)


async def edited_srt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    if not user or not message or not message.document:
        return

    try:
        project = get_project(user.id)
    except RuntimeError as exc:
        await message.reply_text(str(exc))
        return

    if not project.waiting_for_srt:
        await message.reply_text(
            "ចុច “Upload Edited SRT” ជាមុនសិន ដើម្បីជំនួសឯកសារ SRT។"
        )
        return

    if project.processing:
        await message.reply_text("⏳ Bot កំពុងដំណើរការ។ សូមរង់ចាំ។")
        return

    uploaded = project.root / "uploaded_edited.srt"
    try:
        telegram_file = await message.document.get_file()
        await telegram_file.download_to_drive(custom_path=str(uploaded))
        raw = uploaded.read_text(encoding="utf-8-sig")
        subtitles = parse_srt(raw)
        canonical = subtitles_to_srt(subtitles)

        final_srt = project.root / "KhmerDubAI_V10_Khmer_Edited.srt"
        final_srt.write_text(canonical, encoding="utf-8")
        project.khmer_srt = final_srt
        project.waiting_for_srt = False
        project.dubbed_mp3 = None
        project.dubbed_mp4 = None

        await message.reply_text(
            f"✅ បានទទួល និងពិនិត្យ SRT រួច៖ {len(subtitles)} ប្លុក។",
            reply_markup=REVIEW_KEYBOARD,
        )
    except (UnicodeError, ValueError, TelegramError) as exc:
        logger.warning("Invalid edited SRT from user %s: %s", user.id, exc)
        await message.reply_text(
            "❌ SRT មិនត្រឹមត្រូវ៖\n"
            f"{str(exc)[:1000]}\n\n"
            "ទម្រង់ត្រូវមានលេខ, timestamp និង [TAG] សន្ទនា។"
        )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    if not query or not user:
        return

    await query.answer()
    data = query.data or ""

    try:
        project = get_project(user.id)
    except RuntimeError as exc:
        await query.message.reply_text(str(exc))
        return

    if data == "v10:cancel":
        project.cancelled = True
        PROJECTS.pop(user.id, None)
        project.cleanup()
        await query.edit_message_text("❌ បានបោះបង់ និងសម្អាតគម្រោង។")
        return

    if data == "v10:back":
        await query.edit_message_text("ជ្រើសរើសជំហានបន្ទាប់៖", reply_markup=REVIEW_KEYBOARD)
        return

    if data == "v10:upload_srt":
        project.waiting_for_srt = True
        await query.edit_message_text(
            "📄 សូមផ្ញើឯកសារ .srt ដែលអ្នកបានកែ មកក្នុង Chat នេះ។\n\n"
            "Bot នឹងពិនិត្យ tag និង timestamp មុនបង្កើតសំឡេង។"
        )
        return

    if project.processing:
        await query.message.reply_text("⏳ គម្រោងកំពុងដំណើរការ។ សូមរង់ចាំ។")
        return

    if data == "v10:mp3":
        await query.edit_message_text("🎧 បានចាប់ផ្ដើមបង្កើត MP3…")
        try:
            await generate_mp3_workflow(context, project)
        except Exception as exc:
            logger.exception("MP3 workflow failed for user %s", user.id)
            await query.message.reply_text(f"❌ បង្កើត MP3 មិនបាន៖\n{str(exc)[:1500]}")
        return

    if data == "v10:mp4":
        if not project.source_path or not is_video(project.source_path):
            await query.message.reply_text("❌ អ្នកបាន Upload សំឡេង មិនមែនវីដេអូទេ។")
            return
        await query.edit_message_text(
            "ជ្រើសរើសរបៀបសំឡេងសម្រាប់ MP4៖",
            reply_markup=AUDIO_MODE_KEYBOARD,
        )
        return

    if data in {"v10:mp4:replace", "v10:mp4:mix"}:
        mode = data.rsplit(":", 1)[-1]
        await query.edit_message_text(f"🎬 បានចាប់ផ្ដើមបង្កើត MP4 ({mode})…")
        try:
            await generate_mp4_workflow(context, project, mode)
        except Exception as exc:
            logger.exception("MP4 workflow failed for user %s", user.id)
            await query.message.reply_text(f"❌ បង្កើត MP4 មិនបាន៖\n{str(exc)[:1500]}")
        return


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = update.effective_message
    if message:
        await message.reply_text(
            "សូមផ្ញើវីដេអូ/សំឡេង ឬប្រើ /start ដើម្បីមើលការណែនាំ។"
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled Telegram error", exc_info=context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                update.effective_chat.id,
                "❌ Bot ជួបបញ្ហាដែលមិនបានរំពឹងទុក។ សូមសាកល្បងម្ដងទៀត។",
            )
        except TelegramError:
            pass


async def post_init(application: Application) -> None:
    require_system_tools()
    me = await application.bot.get_me()
    logger.info("KhmerDubAI V10 started as @%s", me.username)


async def post_shutdown(application: Application) -> None:
    del application
    for project in list(PROJECTS.values()):
        project.cancelled = True
        project.cleanup()
    PROJECTS.clear()
    logger.info("KhmerDubAI V10 shutdown complete.")


def main() -> None:
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .connect_timeout(60)
        .read_timeout(300)
        .write_timeout(300)
        .pool_timeout(60)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("cancel", cancel_handler))
    application.add_handler(CallbackQueryHandler(callback_handler, pattern=r"^v10:"))
    application.add_handler(
        MessageHandler(
            filters.VIDEO
            | filters.AUDIO
            | filters.VOICE
            | filters.Document.ALL,
            upload_handler,
        )
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    application.add_error_handler(error_handler)

    logger.info("Starting KhmerDubAI V10 polling...")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )


if __name__ == "__main__":
    main()
