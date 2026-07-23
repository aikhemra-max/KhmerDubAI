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
from dotenv import load_dotenv
from google import genai
from telegram import Message, ReplyKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ផ្ទុក Environment Variables
load_dotenv()

# ================= ការកំណត់ទូទៅ (Configuration) =================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("KhmerDubAI_MultiLang_V10")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# សំឡេងខ្មែរ AI ផ្លូវការ (Khmer Neural Voices)
MALE_VOICE = os.getenv("MALE_VOICE", "km-KH-PisethNeural")
FEMALE_VOICE = os.getenv("FEMALE_VOICE", "km-KH-SreymomNeural")

NEW_PROJECT_BUTTON = "🆕 ធ្វើថ្មី"
PROJECT_KEYBOARD = ReplyKeyboardMarkup(
    [[NEW_PROJECT_BUTTON]],
    resize_keyboard=True,
    is_persistent=True,
)

sessions: dict[int, dict] = {}

def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    if not api_key:
        raise ValueError("ខ្វះ GEMINI_API_KEY! សូមពិនិត្យ Variables ក្នុង Railway។")
    return genai.Client(api_key=api_key)

# ================= ប្រវត្តិរូបសំឡេងទាំង ១២ និងអារម្មណ៍ទាំង ៩ =================
@dataclass
class SubtitleCue:
    index: int
    start: float
    end: float
    tag: str
    emotion: str
    text: str

VOICE_PROFILES = {
    "M_YOUNG": {"voice": MALE_VOICE, "rate": "+2%", "pitch": "+1Hz"},
    "F_YOUNG": {"voice": FEMALE_VOICE, "rate": "+2%", "pitch": "+1Hz"},
    "M_ADULT": {"voice": MALE_VOICE, "rate": "+0%", "pitch": "+0Hz"},
    "F_ADULT": {"voice": FEMALE_VOICE, "rate": "+0%", "pitch": "+0Hz"},
    "M_OLD":   {"voice": MALE_VOICE, "rate": "-4%", "pitch": "-2Hz"},
    "F_OLD":   {"voice": FEMALE_VOICE, "rate": "-4%", "pitch": "-2Hz"},
    "BOY":     {"voice": MALE_VOICE, "rate": "+4%", "pitch": "+2Hz"},
    "GIRL":    {"voice": FEMALE_VOICE, "rate": "+4%", "pitch": "+2Hz"},
    "M_THINK": {"voice": MALE_VOICE, "rate": "-3%", "pitch": "-1Hz"},
    "F_THINK": {"voice": FEMALE_VOICE, "rate": "-3%", "pitch": "-1Hz"},
    "NARRATOR_M": {"voice": MALE_VOICE, "rate": "-2%", "pitch": "0Hz"},
    "NARRATOR_F": {"voice": FEMALE_VOICE, "rate": "-2%", "pitch": "0Hz"},
}

EMOTION_ADJUSTMENTS = {
    "NEUTRAL": {"rate": 0, "pitch": 0},
    "HAPPY":   {"rate": 2, "pitch": 1},
    "SAD":     {"rate": -3, "pitch": -1},
    "ANGRY":   {"rate": 3, "pitch": 1},
    "FEAR":    {"rate": 2, "pitch": 1},
    "LOVE":    {"rate": -2, "pitch": 0},
    "SARCASM": {"rate": 0, "pitch": 1},
    "CRYING":  {"rate": -4, "pitch": -1},
    "THINKING":{"rate": -3, "pitch": -1},
}

ProgressCallback = Callable[[int, str], Awaitable[None]]

# ================= ការគ្រប់គ្រង Session =================
def get_session(chat_id: int) -> dict:
    if chat_id not in sessions:
        sessions[chat_id] = {
            "active": False,
            "generation": 0,
            "last_activity": 0.0,
            "message_ids": set(),
        }
    return sessions[chat_id]

def track_message(message: Optional[Message]) -> None:
    if message:
        get_session(message.chat_id)["message_ids"].add(message.message_id)

async def require_new_project_started(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    if not session["active"]:
        msg = await update.message.reply_text(
            "⚠️ សូមចុច «🆕 ធ្វើថ្មី» ជាមុនសិន ទើបអាចផ្ញើវីដេអូ ឬសំឡេងបាន។",
            reply_markup=PROJECT_KEYBOARD,
        )
        track_message(msg)
        return False
    track_message(update.message)
    return True

async def new_project(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    session["active"] = True
    session["last_activity"] = time.monotonic()
    
    msg = await update.message.reply_text(
        "✅ **Project ថ្មីរួចរាល់ (Multi-Language to Khmer Dubbing)**\n\n"
        "📤 អាអាចផ្ញើវីដេអូ/សំឡេង (ចិន, កូរ៉េ, ថៃ, វៀតណាម, អង់គ្លេស...) រហូតដល់ ៥ នាទី។\n"
        "🎭 ខ្ញុំនឹងបកប្រែជាភាសានិយាយខ្មែរយ៉ាងរលូន ត្រូវតាមតួអង្គ និងអារម្មណ៍ ១០០%!",
        reply_markup=PROJECT_KEYBOARD,
        parse_mode="Markdown"
    )
    track_message(msg)

# ================= ដំណើរការ FFmpeg =================
def ffprobe_duration(media_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)],
        capture_output=True, text=True, check=True,
    )
    return max(0.01, float(result.stdout.strip()))

def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr[-2000:]}")

def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-q:a", "4", str(audio_path),
    ])

# ================= Prompt ជាភាសាអង់គ្លេស (សម្រាប់ដំណើរការបកប្រែមកជាខ្មែរ) =================
def english_to_khmer_prompt() -> str:
    return """
You are an Expert Movie Subtitler & Dubbing Translator specializing in translating foreign spoken dialogue (Chinese, Korean, Thai, Vietnamese, English, etc.) directly into Natural Spoken Cambodian Khmer.

TASK:
1. Listen carefully to the original audio dialogue provided.
2. Recognize timing, speaker genders, age groups, relationships, and emotional tones directly from the audio.
3. Translate all spoken dialogue directly into natural, fluent, conversational Cambodian Khmer suitable for professional movie dubbing.
4. Output strictly valid SRT format in Khmer.

STRICT KHMER DUBBING RULES:
1. Natural Spoken Khmer (ភាសានិយាយធម្មជាតិ):
   - Use everyday spoken Khmer expressions.
   - Include appropriate spoken Khmer particles based on context (ណា, ណ៎, ហ្មង, តើ, អញ្ចឹង, វើយ, ហាស, ចា៎, ចុះ).
   - NEVER translate word-for-word or use rigid written Khmer grammar.

2. Match Speaker Voice & Pronouns (ត្រូវសំឡេងតួអង្គ និងទំនាក់ទំនង):
   - Accurately determine gender (Male vs Female) and age from the original audio.
   - Use correct Khmer pronouns matching age, rank, and relationship context:
     (បង/អូន, ឯង/អញ, ខ្ញុំ/លោក, ពួកម៉ាក, សម្លាញ់, អា..., លោកម្ចាស់, ព្រះអង្គ, លោកគ្រូ/សិស្ស).

3. Emotional Depth (បញ្ចេញមនោសញ្ចេតនា):
   - Express true scene emotional tone: Anger (ខឹង), Sadness (កើតទុក្ខ/យំ), Happiness (សប្បាយ), Fear (ភ័យ), Love (ផ្អែមល្ហែម), Sarcasm (ចំអក), Thinking (គិត).

4. Subtitle Conciseness (ភាពច្បាស់លាស់):
   - Keep dialogue concise to fit subtitle speed and timestamp boundaries.

TAG RULES (MANDATORY FOR TTS):
Prefix EVERY subtitle line with EXACTLY ONE Voice Tag and ONE Emotion Tag:
- Voice Tags (12 Profiles): [M_YOUNG] [F_YOUNG] [M_ADULT] [F_ADULT] [M_OLD] [F_OLD] [BOY] [GIRL] [M_THINK] [F_THINK] [NARRATOR_M] [NARRATOR_F]
- Emotion Tags (9 Emotions): [NEUTRAL] [HAPPY] [SAD] [ANGRY] [FEAR] [LOVE] [SARCASM] [CRYING] [THINKING]

FORMAT EXAMPLE:
1
00:00:01,000 --> 00:00:03,000
[M_ADULT][ANGRY] ឯងហ៊ានធ្វើបាបនាងផងហាស!

2
00:00:03,200 --> 00:00:05,000
[F_YOUNG][HAPPY] ខ្ញុំអរគុណបងច្រើនហើយណា!

OUTPUT REQUIREMENTS:
- Output ONLY valid SRT content in Cambodian Khmer.
- No Markdown code blocks, no extra explanations.
"""

def clean_gemini_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:srt|text)?\s*", "", text, flags=re.I)
    return re.sub(r"\s*```$", "", text).strip()

def process_audio_with_gemini(audio_path: Path) -> str:
    logger.info("Uploading audio to Gemini API...")
    client = get_gemini_client()
    uploaded_file = client.files.upload(file=str(audio_path))
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, english_to_khmer_prompt()]
        )
        if not response.text:
            raise RuntimeError("Gemini ពុំបានឆ្លើយតបអត្ថបទឡើយ។")
        return clean_gemini_output(response.text)
    finally:
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

# ================= ការបំបែក SRT =================
def time_to_seconds(value: str) -> float:
    match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if not match:
        raise ValueError(f"Timestamp មិនត្រឹមត្រូវ: {value}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

def parse_tagged_srt(srt_text: str) -> list[SubtitleCue]:
    blocks = re.split(r"\n\s*\n", srt_text.replace("\r\n", "\n").strip())
    cues = []
    
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        
        try:
            index = int(lines[0])
        except ValueError:
            continue
        
        time_match = re.search(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})", lines[1])
        if not time_match:
            continue
        
        dialogue = " ".join(lines[2:]).strip()
        tag_match = re.match(r"^\[([A-Z_]+)\]\s*\[([A-Z_]+)\]\s*(.*)$", dialogue)
        
        if tag_match:
            tag, emotion, text = tag_match.group(1), tag_match.group(2), tag_match.group(3).strip()
        else:
            tag, emotion, text = "M_ADULT", "NEUTRAL", dialogue
            
        tag = tag if tag in VOICE_PROFILES else "M_ADULT"
        emotion = emotion if emotion in EMOTION_ADJUSTMENTS else "NEUTRAL"
        text = re.sub(r"<[^>]+>", "", text).strip()
        
        if not text:
            continue
        
        cues.append(SubtitleCue(
            index,
            time_to_seconds(time_match.group(1)),
            time_to_seconds(time_match.group(2)),
            tag, emotion, text
        ))
        
    return sorted(cues, key=lambda cue: (cue.start, cue.index))

# ================= ការបង្កើតសំឡេងខ្មែរ (TTS Alignment) =================
async def synthesize_audio(cue: SubtitleCue, output_path: Path) -> None:
    profile = VOICE_PROFILES[cue.tag]
    emotion = EMOTION_ADJUSTMENTS[cue.emotion]
    
    def parse_signed(val: str) -> int:
        m = re.search(r"[-+]?\d+", val)
        return int(m.group()) if m else 0

    rate_val = max(-10, min(10, parse_signed(profile['rate']) + emotion['rate']))
    pitch_val = max(-5, min(5, parse_signed(profile['pitch']) + emotion['pitch']))
    
    rate_str = f"{rate_val:+d}%"
    pitch_str = f"{pitch_val:+d}Hz"
    
    communicate = edge_tts.Communicate(
        text=cue.text, voice=profile["voice"], rate=rate_str, pitch=pitch_str
    )
    await communicate.save(str(output_path))
    if not output_path.exists() or output_path.stat().st_size < 100:
        raise RuntimeError(f"ការបង្កើតសំឡេងខ្មែរត្រង់ឃ្លា {cue.index} បរាជ័យ")

async def prepare_cue_audio(pos: int, cue: SubtitleCue, workdir: Path, sem: asyncio.Semaphore) -> tuple:
    raw_path, fit_path = workdir / f"cue_{pos:04d}_raw.mp3", workdir / f"cue_{pos:04d}_fit.wav"
    async with sem:
        await synthesize_audio(cue, raw_path)
    
    raw_duration = await asyncio.to_thread(ffprobe_duration, raw_path)
    target_duration = max(0.25, cue.end - cue.start)
    speed = max(0.92, min(1.08, raw_duration / target_duration))
    
    await asyncio.to_thread(
        run_ffmpeg,
        ["ffmpeg", "-y", "-i", str(raw_path), "-filter:a",
         f"atempo={speed:.6f},highpass=f=60,lowpass=f=13000,afade=t=in:st=0:d=0.02,aresample=48000",
         "-ac", "2", "-c:a", "pcm_s16le", str(fit_path)]
    )
    return pos, fit_path, round(cue.start * 1000)

async def create_timed_dub_mp3(cues: list[SubtitleCue], out_path: Path, workdir: Path, progress: ProgressCallback) -> None:
    sem = asyncio.Semaphore(5)
    tasks = [asyncio.create_task(prepare_cue_audio(i, cue, workdir, sem)) for i, cue in enumerate(cues, 1)]
    
    prepared = []
    for count, task in enumerate(asyncio.as_completed(tasks), 1):
        prepared.append(await task)
        await progress(45 + round((count / len(tasks)) * 40), f"កំពុងបង្កើតសំឡេងខ្មែរ ({count}/{len(tasks)})")
        
    prepared.sort(key=lambda x: x[0])
    await progress(88, "កំពុងបញ្ចូលសំឡេងតាម Timeline")
    
    command = ["ffmpeg", "-y"]
    filter_parts, labels, prev_end_ms = [], [], 0
    
    for i, (_, file_path, req_delay_ms) in enumerate(prepared):
        command += ["-i", str(file_path)]
        audio_dur_ms = int(round(await asyncio.to_thread(ffprobe_duration, file_path) * 1000))
        delay_ms = max(req_delay_ms, prev_end_ms + (20 if i else 0))
        prev_end_ms = delay_ms + audio_dur_ms
        filter_parts.append(f"[{i}:a]adelay={delay_ms}:all=1,volume=1[a{i}]")
        labels.append(f"[a{i}]")
        
    filter_parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0.08,dynaudnorm=f=150:g=7[mix]")
    command += ["-filter_complex", ";".join(filter_parts), "-map", "[mix]", "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "160k", str(out_path)]
    await asyncio.to_thread(run_ffmpeg, command)

# ================= ដំណើរការ Media សារ =================
async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await require_new_project_started(update, context):
        return
    
    message = update.message
    media = message.document or message.audio or message.voice or message.video
    if not media:
        return
    
    status = await message.reply_text("⏳ កំពុងចាប់ផ្ដើមដំណើរការ...")
    
    async def progress(percent: int, label: str):
        try:
            bar = "█" * (percent // 10) + "░" * (10 - (percent // 10))
            await status.edit_text(f"⏳ {label}\n`{bar}` **{percent}%**", parse_mode="Markdown")
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        raw_name = getattr(media, "file_name", None) or "input.mp4"
        suffix = Path(raw_name).suffix.lower()
        if not suffix:
            suffix = ".mp4"
            
        source_path = tmpdir / f"input{suffix}"
        
        try:
            await progress(5, "កំពុងទាញយកឯកសារ...")
            tg_file = await media.get_file()
            await tg_file.download_to_drive(custom_path=source_path)
            
            audio_for_gemini = source_path
            if suffix in {".mp4", ".mov", ".mkv", ".avi"}:
                await progress(15, "កំពុងដកសំឡេងចេញពីវីដេអូ...")
                audio_for_gemini = tmpdir / "audio.mp3"
                await asyncio.to_thread(extract_audio, source_path, audio_for_gemini)
                
            await progress(25, "Gemini AI កំពុងវិភាគ និងបកប្រែជាភាសាខ្មែរ...")
            srt_text = await asyncio.wait_for(asyncio.to_thread(process_audio_with_gemini, audio_for_gemini), timeout=180)
            
            cues = parse_tagged_srt(srt_text)
            if not cues:
                raise ValueError("មិនអាចដកស្រង់ SRT ពីការបកប្រែរបស់ AI បានទេ។")
            
            mp3_path = tmpdir / "khmer_dub.mp3"
            await create_timed_dub_mp3(cues, mp3_path, tmpdir, progress)
            
            await progress(95, "កំពុងផ្ញើលទ្ធផល...")
            await message.reply_audio(
                audio=mp3_path.open("rb"),
                filename="Khmer_Dub.mp3",
                title="Khmer Dubbing",
                caption="✅ ការបកប្រែ និងបញ្ចូលសំឡេងខ្មែរជោគជ័យ ១០០%!"
            )
            await status.delete()
            
        except Exception as e:
            logger.error("Processing error:", exc_info=True)
            await status.edit_text(f"❌ មានបញ្ហាបច្ចេកទេស៖\n`{str(e)}`", parse_mode="Markdown")

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN មិនទាន់បានកំណត់ក្នុង Railway Variables ទេ!")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("🤖 ចុច «🆕 ធ្វើថ្មី» រួចផ្ញើឯកសារមក", reply_markup=PROJECT_KEYBOARD)))
    app.add_handler(MessageHandler(filters.Regex(f"^{NEW_PROJECT_BUTTON}$"), new_project))
    app.add_handler(MessageHandler(filters.Document.ALL | filters.AUDIO | filters.VOICE | filters.VIDEO, handle_media))
    
    logger.info("Khmer Dubbing Bot V10 Multi-Lang running...")
    app.run_polling()

if __name__ == "__main__":
    main()
