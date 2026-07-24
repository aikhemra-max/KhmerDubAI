import asyncio
import os
import re
import subprocess
from pathlib import Path

import streamlit as st
from google import genai
import edge_tts

APP_NAME = "AI KHEMRA BRO"
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #171d35 100%);
    }
    .hero {
        padding: 20px;
        border-radius: 20px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        text-align: center;
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 8px 0; }
    div.stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 12px;
        font-weight: 700;
    }
    textarea { font-family: monospace !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <h1>🎬 {APP_NAME}</h1>
        <div>Website បកប្រែរឿង និងបង្កើតសំឡេងខ្មែរ</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def save_upload(uploaded_file) -> Path:
    path = UPLOAD_DIR / safe_filename(uploaded_file.name)
    path.write_bytes(uploaded_file.getbuffer())
    return path


def run_command(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Command failed")


def extract_audio(video_path: Path) -> Path:
    output_path = OUTPUT_DIR / f"{video_path.stem}_audio.mp3"
    run_command([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "24000",
        "-b:a", "128k",
        str(output_path),
    ])
    return output_path


def translate_text_to_khmer(text: str, api_key: str, model: str) -> str:
    if not api_key:
        raise RuntimeError("សូមបញ្ចូល GEMINI_API_KEY ជាមុនសិន។")

    prompt = f"""
You are a professional Chinese-drama subtitle translator.

Translate the following text into natural spoken Khmer.

Rules:
1. Preserve all SRT numbers and timestamps exactly when present.
2. Use natural Khmer suitable for movie dubbing.
3. Do not translate word-for-word.
4. Keep emotion and context.
5. Do not include explanations.
6. Remove all Chinese characters from the Khmer result.
7. Return only the translated text.

TEXT:
{text}
""".strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    translated = (response.text or "").strip()
    if not translated:
        raise RuntimeError("Gemini មិនបានផ្ដល់លទ្ធផល។")
    return translated


async def save_tts(text: str, voice: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))


def create_khmer_mp3(text: str, voice: str) -> Path:
    clean_text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    clean_text = re.sub(
        r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*"
        r"\d{2}:\d{2}:\d{2},\d{3}",
        "",
        clean_text,
    )
    clean_text = re.sub(r"<[^>]+>", "", clean_text)
    clean_text = re.sub(r"\n{2,}", "\n", clean_text).strip()

    if not clean_text:
        raise RuntimeError("មិនមានអត្ថបទសម្រាប់បង្កើតសំឡេងទេ។")

    output_path = OUTPUT_DIR / "khmer_voice.mp3"
    asyncio.run(save_tts(clean_text, voice, output_path))
    return output_path


def merge_audio_with_video(video_path: Path, audio_path: Path) -> Path:
    output_path = OUTPUT_DIR / f"{video_path.stem}_khmer.mp4"
    run_command([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ])
    return output_path


if "video_path" not in st.session_state:
    st.session_state.video_path = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""
if "khmer_text" not in st.session_state:
    st.session_state.khmer_text = ""
if "mp3_path" not in st.session_state:
    st.session_state.mp3_path = ""

with st.sidebar:
    st.header("⚙️ Settings")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
    )
    gemini_model = st.text_input(
        "Gemini Model",
        value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    )
    voice_label = st.selectbox(
        "Khmer Voice",
        ["សំឡេងស្រី — Sreymom", "សំឡេងប្រុស — Piseth"],
    )
    voice = (
        "km-KH-SreymomNeural"
        if voice_label.startswith("សំឡេងស្រី")
        else "km-KH-PisethNeural"
    )

st.subheader("1️⃣ Upload Video")
uploaded_video = st.file_uploader(
    "ជ្រើសរើសវីដេអូ MP4, MOV, MKV ឬ AVI",
    type=["mp4", "mov", "mkv", "avi"],
)

if uploaded_video is not None:
    video_path = save_upload(uploaded_video)
    st.session_state.video_path = str(video_path)
    st.success("✅ Upload វីដេអូបានជោគជ័យ")
    st.video(str(video_path))

st.divider()
st.subheader("2️⃣ Extract Audio")

if st.button(
    "🎵 ទាញសំឡេងចេញពីវីដេអូ",
    disabled=not st.session_state.video_path,
):
    try:
        with st.spinner("កំពុងទាញសំឡេង..."):
            audio_path = extract_audio(Path(st.session_state.video_path))
            st.session_state.audio_path = str(audio_path)
        st.success("✅ ទាញសំឡេងរួចរាល់")
    except Exception as exc:
        st.error(f"❌ {exc}")

if st.session_state.audio_path and Path(st.session_state.audio_path).exists():
    audio_path = Path(st.session_state.audio_path)
    st.audio(str(audio_path))
    st.download_button(
        "📥 Download Original Audio",
        data=audio_path.read_bytes(),
        file_name=audio_path.name,
        mime="audio/mpeg",
    )

st.divider()
st.subheader("3️⃣ បញ្ចូល Subtitle ឬអត្ថបទដើម")

uploaded_srt = st.file_uploader(
    "Upload SRT",
    type=["srt", "txt"],
)

source_text = ""
if uploaded_srt is not None:
    source_text = uploaded_srt.getvalue().decode("utf-8", errors="replace")

source_text = st.text_area(
    "អត្ថបទភាសាចិន/ថៃ/អង់គ្លេស ឬ SRT",
    value=source_text,
    height=260,
    placeholder="Paste subtitle ឬអត្ថបទនៅទីនេះ...",
)

if st.button("🌍 បកប្រែទៅជាភាសាខ្មែរ", disabled=not source_text.strip()):
    try:
        with st.spinner("កំពុងបកប្រែទៅជាភាសាខ្មែរ..."):
            st.session_state.khmer_text = translate_text_to_khmer(
                source_text,
                gemini_api_key,
                gemini_model,
            )
        st.success("✅ បកប្រែរួចរាល់")
    except Exception as exc:
        st.error(f"❌ {exc}")

st.divider()
st.subheader("4️⃣ កែសម្រួល Khmer SRT")

st.session_state.khmer_text = st.text_area(
    "Khmer Subtitle",
    value=st.session_state.khmer_text,
    height=320,
    placeholder="លទ្ធផលបកប្រែនឹងបង្ហាញនៅទីនេះ...",
)

if st.session_state.khmer_text.strip():
    st.download_button(
        "📥 Download Khmer SRT",
        data=st.session_state.khmer_text.encode("utf-8"),
        file_name="khmer_subtitle.srt",
        mime="application/x-subrip",
    )

st.divider()
st.subheader("5️⃣ បង្កើតសំឡេងខ្មែរ MP3")

if st.button(
    "🎙️ បង្កើត MP3",
    disabled=not st.session_state.khmer_text.strip(),
):
    try:
        with st.spinner("កំពុងបង្កើតសំឡេងខ្មែរ..."):
            mp3_path = create_khmer_mp3(
                st.session_state.khmer_text,
                voice,
            )
            st.session_state.mp3_path = str(mp3_path)
        st.success("✅ បង្កើត MP3 រួចរាល់")
    except Exception as exc:
        st.error(f"❌ {exc}")

if st.session_state.mp3_path and Path(st.session_state.mp3_path).exists():
    mp3_path = Path(st.session_state.mp3_path)
    st.audio(str(mp3_path))
    st.download_button(
        "📥 Download Khmer MP3",
        data=mp3_path.read_bytes(),
        file_name=mp3_path.name,
        mime="audio/mpeg",
    )

st.divider()
st.subheader("6️⃣ បញ្ចូលសំឡេងខ្មែរទៅក្នុងវីដេអូ")

if st.button(
    "🎬 បង្កើត Khmer MP4",
    disabled=not (
        st.session_state.video_path
        and st.session_state.mp3_path
    ),
):
    try:
        with st.spinner("កំពុងបញ្ចូលសំឡេងទៅក្នុងវីដេអូ..."):
            output_video = merge_audio_with_video(
                Path(st.session_state.video_path),
                Path(st.session_state.mp3_path),
            )
        st.success("✅ បង្កើតវីដេអូរួចរាល់")
        st.video(str(output_video))
        st.download_button(
            "📥 Download Khmer MP4",
            data=output_video.read_bytes(),
            file_name=output_video.name,
            mime="video/mp4",
        )
    except Exception as exc:
        st.error(f"❌ {exc}")

st.caption("AI KHEMRA BRO • Google Chrome Website")
