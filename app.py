import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import anthropic
import streamlit as st
from faster_whisper import WhisperModel

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

st.set_page_config(page_title="Reel Clipper", page_icon="🎬")
st.title("🎬 Reel Clipper")
st.caption("Video upload karo, AI best clips nikal ke dega.")


# ---------- helpers ----------
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return r.stdout


def video_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


@st.cache_resource
def load_whisper(size):
    # CPU pe int8 chalta hai; GPU ho to device="cuda", compute_type="float16"
    return WhisperModel(size, device="auto", compute_type="int8")


def transcribe(video_path, size):
    wav = Path(video_path).with_suffix(".wav")
    run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    model = load_whisper(size)
    segs, info = model.transcribe(str(wav), vad_filter=True)
    out = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
    return out, info.language


def pick_clips(segments, n, min_len, max_len, style):
    lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    prompt = f"""Neeche ek video ka timestamped transcript hai.
Isme se {n} sabse acche short-form reel clips chuno.

Rules:
- Har clip {min_len}-{max_len} seconds ka ho.
- Clip ek complete thought ho: beech me shuru ya khatam na ho.
- Shuruaat me strong hook ho (sawal, bold claim, kahani, surprise).
- Clips overlap na karein.
- Start/end transcript ke timestamps ke hisaab se rakho.
- Focus: {style}

Sirf valid JSON array return karo, aur kuch nahi. Format:
[{{"start": 12.3, "end": 55.0, "title": "chhota title", "reason": "kyun acha hai"}}]

Transcript:
{lines}"""
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY env se
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError("Claude ne valid JSON nahi diya:\n" + text[:500])
    return json.loads(match.group(0))


def clean_clips(clips, duration, min_len, max_len):
    cleaned = []
    for c in clips:
        s = max(0.0, float(c["start"]))
        e = min(duration, float(c["end"]))
        if e - s > max_len:
            e = s + max_len
        if e - s < min_len * 0.6:
            continue
        cleaned.append({**c, "start": s, "end": e})
    return cleaned


def cut_clip(video, start, end, out, vertical):
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{end - start:.2f}"]
    if vertical:
        cmd += ["-vf", "crop='min(iw,ih*9/16)':ih,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    run(cmd)


# ---------- UI ----------
with st.sidebar:
    n = st.slider("Kitne clips", 1, 10, 5)
    min_len, max_len = st.slider("Clip length (sec)", 10, 120, (20, 60))
    whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=2)
    vertical = st.checkbox("9:16 vertical crop (center)", value=True)
    style = st.text_input("Kaisa content chahiye", "sabse insightful, funny ya emotional moments")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY environment variable set karo.")
    st.stop()

video_file = st.file_uploader("Video upload karo", type=["mp4", "mov", "mkv", "webm", "avi"])

if video_file and st.button("Clips nikalo", type="primary"):
    with tempfile.TemporaryDirectory() as tmp:
        vpath = Path(tmp) / video_file.name
        vpath.write_bytes(video_file.getbuffer())
        duration = video_duration(vpath)

        with st.status("Transcribe ho raha hai...", expanded=False) as status:
            segments, lang = transcribe(vpath, whisper_size)
            status.update(label=f"Transcript ready (language: {lang}, {len(segments)} segments)")

        if not segments:
            st.error("Video me koi speech nahi mili.")
            st.stop()

        with st.spinner("Claude best moments dhoondh raha hai..."):
            clips = clean_clips(pick_clips(segments, n, min_len, max_len, style),
                                duration, min_len, max_len)

        if not clips:
            st.error("Koi valid clip nahi mili, settings badal ke try karo.")
            st.stop()

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for i, c in enumerate(clips, 1):
                out = Path(tmp) / f"reel_{i}.mp4"
                with st.spinner(f"Clip {i}/{len(clips)} kat rahi hai..."):
                    cut_clip(vpath, c["start"], c["end"], out, vertical)
                st.subheader(f"{i}. {c.get('title', 'Clip')}")
                st.caption(f"{c['start']:.0f}s - {c['end']:.0f}s | {c.get('reason', '')}")
                st.video(out.read_bytes())
                zf.write(out, out.name)

        st.download_button("Saari clips (ZIP)", zip_buf.getvalue(),
                           "reels.zip", "application/zip")
