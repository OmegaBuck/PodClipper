import os
import io
import json
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple
import streamlit as st
import numpy as np
import pandas as pd

# Audio / NLP
import librosa
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon", quiet=True)

# --- Transcription: faster-whisper (Windows-friendly, fast CPU/GPU) ---
from faster_whisper import WhisperModel

# ---------------------------
# Data Model
# ---------------------------
@dataclass
class ClipCandidate:
    start: float
    end: float
    text: str
    rating: float
    reasons: List[str]

# ---------------------------
# Transcription
# ---------------------------
def transcribe_with_whisper(video_path: str, model_size: str, device: str):
    size_map = {"base": "base", "small": "small", "medium": "medium"}
    model_name = size_map.get(model_size, "base")

    compute_type = "float16" if device == "cuda" else "int8"
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
                compute_type = "int8"
        except Exception:
            device = "cpu"
            compute_type = "int8"

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    seg_iter, info = model.transcribe(video_path, vad_filter=True)
    segments, full_text = [], []
    for s in seg_iter:
        segments.append({"start": float(s.start), "end": float(s.end), "text": s.text})
        full_text.append(s.text)
    return segments, " ".join(full_text).strip()

def srt_from_segments(segments: List[dict]) -> str:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = seg["start"]; end = seg["end"]; text = seg["text"].strip()
        def ts(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            ms = int(round((t - int(t)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines.append(f"{idx}")
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(text); lines.append("")
    return "\n".join(lines)

# ---------------------------
# Heuristic scoring
# ---------------------------
def extract_audio_to_wav(video_path: str, wav_path: str, sr: int = 16000):
    cmd = ["ffmpeg","-y","-i",video_path,"-ac","1","-ar",str(sr),wav_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def compute_audio_features(wav_path: str) -> dict:
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms_n = (rms - rms.min()) / (rms.ptp() + 1e-9)
    zcr_n = (zcr - zcr.min()) / (zcr.ptp() + 1e-9)
    return {"rms": rms_n, "zcr": zcr_n, "sr": sr, "tempo": float(tempo), "duration": len(y)/sr}

def slice_segments_into_windows(segments: List[dict], window_sec: float = 30.0, stride_sec: float = 10.0) -> List[Tuple[float, float, str]]:
    if not segments:
        return []
    total_end = max(s["end"] for s in segments)
    windows, t = [], 0.0
    while t < total_end:
        start = t
        end = min(t + window_sec, total_end)
        texts = [s["text"].strip() for s in segments if s["end"] >= start and s["start"] <= end]
        window_text = " ".join(texts).strip()
        if len(window_text.split()) > 6:
            windows.append((start, end, window_text))
        t += stride_sec
    return windows

def keyword_score(text: str, keywords: List[str]) -> float:
    text_l = text.lower()
    hit = sum(1 for k in keywords if k in text_l)
    return min(hit / max(1, len(keywords)), 1.0)

def novelty_score(text: str) -> float:
    score = 0.0
    if "?" in text: score += 0.3
    if "!" in text: score += 0.2
    digits = sum(c.isdigit() for c in text)
    if digits >= 2: score += 0.2
    words = len(text.split())
    if 30 <= words <= 80: score += 0.3
    return min(score, 1.0)

def sentiment_score(sia: SentimentIntensityAnalyzer, text: str) -> float:
    s = sia.polarity_scores(text)
    return min(abs(s["compound"]), 1.0)

def audio_dynamics_score(audio_feats: dict, start: float, end: float) -> float:
    duration = audio_feats["duration"]
    n = len(audio_feats["rms"])
    def idx(t): return int(np.clip((t / duration) * (n - 1), 0, n - 1))
    r = audio_feats["rms"][idx(start):idx(end)+1]
    z = audio_feats["zcr"][idx(start):idx(end)+1]
    if len(r) == 0: return 0.0
    var_r = np.var(r); var_z = np.var(z)
    dyn = float(var_r * 0.7 + var_z * 0.3)
    return float(np.tanh(3.0 * dyn))

def score_window(text: str, start: float, end: float, sia: SentimentIntensityAnalyzer, audio_feats: dict, keywords: List[str]) -> Tuple[float, List[str]]:
    reasons = []
    s_sent = sentiment_score(sia, text); reasons.append(f"Emotion: {s_sent:.2f}")
    s_kw = keyword_score(text, keywords); reasons.append(f"Keywords: {s_kw:.2f}")
    s_nov = novelty_score(text); reasons.append(f"Novelty: {s_nov:.2f}")
    s_dyn = audio_dynamics_score(audio_feats, start, end); reasons.append(f"Audio dynamics: {s_dyn:.2f}")
    rating = 0.35 * s_sent + 0.30 * s_kw + 0.20 * s_nov + 0.15 * s_dyn
    return float(rating), reasons

# ---------------------------
# Rendering (FFmpeg)
# ---------------------------
def burn_subtitles_ffmpeg(input_video: str, srt_path: str, start: float, end: float, out_path: str, scale_width: int = 1080):
    srt_for_ffmpeg = srt_path.replace("\\", "/")
    trim_kwargs = ["-ss", f"{start}", "-to", f"{end}"]
    vf = (
        f"scale={scale_width}:-2,"
        f"subtitles='{srt_for_ffmpeg}':"
        f"force_style='Fontsize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00111111,BorderStyle=3,Outline=2,Shadow=0'"
    )
    cmd = [
        "ffmpeg", "-y",
        *trim_kwargs,
        "-i", input_video,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

# ---------------------------
# UI Helpers
# ---------------------------
def init_state():
    for k, v in {
        "tmpdir": None,
        "in_path": None,
        "segments": None,
        "srt_path": None,
        "audio_feats": None,
        "candidates": None,
        "selected": None
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------
# App
# ---------------------------
def main():
    st.set_page_config(page_title="PodClipper (Local)", page_icon="🎬")
    init_state()

    st.title("🎬 PodClipper (Local)")
    st.caption("Step-by-step: Upload → Analyze & Find Clips → Select → Render with Subtitles → Download")

    with st.sidebar:
        st.header("Settings")
        model_size = st.selectbox("Transcription model", ["base", "small", "medium"], index=0)
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
        window_sec = st.slider("Clip length (sec)", 20, 60, 30, 1)
        stride_sec = st.slider("Stride (sec)", 5, 30, 10, 1)
        scale_width = st.selectbox("Output width", [720, 1080, 1440], index=1)
        user_keywords = st.text_input("Boost keywords (comma-separated)", value="funny,wild,unbelievable,story,secret,tip,how to,viral,crazy,insane,controversial,scandal")
        top_k = st.slider("Keep top N clips", 1, 20, 8, 1)

    st.markdown("### 1) Upload")
    up = st.file_uploader("Upload a podcast video (mp4/mov/mkv)", type=["mp4", "mov", "mkv"])

    if up and st.session_state.tmpdir is None:
        st.session_state.tmpdir = tempfile.mkdtemp(prefix="podclipper_")
        st.session_state.in_path = os.path.join(st.session_state.tmpdir, up.name)
        with open(st.session_state.in_path, "wb") as f:
            f.write(up.read())
        st.success("Video staged locally. Ready to analyze.")

    st.markdown("---")
    st.markdown("### 2) Analyze & Find Clips")
    analyze_clicked = st.button("🔍 Analyze video & find best moments")

    if analyze_clicked and st.session_state.in_path:
        total_steps = 5
        pb = st.progress(0, text="Starting…")
        step = 0
        def tick(msg):
            nonlocal step
            step += 1
            pb.progress(min(step/total_steps, 1.0), text=msg)

        with st.status("Analyzing…", expanded=True) as s:
            st.write("Transcribing audio → text")
            tick("Transcribing with faster-whisper…")
            segments, _ = transcribe_with_whisper(st.session_state.in_path, model_size, device)
            if not segments:
                s.update(label="No speech found", state="error")
                st.stop()
            st.session_state.segments = segments
            st.write(f"Found {len(segments)} transcript segments")

            st.write("Writing SRT subtitles")
            tick("Building SRT…")
            srt_text = srt_from_segments(segments)
            srt_path = os.path.join(st.session_state.tmpdir, "full.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
            st.session_state.srt_path = srt_path
            st.write("SRT ready")

            st.write("Extracting audio & computing features")
            tick("Audio analysis…")
            wav_path = os.path.join(st.session_state.tmpdir, "audio.wav")
            extract_audio_to_wav(st.session_state.in_path, wav_path)
            audio_feats = compute_audio_features(wav_path)
            st.session_state.audio_feats = audio_feats
            st.write("Audio features computed")

            st.write("Generating candidate windows")
            tick("Slicing into windows…")
            windows = slice_segments_into_windows(segments, window_sec=window_sec, stride_sec=stride_sec)
            if not windows:
                s.update(label="No speechy windows found", state="error")
                st.stop()
            st.write(f"{len(windows)} windows")

            st.write("Scoring windows for 'virality'")
            tick("Scoring…")
            sia = SentimentIntensityAnalyzer()
            keywords = [k.strip().lower() for k in user_keywords.split(",") if k.strip()]
            cands: List[ClipCandidate] = []
            for start, end, text in windows:
                rating, reasons = score_window(text, start, end, sia, audio_feats, keywords)
                cands.append(ClipCandidate(start, end, text, rating, reasons))
            cands.sort(key=lambda c: c.rating, reverse=True)
            cands = cands[:top_k]
            st.session_state.candidates = cands
            s.update(label="Analysis complete", state="complete")
            pb.progress(1.0, text="Done")

    if st.session_state.candidates:
        st.markdown("---")
        st.markdown("### 3) Select Clips to Render (with Subtitles)")
        df = pd.DataFrame([{
            "Clip #": i+1,
            "Start (s)": round(c.start, 2),
            "End (s)": round(c.end, 2),
            "Length (s)": round(c.end - c.start, 2),
            "Rating": round(c.rating, 3),
            "Reasons": " | ".join(c.reasons),
            "Preview Text": (c.text[:160] + ("…" if len(c.text) > 160 else "")),
        } for i, c in enumerate(st.session_state.candidates)])
        st.dataframe(df, use_container_width=True)

        if st.session_state.selected is None:
            st.session_state.selected = [True] * len(st.session_state.candidates)

        cols = st.columns([1,3,3,2,2])
        with cols[0]:
            all_toggle = st.checkbox("Select all", value=all(st.session_state.selected))
            if all_toggle and not all(st.session_state.selected):
                st.session_state.selected = [True]*len(st.session_state.selected)
            elif not all_toggle and all(st.session_state.selected):
                st.session_state.selected = [False]*len(st.session_state.selected)

        st.write("Tick which clips you want to render:")
        for i, c in enumerate(st.session_state.candidates, start=1):
            st.session_state.selected[i-1] = st.checkbox(
                f"Clip {i}: {c.start:.2f}s → {c.end:.2f}s  (Rating {c.rating:.3f})",
                value=st.session_state.selected[i-1]
            )

        st.markdown("### 4) Render Selected Clips (burn subtitles)")
        render_clicked = st.button("🎬 Render selected clips")

        if render_clicked:
            chosen = [ (i,c) for i,c in enumerate(st.session_state.candidates) if st.session_state.selected[i] ]
            if not chosen:
                st.warning("Select at least one clip.")
            else:
                render_pb = st.progress(0, text="Rendering…")
                total = len(chosen)
                done = 0
                with st.status("Rendering with FFmpeg…", expanded=True) as s:
                    for idx, c in chosen:
                        clip_out = os.path.join(st.session_state.tmpdir, f"clip_{idx+1:02d}.mp4")
                        try:
                            st.write(f"Rendering Clip {idx+1} → {clip_out}")
                            burn_subtitles_ffmpeg(
                                st.session_state.in_path,
                                st.session_state.srt_path,
                                c.start, c.end,
                                clip_out,
                                scale_width=scale_width
                            )
                            with open(clip_out, "rb") as f:
                                st.video(f.read())
                            st.download_button(
                                f"Download Clip {idx+1}",
                                data=open(clip_out, "rb").read(),
                                file_name=f"clip_{idx+1:02d}.mp4",
                                mime="video/mp4"
                            )
                        except Exception as e:
                            st.error(f"FFmpeg failed for Clip {idx+1}: {e}")
                        finally:
                            done += 1
                            render_pb.progress(done/total, text=f"Rendered {done}/{total}")
                    s.update(label="Rendering complete", state="complete")
                st.success("All selected clips processed.")

if __name__ == "__main__":
    main()
