
# PodClipper (Local)

A simple local app that turns a long-form podcast video into multiple ~30s reel candidates with burned-in subtitles and a basic "virality" rating.

## Features
- **Local transcription** using Whisper (no cloud).
- **Automatic clip suggestions**: sliding 30s windows, scored by a blend of emotion (sentiment), keywords, novelty/punchiness, and audio dynamics.
- **Burned-in subtitles** via ffmpeg (SRT -> on-video).
- **One-click MP4 downloads** for each clip.
- **Keyword boosting** to prioritize topics (e.g., "story, crazy, unbelievable").

## Install
1. Install **FFmpeg** and ensure it's on your PATH.
   - macOS: `brew install ffmpeg`
   - Windows: Use a static build from `gyan.dev` or `BtbN`, then add the `bin` folder to PATH.
   - Linux: `sudo apt-get install ffmpeg`

2. Create & activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```

4. (First run only) Whisper will download the chosen model. After that, it runs fully offline.
   - To pre-download: run a quick `python -c "import whisper; whisper.load_model('base')"`

## Run
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Tips
- Try model **small** or **medium** for better accuracy if your machine can handle it.
- Tweak **keywords** in the sidebar to reflect your niche (e.g., "booking, indie, backstage, kayfabe").
- The rating is **heuristic**, not a guarantee. Use it to shortlist, then trust your eye/ear.
- Change **clip length** and **stride** to produce more or fewer candidates.

## Export Style
- Output width defaults to 1080px (changeable).
- Subtitles use a readable ASS style via ffmpeg's `force_style` parameter. Adjust in `burn_subtitles_ffmpeg` if you want different fonts/colors.

## Offline Models
If you prefer `faster-whisper` (CTranslate2), replace the transcription block in `app.py` and install `faster-whisper`. This can be faster on CPU and very fast on GPU with CUDA.
