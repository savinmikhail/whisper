Whisper (faster-whisper) Dockerized Transcriber
===============================================

This project provides a minimal Docker image to transcribe Russian audio using the small multilingual Whisper model via faster-whisper (CPU-only by default).

Features
- Small multilingual model suitable for Russian (`small`).
- CPU-friendly with INT8 quantization for speed.
- Outputs: plain text, SRT, VTT, or JSON.
- Optional VAD filtering for cleaner segments.
 - Optional speaker diarization (pyannote) and readable paragraph formatting.
 - Progress estimates during transcription with ETA.

Build
- Build with model pre-cache (default `small`):
  docker build -t whisper-ru .

- To skip pre-downloading the model (download on first run):
  docker build -t whisper-ru --build-arg MODEL_SIZE=none .

Run
Assume your audio file is at `./audio/interview.mp3`.

- Plain text to stdout (Russian, small model):
  docker run --rm -v "$PWD":/app whisper-ru /app/audio/interview.mp3 --language ru --model small --format txt

- Save to a text file:
  docker run --rm -v "$PWD":/app whisper-ru /app/audio/interview.mp3 --language ru --model small -o /app/outputs/interview.txt

- Generate SRT captions:
  docker run --rm -v "$PWD":/app whisper-ru /app/audio/interview.mp3 --language ru --model small --format srt -o /app/outputs/interview.srt

Makefile
- Build image:
  make build

- Transcribe and save to a file (choose format via `FORMAT`):
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt

- Other examples:
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.srt" FORMAT=srt VAD=1
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.json" FORMAT=json MODEL=small LANG=ru
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt PROGRESS=1 PROGRESS_INTERVAL=2
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt WARN=0   # suppress deprecation/info warnings
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt TTY=1    # single-line updating progress
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt SPEAKERS=1 NUM_SPEAKERS=2 DIARIZE_RTF=0.35  # better diarization ETA
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt SPEAKERS=1 NUM_SPEAKERS=2 DIARIZE_PROGRESS=elapsed  # elapsed-only diarization progress
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt TXT_TIMESTAMPS=start  # add [mm:ss] before each paragraph

Speakers (Diarization)
- Enable speaker labels with pyannote (CPU, slower, large deps):
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt SPEAKERS=1 HF_TOKEN=hf_your_token

- SRT/VTT with speakers (prefixes captions with speaker):
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.srt" FORMAT=srt SPEAKERS=1 HF_TOKEN=hf_your_token

- Force exactly two speakers:
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt SPEAKERS=1 NUM_SPEAKERS=2 HF_TOKEN=hf_your_token

- Notes:
  - You need a Hugging Face token with access to `pyannote/speaker-diarization-3.1`.
  - When diarization is enabled, the script extracts audio to a temporary 16kHz mono WAV via ffmpeg for compatibility (MP4/AAC is not directly supported by the default backend).
  - Diarization progress: default is elapsed-only via `DIARIZE_PROGRESS=elapsed`. For ETA-based estimates, set `DIARIZE_PROGRESS=estimate` (optionally add `DIARIZE_RTF=0.35`). Set `DIARIZE_PROGRESS=off` to hide it.
  - Time marks in TXT: controlled by `TXT_TIMESTAMPS` (`off`, `start`, `range`). The Makefile defaults to `start`, producing lines like `[03:28] Speaker 1: …`.
  - By default, TXT is formatted into readable paragraphs; set `TXT_GROUPING=segments` for line-per-segment or `TXT_GROUPING=none` for a single line.
  - Tweak paragraphing via `MAX_GAP` (default 1.0s), `MAX_PARAGRAPH_SECONDS` (30s), and `MIN_PARAGRAPH_CHARS` (80).
  - To hide noisy library warnings, use `WARN=0` (default). To see them for debugging, set `WARN=1`.
  - `TTY=1` allocates an interactive terminal for the container (`-it`), enabling a single updating progress line. Without it, progress prints as periodic lines.
  - Progress prints to stderr. In non-interactive runs it emits periodic lines; in a TTY it updates a single line with percentage and ETA.

Notes
- The first run downloads model weights to the container cache (unless pre-cached at build time). Subsequent runs are faster.
- For best quality on noisy recordings, add `--vad`.
- If you have an NVIDIA GPU and want GPU acceleration, build a CUDA-enabled image and run with the NVIDIA container runtime; this setup is CPU-only by default.
 - The Makefile sets `HF_HOME`/`XDG_CACHE_HOME` to `/app/.cache` so models cache into your project folder.

CLI Help
  docker run --rm whisper-ru --help
