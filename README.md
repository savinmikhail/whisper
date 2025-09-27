Whisper (faster-whisper) Dockerized Transcriber
===============================================

This project provides a minimal Docker image to transcribe Russian audio using the small multilingual Whisper model via faster-whisper (CPU-only by default).

Features
- Small multilingual model suitable for Russian (`small`).
- CPU-friendly with INT8 quantization for speed.
- Outputs: plain text, SRT, VTT, or JSON.
- Optional VAD filtering for cleaner segments.
 - Optional speaker diarization (pyannote) and readable paragraph formatting.

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

Speakers (Diarization)
- Enable speaker labels with pyannote (CPU, slower, large deps):
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt SPEAKERS=1 HF_TOKEN=hf_your_token

- SRT/VTT with speakers (prefixes captions with speaker):
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.srt" FORMAT=srt SPEAKERS=1 HF_TOKEN=hf_your_token

- Force exactly two speakers:
  make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" FORMAT=txt SPEAKERS=1 NUM_SPEAKERS=2 HF_TOKEN=hf_your_token

- Notes:
  - You need a Hugging Face token with access to `pyannote/speaker-diarization-3.1`.
  - By default, TXT is formatted into readable paragraphs; set `TXT_GROUPING=segments` for line-per-segment or `TXT_GROUPING=none` for a single line.
  - Tweak paragraphing via `MAX_GAP` (default 1.0s), `MAX_PARAGRAPH_SECONDS` (30s), and `MIN_PARAGRAPH_CHARS` (80).

Notes
- The first run downloads model weights to the container cache (unless pre-cached at build time). Subsequent runs are faster.
- For best quality on noisy recordings, add `--vad`.
- If you have an NVIDIA GPU and want GPU acceleration, build a CUDA-enabled image and run with the NVIDIA container runtime; this setup is CPU-only by default.
 - The Makefile sets `HF_HOME`/`XDG_CACHE_HOME` to `/app/.cache` so models cache into your project folder.

CLI Help
  docker run --rm whisper-ru --help
