Whisper (faster-whisper) Dockerized Transcriber
===============================================

This project provides a minimal Docker image to transcribe Russian audio using the small multilingual Whisper model via faster-whisper (CPU-only by default).

Features
- Small multilingual model suitable for Russian (`small`).
- CPU-friendly with INT8 quantization for speed.
- Outputs: plain text, SRT, VTT, or JSON.
- Optional VAD filtering for cleaner segments.

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

Notes
- The first run downloads model weights to the container cache (unless pre-cached at build time). Subsequent runs are faster.
- For best quality on noisy recordings, add `--vad`.
- If you have an NVIDIA GPU and want GPU acceleration, build a CUDA-enabled image and run with the NVIDIA container runtime; this setup is CPU-only by default.
 - The Makefile sets `HF_HOME`/`XDG_CACHE_HOME` to `/app/.cache` so models cache into your project folder.

CLI Help
  docker run --rm whisper-ru --help
