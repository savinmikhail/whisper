FROM python:3.11-slim

# Build-time options
ARG MODEL_SIZE=small

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    WHISPER_MODEL=${MODEL_SIZE} \
    WHISPER_LANGUAGE=ru

WORKDIR /app

# System deps: ffmpeg for audio decoding, and basic certificates
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY transcribe.py ./

# Default command shows help
ENTRYPOINT ["python", "/app/transcribe.py"]
