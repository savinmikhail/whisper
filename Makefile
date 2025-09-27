# Загружаем переменные из .env
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

IMAGE ?= whisper-ru
MODEL ?= small
LANG ?= ru
FORMAT ?= txt
COMPUTE ?= int8
BEAM ?= 5
UID ?= $(shell id -u)
GID ?= $(shell id -g)
SPEAKERS ?=
HF_TOKEN ?=
DIARIZE_MODEL ?= pyannote/speaker-diarization-3.1
NUM_SPEAKERS ?=
TXT_GROUPING ?= paragraphs
MAX_GAP ?= 1.0
MAX_PARAGRAPH_SECONDS ?= 30
MIN_PARAGRAPH_CHARS ?= 80

.PHONY: build transcribe help

build:
	docker build -t $(IMAGE) .

# Usage:
#   make transcribe FILE="audio/interview.mp3" OUT="outputs/interview.txt" [MODEL=small] [LANG=ru] [FORMAT=txt|srt|vtt|json] [VAD=1]
transcribe:
	@test -n "$(FILE)" || (echo "Error: set FILE=path/to/input audio" && exit 1)
	@test -n "$(OUT)" || (echo "Error: set OUT=path/to/output file" && exit 1)
	@mkdir -p "$(dir $(OUT))"
	docker run --rm -u $(UID):$(GID) -v "$(CURDIR)":/app \
		-e HF_HOME=/app/.cache/huggingface -e XDG_CACHE_HOME=/app/.cache -e TRANSFORMERS_CACHE=/app/.cache/huggingface \
		-e TXT_GROUPING=$(TXT_GROUPING) -e MAX_GAP=$(MAX_GAP) -e MAX_PARAGRAPH_SECONDS=$(MAX_PARAGRAPH_SECONDS) -e MIN_PARAGRAPH_CHARS=$(MIN_PARAGRAPH_CHARS) \
		$(if $(HF_TOKEN),-e HF_TOKEN=$(HF_TOKEN),) \
		$(IMAGE) \
		"/app/$(FILE)" \
		--language $(LANG) --model $(MODEL) --compute-type $(COMPUTE) --beam-size $(BEAM) \
		$(if $(VAD),--vad,) $(if $(SPEAKERS),--diarize --diarize-model $(DIARIZE_MODEL) $(if $(NUM_SPEAKERS),--num-speakers $(NUM_SPEAKERS),),) --format $(FORMAT) -o "/app/$(OUT)"

help:
	@echo "Usage:"
	@echo "  make build"
	@echo "  make transcribe FILE=audio/interview.mp3 OUT=outputs/interview.txt [MODEL=small] [LANG=ru] [FORMAT=txt|srt|vtt|json] [VAD=1]"
