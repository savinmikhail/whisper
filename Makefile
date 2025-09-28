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
PROGRESS ?= 1
PROGRESS_INTERVAL ?= 1.0
WARN ?= 0
TTY ?= 0

# Internal: add -it when TTY=1 for single-line progress updates
DOCKER_TTY := $(if $(filter 1 true yes on,$(TTY)),-it,)
SPEAKERS ?=
HF_TOKEN ?=
DIARIZE_MODEL ?= pyannote/speaker-diarization-3.1
NUM_SPEAKERS ?=
DIARIZE_RTF ?=
DIARIZE_PROGRESS ?= elapsed
TXT_GROUPING ?= paragraphs
TXT_TIMESTAMPS ?= start
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
	@mkdir -p "$(dir $(OUT))" \
		"$(CURDIR)/.cache/huggingface" "$(CURDIR)/.cache/matplotlib" "$(CURDIR)/.cache/torch" "$(CURDIR)/.cache/pyannote" "$(CURDIR)/.config/matplotlib"
	@docker run $(DOCKER_TTY) --rm -u $(UID):$(GID) -v "$(CURDIR)":/app \
		-e HOME=/app -e XDG_CONFIG_HOME=/app/.config \
		-e HF_HOME=/app/.cache/huggingface -e HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface -e XDG_CACHE_HOME=/app/.cache -e TRANSFORMERS_CACHE=/app/.cache/huggingface \
		-e PYANNOTE_CACHE=/app/.cache/pyannote -e TORCH_HOME=/app/.cache/torch -e MPLCONFIGDIR=/app/.cache/matplotlib \
		-e NO_WARNINGS=$(WARN) $(if $(filter 1 true yes on,$(WARN)),,-e ORT_LOG_SEVERITY_LEVEL=3 -e PYTHONWARNINGS=ignore) \
		-e TXT_GROUPING=$(TXT_GROUPING) -e TXT_TIMESTAMPS=$(TXT_TIMESTAMPS) -e MAX_GAP=$(MAX_GAP) -e MAX_PARAGRAPH_SECONDS=$(MAX_PARAGRAPH_SECONDS) -e MIN_PARAGRAPH_CHARS=$(MIN_PARAGRAPH_CHARS) \
		-e PROGRESS=$(PROGRESS) -e PROGRESS_INTERVAL=$(PROGRESS_INTERVAL) \
		$(if $(DIARIZE_RTF),-e DIARIZE_RTF=$(DIARIZE_RTF),) -e DIARIZE_PROGRESS=$(DIARIZE_PROGRESS) \
		$(if $(HF_TOKEN),-e HF_TOKEN=$(HF_TOKEN),) \
		$(IMAGE) \
		"/app/$(FILE)" \
		--language $(LANG) --model $(MODEL) --compute-type $(COMPUTE) --beam-size $(BEAM) \
		$(if $(VAD),--vad,) $(if $(SPEAKERS),--diarize --diarize-model $(DIARIZE_MODEL) $(if $(NUM_SPEAKERS),--num-speakers $(NUM_SPEAKERS),),) $(if $(PROGRESS),--progress --progress-interval $(PROGRESS_INTERVAL),) $(if $(filter 1 true yes on,$(WARN)),, --no-warnings) --format $(FORMAT) -o "/app/$(OUT)"

help:
	@echo "Usage:"
	@echo "  make build"
	@echo "  make transcribe FILE=audio/interview.mp3 OUT=outputs/interview.txt [MODEL=small] [LANG=ru] [FORMAT=txt|srt|vtt|json] [VAD=1]"
