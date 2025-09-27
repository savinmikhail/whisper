IMAGE ?= whisper-ru
MODEL ?= small
LANG ?= ru
FORMAT ?= txt
COMPUTE ?= int8
BEAM ?= 5
UID ?= $(shell id -u)
GID ?= $(shell id -g)

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
		$(IMAGE) \
		"/app/$(FILE)" \
		--language $(LANG) --model $(MODEL) --compute-type $(COMPUTE) --beam-size $(BEAM) \
		$(if $(VAD),--vad,) --format $(FORMAT) -o "/app/$(OUT)"

help:
	@echo "Usage:"
	@echo "  make build"
	@echo "  make transcribe FILE=audio/interview.mp3 OUT=outputs/interview.txt [MODEL=small] [LANG=ru] [FORMAT=txt|srt|vtt|json] [VAD=1]"
