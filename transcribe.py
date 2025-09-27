#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import timedelta

from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio to text using faster-whisper (CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to input audio/video file")
    parser.add_argument(
        "--model",
        default=os.getenv("WHISPER_MODEL", "small"),
        help="Whisper model size/name (e.g. tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("WHISPER_LANGUAGE", "ru"),
        help="Spoken language code (e.g. ru, en, de)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("WHISPER_DEVICE", "cpu"),
        help="Device to run on (cpu, cuda)",
    )
    parser.add_argument(
        "--compute-type",
        dest="compute_type",
        default=os.getenv("WHISPER_COMPUTE", "int8"),
        help="Quantization compute type (int8, int16, float32, etc.)",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task: transcribe in-source language or translate to English",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable VAD filtering to trim non-speech (slower, better quality)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write output to file path (otherwise prints to stdout)",
    )
    return parser.parse_args()


def srt_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    td = timedelta(seconds=max(0, seconds))
    # SRT uses comma as decimal separator
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(round((td.total_seconds() - total_seconds) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def vtt_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    td = timedelta(seconds=max(0, seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(round((td.total_seconds() - total_seconds) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def format_output(segments, fmt: str):
    if fmt == "txt":
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())
        return " ".join(t for t in texts if t)
    elif fmt == "srt":
        lines = []
        idx = 1
        for seg in segments:
            start = srt_timestamp(seg.start)
            end = srt_timestamp(seg.end)
            text = seg.text.strip()
            if not text:
                continue
            lines.append(str(idx))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
            idx += 1
        return "\n".join(lines).rstrip() + "\n"
    elif fmt == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segments:
            start = vtt_timestamp(seg.start)
            end = vtt_timestamp(seg.end)
            text = seg.text.strip()
            if not text:
                continue
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    elif fmt == "json":
        data = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments
            if seg.text and seg.text.strip()
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    print(
        f"Loading model '{args.model}' on {args.device} (compute={args.compute_type})...",
        file=sys.stderr,
    )

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    print(
        f"Transcribing '{args.input}' (language={args.language}, task={args.task})...",
        file=sys.stderr,
    )

    segments, info = model.transcribe(
        args.input,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        # condition_on_previous_text improves coherence on longer files
        condition_on_previous_text=True,
    )

    # Collect segments into a list to format/output more than once if needed
    seg_list = list(segments)

    output_text = format_output(seg_list, args.format)

    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Wrote {args.format} to: {out_path}", file=sys.stderr)
    else:
        # Write to stdout
        if sys.stdout.isatty() and args.format in {"srt", "vtt"}:
            # Avoid clutter: hint when printing captions to terminal
            print("[captions output — redirect to a file with -o]", file=sys.stderr)
        sys.stdout.write(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

