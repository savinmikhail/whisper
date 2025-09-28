#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import timedelta
import time
import math
import subprocess
import tempfile
import shlex

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
        "--txt-grouping",
        choices=["none", "segments", "paragraphs"],
        default=os.getenv("TXT_GROUPING", "paragraphs"),
        help="How to group TXT output: single line, per segment, or paragraphs",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=float(os.getenv("MAX_GAP", 1.0)),
        help="New paragraph if gap between segments exceeds seconds",
    )
    parser.add_argument(
        "--max-paragraph-seconds",
        type=float,
        default=float(os.getenv("MAX_PARAGRAPH_SECONDS", 30.0)),
        help="Soft cap for paragraph duration; allows break after sentence end",
    )
    parser.add_argument(
        "--min-paragraph-chars",
        type=int,
        default=int(os.getenv("MIN_PARAGRAPH_CHARS", 80)),
        help="Prefer break after sentence if paragraph has at least this length",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization using pyannote (requires HF token)",
    )
    parser.add_argument(
        "--diarize-model",
        default=os.getenv("DIARIZE_MODEL", "pyannote/speaker-diarization-3.1"),
        help="Pyannote diarization pipeline name",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=(int(os.getenv("NUM_SPEAKERS")) if os.getenv("NUM_SPEAKERS") else None),
        help="Fix the number of speakers for diarization (e.g., 2)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (env HF_TOKEN) for gated diarization models",
    )
    parser.add_argument(
        "--speaker-prefix",
        default=os.getenv("SPEAKER_PREFIX", "Speaker"),
        help="Prefix for speaker labels in output (e.g. 'Speaker', 'SPK')",
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
        "--progress",
        action="store_true",
        default=os.getenv("PROGRESS") in {"1", "true", "yes", "on"},
        help="Show rough progress updates during transcription",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=float(os.getenv("PROGRESS_INTERVAL", 1.0)),
        help="Seconds between progress updates (non-TTY)",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        default=os.getenv("NO_WARNINGS") in {"1", "true", "yes", "on"},
        help="Suppress third-party deprecation/info warnings",
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


def hms(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    seconds = max(0.0, float(seconds))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02}:{m:02}:{s:02}"


def configure_quiet(no_warnings: bool) -> None:
    if not no_warnings:
        return
    # Reduce noisy library warnings
    import warnings

    try:
        # Common deprecation/info warnings from audio stack
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=r".*deprecated.*torchaudio.*")
        warnings.filterwarnings("ignore", module=r"torchaudio(\.|$)")
        warnings.filterwarnings("ignore", module=r"pyannote(\.|$)")
        warnings.filterwarnings("ignore", module=r"speechbrain(\.|$)")
    except Exception:
        pass

    # Quieten onnxruntime logs (0=verbose,1=info,2=warning,3=error,4=fatal)
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    # Minor noise reductions
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _group_paragraphs(segments, *, max_gap: float, max_sec: float, min_chars: int, by_speaker: bool):
    groups = []
    current = None
    punct = ".?!…"

    for seg in segments:
        if not seg.text or not seg.text.strip():
            continue
        gap = 0.0
        if current is not None:
            gap = max(0.0, (seg.start or 0.0) - (current["end"] or 0.0))

        start_new = False
        if current is None:
            start_new = True
        else:
            if by_speaker and (getattr(seg, "speaker", None) != current.get("speaker")):
                start_new = True
            elif gap > max_gap:
                start_new = True
            else:
                duration = (current["end"] or 0.0) - (current["start"] or 0.0)
                ends_with_sentence = current["text"].rstrip().endswith(tuple(punct))
                if ends_with_sentence and (len(current["text"]) >= min_chars or duration >= max_sec):
                    start_new = True

        if start_new:
            current = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "speaker": getattr(seg, "speaker", None),
            }
            groups.append(current)
        else:
            current["end"] = seg.end
            if current["text"]:
                # Keep natural spacing; avoid double spaces
                if not current["text"].endswith(" "):
                    current["text"] += " "
            current["text"] += seg.text.strip()

    return groups


def format_output(segments, fmt: str, *, txt_grouping: str, max_gap: float, max_sec: float, min_chars: int, diarized: bool, speaker_prefix: str):
    if fmt == "txt":
        if txt_grouping == "none":
            return " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip())
        elif txt_grouping == "segments":
            lines = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
            return "\n".join(lines) + ("\n" if lines else "")
        else:  # paragraphs
            groups = _group_paragraphs(
                segments,
                max_gap=max_gap,
                max_sec=max_sec,
                min_chars=min_chars,
                by_speaker=diarized,
            )
            lines = []
            for g in groups:
                prefix = f"{speaker_prefix} {g['speaker']}: " if diarized and g.get("speaker") is not None else ""
                lines.append(prefix + g["text"].strip())
            return "\n\n".join(lines) + ("\n" if lines else "")
    elif fmt == "srt":
        lines = []
        idx = 1
        for seg in segments:
            start = srt_timestamp(seg.start)
            end = srt_timestamp(seg.end)
            text = seg.text.strip()
            if diarized and getattr(seg, "speaker", None) is not None:
                text = f"{speaker_prefix} {seg.speaker}: " + text
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
            if diarized and getattr(seg, "speaker", None) is not None:
                text = f"{speaker_prefix} {seg.speaker}: " + text
            if not text:
                continue
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
    elif fmt == "json":
        data = []
        for seg in segments:
            if not seg.text or not seg.text.strip():
                continue
            item = {"start": seg.start, "end": seg.end, "text": seg.text}
            if diarized and getattr(seg, "speaker", None) is not None:
                item["speaker"] = str(seg.speaker)
            data.append(item)
        return json.dumps(data, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main() -> int:
    args = parse_args()
    configure_quiet(args.no_warnings)

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

    t0 = time.time()
    segments, info = model.transcribe(
        args.input,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        # condition_on_previous_text improves coherence on longer files
        condition_on_previous_text=True,
    )

    # Lightweight segment holder that allows attaching 'speaker'
    class OutSeg:
        __slots__ = ("start", "end", "text", "speaker")
        def __init__(self, start, end, text, speaker=None):
            self.start = start
            self.end = end
            self.text = text
            self.speaker = speaker

    # Consume segments with optional progress display
    seg_list = []
    max_end = 0.0
    total_dur = float(getattr(info, "duration", 0.0) or 0.0)
    is_tty = sys.stderr.isatty()
    last_emit = 0.0

    def emit_progress(force=False):
        nonlocal last_emit
        if not args.progress or total_dur <= 0:
            return
        now = time.time()
        if not force and not is_tty and (now - last_emit) < args.progress_interval:
            return
        progress = min(1.0, max_end / total_dur if total_dur > 0 else 0.0)
        elapsed = now - t0
        speed = (max_end / elapsed) if elapsed > 0 else 0.0
        remaining = (total_dur - max_end) / speed if speed > 0 else float('inf')
        percent = int(progress * 100)
        eta_str = hms(remaining) if math.isfinite(remaining) else "??:??:??"
        msg = f"[{percent:3d}%] {hms(max_end)}/{hms(total_dur)} ETA {eta_str}"
        if is_tty:
            sys.stderr.write("\r" + msg)
            sys.stderr.flush()
        else:
            print(msg, file=sys.stderr)
        last_emit = now

    for seg in segments:
        out = OutSeg(seg.start, seg.end, seg.text)
        seg_list.append(out)
        try:
            if out.end is not None:
                max_end = max(max_end, float(out.end))
                emit_progress()
        except Exception:
            pass

    emit_progress(force=True)
    if is_tty and args.progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    # Optional speaker diarization
    diarized = False
    if args.diarize:
        if not args.hf_token:
            print("Diarization requires a Hugging Face token. Set HF_TOKEN or pass --hf-token.", file=sys.stderr)
            return 2
        try:
            from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore
        except Exception as e:
            print(f"Diarization unavailable: failed to import pyannote.audio: {e}", file=sys.stderr)
            print("Try rebuilding image, or run: python -c 'import pyannote.audio' inside the container to see details.", file=sys.stderr)
            return 2
        # Ensure audio is in a format pyannote/torchaudio can read reliably (WAV)
        # Many containers (e.g., MP4/AAC) are not supported by soundfile backend.
        audio_path = args.input
        cleanup = None
        if not str(audio_path).lower().endswith((".wav", ".flac", ".ogg")):
            try:
                tmp = tempfile.NamedTemporaryFile(prefix="pyannote_audio_", suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                cmd = [
                    "ffmpeg", "-y", "-i", audio_path, "-vn",
                    "-ac", "1", "-ar", "16000", "-f", "wav", tmp_path,
                ]
                print("Preparing audio for diarization (ffmpeg -> wav)...", file=sys.stderr)
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    print("ffmpeg failed to extract audio for diarization:", file=sys.stderr)
                    try:
                        err_txt = proc.stderr.decode("utf-8", errors="ignore")
                    except Exception:
                        err_txt = str(proc.stderr)
                    print(err_txt, file=sys.stderr)
                    return 2
                audio_path = tmp_path
                cleanup = tmp_path
            except Exception as e:
                print(f"Failed to prepare audio for diarization: {e}", file=sys.stderr)
                return 2

        print("Running speaker diarization (pyannote)...", file=sys.stderr)
        try:
            pipeline = PyannotePipeline.from_pretrained(args.diarize_model, use_auth_token=args.hf_token)
            # Use explicit file mapping to support a wide range of formats
            call_kwargs = {}
            if args.num_speakers is not None:
                call_kwargs["num_speakers"] = int(args.num_speakers)
            diar = pipeline({"audio": audio_path}, **call_kwargs)
        except Exception as e:
            print(f"Diarization failed: {e}", file=sys.stderr)
            if cleanup:
                try:
                    os.unlink(cleanup)
                except Exception:
                    pass
            return 2

        # Build list of diarization segments with labels
        diar_segs = []
        try:
            for turn, _, speaker in diar.itertracks(yield_label=True):
                diar_segs.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                })
        except Exception:
            # Backwards compatibility if API differs
            for segment, _, label in diar.itertracks(yield_label=True):
                diar_segs.append({
                    "start": float(getattr(segment, "start", 0.0)),
                    "end": float(getattr(segment, "end", 0.0)),
                    "speaker": str(label),
                })

        # Normalize speaker labels to integers starting at 1 for nicer display
        speaker_map = {}
        next_id = 1
        for d in diar_segs:
            if d["speaker"] not in speaker_map:
                speaker_map[d["speaker"]] = next_id
                next_id += 1

        # Assign speaker to each whisper segment via maximum overlap
        def overlap(a_start, a_end, b_start, b_end):
            return max(0.0, min(a_end, b_end) - max(a_start, b_start))

        for seg in seg_list:
            best_label = None
            best_ov = 0.0
            s0, s1 = float(seg.start or 0.0), float(seg.end or 0.0)
            for d in diar_segs:
                ov = overlap(s0, s1, d["start"], d["end"])
                if ov > best_ov:
                    best_ov = ov
                    best_label = d["speaker"]
            if best_label is not None:
                # Attach label for downstream formatting
                seg.speaker = speaker_map[best_label]
                diarized = True
        if cleanup:
            try:
                os.unlink(cleanup)
            except Exception:
                pass

    output_text = format_output(
        seg_list,
        args.format,
        txt_grouping=args.txt_grouping,
        max_gap=args.max_gap,
        max_sec=args.max_paragraph_seconds,
        min_chars=args.min_paragraph_chars,
        diarized=diarized,
        speaker_prefix=args.speaker_prefix,
    )

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
