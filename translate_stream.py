#!/usr/bin/env python3
"""Real-time uk/ru -> English translation of desktop (Signal) call audio.

Pipeline:  PulseAudio/PipeWire monitor  ->  energy VAD (utterance segmentation)
           ->  whisper.cpp server (ASR, iGPU)  ->  llama.cpp server (translation)
           ->  scrolling terminal output.

The mic is ignored on purpose: in a call you speak English, so we tap the
*playback* the remote uk/ru speaker comes out of (a sink monitor), not the mic.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from collections import deque

import numpy as np
import requests

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000          # 480
FRAME_BYTES = FRAME_SAMPLES * 2                          # s16le mono


# --------------------------------------------------------------------------- #
# Audio capture
# --------------------------------------------------------------------------- #
def default_monitor_source() -> str:
    """Monitor of the current default sink (what you hear = all system audio)."""
    sink = subprocess.check_output(["pactl", "get-default-sink"], text=True).strip()
    return f"{sink}.monitor"


def find_signal_node() -> tuple[int, str] | None:
    """Locate Signal's call *output* stream (the remote party you hear).

    Signal plays call audio through a `ringrtc` node (media.class
    Stream/Output/Audio). Returns (object.serial, media.name) or None if no
    call is active. The object *id* drifts between calls, so we key on serial.
    """
    try:
        dump = json.loads(subprocess.check_output(["pw-dump"]))
    except Exception:
        return None
    for obj in dump:
        if obj.get("type") != "PipeWire:Interface:Node":
            continue
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("media.class") != "Stream/Output/Audio":
            continue
        if props.get("node.name") == "ringrtc" or \
           props.get("application.process.binary") == "signal-desktop":
            serial = props.get("object.serial")
            if serial is not None:
                return int(serial), props.get("media.name", "signal")
    return None


def _open_wav(path) -> wave.Wave_write:
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SAMPLE_RATE)
    return w


def tee_record(frames, wav_path):
    """Pass frames through while also writing a continuous session recording."""
    w = _open_wav(wav_path)
    try:
        for f in frames:
            w.writeframes(f)
            yield f
    finally:
        w.close()


def _frames_from_proc(cmd: list[str]):
    """Yield fixed-size s16le mono frames from a capture subprocess, forever."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(FRAME_BYTES)
            if not buf or len(buf) < FRAME_BYTES:
                break
            yield buf
    finally:
        proc.terminate()


def capture_frames(source: str):
    """Capture a PulseAudio source (sink monitor) via parec."""
    return _frames_from_proc([
        "parec", "-d", source,
        "--format=s16le", f"--rate={SAMPLE_RATE}",
        "--channels=1", "--latency-msec=30",
    ])


def capture_frames_signal(serial: int):
    """Tap Signal's call output directly via pw-record (raw PCM, no null sink).

    Non-destructive fan-out: Signal keeps playing to your speakers untouched;
    we just capture a copy of the same stream.
    """
    return _frames_from_proc([
        "pw-record", "--target", str(serial),
        "--rate", str(SAMPLE_RATE), "--channels", "1",
        "--format", "s16", "--raw", "--latency", "30ms", "-",
    ])


# --------------------------------------------------------------------------- #
# Energy VAD -> utterance segmentation
# --------------------------------------------------------------------------- #
class Utterance:
    """Accumulates speech frames and emits a WAV blob."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []

    def add(self, frame: bytes) -> None:
        self.frames.append(frame)

    def duration_s(self) -> float:
        return len(self.frames) * FRAME_MS / 1000.0

    def to_wav(self) -> bytes:
        pcm = b"".join(self.frames)
        bio = io.BytesIO()
        with wave.open(bio, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        return bio.getvalue()


def segment(frames, args, out_q: "queue.Queue[bytes]") -> None:
    """Split the frame stream into utterances on silence boundaries."""
    noise = float(args.noise_init)          # adaptive noise floor (RMS)
    preroll = deque(maxlen=int(args.preroll_ms / FRAME_MS))
    utt: Utterance | None = None
    silence_ms = 0

    for frame in frames:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        thresh = max(args.rms_floor, noise * args.speech_mult)
        speaking = rms > thresh

        if utt is None:
            preroll.append(frame)
            if speaking:
                utt = Utterance()
                utt.frames.extend(preroll)   # keep the onset
                preroll.clear()
                silence_ms = 0
            else:
                # adapt noise floor only while it's quiet
                noise = 0.95 * noise + 0.05 * rms
        else:
            utt.add(frame)
            if speaking:
                silence_ms = 0
            else:
                silence_ms += FRAME_MS
            end = silence_ms >= args.silence_ms
            too_long = utt.duration_s() >= args.max_utt_s
            if end or too_long:
                if utt.duration_s() >= args.min_utt_s:
                    out_q.put(utt.to_wav())
                utt = None
                silence_ms = 0


# --------------------------------------------------------------------------- #
# Stage 1 (ASR) + Stage 2 (translate) worker
# --------------------------------------------------------------------------- #
# Whisper silence/noise hallucination markers and content-free outputs.
_BLANK_MARKERS = {"[blank_audio]", "[ silence ]", "(silence)", "[silence]",
                  "[music]", "(music)", "[applause]", "thank you.", "you"}


def is_meaningful(text: str) -> bool:
    """Reject empty, punctuation-only, or known blank/hallucination outputs.

    Accepts any alphanumeric content — including numbers-only lines like
    "1, 2, 3" (whisper normalizes spoken numerals to digits), which must NOT
    be dropped.
    """
    t = text.strip()
    if not t:
        return False
    if t.lower() in _BLANK_MARKERS:
        return False
    # require some letters or digits (drops ".", "-", "...", whitespace)
    return re.search(r"\w", t) is not None


def transcribe(wav: bytes, args) -> tuple[str, str]:
    r = requests.post(
        f"{args.whisper_url}/inference",
        files={"file": ("utt.wav", wav, "audio/wav")},
        data={"temperature": "0.0", "response_format": "verbose_json",
              "language": args.language},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    return (j.get("text") or "").strip(), (j.get("language") or "").strip()


def translate(text: str, context: list[str], args) -> str:
    sys_prompt = (
        "You are a translator for a live phone call. Translate the user text "
        "(Ukrainian or Russian) into natural, fluent English. Output ONLY the "
        "English translation, no notes, no quotes, no source text."
    )
    msgs = [{"role": "system", "content": sys_prompt}]
    if context:
        msgs.append({"role": "system",
                     "content": "Recent context (already translated):\n" + "\n".join(context)})
    msgs.append({"role": "user", "content": text})
    r = requests.post(
        f"{args.llm_url}/v1/chat/completions",
        json={
            "messages": msgs,
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},  # CRITICAL for this model
        },
        timeout=120,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def worker(in_q: "queue.Queue[bytes]", args) -> None:
    context: deque[str] = deque(maxlen=args.context_lines)
    n = 0
    while True:
        wav = in_q.get()
        if wav is None:
            return
        n += 1
        # Save the exact audio sent to ASR *before* filtering, so dropped
        # utterances are still inspectable.
        if args.save_audio:
            with open(f"{args.save_audio}/utt_{n:04d}.wav", "wb") as fh:
                fh.write(wav)
        try:
            src, lang = transcribe(wav, args)
            meaningful = is_meaningful(src)
            eng = translate(src, list(context), args) if meaningful else ""
            ts = time.strftime("%H:%M:%S")
            if args.transcript:
                with open(args.transcript, "a") as fh:
                    fh.write(json.dumps({"n": n, "ts": ts, "lang": lang,
                                         "source": src, "english": eng,
                                         "kept": bool(meaningful and eng)},
                                        ensure_ascii=False) + "\n")
            if meaningful and eng:
                context.append(eng)
                tag = f"({lang}) " if lang else ""
                if args.show_source:
                    print(f"[{ts}] {tag}{src}", flush=True)
                print(f"[{ts}] {tag}{eng}", flush=True)
        except Exception as exc:  # keep the loop alive on transient errors
            print(f"[warn] {exc}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
def check_servers(args) -> bool:
    ok = True
    try:
        requests.get(f"{args.whisper_url}/", timeout=3)
    except Exception:
        print(f"[error] whisper-server not reachable at {args.whisper_url} "
              f"(start it on that port).", file=sys.stderr)
        ok = False
    try:
        requests.get(f"{args.llm_url}/v1/models", timeout=3).raise_for_status()
    except Exception:
        print(f"[error] llama-server not reachable at {args.llm_url}.", file=sys.stderr)
        ok = False
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # Default: capture ONLY Signal's call audio (the remote party). The overrides
    # below switch to whole-system / a specific source instead.
    p.add_argument("--monitor", action="store_true",
                   help="Override: capture the whole default-sink monitor "
                        "(all system audio) instead of just the Signal call.")
    p.add_argument("--source", default=None,
                   help="Override: capture this specific PulseAudio source "
                        "instead of the Signal call.")
    p.add_argument("--whisper-url", default="http://127.0.0.1:8081")
    p.add_argument("--llm-url", default="http://127.0.0.1:8080")
    p.add_argument("--language", default="auto", help="'auto', 'uk', or 'ru'.")
    # VAD tuning
    p.add_argument("--silence-ms", type=int, default=1000,
                   help="Silence needed to end an utterance. Higher = fewer "
                        "mid-sentence splits (better fluency), a bit more lag.")
    p.add_argument("--min-utt-s", type=float, default=0.4)
    p.add_argument("--max-utt-s", type=float, default=12.0)
    p.add_argument("--preroll-ms", type=int, default=300)
    p.add_argument("--rms-floor", type=float, default=180.0,
                   help="Absolute minimum RMS to count as speech.")
    p.add_argument("--speech-mult", type=float, default=3.0,
                   help="Speech threshold = noise_floor * this.")
    p.add_argument("--noise-init", type=float, default=120.0)
    p.add_argument("--context-lines", type=int, default=4,
                   help="Previous translations fed back for coherence.")
    # Recording / logging — always on, one timestamped dir per session,
    # created directly in the current directory.
    p.add_argument("--no-record", action="store_true",
                   help="Disable recording entirely (rarely needed).")
    p.add_argument("--show-source", action="store_true",
                   help="Also print the source-language transcript to the terminal.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not check_servers(args):
        return 1

    if args.source or args.monitor:
        source = args.source or default_monitor_source()
        frames = capture_frames(source)
        src_desc = f"{source} (whole-system override)"
    else:
        found = find_signal_node()
        if found is None:
            print("[error] No active Signal call found. Start a call first, or "
                  "use --monitor / --source to capture system audio instead.",
                  file=sys.stderr)
            return 1
        serial, name = found
        frames = capture_frames_signal(serial)
        src_desc = f"Signal call ({name}, serial {serial}) — remote party only"

    session_dir = None
    if not args.no_record:
        import os
        session_dir = time.strftime("%y%m%d-%H%M%S")
        os.makedirs(session_dir, exist_ok=True)
        args.save_audio = session_dir
        args.transcript = os.path.join(session_dir, "transcript.jsonl")
        frames = tee_record(frames, os.path.join(session_dir, "session.wav"))
    else:
        args.save_audio = None
        args.transcript = None

    q: "queue.Queue[bytes]" = queue.Queue()
    threading.Thread(target=worker, args=(q, args), daemon=True).start()

    print(f"source   = {src_desc}")
    print(f"whisper  = {args.whisper_url}   llm = {args.llm_url}")
    print(f"language = {args.language}   (Ctrl+C to stop)")
    if session_dir:
        print(f"recording -> {session_dir}/  (session.wav, utt_NNNN.wav, transcript.jsonl)")
    print()

    try:
        segment(frames, args, q)
    except KeyboardInterrupt:
        print("\nStopping...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
