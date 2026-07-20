# Session log — 2026-07-19

## Goal

Pick the project back up after losing track of it: real-time Ukrainian and
Russian speech → English translation, needed for Signal calls.

## What happened, in order

1. **Recovered context.** Read existing `realtime_translate.py`
   (faster-whisper / OpenAI backends) and `qwen3_omni_translate.py`
   (Qwen3-Omni, shelved — segfaults on ROCm GPU load).
2. **Reassessed against the actual hardware** (Strix Halo, Ryzen AI Max+ 395,
   gfx1151 iGPU, 128 GB unified memory). Found faster-whisper was almost
   certainly running CPU-only; the better-supported native path is
   whisper.cpp built with ROCm.
3. **Locked requirements:** fluency over latency, 100% offline, source is
   live Signal calls. This ruled out the OpenAI backends and Whisper's
   built-in one-shot `translate` task, and pointed to a two-stage design.
4. **Landed on the architecture:** whisper.cpp (ROCm, large-v3) for ASR,
   reusing the already-running llama-server (Qwen3 27B) for translation —
   no new model to stand up.
5. **Built and validated Stage 1.** Compiled whisper.cpp with HIP for
   gfx1151, confirmed GPU offload (~10x real-time on `large-v3`), then
   validated ASR quality on real Ukrainian and Russian audio pulled from
   Wikimedia Commons.
6. **Validated Stage 2.** Hit an early failure: the Qwen3 model is a
   reasoning model and was burning the whole token budget on hidden
   `reasoning_content`, returning empty translations. Fixed by sending
   `"chat_template_kwargs": {"enable_thinking": false}` on every request
   (the `/no_think` prompt trick does not work on this finetune). With that,
   translation is fast (~2s) and fluent on both languages.
7. **Wired the live streaming loop** (`translate_stream.py` + `run.sh`):
   energy-based VAD segments the audio into utterances, which flow through
   whisper-server → llama-server → the terminal, with a rolling translation
   context for coherence.
8. **Made capture Signal-only by default.** Discovered Signal routes call
   audio through a `ringrtc` PipeWire node; the orchestrator finds it
   dynamically each run (the node's id/serial changes call to call) and taps
   it directly with `pw-record`, non-destructively — Signal keeps playing to
   the speakers as normal.
9. **Fixed a real silence-hallucination problem**, found via live testing on
   an actual call: Whisper invents text ("Great Britain", "Thank you") during
   call pauses. Fixed with whisper-server's native Silero VAD plus a
   client-side filter.
10. **Live-call testing surfaced a genuine bug and a false alarm.** The false
    alarm: garbled-looking output was actually correct — the test speech was
    Spanish, and it was detected and translated correctly. The real bug: the
    meaningful-text filter required 2+ letters, so a spoken-numbers utterance
    ("1, 2, 3...") was wrongly dropped as junk. Fixed to accept digits too.
11. **Added always-on session recording** per request — every run now saves
    `session.wav`, per-utterance WAVs, and a `transcript.jsonl`, initially
    under `recordings/`, then moved to a bare `<yymmdd-HHMMSS>/` directory
    created directly in the run location (no `recordings/` wrapper, no
    `--record-dir` flag — removed since it would never be used).
12. **Confirmed working on a real Signal call.**
13. **Created the git repo and pushed to GitHub** (public,
    `DanCard/realtime-translate`), with `.venv/`, `test-audio/`, logs, and
    the timestamped recording directories all gitignored — call audio and
    transcripts never leave the machine.
14. **Made `run.sh` symlink-safe** (`readlink -f` instead of raw `$0`) and
    symlinked it to `~/bin/translate`, so the whole pipeline can be launched
    from anywhere with one command.

## State at end of session

Working end-to-end, offline, on both Ukrainian and Russian, validated on a
real Signal call. Launch with `translate` (or `~/dtu/realtime-translate/run.sh`)
from any directory; each run's recording lands in `<yymmdd-HHMMSS>/` under the
current directory.

## Open items (non-blocking)

- Auto-detected language label can flip-flop on very short/ambiguous phrases
  (translation itself is unaffected). Pin `--language uk` or `--language ru`
  for a call known to be one language if this ever matters.
- Optional: subtitle-overlay window instead of scrolling terminal output.
- Optional: delete the archived v1 prototype code in `archive/v1-prototype/`.
