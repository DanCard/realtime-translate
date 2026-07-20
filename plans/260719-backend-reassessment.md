# Backend Reassessment — Real-time uk+ru → English on Strix Halo

Date: 2026-07-19

## Requirement

Real-time translation of **Ukrainian and Russian speech → English** from Linux
desktop audio, running on a **Strix Halo** system (Ryzen AI Max+ 395).

### Locked priorities (2026-07-19)

1. **Fluency is the top priority** — natural, readable English wins over
   shaving latency. A few seconds of lag behind live is acceptable.
2. **100% offline** — no cloud/API. This removes the OpenAI backends from
   consideration; they become dead code to delete.

Consequence: the direct Whisper-`translate` path (latency-optimal) is **not**
the pick. Fluency-first + offline selects the **two-stage** design below.

## What Strix Halo actually is (and what that implies)

Ryzen AI Max+ 395: Radeon 8060S iGPU (gfx1151, RDNA 3.5), up to 128 GB
**unified** memory, ~256 GB/s bandwidth, plus an XDNA2 NPU. Three consequences:

- **The iGPU is the workhorse** — but only through Vulkan or ROCm. Anything
  CUDA-only, or CPU-only, wastes it.
- **Unified memory is a superpower**: can hold models that would OOM a 24 GB
  discrete card. Favors *bigger local models*, not smaller ones.
- **The NPU is a trap on Linux** — the Ryzen AI stack is Windows-first; ignore
  it for now.

## Grading the choices already made

**`faster-whisper` (CTranslate2) as the local default — the weak link.**
CTranslate2's AMD story has historically been poor, so `--device auto
--compute-type int8` was almost certainly running **CPU-only**, ignoring the
iGPU. Working recipes now exist both ways, but the better-supported native path
is **whisper.cpp built for gfx1151 with ROCm** — confirmed working (1.8.0 +
ROCm 7) and hitting **~4.5–11× real-time on large-v3** on this exact chip. A
large-quality model in real time on the iGPU changes the whole calculus.

**Qwen3-Omni — right to shelve it.** A 30B MoE that segfaults on ROCm load is a
research rabbit hole. Even working, it's overkill and latency-prone for
streaming. Unified memory *can* hold it, but ~256 GB/s bandwidth caps token
throughput — not where you want to be for real-time.

**OpenAI backends — delete.** The 100%-offline requirement rules them out
entirely. Keep only for a one-off manual quality spot-check if ever needed, but
they are not part of the shipping design.

**Two things that matter more than backend choice:**

1. **Fixed 4-second ffmpeg chunks cut words at boundaries** — a real quality
   tax, and doubly bad for fluency because the LLM translator needs *complete
   thoughts*. Replace with **VAD / silence-boundary segmentation** so each unit
   fed downstream is a full utterance, not an arbitrary 4 s slice.
2. **`language="uk"` is hardcoded.** In the two-stage design, run ASR as
   `task=transcribe` with **auto-detect**, producing accurate native uk/ru
   text; the LLM handles uk *and* ru → English with no per-chunk flagging.

## Recommended architecture — two-stage (fluency-first, fully offline)

Because fluency wins and latency is negotiable, transcribe in the *source*
language and let a strong local LLM do the translating. A dedicated LLM
produces far more natural English than Whisper's literal built-in `translate`,
and can use surrounding context for coherence.

**Stage 1 — ASR:** whisper.cpp (ROCm, gfx1151) → `large-v3`, `task=transcribe`,
auto-detect language. Emits accurate Ukrainian/Russian text. Runs comfortably
faster than real-time on the iGPU.

**Stage 2 — Translation:** reuse the **llama-server already running** rather
than standing up a second model. Current daily driver:

```
~/github/llama.cpp/build/bin/llama-server \
  -m ~/llms/qwen3/6/Q4_K_M-27B-uncensored-heretic-v2-Native-MTP.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 999 -c 256000 -fa on -ctk q8_0 -ctv q8_0 \
  --no-mmap --temp 0 --webui-mcp-proxy
```

Why this fits the requirement well:
- **Already offline** — llama-server exposes an OpenAI-compatible endpoint on
  localhost (default `:8080/v1`). Stage 2 = an HTTP POST to
  `/v1/chat/completions`. No cloud, no new download, no extra resident VRAM.
- **Uncensored (heretic) is a feature here**, not a liability: uk/ru war/news
  audio is often graphic or profane; a base model may refuse or sanitize,
  hurting *fidelity*. Faithful translation is exactly what we want.
- **`--temp 0`** → deterministic, consistent translations (good for
  translation; we do not want creative variance).
- **Native-MTP speculative decoding** (`--spec-type draft-mtp`) speeds
  generation → helps the latency budget.
- **256k context** is far more than needed; a rolling window of the previous
  few sentences costs almost nothing here.

Design details:
- Feed a **rolling window of the previous few translated sentences** as context
  so pronouns/terminology stay coherent across utterances — a big fluency win.
- System prompt: "Translate the Ukrainian or Russian input to natural English.
  Output only the translation, nothing else." Keep it faithful, not
  interpretive.
- **Fidelity spot-check:** abliterated finetunes can drift from the source.
  Validate uk/ru→en faithfulness on real clips at temp 0. Only if it
  disappoints, bake off against base **Qwen3 32B** or **Gemma 3 27B** — but
  start with what is already loaded.

**Segmentation glue:** VAD (e.g. Silero) buffers audio to natural sentence
boundaries before Stage 1, so the LLM always translates whole thoughts.

### Latency budget (sanity check)

VAD wait to end-of-utterance (~0.5–2 s) + large-v3 ASR (near real-time) + LLM
translate (~40 tok/s on this chip → ~1 s for a typical sentence) ≈ **a few
seconds behind live**. Acceptable under the locked priorities.

## Source: live Signal calls (resolved 2026-07-19)

Primary use case is **translating the remote party during Signal desktop
calls.** This settles capture and VAD design:

### Audio capture — tap Signal's playback, not the mic

The uk/ru voice you need is what Signal *plays back* to you; your own mic is
English and irrelevant. So capture the output side:

- **Simple (start here):** capture the default sink's `.monitor` source with
  ffmpeg/pw. Works immediately. Downside: also grabs music, notifications, any
  other system audio during the call.
- **Clean (recommended once working):** create a dedicated null sink, route
  *only* Signal's output stream into it (`pactl move-sink-input`, or `pw-link`
  on PipeWire), and capture that null sink's monitor. Isolates the call from
  other audio. On PipeWire, `pw-record --target <signal-output-node>` can grab
  Signal's node directly.
- Headset vs. speakers doesn't matter — we tap the stream before the speaker,
  so there's no mic echo/re-recording problem.

This is a first-party, on-device accessibility use (you're a call participant,
translation never leaves the machine).

### VAD / latency tuning for live conversation

- Live dialogue has natural turn/sentence pauses — segment on those. Tune
  Silero-VAD **moderately aggressive** so utterance-end is detected quickly;
  latency matters more here than for passive broadcast monitoring (you need to
  respond in the call).
- If lag feels too long, the ASR latency knob is **`large-v3-turbo`** instead of
  `large-v3` — transcription (not translate) quality on turbo is strong, and it
  shaves time. Keep `large-v3` as the fluency default, turbo as the fast option.
- Signal call audio can be noisy / packet-lossy; large-v3's robustness is a
  reason to prefer it over smaller models for Stage 1.

## Stage 1 — DONE (2026-07-19): whisper.cpp ROCm on gfx1151

Validated. large-v3 runs on the iGPU faster than real-time.

Environment: ROCm 7 (HIP 7.14), AMD clang 23, cmake 4.3, gcc 15.3.
Checkout: `~/github/whisper.cpp` (ggml 0.16.0, commit 080bbbe).

Build recipe:
```
cd ~/github/whisper.cpp
cmake -B build-rocm -S . \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build-rocm --config Release -j$(nproc)
sh ./models/download-ggml-model.sh large-v3   # models/ggml-large-v3.bin, 2.9GB
```

Verification (`whisper-cli` on samples/jfk.wav):
- GPU detected: `AMD Radeon 8060S Graphics, gfx1151`, backend `ROCm0`,
  122880 MiB VRAM visible (unified).
- 11 s clip → encode 385 ms, ~1.1 s inference (excl. 1.14 s one-time load)
  ≈ ~10× real-time. Ample headroom for live calls.

Binaries available in `build-rocm/bin/`:
- `whisper-cli` — one-shot / testing.
- `whisper-server` — OpenAI-compatible HTTP server; **use this for Stage 1**
  (model stays resident on the iGPU; orchestrator POSTs audio segments,
  mirroring the llama-server pattern).
- `whisper-vad-speech-segments`, `test-vad` — **native Silero VAD is built in**,
  so a separate Python VAD layer may be unnecessary.

Not built: `whisper-stream` (SDL2 mic-capture example) — not needed; our capture
is Signal's sink monitor, not a mic.

Still TODO for Stage 1: validate transcription *quality* on real uk + ru audio
(jfk was English only).

## Stage 2 — VALIDATED (2026-07-19): llama-server translation

End-to-end pipeline proven: uk audio → whisper large-v3 (iGPU) → llama-server
Qwen3 27B → fluent English.

**CRITICAL config finding:** this model (`Qwen3 27B uncensored-heretic`) is a
*reasoning* model — it emits chain-of-thought into `reasoning_content` and, if
`max_tokens` is hit mid-thought, `content` comes back **empty**. Thinking also
adds ~8 s of latency per utterance. Fixes tried:
- `/no_think` soft switch in the prompt → **ignored** by this finetune's template.
- `"chat_template_kwargs": {"enable_thinking": false}` in the request body →
  **works.** Clean translation in `content`, empty `reasoning_content`.

So every Stage-2 request MUST include:
```json
"temperature": 0,
"chat_template_kwargs": {"enable_thinking": false}
```

Measured: 3-sentence utterance → **1.8 s** total, 38 tokens @ ~33 tok/s
(MTP speculative decode, ~84% draft acceptance). With thinking on it was 10 s.

Quality (uk→en), verbatim output:
> "The Russians are using everything they have against Bakhmut and our other
> cities. The occupiers have a significant artillery advantage. They have far
> more missiles and aircraft than we ever did."

Fluent, natural, faithful — meets the fluency-first bar.

Stage 1 ASR quality (uk) also confirmed on real Ukrainian narration: accurate
text incl. specialized vocab (візантійської=Byzantine, знаменний спів=znamenny
chant, бароко=Baroque). Auto-detect returned `uk` at p=0.97.

**Russian validated too** (2026-07-19): auto-detect `ru` at p=0.9997, accurate
ASR, and Stage-2 translation in 2.8 s:
> "Russian is one of the East Slavic languages and the national language of the
> Russian people. It is one of the most widely spoken languages in the world,
> ranking sixth by total number of speakers and eighth by the number of native
> speakers. Russian is also the most widely spoken Slavic language and the most
> widely spoken language in Europe."

Both uk and ru now proven through the full two-stage pipeline. Test clips live
in `test-audio/` (Wikimedia Commons, freely licensed).

**Stages 1 + 2 are DONE. Remaining work is the streaming orchestration** (capture
+ VAD + glue), not the models.

## Live loop — WORKING + REAL-CALL VALIDATED (2026-07-19)

Confirmed working on a real Signal call by the user. Core project goal met:
offline real-time uk/ru→English of call audio on Strix Halo.


Full end-to-end streaming loop runs. Files:
- `translate_stream.py` — orchestrator: parec (sink monitor) → energy VAD →
  whisper-server (:8081) → llama-server (:8080, enable_thinking=false) →
  scrolling terminal, with a rolling context window.
- `run.sh` — brings up whisper-server on :8081 if needed, then runs the loop.
  (Stage 2 llama-server on :8080 is your existing daily driver.)

Deps: fresh `.venv` (Python 3.14) with numpy + requests. The OLD `.venv` was
broken — its interpreter still pointed at the pre-move `~/projects/dtu/...`
path; recreated.

VAD is energy-based (numpy only) — chosen because torch/onnxruntime lag on
Python 3.14. Tunable via CLI flags (`--silence-ms`, `--speech-mult`, etc.).

Verified test: played uk narration to the default sink; the monitor capture
produced live English, e.g. "chant, which was shaped under the influence of
Byzantine and Ukrainian folk music."

Known quality nits (tuning, not architecture):
- **Fragmentation:** VAD splits at mid-sentence pauses, so the LLM sometimes
  translates a fragment ("...influence on talant." / "The creativity of
  composers of the lower epoch."). Default `--silence-ms` raised 700→1000 to
  reduce this; further option is merging short adjacent segments.
- **uk/ru mislabel on short clips:** whisper auto-detect is unreliable on <~2 s
  audio (tagged some uk segments "russian"). Harmless to translation (LLM takes
  both); only the display tag is wrong. Could pin `--language uk` if a call is
  known-Ukrainian, or detect on longer buffers.
- One number error (XII–XVII → "19th century"): ASR nit on spoken numerals.

Ports: whisper-server MUST NOT use 8080 (llama-server lives there) — using 8081.

## Signal-only capture — DONE (2026-07-19), now the default

`translate_stream.py` now captures ONLY the Signal call by default (overrides:
`--monitor` for whole system, `--source X` for a specific source).

How it works:
- Signal plays call audio through a **`ringrtc`** node
  (`media.class=Stream/Output/Audio`, `application.process.binary=signal-desktop`).
- `find_signal_node()` scans `pw-dump` for it and reads its `object.serial`
  (the object *id* AND serial both drift every call — 138→134, 8974→9003 —
  so discovery MUST be dynamic each run).
- `pw-record --target <serial> --raw` taps it directly. **Non-destructive
  fan-out**: Signal keeps playing to the speakers untouched; we capture a copy.
  No null sink / no module-load needed (an earlier null-sink attempt was
  abandoned as unnecessary).

**IMPORTANT semantics:** `ringrtc output` = the audio Signal plays to you = the
**remote** party's voice (correct target — they speak uk/ru). Your OWN mic goes
to Signal's *input* node, which we do NOT tap. So to test, audio must come from
the OTHER end; talking into your own mic won't appear in the capture.

### Silence-hallucination fix (critical for real calls)

Real calls have pauses; Whisper hallucinates junk on silence/comfort-noise
(observed live: "(nynorsk) Thanks for the media text", "Great Britain", etc.).
Two-layer defense, both in place:
1. **whisper-server native Silero VAD** — `--vad --vad-model
   ggml-silero-v5.1.2.bin --vad-threshold 0.5`. Verified: pure silence and pink
   noise now return "" instead of hallucinated text. Real speech still passes.
   Baked into `run.sh`.
2. **Client filter** `is_meaningful()` — drops empty / punctuation-only /
   blank-audio markers as defense-in-depth.

## Recording — always on (2026-07-19)

Every run auto-records into `<yymmdd-HHMMSS>/`, created directly in the
directory the script is run from (no `recordings/` wrapper dir, no
`--record-dir` flag — user will never use a custom base dir, so it was
removed entirely): `session.wav` (full continuous capture), `utt_NNNN.wav`
per detected utterance (saved *before* the meaningful-text filter, so dropped
utterances stay inspectable), and `transcript.jsonl` (timestamp, detected
language, source text, English translation, kept/dropped flag). `--no-record`
disables; `--show-source` also prints source-language text to the terminal.
`run.sh` needed no change — it already forwards args transparently.

`.gitignore` matches these session dirs by the `yymmdd-HHMMSS/` name pattern
at repo root (glob: `[0-9]{6}-[0-9]{6}/`) so real call recordings never get
committed.

## Filter bug fixed (2026-07-19)

`is_meaningful()` originally required a run of 2+ letters, so numbers-only ASR
output (spoken numerals get rendered as digits, e.g. counting "1, 2, 3...") was
wrongly dropped as if it were silence-hallucination junk. Fixed: now accepts
any `\w` (letters or digits); still correctly rejects empty/punctuation-only/
blank-audio markers.

## Build order

1. Stand up whisper.cpp with ROCm for gfx1151; confirm large-v3 transcribes
   uk + ru faster than real-time on the iGPU.
2. Point Stage 2 at the already-running llama-server (`localhost:8080/v1`);
   fidelity spot-check the loaded Qwen3 27B on captured clips at temp 0.
3. Add Silero-VAD segmentation to replace fixed ffmpeg chunks.
4. Wire the streaming loop: audio → VAD → ASR → LLM (with rolling context) →
   terminal output.
5. Delete the OpenAI backends and the `language="uk"` hardcoding from the old
   script; keep faster-whisper only if it earns its place vs. whisper.cpp-ROCm.

Note on GPU coexistence: whisper large-v3 (~few GB) + the resident Qwen3 27B
Q4 + its KV cache all share the 128 GB unified pool — no VRAM contention. ASR is
bursty and the LLM is idle between utterances, so they interleave fine on the
one iGPU.

## Sources

- whisper.cpp 1.8.0 on Strix Halo w/ ROCm 7 (Discussion #3460):
  https://github.com/ggml-org/whisper.cpp/discussions/3460
- faster-whisper-rocm-strix-halo recipe (gfx1151):
  https://github.com/nabe2030/faster-whisper-rocm-strix-halo
- lemonade-sdk/whisper.cpp-rocm:
  https://github.com/lemonade-sdk/whisper.cpp-rocm
- Strix Halo ROCm working guide — 40 tok/s on 30B (ollama #14855):
  https://github.com/ollama/ollama/issues/14855
- llm-tracker.info — Strix Halo:
  https://llm-tracker.info/_TOORG/Strix-Halo
