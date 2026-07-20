#!/usr/bin/env bash
# Launch the real-time uk/ru -> English call translator.
#
# Stage 2 (llama-server, Qwen3 27B) is assumed already running on :8080
# (your usual daily-driver command). This script only brings up Stage 1
# (whisper-server on :8081) if it isn't already up, then runs the orchestrator.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WHISPER_BIN="${WHISPER_BIN:-$HOME/github/whisper.cpp/build-rocm/bin/whisper-server}"
WHISPER_MODEL="${WHISPER_MODEL:-$HOME/github/whisper.cpp/models/ggml-large-v3.bin}"
WHISPER_VAD_MODEL="${WHISPER_VAD_MODEL:-$HOME/github/whisper.cpp/models/ggml-silero-v5.1.2.bin}"
WHISPER_PORT="${WHISPER_PORT:-8081}"

if ! curl -sf "http://127.0.0.1:${WHISPER_PORT}/" >/dev/null 2>&1; then
  echo "Starting whisper-server on :${WHISPER_PORT} (large-v3 + Silero VAD, gfx1151)..."
  "$WHISPER_BIN" \
    -m "$WHISPER_MODEL" \
    --host 127.0.0.1 --port "$WHISPER_PORT" \
    -l auto -t 4 \
    --vad --vad-model "$WHISPER_VAD_MODEL" --vad-threshold 0.5 \
    >"$HERE/whisper-server.log" 2>&1 &
  echo "  logging to whisper-server.log; waiting for load..."
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${WHISPER_PORT}/" >/dev/null 2>&1 && break
    sleep 1
  done
fi

exec "$HERE/.venv/bin/python" "$HERE/translate_stream.py" \
  --whisper-url "http://127.0.0.1:${WHISPER_PORT}" "$@"
