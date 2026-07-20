# Realtime Ukrainian -> English Translation

This directory contains a small real-time translation harness for Linux desktop audio using PulseAudio or PipeWire.

It captures audio in short chunks with `ffmpeg`, translates each chunk, and prints incremental English output to the terminal.

## What You Can Compare

You said you are willing to test multiple options. This script supports three useful paths:

1. `faster-whisper`
   Local translation with Whisper-compatible models.
2. `openai-translate`
   OpenAI audio translation in one step.
3. `openai-transcribe-translate`
   OpenAI transcription first, then a text model translates to English.

## Requirements

- Linux with PulseAudio or PipeWire Pulse compatibility
- `ffmpeg`
- Python 3.10+
- For OpenAI backends: `OPENAI_API_KEY`

## Setup

```bash
cd /home/dcar/projects/dtu/realtime-translate
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Find audio sources:

```bash
pactl list short sources
```

If you want a persistent default source, export it:

```bash
export REALTIME_TRANSLATE_SOURCE="alsa_output.pci-0000_c6_00.6.analog-stereo.monitor"
```

Or use your microphone source instead:

```bash
export REALTIME_TRANSLATE_SOURCE="alsa_input.usb-EMEET_HD_Webcam_eMeet_C950_A230803002402311-02.analog-stereo"
```

## Recommended Test Matrix

### 1. Fast local baseline

```bash
python3 realtime_translate.py \
  --backend faster-whisper \
  --model small \
  --source "$REALTIME_TRANSLATE_SOURCE"
```

### 2. Better local quality

```bash
python3 realtime_translate.py \
  --backend faster-whisper \
  --model medium \
  --source "$REALTIME_TRANSLATE_SOURCE"
```

### 3. Highest local quality to try

```bash
python3 realtime_translate.py \
  --backend faster-whisper \
  --model large-v3 \
  --compute-type float16 \
  --source "$REALTIME_TRANSLATE_SOURCE"
```

### 4. OpenAI one-step translation

```bash
export OPENAI_API_KEY="your-key-here"
python3 realtime_translate.py \
  --backend openai-translate \
  --model whisper-1 \
  --source "$REALTIME_TRANSLATE_SOURCE"
```

### 5. OpenAI transcription model plus text translation

```bash
export OPENAI_API_KEY="your-key-here"
python3 realtime_translate.py \
  --backend openai-transcribe-translate \
  --model gpt-4o-mini-transcribe \
  --text-model gpt-4.1-mini \
  --source "$REALTIME_TRANSLATE_SOURCE"
```

You can also swap `gpt-4o-mini-transcribe` for `gpt-4o-transcribe`.

## Notes

- Shorter `--chunk-seconds` reduces latency but may lower translation quality.
- Longer chunks improve context but add delay.
- Good starting values are `3.0` to `5.0` seconds.
- System audio monitors are useful if you want to translate the remote speaker in a call.
- Microphone sources are useful if you want to test live spoken Ukrainian directly.

## Example With Lower Latency

```bash
python3 realtime_translate.py \
  --backend faster-whisper \
  --model small \
  --chunk-seconds 3 \
  --source "$REALTIME_TRANSLATE_SOURCE"
```

## Troubleshooting

If `ffmpeg` cannot open the source, verify the exact source name with:

```bash
pactl list short sources
```

If local models are too slow:

- Try `--model small`
- Keep `--compute-type int8`
- Increase `--chunk-seconds` to `5`

If OpenAI is too expensive or too slow:

- Use `faster-whisper small` as the baseline
- Compare it against `medium` before trying `large-v3`

## Qwen3-Omni Local Setup

This repo also includes a first-pass local test script for `Qwen3-Omni-30B-A3B-Instruct`.

Environment:

```bash
~/.pyenv/versions/3.10.6/bin/python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.26.4
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
python3 -m pip install git+https://github.com/huggingface/transformers
python3 -m pip install accelerate qwen-omni-utils soundfile
```

Model download target:

```bash
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --local-dir /home/dcar/llms/qwen3/3.5/Qwen3-Omni-30B-A3B-Instruct
```

Translate a local audio file:

```bash
source .venv-qwen/bin/activate
python3 qwen3_omni_translate.py /path/to/audio.wav
```

Force CPU fallback while debugging model-load issues:

```bash
source .venv-qwen/bin/activate
python3 qwen3_omni_translate.py /path/to/audio.wav --device-map cpu
```

Check whether the environment sees the AMD device:

```bash
source .venv-qwen/bin/activate
python3 qwen_env_check.py
```

Notes:

- This script uses the official Hugging Face `transformers` path for Qwen3-Omni.
- Qwen3-Omni currently requires `transformers` installed from GitHub source, not the older PyPI release.
- It tests local file-based audio translation first, not live microphone streaming.
- For this machine, use a ROCm-compatible PyTorch build in `.venv-qwen`; the generic CUDA wheel path will not expose the AMD GPU.
- On this AMD system, ROCm PyTorch is visible and working, but `Qwen3-Omni-30B-A3B-Instruct` currently segfaults during GPU model load with `device_map auto`. CPU fallback loads successfully and is useful for debugging, but will not be real-time.
