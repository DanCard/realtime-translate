#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TranslationBackend(Protocol):
    def translate_file(self, audio_path: Path) -> str: ...


@dataclass
class FasterWhisperBackend:
    model_name: str
    device: str
    compute_type: str
    beam_size: int

    def __post_init__(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: faster-whisper. Install with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def translate_file(self, audio_path: Path) -> str:
        segments, _info = self._model.transcribe(
            str(audio_path),
            language="uk",
            task="translate",
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


@dataclass
class OpenAITranslateBackend:
    model_name: str

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: openai. Install with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        self._client = OpenAI()

    def translate_file(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            result = self._client.audio.translations.create(
                model=self.model_name,
                file=audio_file,
                prompt=(
                    "Translate spoken Ukrainian into concise natural English. "
                    "Do not explain the translation."
                ),
            )

        if isinstance(result, str):
            return result.strip()
        return getattr(result, "text", str(result)).strip()


@dataclass
class OpenAITranscribeThenTranslateBackend:
    transcription_model: str
    text_model: str

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: openai. Install with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        self._client = OpenAI()

    def translate_file(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            transcription = self._client.audio.transcriptions.create(
                model=self.transcription_model,
                file=audio_file,
                language="uk",
                prompt="The audio is Ukrainian.",
            )

        transcript_text = self._extract_text(transcription)
        if not transcript_text:
            return ""

        response = self._client.responses.create(
            model=self.text_model,
            input=(
                "Translate the following Ukrainian speech to English. "
                "Return only the translation.\n\n"
                f"{transcript_text}"
            ),
        )
        return self._extract_text(response)

    @staticmethod
    def _extract_text(result: object) -> str:
        if isinstance(result, str):
            return result.strip()
        output_text = getattr(result, "output_text", None)
        if isinstance(output_text, str):
            return output_text.strip()
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text.strip()
        return str(result).strip()


class ChunkCapture:
    def __init__(
        self,
        source: str,
        sample_rate: int,
        channels: int,
        chunk_seconds: float,
        chunk_dir: Path,
    ) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        self.chunk_dir = chunk_dir
        self.process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(self.chunk_dir / "chunk_%06d.wav")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "pulse",
            "-i",
            self.source,
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-f",
            "segment",
            "-segment_time",
            str(self.chunk_seconds),
            "-reset_timestamps",
            "1",
            pattern,
        ]
        self.process = subprocess.Popen(command)

    def stop(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def build_backend(args: argparse.Namespace) -> TranslationBackend:
    if args.backend == "faster-whisper":
        return FasterWhisperBackend(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
        )
    if args.backend == "openai-translate":
        return OpenAITranslateBackend(model_name=args.model)
    if args.backend == "openai-transcribe-translate":
        return OpenAITranscribeThenTranslateBackend(
            transcription_model=args.model,
            text_model=args.text_model,
        )
    raise SystemExit(f"Unsupported backend: {args.backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture audio from PulseAudio/PipeWire and translate Ukrainian speech to English in near real time.",
    )
    parser.add_argument(
        "--backend",
        choices=[
            "faster-whisper",
            "openai-translate",
            "openai-transcribe-translate",
        ],
        default="faster-whisper",
        help="Translation backend to use.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help=(
            "Model name for the selected backend. Examples: "
            "`small`, `medium`, `large-v3`, `whisper-1`, `gpt-4o-mini-transcribe`."
        ),
    )
    parser.add_argument(
        "--text-model",
        default="gpt-4.1-mini",
        help="Text model used only by openai-transcribe-translate.",
    )
    parser.add_argument(
        "--source",
        default=os.environ.get("REALTIME_TRANSLATE_SOURCE", "default"),
        help="PulseAudio source name. Use `pactl list short sources` to inspect available sources.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Capture sample rate.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of audio channels to capture.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=4.0,
        help="Length of each audio chunk in seconds.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="How often to poll for newly finished chunks.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device for faster-whisper, for example `auto`, `cpu`, or `cuda`.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type, for example `int8`, `float16`, or `int8_float16`.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam size for faster-whisper translation.",
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep recorded chunk wav files instead of deleting them after translation.",
    )
    return parser


def should_process(path: Path, processed: set[Path], current_time: float) -> bool:
    if path in processed:
        return False
    if not path.is_file():
        return False
    age_seconds = current_time - path.stat().st_mtime
    return age_seconds >= 0.5


def looks_like_duplicate(text: str, last_text: str) -> bool:
    normalized = " ".join(text.lower().split())
    last_normalized = " ".join(last_text.lower().split())
    return bool(normalized) and normalized == last_normalized


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    backend = build_backend(args)

    with tempfile.TemporaryDirectory(prefix="realtime-translate-") as temp_dir:
        chunk_dir = Path(temp_dir)
        capture = ChunkCapture(
            source=args.source,
            sample_rate=args.sample_rate,
            channels=args.channels,
            chunk_seconds=args.chunk_seconds,
            chunk_dir=chunk_dir,
        )

        processed: set[Path] = set()
        last_text = ""

        print(f"backend={args.backend} model={args.model}")
        if args.backend == "openai-transcribe-translate":
            print(f"text_model={args.text_model}")
        print(f"source={args.source} chunk_seconds={args.chunk_seconds}")
        print("Press Ctrl+C to stop.\n")

        try:
            capture.start()
            while True:
                now = time.time()
                chunk_paths = sorted(chunk_dir.glob("chunk_*.wav"))
                for chunk_path in chunk_paths:
                    if not should_process(chunk_path, processed, now):
                        continue
                    processed.add(chunk_path)
                    translated = backend.translate_file(chunk_path).strip()
                    if translated and not looks_like_duplicate(translated, last_text):
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] {translated}", flush=True)
                        last_text = translated
                    if not args.keep_chunks:
                        chunk_path.unlink(missing_ok=True)
                time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            capture.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
