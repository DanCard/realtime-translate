#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


DEFAULT_MODEL_PATH = "/home/dcar/llms/qwen3/3.5/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_PROMPT = (
    "Translate the spoken Ukrainian audio to concise natural English text. "
    "Return only the English translation."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Qwen3-Omni audio-to-English translation on a WAV or audio file."
    )
    parser.add_argument("audio", help="Path to a local audio file.")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local path to the Qwen3-Omni model directory.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instruction prompt passed alongside the audio.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation, for example `sdpa` or `flash_attention_2`.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device map, for example `auto` or `cpu`.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype, for example `auto`, `bfloat16`, or `float16`.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise SystemExit(f"Model path not found: {model_path}")

    load_started = time.time()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    model.disable_talker()
    print(
        f"model_loaded_seconds={time.time() - load_started:.1f}",
        flush=True,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path))

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=False,
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    generated, _audio = model.generate(
        **inputs,
        return_audio=False,
        thinker_return_dict_in_generate=True,
        use_audio_in_video=False,
        max_new_tokens=args.max_new_tokens,
    )
    output = processor.batch_decode(
        generated.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output[0].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
