#!/usr/bin/env python3
"""Persistent-process MOSS-TTS-Nano voice-cloning worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from moss_tts_nano.defaults import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
)


MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"


def read_jsonl(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument(
        "--audio-tokenizer",
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
    )
    args = parser.parse_args()

    entries = read_jsonl(args.input_jsonl)
    if not entries:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa")
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(entries, start=1):
        seed = int(item.get("seed", index))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        output_path = args.output_dir / f"{item['id']}.wav"
        model.inference(
            text=str(item["text"]),
            output_audio_path=str(output_path),
            mode="voice_clone",
            prompt_audio_path=str(item["ref_audio"]),
            reference_audio_path=None,
            text_tokenizer_path=None,
            audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
            audio_tokenizer_pretrained_name_or_path=args.audio_tokenizer,
            device=device,
            nq=None,
            max_new_frames=64,
            voice_clone_max_text_tokens=32,
            voice_clone_max_memory_per_sample_gb=1.0,
            do_sample=True,
            use_kv_cache=True,
            text_temperature=1.0,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=0.7,
            audio_top_p=0.9,
            audio_top_k=25,
            audio_repetition_penalty=1.3,
        )
        if index % 10 == 0 or index == len(entries):
            print(f"MOSS generated {index}/{len(entries)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
