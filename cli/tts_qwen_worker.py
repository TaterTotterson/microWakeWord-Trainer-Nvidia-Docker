#!/usr/bin/env python3
"""Persistent-process Qwen3-TTS worker used by the sample orchestrator."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def read_jsonl(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def chunks(values: list, size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def runtime() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda:0", dtype
    return "cpu", torch.float32


def load_model(model_id: str) -> Qwen3TTSModel:
    device, dtype = runtime()
    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )


def build_bank(entries: list[dict], output_dir: Path, batch_size: int) -> None:
    model = load_model(VOICE_DESIGN_MODEL)
    output_dir.mkdir(parents=True, exist_ok=True)
    for batch in chunks(entries, max(1, batch_size)):
        seed = int(batch[0].get("seed", 0))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        wavs, sample_rate = model.generate_voice_design(
            text=[str(item["text"]) for item in batch],
            language=[str(item["language_name"]) for item in batch],
            instruct=[str(item["instruct"]) for item in batch],
        )
        for item, wav in zip(batch, wavs):
            sf.write(output_dir / f"{item['id']}.wav", wav, sample_rate)


def generate_direct(entries: list[dict], output_dir: Path, batch_size: int) -> None:
    """Create every final corpus candidate with a fresh voice design."""

    model = load_model(VOICE_DESIGN_MODEL)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for batch in chunks(entries, max(1, batch_size)):
        seed = int(batch[0].get("seed", completed + 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        wavs, sample_rate = model.generate_voice_design(
            text=[str(item["text"]) for item in batch],
            language=[str(item["language_name"]) for item in batch],
            instruct=[str(item["instruct"]) for item in batch],
            # Qwen emits 12 acoustic frames per second. Four seconds is a hard
            # wake-phrase ceiling and prevents decoder rambling.
            max_new_tokens=48,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.12,
        )
        for item, wav in zip(batch, wavs):
            sf.write(output_dir / f"{item['id']}.wav", wav, sample_rate)
            completed += 1
        if completed % 25 == 0 or completed == len(entries):
            print(f"Qwen direct generation created {completed}/{len(entries)}", flush=True)


def generate(entries: list[dict], output_dir: Path, batch_size: int) -> None:
    model = load_model(VOICE_CLONE_MODEL)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in entries:
        key = (
            str(item["ref_audio"]),
            str(item["ref_text"]),
            str(item["language_name"]),
        )
        grouped[key].append(item)

    for (ref_audio, ref_text, language_name), group in grouped.items():
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        for batch in chunks(group, max(1, batch_size)):
            seed = int(batch[0].get("seed", 0))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            wavs, sample_rate = model.generate_voice_clone(
                text=[str(item["text"]) for item in batch],
                language=[language_name] * len(batch),
                voice_clone_prompt=prompt,
            )
            for item, wav in zip(batch, wavs):
                sf.write(output_dir / f"{item['id']}.wav", wav, sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bank", "direct", "generate"), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    entries = read_jsonl(args.input_jsonl)
    if not entries:
        return 0
    if args.mode == "bank":
        build_bank(entries, args.output_dir, args.batch_size)
    elif args.mode == "direct":
        generate_direct(entries, args.output_dir, args.batch_size)
    else:
        generate(entries, args.output_dir, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
