#!/usr/bin/env python3
"""Build deterministic wake+speech-tail hard negatives for endpoint training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MIN_TAIL_SAMPLES = SAMPLE_RATE // 4
MAX_TAIL_SAMPLES = SAMPLE_RATE


def read_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        pcm = source.readframes(source.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got {width * 8}-bit")
    values = np.frombuffer(pcm, dtype="<i2")
    if channels > 1:
        values = values.reshape(-1, channels).astype(np.float32).mean(axis=1)
    elif channels != 1:
        raise ValueError(f"{path}: WAV has no audio channels")
    if rate != SAMPLE_RATE:
        source_x = np.arange(len(values), dtype=np.float64)
        target_count = max(1, round(len(values) * SAMPLE_RATE / rate))
        target_x = np.linspace(0, max(0, len(values) - 1), target_count)
        values = np.interp(target_x, source_x, values)
    return np.clip(values, -32768, 32767).astype("<i2").tobytes()


def trim_silence(pcm: bytes) -> bytes:
    if not pcm:
        return b""
    samples = memoryview(pcm).cast("h")
    peak = max((abs(int(value)) for value in samples), default=0)
    threshold = max(192, int(peak * 0.025))
    active = [index for index, value in enumerate(samples) if abs(int(value)) >= threshold]
    if not active:
        return b""
    pad = SAMPLE_RATE // 200  # 5 ms preserves final consonants without a long gap.
    start = max(0, active[0] - pad)
    end = min(len(samples), active[-1] + 1 + pad)
    return bytes(samples[start:end])


def write_pcm16(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm)


def build(positive_dirs: list[Path], output_dir: Path, variants: int) -> dict[str, object]:
    sources = sorted({
        path.resolve()
        for directory in positive_dirs
        if directory.is_dir()
        for path in directory.glob("*.wav")
    })
    if len(sources) < 2:
        raise RuntimeError("wake boundary training needs at least two positive WAV files")

    clips: list[tuple[Path, bytes, str]] = []
    for source in sources:
        try:
            pcm = trim_silence(read_pcm16(source))
        except (OSError, ValueError, wave.Error):
            continue
        if len(pcm) >= SAMPLE_RATE // 5 * 2:
            clips.append((source, pcm, hashlib.sha256(pcm).hexdigest()))
    if len(clips) < 2:
        raise RuntimeError("wake boundary training could not read two usable positive WAV files")

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    for wake_index, (wake_path, wake_pcm, wake_sha256) in enumerate(clips):
        for variant in range(max(1, variants)):
            tail_path, tail_pcm, tail_sha256 = clips[(wake_index + 1 + variant * 17) % len(clips)]
            tail_samples = len(tail_pcm) // 2
            wanted = min(MAX_TAIL_SAMPLES, max(MIN_TAIL_SAMPLES, tail_samples // 2))
            rng = random.Random((wake_index + 1) * 1009 + variant * 9176)
            start_max = max(0, tail_samples - wanted)
            start = rng.randint(0, start_max) if start_max else 0
            tail = tail_pcm[start * 2 : (start + wanted) * 2]
            combined = wake_pcm + tail
            name = f"boundary-negative-{wake_index:06d}-{variant}.wav"
            write_pcm16(temporary / name, combined)
            rows.append({
                "file": name,
                "wake_source": wake_path.name,
                "tail_source": tail_path.name,
                "wake_samples": len(wake_pcm) // 2,
                "tail_samples": len(tail) // 2,
                "wake_sha256": wake_sha256,
                "tail_sha256": tail_sha256,
            })

    identity = hashlib.sha256(
        "\n".join(
            f"{row['wake_source']}:{row['tail_source']}:{row['wake_samples']}:"
            f"{row['tail_samples']}:{row['wake_sha256']}:{row['tail_sha256']}"
            for row in rows
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "strategy": "endpoint_hard_negative_v1",
        "sample_rate": SAMPLE_RATE,
        "count": len(rows),
        "identity": identity,
        "samples": rows,
    }
    (temporary / ".generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    existing_manifest = output_dir / ".generation_manifest.json"
    if existing_manifest.is_file():
        try:
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("identity") == identity:
            shutil.rmtree(temporary)
            return existing
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary.replace(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", type=int, default=2)
    args = parser.parse_args()
    manifest = build(
        [Path(value) for value in args.positive_dir],
        Path(args.output_dir),
        max(1, min(4, args.variants)),
    )
    print(f"✅ Built {manifest['count']} endpoint hard negatives in {args.output_dir}")


if __name__ == "__main__":
    main()
