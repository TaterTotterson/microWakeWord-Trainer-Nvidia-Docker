#!/usr/bin/env python3
"""Build a reusable multi-phrase TTS corpus for MicroWakeWord hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path


MANIFEST_NAME = ".generation_manifest.json"
MAX_TOTAL_SAMPLES = 2048
MAX_SAMPLES_PER_PHRASE = 256
MIN_SAMPLES_PER_PHRASE = 32


def normalize_phrase(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def phrase_slug(phrase: str) -> str:
    ascii_phrase = unicodedata.normalize("NFKD", phrase).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^a-z0-9]+", "_", ascii_phrase.casefold()).strip("_") or "phrase"
    digest = hashlib.sha256(phrase.casefold().encode("utf-8")).hexdigest()[:8]
    return f"{stem[:48]}_{digest}"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_phrases(raw_json: str, target_phrase: str) -> list[str]:
    try:
        payload = json.loads(raw_json or "[]")
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid generated-negative phrase JSON: {error}") from error
    if not isinstance(payload, list):
        raise SystemExit("Generated-negative phrase JSON must be a list")
    target_key = normalize_phrase(target_phrase).casefold()
    phrases: dict[str, str] = {}
    for raw_value in payload:
        phrase = normalize_phrase(raw_value)
        key = phrase.casefold()
        if phrase and key != target_key:
            phrases.setdefault(key, phrase)
    return sorted(phrases.values(), key=str.casefold)


def clear_managed_outputs(root: Path) -> None:
    for child in (root / "by_phrase", root / "combined"):
        shutil.rmtree(child, ignore_errors=True)
    (root / MANIFEST_NAME).unlink(missing_ok=True)


def cache_is_ready(root: Path, signature: dict[str, object], expected_count: int) -> bool:
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    combined = root / "combined"
    return manifest.get("signature") == signature and len(list(combined.glob("*.wav"))) == expected_count


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--phrases-json", default="[]")
    result.add_argument("--target-phrase", required=True)
    result.add_argument("--root-dir", type=Path, required=True)
    result.add_argument("--generator-script", type=Path, required=True)
    result.add_argument("--data-dir", type=Path, required=True)
    result.add_argument("--language", default="en")
    result.add_argument("--english-accent", default="mixed")
    result.add_argument("--tts-mode", default="modern")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--voice-count", type=int, default=128)
    result.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    return result


def main() -> int:
    args = parser().parse_args()
    phrases = parse_phrases(args.phrases_json, args.target_phrase)
    root = args.root_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not phrases:
        clear_managed_outputs(root)
        print("ℹ️ No generated hard-negative phrases selected for MicroWakeWord.", flush=True)
        return 0

    samples_per_phrase = max(
        MIN_SAMPLES_PER_PHRASE,
        min(MAX_SAMPLES_PER_PHRASE, MAX_TOTAL_SAMPLES // len(phrases)),
    )
    expected_count = samples_per_phrase * len(phrases)
    generator_script = args.generator_script.resolve()
    signature: dict[str, object] = {
        "schema": 1,
        "phrases": phrases,
        "target_phrase": normalize_phrase(args.target_phrase),
        "samples_per_phrase": samples_per_phrase,
        "language": args.language,
        "english_accent": args.english_accent,
        "tts_mode": args.tts_mode,
        "batch_size": args.batch_size,
        "voice_count": args.voice_count,
        "generator_sha256": file_digest(generator_script),
    }
    if cache_is_ready(root, signature, expected_count):
        print(
            f"✅ Reusing {expected_count} generated MicroWakeWord hard-negative samples "
            f"for {len(phrases)} phrase(s).",
            flush=True,
        )
        return 0

    by_phrase = root / "by_phrase"
    by_phrase.mkdir(parents=True, exist_ok=True)
    selected_slugs = {phrase_slug(phrase) for phrase in phrases}
    for child in by_phrase.iterdir():
        if child.is_dir() and child.name not in selected_slugs:
            shutil.rmtree(child)

    phrase_files: list[tuple[str, list[Path]]] = []
    for phrase in phrases:
        slug = phrase_slug(phrase)
        output_dir = by_phrase / slug
        command = [
            sys.executable,
            str(generator_script),
            phrase,
            "--language", args.language,
            "--english-accent", args.english_accent,
            "--tts-mode", args.tts_mode,
            "--samples", str(samples_per_phrase),
            "--batch-size", str(args.batch_size),
            "--voice-count", str(args.voice_count),
            "--data-dir", str(args.data_dir.resolve()),
            "--output-dir", str(output_dir),
            "--ffmpeg", args.ffmpeg,
        ]
        print(f"🎭 Generating hard-negative TTS for: {phrase}", flush=True)
        subprocess.run(command, check=True)
        files = sorted(output_dir.glob("*.wav"))
        if len(files) != samples_per_phrase:
            raise RuntimeError(
                f"Expected {samples_per_phrase} hard-negative samples for {phrase!r}, found {len(files)}"
            )
        phrase_files.append((slug, files))

    build_dir = root / ".combined.build"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)
    for slug, files in phrase_files:
        for index, source in enumerate(files):
            link_or_copy(source, build_dir / f"{slug}_{index:04d}.wav")
    combined = root / "combined"
    old_combined = root / ".combined.previous"
    shutil.rmtree(old_combined, ignore_errors=True)
    if combined.exists():
        combined.rename(old_combined)
    build_dir.rename(combined)
    shutil.rmtree(old_combined, ignore_errors=True)

    manifest = {"signature": signature, "sample_count": expected_count}
    temporary_manifest = root / f"{MANIFEST_NAME}.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_manifest.replace(root / MANIFEST_NAME)
    print(
        f"✅ Prepared {expected_count} generated MicroWakeWord hard-negative samples "
        f"across {len(phrases)} phrase(s).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
