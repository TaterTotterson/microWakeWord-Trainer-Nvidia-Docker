#!/usr/bin/env python3
"""Publish a boundary-only model into an existing wake-word bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wake_bundle import load_bundle, publish_wake_boundary  # noqa: E402


def _threshold(path: Path) -> tuple[float, dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.80, {}
    threshold = float(value.get("probability_cutoff", 0.80))
    return (threshold if 0.0 < threshold <= 1.0 else 0.80), value


def publish(*, phrase: str, safe_word: str, output_dir: Path, model: Path, calibration: Path) -> Path:
    bundle = load_bundle(output_dir, safe_word)
    if not bundle or not any(isinstance(bundle.get(key), dict) for key in ("micro_wake_word", "open_wake_word", "nano_wake_word")):
        raise RuntimeError(
            f"Boundary-only training needs an existing detector bundle: {output_dir / f'{safe_word}.wake-bundle.json'}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{safe_word}.boundary.tflite"
    shutil.copy2(model, destination)
    threshold, report = _threshold(calibration)
    return publish_wake_boundary(
        trained_dir=output_dir,
        safe_word=safe_word,
        phrase=str(bundle.get("wake_word") or phrase),
        model_path=destination,
        metadata={
            "threshold": threshold,
            "tensor_arena_size": 64 * 1024,
            "training_strategy": "endpoint_hard_negative_v1",
            "quality_summary": report.get("selected_metrics", {}),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phrase")
    parser.add_argument("--safe-word", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    path = publish(
        phrase=args.phrase,
        safe_word=args.safe_word,
        output_dir=args.output_dir.resolve(),
        model=args.model.resolve(),
        calibration=args.calibration.resolve(),
    )
    print(f"✅ Updated wake boundary in {path}")


if __name__ == "__main__":
    main()
