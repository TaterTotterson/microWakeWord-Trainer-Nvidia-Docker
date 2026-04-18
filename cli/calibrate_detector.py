#!/usr/bin/env python3
"""Choose detector metadata that better matches the trained model."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from microwakeword.data import FeatureHandler
from microwakeword.inference import Model


DEFAULT_WINDOW_SIZES = [3, 4, 5, 6, 7]
DEFAULT_TARGET_FAPH = float(os.environ.get("MWW_CALIBRATION_TARGET_FAPH", "1.0"))
DEFAULT_COOLDOWN_SLICES = int(os.environ.get("MWW_CALIBRATION_COOLDOWN_SLICES", "25"))
DEFAULT_POSITIVE_SKIP_SLICES = int(
    os.environ.get("MWW_CALIBRATION_POSITIVE_SKIP_SLICES", "25")
)
DEFAULT_CUTOFF_STEP = float(os.environ.get("MWW_CALIBRATION_CUTOFF_STEP", "0.01"))
DEFAULT_CUTOFF_MIN = float(os.environ.get("MWW_CALIBRATION_CUTOFF_MIN", "0.00"))
DEFAULT_CUTOFF_MAX = float(os.environ.get("MWW_CALIBRATION_CUTOFF_MAX", "1.00"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate microWakeWord detector metadata from validation data."
    )
    parser.add_argument(
        "--training-config",
        default="trained_models/wakeword/training_config.yaml",
        help="Path to the saved microWakeWord training_config.yaml file.",
    )
    parser.add_argument(
        "--model",
        default=(
            "trained_models/wakeword/tflite_stream_state_internal_quant/"
            "stream_state_internal_quant.tflite"
        ),
        help="Path to the quantized streaming TFLite model.",
    )
    parser.add_argument(
        "--output",
        default=(
            "trained_models/wakeword/tflite_stream_state_internal_quant/"
            "detection_calibration.json"
        ),
        help="Where to write the selected detector settings as JSON.",
    )
    parser.add_argument(
        "--window-sizes",
        default=",".join(str(value) for value in DEFAULT_WINDOW_SIZES),
        help="Comma-separated sliding window sizes to evaluate.",
    )
    parser.add_argument(
        "--target-faph",
        type=float,
        default=DEFAULT_TARGET_FAPH,
        help="Target ambient false accepts per hour for the selected operating point.",
    )
    parser.add_argument(
        "--cooldown-slices",
        type=int,
        default=DEFAULT_COOLDOWN_SLICES,
        help="Cooldown slices to use when estimating false accepts per hour.",
    )
    parser.add_argument(
        "--positive-skip-slices",
        type=int,
        default=DEFAULT_POSITIVE_SKIP_SLICES,
        help="Initial streaming slices to ignore when scoring positive examples.",
    )
    parser.add_argument(
        "--cutoff-step",
        type=float,
        default=DEFAULT_CUTOFF_STEP,
        help="Cutoff increment to evaluate between cutoff-min and cutoff-max.",
    )
    parser.add_argument(
        "--cutoff-min",
        type=float,
        default=DEFAULT_CUTOFF_MIN,
        help="Minimum cutoff to evaluate.",
    )
    parser.add_argument(
        "--cutoff-max",
        type=float,
        default=DEFAULT_CUTOFF_MAX,
        help="Maximum cutoff to evaluate.",
    )
    return parser.parse_args()


def _parse_window_sizes(raw: str) -> list[int]:
    values = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError("window sizes must be >= 1")
        values.append(value)
    if not values:
        raise ValueError("at least one window size is required")
    return sorted(set(values))


def _moving_average(values: Sequence[float], window_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    if window_size <= 1:
        return array
    if array.size < window_size:
        return np.asarray([float(array.mean())], dtype=np.float32)
    cumsum = np.cumsum(np.insert(array, 0, 0.0))
    averaged = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return averaged.astype(np.float32)


def _compute_false_accepts_per_hour(
    probabilities_per_track: Iterable[np.ndarray],
    cutoffs: np.ndarray,
    cooldown_slices: int,
    stride: int,
    step_seconds: float,
) -> tuple[np.ndarray, float]:
    cutoffs = np.asarray(cutoffs, dtype=np.float32)
    false_accepts = np.zeros(cutoffs.shape[0], dtype=np.float64)
    duration_hours = 0.0

    for track_probabilities in probabilities_per_track:
        if track_probabilities.size == 0:
            continue
        duration_hours += (
            len(track_probabilities) * stride * step_seconds / 3600.0
        )
        cooldown = np.full(cutoffs.shape[0], cooldown_slices, dtype=np.int32)
        for probability in track_probabilities:
            cooldown = np.maximum(cooldown - 1, 0)
            accepted = (cooldown == 0) & (probability > cutoffs)
            false_accepts += accepted.astype(np.float64)
            cooldown = np.where(accepted, cooldown_slices, cooldown)

    if duration_hours <= 0:
        return np.full(cutoffs.shape[0], math.inf, dtype=np.float64), 0.0

    return false_accepts / duration_hours, duration_hours


def _select_best_candidate(
    candidates: list[dict[str, float]],
    target_faph: float,
) -> tuple[dict[str, float], float]:
    fallback_limits = [
        target_faph,
        max(target_faph * 2.0, target_faph + 0.5),
        max(target_faph * 4.0, 2.0),
    ]

    def tier(candidate: dict[str, float]) -> int:
        for index, limit in enumerate(fallback_limits):
            if candidate["false_accepts_per_hour"] <= limit + 1e-9:
                return index
        return len(fallback_limits)

    best = min(
        candidates,
        key=lambda candidate: (
            tier(candidate),
            -candidate["recall"],
            candidate["false_accepts_per_hour"],
            abs(candidate["sliding_window_size"] - 5),
            -candidate["probability_cutoff"],
        ),
    )

    tier_index = tier(best)
    if tier_index < len(fallback_limits):
        return best, fallback_limits[tier_index]
    return best, float("inf")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle.read(), Loader=yaml.Loader)


def _load_eval_sets(
    handler: FeatureHandler,
    config: dict,
) -> tuple[str, str, list[np.ndarray], list[np.ndarray]]:
    for positive_mode, ambient_mode in (
        ("validation", "validation_ambient"),
        ("testing", "testing_ambient"),
    ):
        positive_tracks, labels, _ = handler.get_data(
            positive_mode,
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="none",
        )
        ambient_tracks, _, _ = handler.get_data(
            ambient_mode,
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="none",
        )
        positives = [
            np.asarray(track)
            for track, label in zip(positive_tracks, labels)
            if bool(label)
        ]
        ambient = [np.asarray(track) for track in ambient_tracks]
        if positives and ambient:
            return positive_mode, ambient_mode, positives, ambient
    raise RuntimeError(
        "No suitable validation/testing data was found for detector calibration."
    )


def _predict_tracks(
    model: Model,
    tracks: Sequence[np.ndarray],
    label: str,
) -> list[np.ndarray]:
    predictions: list[np.ndarray] = []
    total = len(tracks)
    print(f"→ Running streaming inference on {total} {label} track(s)")
    for index, track in enumerate(tracks, start=1):
        values = np.asarray(model.predict_spectrogram(track), dtype=np.float32)
        predictions.append(values)
        if index == total or index % 25 == 0:
            print(f"   {label}: {index}/{total}")
    return predictions


def main() -> int:
    args = parse_args()
    window_sizes = _parse_window_sizes(args.window_sizes)
    if args.cutoff_step <= 0:
        raise ValueError("cutoff-step must be > 0")
    if args.cutoff_max < args.cutoff_min:
        raise ValueError("cutoff-max must be >= cutoff-min")

    config_path = Path(args.training_config)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Streaming TFLite model not found: {model_path}")

    cutoffs = np.arange(
        args.cutoff_min,
        args.cutoff_max + (args.cutoff_step / 2.0),
        args.cutoff_step,
        dtype=np.float32,
    )
    cutoffs = np.clip(cutoffs, 0.0, 1.0)
    cutoffs = np.unique(np.round(cutoffs, 4))

    print("===== Detector Calibration =====")
    print(f"→ Model: {model_path}")
    print(f"→ Training config: {config_path}")
    print(
        f"→ Evaluating window sizes {window_sizes} with target <= "
        f"{args.target_faph:.2f} false accepts/hour"
    )

    config = _load_config(config_path)
    config["flags"] = config.get("flags", {})
    handler = FeatureHandler(config)

    positive_mode, ambient_mode, positive_tracks, ambient_tracks = _load_eval_sets(
        handler, config
    )

    print(
        f"→ Using {positive_mode} positives ({len(positive_tracks)}) and "
        f"{ambient_mode} ambient tracks ({len(ambient_tracks)})"
    )

    model = Model(str(model_path), stride=config["stride"])
    positive_predictions = _predict_tracks(model, positive_tracks, "positive")
    ambient_predictions = _predict_tracks(model, ambient_tracks, "ambient")

    candidates: list[dict[str, float]] = []
    best_by_window: list[dict[str, float]] = []
    step_seconds = config["window_step_ms"] / 1000.0

    for window_size in window_sizes:
        ambient_averages = [
            _moving_average(track, window_size) for track in ambient_predictions
        ]
        positive_maxima = []
        for track in positive_predictions:
            search = (
                track[args.positive_skip_slices :]
                if track.size > args.positive_skip_slices
                else track
            )
            averaged = _moving_average(search, window_size)
            if averaged.size == 0:
                averaged = _moving_average(track, window_size)
            positive_maxima.append(float(np.max(averaged)) if averaged.size else 0.0)

        positive_maxima_array = np.asarray(positive_maxima, dtype=np.float32)
        recall_by_cutoff = np.mean(
            positive_maxima_array[None, :] > cutoffs[:, None], axis=1
        )
        faph_by_cutoff, ambient_hours = _compute_false_accepts_per_hour(
            ambient_averages,
            cutoffs,
            args.cooldown_slices,
            stride=config["stride"],
            step_seconds=step_seconds,
        )

        window_candidates = []
        for cutoff, recall, faph in zip(cutoffs, recall_by_cutoff, faph_by_cutoff):
            candidate = {
                "probability_cutoff": float(round(float(cutoff), 2)),
                "sliding_window_size": int(window_size),
                "recall": float(recall),
                "false_accepts_per_hour": float(faph),
                "ambient_hours": float(ambient_hours),
            }
            candidates.append(candidate)
            window_candidates.append(candidate)

        best_window, _ = _select_best_candidate(window_candidates, args.target_faph)
        best_by_window.append(best_window)
        print(
            "   window={window}: cutoff={cutoff:.2f}; recall={recall:.2%}; "
            "ambient_faph={faph:.3f}".format(
                window=window_size,
                cutoff=best_window["probability_cutoff"],
                recall=best_window["recall"],
                faph=best_window["false_accepts_per_hour"],
            )
        )

    best, selected_limit = _select_best_candidate(candidates, args.target_faph)
    if best["false_accepts_per_hour"] > args.target_faph + 1e-9:
        print(
            "⚠️  No candidate met the target false accepts/hour budget; "
            "using the best fallback operating point."
        )

    print(
        "✓ Selected cutoff={cutoff:.2f}, window={window}, recall={recall:.2%}, "
        "ambient_faph={faph:.3f}".format(
            cutoff=best["probability_cutoff"],
            window=best["sliding_window_size"],
            recall=best["recall"],
            faph=best["false_accepts_per_hour"],
        )
    )

    output = {
        "probability_cutoff": best["probability_cutoff"],
        "sliding_window_size": best["sliding_window_size"],
        "target_false_accepts_per_hour": float(args.target_faph),
        "selected_false_accepts_per_hour_limit": (
            None if math.isinf(selected_limit) else float(selected_limit)
        ),
        "selected_metrics": {
            "recall": round(best["recall"], 6),
            "false_accepts_per_hour": round(best["false_accepts_per_hour"], 6),
            "ambient_hours": round(best["ambient_hours"], 6),
        },
        "evaluation": {
            "positive_dataset": positive_mode,
            "ambient_dataset": ambient_mode,
            "positive_tracks": len(positive_tracks),
            "ambient_tracks": len(ambient_tracks),
            "cooldown_slices": int(args.cooldown_slices),
            "positive_skip_slices": int(args.positive_skip_slices),
            "window_sizes": window_sizes,
            "cutoff_min": round(float(cutoffs[0]), 4),
            "cutoff_max": round(float(cutoffs[-1]), 4),
            "cutoff_step": float(args.cutoff_step),
        },
        "per_window_best": best_by_window,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"📝 Wrote calibration to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
