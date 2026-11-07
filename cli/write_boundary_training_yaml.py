#!/usr/bin/env python3
"""Derive an endpoint-focused microWakeWord training config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def build(source: Path, output: Path, train_dir: Path, features_dir: Path, steps: int) -> None:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{source} did not contain a training config")
    config = copy.deepcopy(payload)
    features = list(config.get("features") or [])
    if not features:
        raise RuntimeError("primary training config has no feature providers")
    boundary_features = {
        "features_dir": str(features_dir.resolve()),
        "sampling_weight": 18.0,
        "penalty_weight": 2.0,
        "truth": False,
        "truncation_strategy": "truncate_start",
        "type": "mmap",
    }
    features = [
        row for row in features
        if not isinstance(row, dict)
        or not str(row.get("features_dir") or "").endswith("boundary_negative_features")
    ]
    features.insert(1, boundary_features)
    config["features"] = features
    config["train_dir"] = str(train_dir.resolve())
    config["training_steps"] = [max(1000, int(steps))]
    config["early_stopping"] = {
        "enabled": True,
        "min_steps": max(1000, min(int(steps), int(steps * 0.6))),
        "patience_steps": max(1000, min(4000, int(steps * 0.2))),
        "min_delta": 0.001,
    }
    config["learning_rates"] = [0.001]
    config["negative_class_weight"] = [12]
    config["positive_class_weight"] = [1]
    config["minimum_positive_samples_per_batch"] = max(
        2, int(config.get("minimum_positive_samples_per_batch", 2))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"📝 Wrote endpoint training config: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--steps", type=int, default=20000)
    args = parser.parse_args()
    build(Path(args.source), Path(args.output), Path(args.train_dir), Path(args.features_dir), args.steps)


if __name__ == "__main__":
    main()
