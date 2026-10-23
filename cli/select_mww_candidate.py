#!/usr/bin/env python3
"""Select and restore the better calibrated MicroWakeWord candidate."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    name: str
    model: Path
    calibration: Path
    recall: float
    false_accepts_per_hour: float

    @property
    def metrics(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "recall": self.recall,
            "false_accepts_per_hour": self.false_accepts_per_hour,
        }


def load_candidate(name: str, model: Path, calibration: Path) -> Candidate:
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    metrics = payload.get("selected_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{name} calibration has no selected_metrics")
    return Candidate(
        name=name,
        model=model,
        calibration=calibration,
        recall=float(metrics["recall"]),
        false_accepts_per_hour=float(metrics["false_accepts_per_hour"]),
    )


def choose_candidate(
    baseline: Candidate,
    corrected: Candidate,
    *,
    minimum_recall: float,
    target_faph: float,
    recall_margin: float,
) -> Candidate:
    candidates = [baseline, corrected]

    passing = [
        item
        for item in candidates
        if item.recall >= minimum_recall
        and item.false_accepts_per_hour <= target_faph
    ]
    if len(passing) == 1:
        return passing[0]
    if passing:
        best_recall = max(item.recall for item in passing)
        viable = [item for item in passing if item.recall >= best_recall - recall_margin]
        return min(viable, key=lambda item: (item.false_accepts_per_hour, -item.recall))

    within_false_accept_target = [
        item for item in candidates if item.false_accepts_per_hour <= target_faph
    ]
    if len(within_false_accept_target) == 1:
        return within_false_accept_target[0]
    if within_false_accept_target:
        return max(
            within_false_accept_target,
            key=lambda item: (item.recall, -item.false_accepts_per_hour),
        )

    return min(
        candidates,
        key=lambda item: (item.false_accepts_per_hour, -item.recall),
    )


def copy_if_different(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--baseline-calibration", required=True, type=Path)
    parser.add_argument("--corrected-model", required=True, type=Path)
    parser.add_argument("--corrected-calibration", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-calibration", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-recall", type=float, default=0.90)
    parser.add_argument("--target-faph", type=float, default=0.25)
    parser.add_argument("--recall-margin", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_candidate(
        "initial", args.baseline_model, args.baseline_calibration
    )
    corrected = load_candidate(
        "corrective", args.corrected_model, args.corrected_calibration
    )
    selected = choose_candidate(
        baseline,
        corrected,
        minimum_recall=args.minimum_recall,
        target_faph=args.target_faph,
        recall_margin=args.recall_margin,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
    copy_if_different(selected.model, args.output_model)
    copy_if_different(selected.calibration, args.output_calibration)

    report = {
        "selected": selected.name,
        "minimum_recall": args.minimum_recall,
        "target_false_accepts_per_hour": args.target_faph,
        "candidates": [baseline.metrics, corrected.metrics],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selected {selected.name} MicroWakeWord candidate: "
        f"recall={selected.recall:.3%}, "
        f"false_accepts={selected.false_accepts_per_hour:.3f}/hr"
    )


if __name__ == "__main__":
    main()
