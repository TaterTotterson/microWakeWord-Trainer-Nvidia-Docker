"""Create a compact, user-facing quality summary for a completed training run."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_LABELS = {
    "micro": "MicroWakeWord",
    "open": "openWakeWord",
    "nano_cnn": "NanoWakeWord CNN",
    "nano_gru": "NanoWakeWord GRU",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rating(recall: float | None, negative_value: float | None, *, negative_kind: str) -> tuple[str, str]:
    if recall is None and negative_value is None:
        return "Completed", "neutral"
    excellent_negative = 0.25 if negative_kind == "hour" else 0.05
    good_negative = 1.0 if negative_kind == "hour" else 0.10
    if recall is not None and recall >= 0.95 and (negative_value is None or negative_value <= excellent_negative):
        return "Excellent", "success"
    if recall is not None and recall >= 0.90 and (negative_value is None or negative_value <= good_negative):
        return "Good", "success"
    return "Needs attention", "warning"


def _metric(label: str, value: str) -> dict[str, str]:
    return {"label": label, "value": value}


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _successful_model(
    model_id: str,
    *,
    data_dir: Path,
    trained_dir: Path,
    safe_word: str,
) -> dict[str, Any]:
    metrics: list[dict[str, str]] = []
    recall: float | None = None
    negative_value: float | None = None
    negative_kind = "rate"

    if model_id == "micro":
        bundle = _read_json(trained_dir / f"{safe_word}.wake-bundle.json")
        micro = bundle.get("micro_wake_word") if isinstance(bundle.get("micro_wake_word"), dict) else {}
        calibration = micro.get("calibration") if isinstance(micro.get("calibration"), dict) else {}
        runtime = micro.get("tater_native") if isinstance(micro.get("tater_native"), dict) else {}
        micro_config = micro.get("micro") if isinstance(micro.get("micro"), dict) else {}
        recall = _number(calibration.get("recall"))
        negative_value = _number(calibration.get("false_accepts_per_hour"))
        negative_kind = "hour"
        threshold = _number(runtime.get("wake_threshold"))
        if threshold is None:
            threshold = _number(micro_config.get("probability_cutoff"))
        window = _integer(runtime.get("wake_sliding_window"))
        if window is None:
            window = _integer(micro_config.get("sliding_window_size"))
        if recall is not None:
            metrics.append(_metric("Recall", _percent(recall)))
        if negative_value is not None:
            metrics.append(_metric("False accepts", f"{negative_value:.2f}/hr"))
        if threshold is not None:
            metrics.append(_metric("Threshold", f"{threshold:.2f}"))
        if window is not None:
            metrics.append(_metric("Window", str(window)))

    elif model_id == "open":
        metadata = _read_json(data_dir / "openwakeword" / "publish-staging" / safe_word / f"{safe_word}.json")
        calibration = metadata.get("calibration") if isinstance(metadata.get("calibration"), dict) else {}
        recommended = calibration.get("recommended_metrics") if isinstance(calibration.get("recommended_metrics"), dict) else {}
        recall = _number(recommended.get("positive_recall"))
        negative_value = _number(recommended.get("false_positives_per_hour"))
        negative_kind = "hour"
        threshold = _number(calibration.get("recommended_threshold"))
        patience = _integer(calibration.get("recommended_patience"))
        if recall is not None:
            metrics.append(_metric("Recall", _percent(recall)))
        if negative_value is not None:
            metrics.append(_metric("False accepts", f"{negative_value:.2f}/hr"))
        if threshold is not None:
            metrics.append(_metric("Threshold", f"{threshold:.2f}"))
        if patience is not None:
            metrics.append(_metric("Patience", str(patience)))

    else:
        architecture = model_id.removeprefix("nano_")
        report = _read_json(
            data_dir / "nanowakeword" / "output" / architecture / safe_word / f"{safe_word}.quality.json"
        )
        validation = report.get("validation") if isinstance(report.get("validation"), dict) else {}
        positive = validation.get("positive") if isinstance(validation.get("positive"), dict) else {}
        negative = validation.get("negative") if isinstance(validation.get("negative"), dict) else {}
        contract = validation.get("streaming_contract") if isinstance(validation.get("streaming_contract"), dict) else {}
        recall = _number(positive.get("rate"))
        negative_value = _number(negative.get("rate"))
        threshold = _number(contract.get("threshold"))
        patience = _integer(contract.get("patience"))
        if recall is not None:
            metrics.append(_metric("Recall", _percent(recall)))
        if negative_value is not None:
            metrics.append(_metric("False-trigger test", _percent(negative_value)))
        if threshold is not None:
            metrics.append(_metric("Threshold", f"{threshold:.2f}"))
        if patience is not None:
            metrics.append(_metric("Patience", str(patience)))

    rating, tone = _rating(recall, negative_value, negative_kind=negative_kind)
    return {
        "id": model_id,
        "label": MODEL_LABELS.get(model_id, model_id),
        "status": "passed",
        "rating": rating,
        "tone": tone,
        "metrics": metrics,
        "message": "Quality validation passed." if metrics else "Training completed; detailed quality metrics were unavailable.",
    }


def build_training_summary(
    *,
    data_dir: Path,
    trained_dir: Path,
    safe_word: str,
    phrase: str,
    selected_models: Sequence[str],
    outcomes: Mapping[str, int | None],
) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for model_id in selected_models:
        code = outcomes.get(model_id)
        if code is None:
            models.append({
                "id": model_id,
                "label": MODEL_LABELS.get(model_id, model_id),
                "status": "skipped",
                "rating": "Not completed",
                "tone": "warning",
                "metrics": [],
                "message": "A required earlier stage did not finish.",
            })
        elif int(code) != 0:
            models.append({
                "id": model_id,
                "label": MODEL_LABELS.get(model_id, model_id),
                "status": "failed",
                "rating": "Failed",
                "tone": "error",
                "metrics": [_metric("Exit code", str(code))],
                "message": "The previous published model was preserved.",
            })
        else:
            models.append(_successful_model(
                model_id,
                data_dir=Path(data_dir),
                trained_dir=Path(trained_dir),
                safe_word=safe_word,
            ))

    if any(model["status"] == "failed" for model in models):
        overall, tone = "Needs attention", "error"
    elif any(model["status"] == "skipped" or model["rating"] == "Needs attention" for model in models):
        overall, tone = "Needs attention", "warning"
    elif models and all(model["rating"] == "Excellent" for model in models):
        overall, tone = "Excellent", "success"
    elif models and all(model["status"] == "passed" for model in models):
        overall, tone = "Good", "success"
    else:
        overall, tone = "No result", "neutral"

    passed = sum(model["status"] == "passed" for model in models)
    return {
        "wake_word": phrase or safe_word.replace("_", " "),
        "safe_word": safe_word,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "tone": tone,
        "message": f"{passed} of {len(models)} selected model{'s' if len(models) != 1 else ''} passed quality validation.",
        "models": models,
    }


def training_summary_log_lines(summary: Mapping[str, Any]) -> list[str]:
    lines = ["", "===== Training quality summary =====", f"Wake phrase: {summary.get('wake_word', '')}"]
    for model in summary.get("models", []):
        if not isinstance(model, Mapping):
            continue
        marker = "✓" if model.get("status") == "passed" else ("✗" if model.get("status") == "failed" else "–")
        details = [str(model.get("rating") or "")]
        for metric in model.get("metrics", []):
            if isinstance(metric, Mapping):
                details.append(f"{metric.get('label')}: {metric.get('value')}")
        lines.append(f"{marker} {model.get('label', 'Model')} — " + " · ".join(filter(None, details)))
    lines.append(f"Overall: {summary.get('overall', 'No result')} — {summary.get('message', '')}")
    lines.append("====================================")
    return lines
