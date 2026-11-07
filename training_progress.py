"""Structured progress events shared by trainer workers and the local server."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any, TypeVar


PROGRESS_PREFIX = "@@trainer-progress "
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DETERMINATE_RE = re.compile(
    r"^(?P<label>.*?)\s*(?P<percent>\d+(?:\.\d+)?)%\|[^|]*\|\s*"
    r"(?P<current>[\d.,]+[kKmM]?)/(?P<total>[\d.,]+[kKmM]?)\s*"
    r"\[(?P<elapsed>[^<,\]]+)(?:<(?P<eta>[^,\]]+))?,\s*(?P<rate>[^\]]+)\]\s*$"
)
_INDETERMINATE_RE = re.compile(
    r"^(?P<current>[\d.,]+)it\s+\[(?P<elapsed>[^,\]]+),\s*(?P<rate>[^\]]+)\]\s*$"
)
_SIMPLE_COUNT_RE = re.compile(
    r"^(?P<label>.*?\D)\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)(?P<suffix>.*)$"
)
_BARE_COUNT_RE = re.compile(
    r"^(?P<current>\d+)\s*/\s*(?P<total>\d+)(?P<suffix>\s+.*)$"
)
_STEP_RE = re.compile(r"(?:^|:)Step\s+#(?P<current>\d+)(?::|\s|$)")
_EARLY_STOP_RE = re.compile(
    r"Early stopping(?: openWakeWord main sequence)? at step "
    r"(?P<current>\d+)/(?P<total>\d+).*?best(?: acceptable)? checkpoint step "
    r"(?P<best>\d+)",
    re.IGNORECASE,
)
_STAGE_PATTERNS = (
    (re.compile(r"^===== Generating\s+\d+\s+wake-word samples"), "tts", "Generating wake-word speech"),
    (re.compile(r"^===== Preparing isolated\s+(.+?)\s+TTS environment"), "tts", "Preparing {0} speech engine"),
    (re.compile(r"^🎭\s*Generating hard-negative TTS for:\s*(.+)$"), "tts", "Generating hard-negative speech · {0}"),
    (re.compile(r"^\s*Augmenting\s+(.+?)\s*\((.+?)\)\s*$"), "features", "Building {1} · {0} features"),
    (re.compile(r"^===== Setting up Training Datasets"), "datasets", "Preparing training datasets"),
    (re.compile(r"^===== Checking negative datasets"), "datasets", "Preparing negative datasets"),
    (re.compile(r"^===== Shared TTS / MicroWakeWord stage"), "tts", "Preparing shared speech data"),
    (re.compile(r"^===== Starting\s+\d+\s+training steps"), "model", "Training wake-word detector"),
    (re.compile(r"^===== Attempt\s+\d+/\d+:.*training"), "model", "Training wake-word detector"),
    (re.compile(r"^===== CPU fallback:\s*training"), "model", "Training wake-word detector"),
    (re.compile(r"^===== Corrective training:"), "model", "Corrective detector fine-tuning"),
    (re.compile(r"^🏋️\s*Starting model training"), "model", "Training wake-word detector"),
    (re.compile(r"^📝\s*Wrote endpoint training config"), "boundary", "Training wake boundary"),
    (re.compile(r"^.*Training dedicated wake-end boundary model"), "boundary", "Training wake boundary"),
    (re.compile(r"^.*Calibrating.*boundary", re.IGNORECASE), "calibration", "Calibrating wake boundary"),
    (re.compile(r"^.*Calibrating.*detector", re.IGNORECASE), "calibration", "Calibrating wake-word detector"),
    (re.compile(r"^.*Packaging.*model", re.IGNORECASE), "package", "Packaging trained models"),
)

T = TypeVar("T")


def iter_progress_lines(stream) -> Iterator[str]:
    """Yield both newline-delimited logs and carriage-return terminal redraws."""
    buffer: list[str] = []
    previous_was_carriage_return = False
    while True:
        character = stream.read(1)
        if character == "":
            if buffer:
                yield "".join(buffer)
            return
        if character == "\r":
            if buffer:
                yield "".join(buffer) + "\n"
                buffer.clear()
            previous_was_carriage_return = True
            continue
        if character == "\n":
            if buffer:
                yield "".join(buffer) + "\n"
                buffer.clear()
            elif not previous_was_carriage_return:
                yield "\n"
            previous_was_carriage_return = False
            continue
        previous_was_carriage_return = False
        buffer.append(character)


def _duration(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    value = max(0, int(round(float(seconds))))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_value:02d}"
    return f"{minutes:02d}:{seconds_value:02d}"


def _scaled_number(value: str) -> float:
    token = value.strip().replace(",", "")
    multiplier = 1.0
    if token[-1:].lower() == "k":
        token, multiplier = token[:-1], 1_000.0
    elif token[-1:].lower() == "m":
        token, multiplier = token[:-1], 1_000_000.0
    return float(token) * multiplier


def progress_event(*, stage: str, label: str, current: int, total: int, elapsed_seconds: float, unit: str) -> str:
    total = max(0, int(total))
    current = max(0, min(int(current), total)) if total else max(0, int(current))
    elapsed = max(0.0, float(elapsed_seconds))
    rate = current / elapsed if current and elapsed > 0 else 0.0
    remaining = (total - current) / rate if total and rate > 0 else None
    payload = {
        "stage": stage,
        "label": label,
        "current": current,
        "total": total,
        "percent": round(current * 100.0 / total, 1) if total else None,
        "elapsed": _duration(elapsed),
        "eta": _duration(remaining),
        "rate": round(rate, 1) if rate else None,
        "unit": unit,
    }
    return PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def tracked_generator(
    samples: Iterable[T], *, total: int, stage: str, label: str, unit: str = "items",
    emit: Callable[[str], Any] = print, update_interval_seconds: float = 1.0,
) -> Iterator[T]:
    """Yield samples while emitting bounded, machine-readable progress lines."""
    started = time.monotonic()
    last_update = started
    current = 0
    emit(progress_event(stage=stage, label=label, current=0, total=total, elapsed_seconds=0, unit=unit))
    for current, sample in enumerate(samples, start=1):
        now = time.monotonic()
        if current == total or now - last_update >= update_interval_seconds:
            emit(progress_event(stage=stage, label=label, current=current, total=total, elapsed_seconds=now - started, unit=unit))
            last_update = now
        yield sample
    if current != total:
        emit(progress_event(stage=stage, label=label, current=current, total=total, elapsed_seconds=time.monotonic() - started, unit=unit))


def parse_progress_line(line: str) -> tuple[dict[str, Any] | None, bool]:
    """Return (progress, transient); transient lines should not flood the console."""
    raw = _ANSI_RE.sub("", str(line or ""))
    fragments = [fragment.strip() for fragment in raw.replace("\r", "\n").splitlines() if fragment.strip()]
    clean = fragments[-1] if fragments else ""
    if not clean:
        return None, False
    if clean.startswith(PROGRESS_PREFIX):
        try:
            payload = json.loads(clean[len(PROGRESS_PREFIX):])
        except (TypeError, json.JSONDecodeError):
            return None, False
        return (payload if isinstance(payload, dict) else None), True
    match = _EARLY_STOP_RE.search(clean)
    if match:
        current = int(match.group("current"))
        total = int(match.group("total"))
        best = int(match.group("best"))
        return {
            "stage": "worker", "label": f"Stopped early · best checkpoint {best:,}",
            "current": current, "total": total,
            "percent": round(current * 100.0 / total, 1) if total else None,
            "elapsed": "", "eta": "", "unit": "steps",
            "status": "stopped_early", "best_step": best,
        }, False
    match = _DETERMINATE_RE.match(clean)
    if match:
        current = _scaled_number(match.group("current"))
        total = _scaled_number(match.group("total"))
        return {
            "stage": "worker", "label": match.group("label").strip(" :-") or "Processing",
            "current": current, "total": total, "percent": float(match.group("percent")),
            "elapsed": match.group("elapsed"), "eta": match.group("eta") or "",
            "rate_text": match.group("rate"), "unit": "items",
        }, True
    match = _INDETERMINATE_RE.match(clean)
    if match:
        return {
            "stage": "worker", "label": "Processing", "current": _scaled_number(match.group("current")),
            "total": None, "percent": None, "elapsed": match.group("elapsed"), "eta": "",
            "rate_text": match.group("rate"), "unit": "items",
        }, True
    match = _STEP_RE.search(clean)
    if match:
        return {
            "stage": "worker", "label": "Processing", "current": int(match.group("current")),
            "total": None, "percent": None, "elapsed": "", "eta": "", "unit": "steps",
        }, True
    for pattern, stage, template in _STAGE_PATTERNS:
        match = pattern.search(clean)
        if match:
            label = template.format(*match.groups()) if match.groups() else template
            return {
                "stage": stage, "label": label, "current": 0, "total": None,
                "percent": None, "elapsed": "", "eta": "", "unit": "items",
            }, False
    match = _SIMPLE_COUNT_RE.match(clean)
    if match:
        current, total = int(match.group("current")), int(match.group("total"))
        if total > 0 and current <= total:
            progress = {
                "stage": "worker", "label": match.group("label").strip(" :-→") or "Processing",
                "current": current, "total": total, "percent": round(current * 100.0 / total, 1),
                "elapsed": "", "eta": "", "unit": "items",
            }
            label_key = progress["label"].casefold()
            if "attempt" in label_key or "recovery" in label_key:
                return None, False
            transient = total >= 10 or any(
                word in label_key
                for word in ("progress", "generated", "created", "accepted", "batch", "sample", "clip")
            )
            return progress, transient
    match = _BARE_COUNT_RE.match(clean)
    if match:
        current, total = int(match.group("current")), int(match.group("total"))
        if total > 0 and current <= total:
            return {
                "stage": "worker", "label": "Processing", "current": current, "total": total,
                "percent": round(current * 100.0 / total, 1), "elapsed": "", "eta": "", "unit": "steps",
            }, True
    return None, False
