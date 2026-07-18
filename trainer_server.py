#!/usr/bin/env python3

# trainer_server.py
import contextlib
import io
import os
import queue
import re
import json
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from array import array
from datetime import datetime, timedelta, timezone
from math import isfinite, log10
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request as URLRequest, urlopen

from fastapi import FastAPI, UploadFile, File, Form, Header, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent

# In Docker, /data is the persistent workspace mounted by the user.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data")).resolve()
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT_DIR / "static"))).resolve()
PERSONAL_DIR = Path(os.environ.get("PERSONAL_DIR", str(DATA_DIR / "personal_samples"))).resolve()
CAPTURED_DIR = Path(os.environ.get("CAPTURED_DIR", str(DATA_DIR / "captured_audio"))).resolve()
NEGATIVE_DIR = Path(os.environ.get("NEGATIVE_DIR", str(DATA_DIR / "negative_samples"))).resolve()
TRIM_HISTORY_DIR = Path(os.environ.get("TRIM_HISTORY_DIR", str(DATA_DIR / "trim_history"))).resolve()
TRIM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_WAKE_WORDS_DIR = Path(
    os.environ.get("TRAINED_WAKE_WORDS_DIR", str(DATA_DIR / "trained_wake_words"))
).resolve()
AUTO_TRAIN_CONFIG_FILE = Path(
    os.environ.get("AUTO_TRAIN_CONFIG_FILE", str(DATA_DIR / "auto_train_config.json"))
).resolve()
AUTO_TRAIN_STATE_FILE = Path(
    os.environ.get("AUTO_TRAIN_STATE_FILE", str(DATA_DIR / "auto_train_state.json"))
).resolve()
AUTO_TRAIN_MODEL_DIR = Path(
    os.environ.get("AUTO_TRAIN_MODEL_DIR", str(DATA_DIR / "auto_train_models"))
).resolve()
CLI_DIR = Path(os.environ.get("CLI_DIR", str(ROOT_DIR / "cli"))).resolve()
PIPER_ROOT = DATA_DIR / "tools" / "piper-sample-generator"
PIPER_VOICES_DIR = PIPER_ROOT / "voices"
PIPER_VOICES_INDEX_URL = os.environ.get(
    "PIPER_VOICES_INDEX_URL",
    "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json",
)
PIPER_VOICES_ROOT_URL = os.environ.get(
    "PIPER_VOICES_ROOT_URL",
    "https://huggingface.co/rhasspy/piper-voices/resolve/main",
)
PIPER_CATALOG_CACHE_TTL_SECONDS = int(os.environ.get("PIPER_CATALOG_CACHE_TTL_SECONDS", "900"))
PIPER_CATALOG_CACHE_FILE = Path(
    os.environ.get(
        "PIPER_CATALOG_CACHE_FILE",
        str(DATA_DIR / ".cache" / "piper_voices_catalog.json"),
    )
).resolve()

DATASET_CLEANUP_ARCHIVES = os.environ.get("REC_DATASET_CLEANUP_ARCHIVES", "false").lower() in ("1", "true", "yes", "y")
DATASET_CLEANUP_INTERMEDIATE = os.environ.get("REC_DATASET_CLEANUP_INTERMEDIATE_FILES", "false").lower() in ("1", "true", "yes", "y")

TRAIN_CMD = os.environ.get(
    "TRAIN_CMD",
    f"source '{DATA_DIR}/.venv/bin/activate' && train_wake_word --data-dir '{DATA_DIR}'",
)
DEFAULT_LANGUAGE = os.environ.get("MWW_LANGUAGE", "en")

TAKES_PER_SPEAKER_DEFAULT = int(os.environ.get("REC_TAKES_PER_SPEAKER", "10"))
SPEAKERS_TOTAL_DEFAULT = int(os.environ.get("REC_SPEAKERS_TOTAL", "1"))
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2
CAPTURE_GAIN_PROFILE = "capture_rms_v1"
DEFAULT_FASTER_WHISPER_MODEL = os.environ.get("AUTO_TRAIN_STT_MODEL", "small.en")

AUTO_TRAIN_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "wake_phrase": "",
    "language": DEFAULT_LANGUAGE,
    "stt_model": DEFAULT_FASTER_WHISPER_MODEL,
    "stt_device": "auto",
    "stt_compute_type": "auto",
    "minimum_transcript_chars": 2,
    "delete_confirmed_wakes": False,
    "promote_close_misses": False,
    "schedule_hours": 24,
    "minimum_new_negatives": 3,
    "advertised_base_url": "",
    "tater_url": "http://127.0.0.1:8501",
    "tater_selector": "",
    "tater_api_token": "",
    "notify_satellites": True,
}

AUTO_TRAIN_DEFAULT_STATE: Dict[str, Any] = {
    "pending_negative_count": 0,
    "next_run_at": "",
    "last_review_at": "",
    "last_review_file": "",
    "last_review_transcript": "",
    "last_review_result": "",
    "last_review_error": "",
    "last_stt_device": "",
    "last_stt_compute_type": "",
    "last_train_started_at": "",
    "last_train_finished_at": "",
    "last_train_exit_code": None,
    "last_notify_at": "",
    "last_notify_count": None,
    "last_notify_error": "",
}


app = FastAPI(title="microWakeWord Personal Samples")

# Serve static UI
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def safe_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"^_+|_+$", "", s)
    return s or "wakeword"


# -------------------- In-memory session state --------------------
STATE: Dict[str, Any] = {
    "raw_phrase": None,
    "safe_word": None,
    "language": DEFAULT_LANGUAGE,

    # multi-speaker
    "speakers_total": SPEAKERS_TOTAL_DEFAULT,
    "takes_per_speaker": TAKES_PER_SPEAKER_DEFAULT,

    # recording progress
    "takes_received": 0,   # total across all speakers
    "takes": [],           # list of saved filenames

    "training": {
        "running": False,
        "exit_code": None,
        "log_lines": [],
        "log_path": None,
        "safe_word": None,
    },
}

STATE_LOCK = threading.Lock()
SAMPLES_LOCK = threading.Lock()
PIPER_CATALOG_LOCK = threading.Lock()
AUTO_TRAIN_LOCK = threading.RLock()
AUTO_TRAIN_WAKE_EVENT = threading.Event()
AUTO_TRAIN_STOP_EVENT = threading.Event()
AUTO_TRAIN_REVIEW_QUEUE: queue.Queue[str] = queue.Queue()
AUTO_TRAIN_QUEUED_FILES: set[str] = set()
AUTO_TRAIN_WORKER: threading.Thread | None = None
AUTO_TRAIN_RUNTIME: Dict[str, Any] = {
    "review_running": False,
    "review_file": "",
    "scheduler_running": False,
    "training_pending_consumed": 0,
}
LAN_ADDRESS_CACHE: Dict[str, Any] = {"value": "", "fetched_at": 0.0}
FASTER_WHISPER_MODEL_LOCK = threading.RLock()
FASTER_WHISPER_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
PIPER_CATALOG_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "entries": None,
}

# --- Silero VAD (lazy-loaded) ---
_silero_vad_model = None
_silero_vad_utils = None
_SILERO_VAD_LOCK = threading.Lock()
VAD_SELECTION_PAD_START_S = 0.08
VAD_SELECTION_PAD_END_S = 0.08


def _load_silero_vad():
    """Lazy-load Silero VAD model on first use. Returns (model, utils)."""
    global _silero_vad_model, _silero_vad_utils
    if _silero_vad_model is not None:
        return _silero_vad_model, _silero_vad_utils
    with _SILERO_VAD_LOCK:
        if _silero_vad_model is not None:
            return _silero_vad_model, _silero_vad_utils
        import torch
        import silero_vad
        model = silero_vad.load_silero_vad()
        model.eval()
        _silero_vad_model = model
        _silero_vad_utils = {"torch": torch}
        return model, _silero_vad_utils


def _detect_speech_segments(wav_bytes: bytes) -> List[Dict[str, float]]:
    """Run Silero VAD on 16 kHz mono WAV bytes. Return {start, end} seconds."""
    model, utils = _load_silero_vad()
    torch = utils["torch"]
    import numpy as np
    from silero_vad.utils_vad import get_speech_timestamps

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(samples)

    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=150,
        min_silence_duration_ms=100,
        return_seconds=True,
    )
    return [{"start": round(ts["start"], 3), "end": round(ts["end"], 3)} for ts in timestamps]



def _reset_personal_samples_dir():
    _reset_audio_dir(PERSONAL_DIR)


def _reset_audio_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in {".wav", ".json"}:
            try:
                p.unlink()
            except Exception:
                pass


def _list_audio_samples(directory: Path) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in directory.glob("*.wav"))


def _list_personal_samples() -> List[str]:
    return _list_audio_samples(PERSONAL_DIR)


def _list_negative_samples() -> List[str]:
    return _list_audio_samples(NEGATIVE_DIR)


def _list_captured_sample_names() -> List[str]:
    return _list_audio_samples(CAPTURED_DIR)


def _sync_trained_wake_word_artifacts() -> None:
    """Mirror generated output artifacts into /data/trained_wake_words for live wake-word links."""
    TRAINED_WAKE_WORDS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_jsons: list[Path] = []

    output_dir = DATA_DIR / "output"
    if output_dir.exists():
        candidate_jsons.extend(output_dir.rglob("*.json"))

    # One-time migration for older root-level outputs.
    candidate_jsons.extend(ROOT_DIR.glob("*.json"))

    for json_path in sorted(candidate_jsons):
        if TRAINED_WAKE_WORDS_DIR in json_path.parents:
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue

        model_name = str(meta.get("model") or json_path.with_suffix(".tflite").name).strip()
        tflite_path = (json_path.parent / Path(model_name).name).resolve()
        if not tflite_path.exists():
            fallback = json_path.with_suffix(".tflite")
            if fallback.exists():
                tflite_path = fallback.resolve()
            else:
                continue

        for source_path in (json_path, tflite_path):
            dest_path = TRAINED_WAKE_WORDS_DIR / source_path.name
            if not dest_path.exists() or source_path.stat().st_mtime > dest_path.stat().st_mtime:
                shutil.copy2(source_path, dest_path)

        if json_path.parent == ROOT_DIR:
            with contextlib.suppress(Exception):
                json_path.unlink()
            with contextlib.suppress(Exception):
                tflite_path.unlink()


def _metadata_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(out):
        return None
    return out


def _metadata_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _list_trained_wake_words(base_url: str = "") -> List[Dict[str, Any]]:
    _sync_trained_wake_word_artifacts()
    base = str(base_url or "").rstrip("/")
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for json_path in sorted(TRAINED_WAKE_WORDS_DIR.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue

        model_name = str(meta.get("model") or json_path.with_suffix(".tflite").name).strip()
        model_path = TRAINED_WAKE_WORDS_DIR / Path(model_name).name
        if not model_path.exists():
            continue

        safe = json_path.stem
        if safe in seen:
            continue
        seen.add(safe)

        wake_word = str(meta.get("wake_word") or safe.replace("_", " ")).strip()
        micro = meta.get("micro") if isinstance(meta.get("micro"), dict) else {}
        native = meta.get("tater_native") if isinstance(meta.get("tater_native"), dict) else {}
        calibration = meta.get("calibration") if isinstance(meta.get("calibration"), dict) else {}
        threshold = _metadata_float(native.get("wake_threshold"))
        if threshold is None:
            threshold = _metadata_float(micro.get("probability_cutoff"))
        sliding_window = _metadata_int(native.get("wake_sliding_window"))
        if sliding_window is None:
            sliding_window = _metadata_int(micro.get("sliding_window_size"))
        close_miss_threshold = _metadata_float(native.get("close_miss_threshold"))
        recall = _metadata_float(calibration.get("recall"))
        false_accepts_per_hour = _metadata_float(calibration.get("false_accepts_per_hour"))
        json_url = f"/api/trained_wake_words/{quote(json_path.name)}"
        model_url = f"/api/trained_wake_words/{quote(model_path.name)}"
        if base:
            json_url = f"{base}{json_url}"
            model_url = f"{base}{model_url}"

        rows.append(
            {
                "key": safe,
                "label": wake_word or safe,
                "wake_word_name": safe,
                "wake_word": wake_word or safe,
                "json_url": json_url,
                "model_url": model_url,
                "json_file": json_path.name,
                "model_file": model_path.name,
                "threshold": round(threshold, 3) if threshold is not None else None,
                "sliding_window": sliding_window,
                "close_miss_threshold": round(close_miss_threshold, 3) if close_miss_threshold is not None else None,
                "quantization": str(meta.get("quantization") or "").strip(),
                "model_format": str(meta.get("model_format") or "").strip(),
                "sample_rate": _metadata_int(meta.get("sample_rate")),
                "calibration_recall": round(recall, 4) if recall is not None else None,
                "calibration_false_accepts_per_hour": (
                    round(false_accepts_per_hour, 6) if false_accepts_per_hour is not None else None
                ),
                "calibration_generated_at": str(calibration.get("generated_at") or "").strip(),
            }
        )
    return rows


def _request_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_object(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    with contextlib.suppress(Exception):
        path.chmod(0o600)


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _config_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"1", "true", "yes", "on", "enabled"}:
        return True
    if token in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(default)


def _normalize_http_base_url(value: Any, *, allow_empty: bool = True) -> str:
    token = str(value or "").strip().rstrip("/")
    if not token and allow_empty:
        return ""
    if not token.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return token


def _normalize_auto_train_config(values: Dict[str, Any] | None, *, base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    incoming = values if isinstance(values, dict) else {}
    source = {**AUTO_TRAIN_DEFAULT_CONFIG, **(base or {}), **incoming}
    schedule_hours = _bounded_int(source.get("schedule_hours"), 24, 0, 24 * 30)
    language = str(source.get("language") or DEFAULT_LANGUAGE).strip().lower().replace("-", "_")
    language = re.sub(r"[^a-z0-9_]", "", language) or DEFAULT_LANGUAGE
    stt_device = str(source.get("stt_device") or "auto").strip().lower()
    if stt_device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Faster Whisper device must be auto, cuda, or cpu.")
    stt_compute_type = str(source.get("stt_compute_type") or "auto").strip().lower()
    if stt_compute_type not in {"auto", "default", "float16", "float32", "int8", "int8_float16"}:
        raise ValueError("Unsupported Faster Whisper compute type.")
    return {
        "enabled": _config_bool(source.get("enabled")),
        "wake_phrase": str(source.get("wake_phrase") or "").strip(),
        "language": language,
        "stt_model": str(source.get("stt_model") or DEFAULT_FASTER_WHISPER_MODEL).strip() or DEFAULT_FASTER_WHISPER_MODEL,
        "stt_device": stt_device,
        "stt_compute_type": stt_compute_type,
        "minimum_transcript_chars": _bounded_int(source.get("minimum_transcript_chars"), 2, 1, 100),
        "delete_confirmed_wakes": _config_bool(source.get("delete_confirmed_wakes")),
        "promote_close_misses": _config_bool(source.get("promote_close_misses")),
        "schedule_hours": schedule_hours,
        "minimum_new_negatives": _bounded_int(source.get("minimum_new_negatives"), 3, 1, 10000),
        "advertised_base_url": _normalize_http_base_url(source.get("advertised_base_url")),
        "tater_url": _normalize_http_base_url(source.get("tater_url"), allow_empty=False),
        "tater_selector": str(source.get("tater_selector") or "").strip(),
        "tater_api_token": str(source.get("tater_api_token") or "").strip(),
        "notify_satellites": _config_bool(source.get("notify_satellites"), True),
    }


try:
    AUTO_TRAIN_CONFIG: Dict[str, Any] = _normalize_auto_train_config(
        _read_json_object(AUTO_TRAIN_CONFIG_FILE)
    )
except ValueError:
    AUTO_TRAIN_CONFIG = dict(AUTO_TRAIN_DEFAULT_CONFIG)
AUTO_TRAIN_STATE: Dict[str, Any] = {
    **AUTO_TRAIN_DEFAULT_STATE,
    **_read_json_object(AUTO_TRAIN_STATE_FILE),
}


def _save_auto_train_config_locked() -> None:
    _write_json_object(AUTO_TRAIN_CONFIG_FILE, AUTO_TRAIN_CONFIG)


def _save_auto_train_state_locked() -> None:
    persisted = {key: AUTO_TRAIN_STATE.get(key) for key in AUTO_TRAIN_DEFAULT_STATE}
    _write_json_object(AUTO_TRAIN_STATE_FILE, persisted)


def _parse_iso_datetime(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _schedule_next_auto_run_locked(*, from_time: datetime | None = None) -> None:
    hours = int(AUTO_TRAIN_CONFIG.get("schedule_hours") or 0)
    if hours <= 0 or not AUTO_TRAIN_CONFIG.get("enabled"):
        AUTO_TRAIN_STATE["next_run_at"] = ""
    else:
        base = from_time or _utc_now()
        AUTO_TRAIN_STATE["next_run_at"] = (base + timedelta(hours=hours)).isoformat()
    _save_auto_train_state_locked()


def _public_auto_train_config() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        config = {key: value for key, value in AUTO_TRAIN_CONFIG.items() if key != "tater_api_token"}
        config["tater_api_token_configured"] = bool(AUTO_TRAIN_CONFIG.get("tater_api_token"))
        return config


def _auto_train_status_payload() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        return {
            "config": _public_auto_train_config(),
            "state": dict(AUTO_TRAIN_STATE),
            "runtime": dict(AUTO_TRAIN_RUNTIME),
            "advertised_base_url": _advertised_base_url(),
        }


def _discover_lan_ipv4() -> str:
    override = str(os.environ.get("REC_ADVERTISED_HOST") or "").strip()
    if override and override not in {"0.0.0.0", "127.0.0.1", "localhost", "::1"}:
        return override

    now = time.time()
    cached_value = str(LAN_ADDRESS_CACHE.get("value") or "")
    if cached_value and (now - float(LAN_ADDRESS_CACHE.get("fetched_at") or 0.0)) < 30:
        return cached_value

    candidates: List[str] = []
    with contextlib.suppress(Exception):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("192.0.2.1", 9))
            candidates.append(str(sock.getsockname()[0]))
        finally:
            sock.close()
    with contextlib.suppress(Exception):
        for row in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM):
            candidates.append(str(row[4][0]))

    with contextlib.suppress(Exception):
        proc = subprocess.run(
            ["/sbin/ifconfig"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        blocks = re.split(r"(?m)(?=^[^\s].*?: flags=)", proc.stdout or "")
        interface_rows: List[tuple[int, str]] = []
        for block in blocks:
            name_match = re.match(r"^([^:]+):", block)
            address_match = re.search(r"(?m)^\s+inet\s+(\d+(?:\.\d+){3})\b", block)
            if not name_match or not address_match or "status: active" not in block:
                continue
            name = name_match.group(1)
            if name == "lo0" or name.startswith(("utun", "awdl", "llw", "ap")):
                continue
            priority = 0 if name == "en0" else 1 if name == "en1" else 10
            interface_rows.append((priority, address_match.group(1)))
        candidates.extend(address for _priority, address in sorted(interface_rows))

    for candidate in candidates:
        if candidate and not candidate.startswith("127.") and candidate != "0.0.0.0":
            LAN_ADDRESS_CACHE["value"] = candidate
            LAN_ADDRESS_CACHE["fetched_at"] = now
            return candidate
    LAN_ADDRESS_CACHE["fetched_at"] = now
    return ""


def _advertised_base_url(request: Request | None = None) -> str:
    env_url = str(os.environ.get("REC_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    with AUTO_TRAIN_LOCK:
        configured_url = str(AUTO_TRAIN_CONFIG.get("advertised_base_url") or "").strip().rstrip("/")
    if configured_url:
        return configured_url
    if env_url:
        return env_url

    request_url = _request_base_url(request) if request is not None else ""
    request_host = str(request.url.hostname or "").lower() if request is not None else ""
    if request_url and request_host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return request_url

    host = _discover_lan_ipv4()
    if not host:
        return request_url
    scheme = str(request.url.scheme or "http") if request is not None else "http"
    port = request.url.port if request is not None else None
    if port is None:
        port = _bounded_int(os.environ.get("REC_PORT"), 8789, 1, 65535)
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    return f"{scheme}://{host}{'' if default_port else f':{port}'}"


def _normalize_transcript_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("_", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _transcript_contains_wake_phrase(transcript: Any, wake_phrase: Any) -> bool:
    normalized_transcript = _normalize_transcript_text(transcript)
    normalized_phrase = _normalize_transcript_text(wake_phrase)
    if not normalized_transcript or not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_transcript} "


def _captured_event_is_close_miss(metadata: Dict[str, Any]) -> bool:
    event_type = str(metadata.get("event_type") or "captured").strip().lower()
    return "close" in event_type


def _captured_event_is_auto_reviewable(
    metadata: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> bool:
    if _parse_bool(metadata.get("blocked_by_vad")):
        return False
    event_type = str(metadata.get("event_type") or "captured").strip().lower()
    if "close" in event_type:
        return bool((config or {}).get("promote_close_misses"))
    return event_type in {"captured", "trigger", "false_trigger"} or "wake" in event_type or "detect" in event_type


def _resolve_faster_whisper_runtime(device_value: Any, compute_value: Any) -> Tuple[str, str]:
    requested_device = str(device_value or "auto").strip().lower()
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Faster Whisper device must be auto, cuda, or cpu.")

    cuda_devices = 0
    with contextlib.suppress(Exception):
        import ctranslate2
        cuda_devices = int(ctranslate2.get_cuda_device_count())

    if requested_device == "cuda" and cuda_devices <= 0:
        raise RuntimeError("CUDA was selected for Faster Whisper, but CTranslate2 cannot see an NVIDIA GPU.")
    device = "cuda" if requested_device == "cuda" or (requested_device == "auto" and cuda_devices > 0) else "cpu"

    requested_compute = str(compute_value or "auto").strip().lower()
    allowed_compute = {"auto", "default", "float16", "float32", "int8", "int8_float16"}
    if requested_compute not in allowed_compute:
        raise ValueError("Unsupported Faster Whisper compute type.")
    compute_type = ("float16" if device == "cuda" else "int8") if requested_compute == "auto" else requested_compute
    return device, compute_type


def _load_faster_whisper_model(*, model_name: str, device: str, compute_type: str):
    cache_key = (model_name, device, compute_type)
    with FASTER_WHISPER_MODEL_LOCK:
        cached = FASTER_WHISPER_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(f"faster-whisper is unavailable: {exc}") from exc

        AUTO_TRAIN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(AUTO_TRAIN_MODEL_DIR),
        )
        FASTER_WHISPER_MODEL_CACHE.clear()
        FASTER_WHISPER_MODEL_CACHE[cache_key] = model
        return model


def _transcribe_capture_with_faster_whisper(audio_path: Path, *, model: str, language: str) -> str:
    with AUTO_TRAIN_LOCK:
        device_value = AUTO_TRAIN_CONFIG.get("stt_device")
        compute_value = AUTO_TRAIN_CONFIG.get("stt_compute_type")
    device, compute_type = _resolve_faster_whisper_runtime(device_value, compute_value)
    whisper_model = _load_faster_whisper_model(
        model_name=model,
        device=device,
        compute_type=compute_type,
    )
    segments, _info = whisper_model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=1,
        condition_on_previous_text=False,
    )
    transcript = re.sub(
        r"\s+",
        " ",
        " ".join(str(segment.text or "").strip() for segment in segments),
    ).strip()
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_STATE["last_stt_device"] = device
        AUTO_TRAIN_STATE["last_stt_compute_type"] = compute_type
        _save_auto_train_state_locked()
    return transcript


def _queue_auto_review(file_name: str) -> bool:
    safe_file_name = Path(str(file_name or "")).name
    if not safe_file_name:
        return False
    with AUTO_TRAIN_LOCK:
        if safe_file_name in AUTO_TRAIN_QUEUED_FILES:
            return False
        AUTO_TRAIN_QUEUED_FILES.add(safe_file_name)
    AUTO_TRAIN_REVIEW_QUEUE.put(safe_file_name)
    AUTO_TRAIN_WAKE_EVENT.set()
    return True


def _queue_pending_auto_reviews(*, force: bool = False) -> int:
    queued = 0
    with AUTO_TRAIN_LOCK:
        config = dict(AUTO_TRAIN_CONFIG)
    if not config.get("enabled"):
        return queued
    CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
    for audio_path in sorted(CAPTURED_DIR.glob("*.wav")):
        metadata = _load_sidecar_json(audio_path)
        if not _captured_event_is_auto_reviewable(metadata, config):
            continue
        status = str(metadata.get("auto_review_status") or "").strip()
        if status == "transcribing":
            metadata.pop("auto_review_status", None)
            _write_sidecar_json(audio_path, metadata)
            status = ""
        if force and status in {"error", "no_speech"}:
            metadata.pop("auto_review_status", None)
            _write_sidecar_json(audio_path, metadata)
            status = ""
        if (
            status == "wake_phrase_detected"
            and config.get("delete_confirmed_wakes")
            and not _captured_event_is_close_miss(metadata)
        ):
            status = ""
        if status:
            continue
        if _queue_auto_review(audio_path.name):
            queued += 1
    return queued


def _record_auto_review_result(*, file_name: str, transcript: str = "", result: str = "", error: str = "") -> None:
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_STATE["last_review_at"] = _iso_now()
        AUTO_TRAIN_STATE["last_review_file"] = file_name
        AUTO_TRAIN_STATE["last_review_transcript"] = transcript
        AUTO_TRAIN_STATE["last_review_result"] = result
        AUTO_TRAIN_STATE["last_review_error"] = error
        _save_auto_train_state_locked()


def _auto_review_capture(file_name: str) -> None:
    try:
        with AUTO_TRAIN_LOCK:
            config = dict(AUTO_TRAIN_CONFIG)
            AUTO_TRAIN_RUNTIME["review_running"] = True
            AUTO_TRAIN_RUNTIME["review_file"] = file_name
        if not config.get("enabled"):
            return
        wake_phrase = str(config.get("wake_phrase") or "").strip()
        if not wake_phrase:
            _record_auto_review_result(file_name=file_name, result="waiting_for_wake_phrase")
            return

        try:
            audio_path = _resolve_audio_path(CAPTURED_DIR, file_name)
        except FileNotFoundError:
            return
        metadata = _load_sidecar_json(audio_path)
        is_close_miss = _captured_event_is_close_miss(metadata)
        status = str(metadata.get("auto_review_status") or "").strip()
        if (
            status == "wake_phrase_detected"
            and config.get("delete_confirmed_wakes")
            and not is_close_miss
        ):
            transcript = str(metadata.get("transcript") or "")
            _remove_audio_with_sidecar(audio_path)
            _record_auto_review_result(
                file_name=file_name,
                transcript=transcript,
                result="deleted_confirmed_wake",
            )
            return
        if status or not _captured_event_is_auto_reviewable(metadata, config):
            return
        captured_wake_phrase = str(metadata.get("wake_word") or "").strip()
        if captured_wake_phrase and _normalize_transcript_text(captured_wake_phrase) != _normalize_transcript_text(wake_phrase):
            metadata["auto_review_status"] = "different_wake_phrase"
            metadata["auto_review_reason"] = (
                f"Capture is for '{captured_wake_phrase}', not configured phrase '{wake_phrase}'; left for manual review."
            )
            metadata["auto_reviewed_at"] = _iso_now()
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(file_name=file_name, result="different_wake_phrase")
            return

        metadata["auto_review_status"] = "transcribing"
        metadata["auto_reviewed_at"] = _iso_now()
        metadata["auto_review_wake_phrase"] = wake_phrase
        metadata["auto_review_stt_model"] = config["stt_model"]
        _write_sidecar_json(audio_path, metadata)

        transcript = _transcribe_capture_with_faster_whisper(
            audio_path,
            model=str(config["stt_model"]),
            language=str(config.get("language") or DEFAULT_LANGUAGE),
        )
        normalized = _normalize_transcript_text(transcript)
        metadata = _load_sidecar_json(audio_path)
        metadata["transcript"] = transcript
        metadata["transcribed_at"] = _iso_now()

        if len(normalized) < int(config.get("minimum_transcript_chars") or 2):
            metadata["auto_review_status"] = "no_speech"
            metadata["auto_review_reason"] = "STT did not return enough text; left for manual review."
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(file_name=file_name, transcript=transcript, result="no_speech")
            return

        if _transcript_contains_wake_phrase(transcript, wake_phrase):
            if is_close_miss:
                metadata["auto_review_status"] = "approved_positive"
                metadata["auto_review_reason"] = (
                    "Close miss contained the configured wake phrase and was promoted to a positive sample."
                )
                metadata["auto_positive"] = True
                _write_sidecar_json(audio_path, metadata)
                _move_captured_audio(
                    file_name,
                    PERSONAL_DIR,
                    target_prefix="sample",
                    review_status="auto_approved_personal",
                )
                _record_auto_review_result(
                    file_name=file_name,
                    transcript=transcript,
                    result="promoted_close_miss",
                )
                return
            if config.get("delete_confirmed_wakes"):
                _remove_audio_with_sidecar(audio_path)
                _record_auto_review_result(
                    file_name=file_name,
                    transcript=transcript,
                    result="deleted_confirmed_wake",
                )
                return
            metadata["auto_review_status"] = "wake_phrase_detected"
            metadata["auto_review_reason"] = "Wake phrase found in transcript; left for manual positive review."
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(file_name=file_name, transcript=transcript, result="wake_phrase_detected")
            return

        if is_close_miss:
            metadata["auto_review_status"] = "close_miss_phrase_not_detected"
            metadata["auto_review_reason"] = (
                "Close miss did not contain the configured wake phrase; left for manual review."
            )
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(
                file_name=file_name,
                transcript=transcript,
                result="close_miss_phrase_not_detected",
            )
            return

        metadata["auto_review_status"] = "approved_negative"
        metadata["auto_review_reason"] = "Wake phrase was not found in the STT transcript."
        metadata["auto_negative"] = True
        _write_sidecar_json(audio_path, metadata)
        _move_captured_audio(
            file_name,
            NEGATIVE_DIR,
            target_prefix="negative",
            review_status="auto_approved_negative",
        )
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["pending_negative_count"] = int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0) + 1
            _save_auto_train_state_locked()
        _record_auto_review_result(file_name=file_name, transcript=transcript, result="approved_negative")
    except Exception as exc:
        error = str(exc)
        with contextlib.suppress(Exception):
            audio_path = _resolve_audio_path(CAPTURED_DIR, file_name)
            metadata = _load_sidecar_json(audio_path)
            metadata["auto_review_status"] = "error"
            metadata["auto_review_error"] = error
            metadata["auto_reviewed_at"] = _iso_now()
            _write_sidecar_json(audio_path, metadata)
        _record_auto_review_result(file_name=file_name, result="error", error=error)
    finally:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_RUNTIME["review_running"] = False
            AUTO_TRAIN_RUNTIME["review_file"] = ""


def _notify_tater_satellites() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        config = dict(AUTO_TRAIN_CONFIG)
    if not config.get("notify_satellites"):
        return {"ok": True, "skipped": True, "message": "Satellite notification is disabled."}

    endpoint = f"{str(config.get('tater_url') or '').rstrip('/')}/api/tater/satellite/v1/settings"
    body = json.dumps(
        {
            "selector": str(config.get("tater_selector") or ""),
            "settings": {},
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "microWakeWord-Trainer/auto-train"}
    token = str(config.get("tater_api_token") or "").strip()
    if token:
        headers["X-Tater-Token"] = token

    try:
        req = URLRequest(endpoint, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        push = payload.get("push") if isinstance(payload, dict) and isinstance(payload.get("push"), dict) else {}
        count = push.get("count")
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["last_notify_at"] = _iso_now()
            AUTO_TRAIN_STATE["last_notify_count"] = count
            AUTO_TRAIN_STATE["last_notify_error"] = ""
            _save_auto_train_state_locked()
        return {"ok": True, "count": count, "response": payload}
    except Exception as exc:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["last_notify_at"] = _iso_now()
            AUTO_TRAIN_STATE["last_notify_count"] = None
            AUTO_TRAIN_STATE["last_notify_error"] = str(exc)
            _save_auto_train_state_locked()
        return {"ok": False, "error": str(exc)}


def _start_auto_training() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        config = dict(AUTO_TRAIN_CONFIG)
    wake_phrase = str(config.get("wake_phrase") or "").strip()
    if not wake_phrase:
        return {"ok": False, "error": "Auto Training needs a wake phrase."}
    safe_word = safe_name(wake_phrase)
    language = str(config.get("language") or DEFAULT_LANGUAGE)
    with STATE_LOCK:
        if STATE["training"]["running"]:
            return {"ok": False, "error": "Training already running."}
        STATE["raw_phrase"] = wake_phrase
        STATE["safe_word"] = safe_word
        STATE["language"] = language
        STATE["training"]["running"] = True
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_STATE["last_train_started_at"] = _iso_now()
        AUTO_TRAIN_RUNTIME["training_pending_consumed"] = int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0)
        _save_auto_train_state_locked()
    threading.Thread(
        target=_run_training_background,
        args=(safe_word, language, True, True),
        daemon=True,
    ).start()
    return {"ok": True, "started": True, "safe_word": safe_word, "language": language}


def _maybe_run_scheduled_auto_training() -> None:
    with AUTO_TRAIN_LOCK:
        if not AUTO_TRAIN_CONFIG.get("enabled"):
            return
        schedule_hours = int(AUTO_TRAIN_CONFIG.get("schedule_hours") or 0)
        if schedule_hours <= 0:
            return
        next_run = _parse_iso_datetime(AUTO_TRAIN_STATE.get("next_run_at"))
        if next_run is None:
            _schedule_next_auto_run_locked()
            return
        now = _utc_now()
        if now < next_run:
            return
        pending = int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0)
        minimum = int(AUTO_TRAIN_CONFIG.get("minimum_new_negatives") or 1)
        if pending < minimum:
            _schedule_next_auto_run_locked(from_time=now)
            return
    result = _start_auto_training()
    with AUTO_TRAIN_LOCK:
        if result.get("started"):
            _schedule_next_auto_run_locked()
        else:
            AUTO_TRAIN_STATE["next_run_at"] = (_utc_now() + timedelta(minutes=10)).isoformat()
            _save_auto_train_state_locked()


def _auto_train_worker_loop() -> None:
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_RUNTIME["scheduler_running"] = True
    _queue_pending_auto_reviews()
    try:
        while not AUTO_TRAIN_STOP_EVENT.is_set():
            try:
                file_name = AUTO_TRAIN_REVIEW_QUEUE.get_nowait()
            except queue.Empty:
                file_name = ""
            if file_name:
                try:
                    _auto_review_capture(file_name)
                finally:
                    with AUTO_TRAIN_LOCK:
                        AUTO_TRAIN_QUEUED_FILES.discard(file_name)
                    AUTO_TRAIN_REVIEW_QUEUE.task_done()
            _maybe_run_scheduled_auto_training()
            AUTO_TRAIN_WAKE_EVENT.wait(1.0)
            AUTO_TRAIN_WAKE_EVENT.clear()
    finally:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_RUNTIME["scheduler_running"] = False


def _start_auto_train_worker() -> None:
    global AUTO_TRAIN_WORKER
    with AUTO_TRAIN_LOCK:
        if AUTO_TRAIN_WORKER is not None and AUTO_TRAIN_WORKER.is_alive():
            return
        AUTO_TRAIN_STOP_EVENT.clear()
        AUTO_TRAIN_WORKER = threading.Thread(
            target=_auto_train_worker_loop,
            name="auto-train-worker",
            daemon=True,
        )
        AUTO_TRAIN_WORKER.start()


def _stop_auto_train_worker() -> None:
    AUTO_TRAIN_STOP_EVENT.set()
    AUTO_TRAIN_WAKE_EVENT.set()



def _sync_personal_samples_state() -> List[str]:
    takes = _list_personal_samples()
    with STATE_LOCK:
        STATE["takes"] = takes
        STATE["takes_received"] = len(takes)
    return takes


def _registered_language_family(language: Dict[str, Any]) -> str:
    family = str(language.get("family") or "").strip().lower()
    if family:
        return family
    code = str(language.get("code") or "").strip()
    return code.split("_", 1)[0].lower() if code else ""


def _register_language(
    languages: Dict[str, Dict[str, Any]],
    *,
    family: str,
    name: str,
    region: str = "",
    count: int = 1,
):
    if not family:
        return
    entry = languages.setdefault(
        family,
        {
            "code": family,
            "label": f"{name} ({family})",
            "name": name,
            "voice_count": 0,
            "regions": [],
        },
    )
    entry["voice_count"] += count
    if region and region not in entry["regions"]:
        entry["regions"].append(region)


def _fetch_piper_catalog() -> Dict[str, Any] | None:
    req = URLRequest(
        PIPER_VOICES_INDEX_URL,
        headers={"User-Agent": "microWakeWord-Trainer/1.0"},
    )
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else None


def _read_cached_piper_catalog_file() -> Dict[str, Any] | None:
    try:
        if not PIPER_CATALOG_CACHE_FILE.exists():
            return None
        data = json.loads(PIPER_CATALOG_CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_cached_piper_catalog_file(data: Dict[str, Any]):
    try:
        PIPER_CATALOG_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PIPER_CATALOG_CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_piper_catalog() -> Dict[str, Any] | None:
    now = time.time()
    with PIPER_CATALOG_LOCK:
        cached = PIPER_CATALOG_CACHE.get("entries")
        fetched_at = float(PIPER_CATALOG_CACHE.get("fetched_at") or 0.0)
        if cached is not None and (now - fetched_at) < PIPER_CATALOG_CACHE_TTL_SECONDS:
            return cached

    disk_cached = _read_cached_piper_catalog_file()

    try:
        fresh = _fetch_piper_catalog()
    except Exception:
        fresh = None

    with PIPER_CATALOG_LOCK:
        if fresh is not None:
            PIPER_CATALOG_CACHE["entries"] = fresh
            PIPER_CATALOG_CACHE["fetched_at"] = now
            _write_cached_piper_catalog_file(fresh)
            return fresh
        if PIPER_CATALOG_CACHE.get("entries") is not None:
            return PIPER_CATALOG_CACHE.get("entries")
        if disk_cached is not None:
            PIPER_CATALOG_CACHE["entries"] = disk_cached
            PIPER_CATALOG_CACHE["fetched_at"] = now
            return disk_cached
        PIPER_CATALOG_CACHE["entries"] = {}
        PIPER_CATALOG_CACHE["fetched_at"] = now
        return PIPER_CATALOG_CACHE.get("entries")


def _available_languages() -> List[Dict[str, Any]]:
    languages: Dict[str, Dict[str, Any]] = {
        "en": {
            "code": "en",
            "label": "English (en)",
            "name": "English",
            "voice_count": 1,
            "regions": [],
        }
    }

    if PIPER_VOICES_DIR.exists():
        for meta_path in sorted(PIPER_VOICES_DIR.glob("*.onnx.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            language = data.get("language") or {}
            family = _registered_language_family(language)
            if not family or family == "en":
                continue

            name = str(language.get("name_english") or language.get("name_native") or family.upper()).strip()
            region = str(language.get("country_english") or language.get("region") or "").strip()
            _register_language(languages, family=family, name=name, region=region, count=1)

    catalog = _load_piper_catalog() or {}
    for entry in catalog.values():
        if not isinstance(entry, dict):
            continue
        language = entry.get("language") or {}
        family = _registered_language_family(language)
        if not family or family == "en":
            continue
        name = str(language.get("name_english") or language.get("name_native") or family.upper()).strip()
        region = str(language.get("country_english") or language.get("region") or "").strip()
        _register_language(languages, family=family, name=name, region=region, count=0)

    ordered = [languages["en"]]
    ordered.extend(
        sorted(
            (entry for code, entry in languages.items() if code != "en"),
            key=lambda entry: (entry["name"].lower(), entry["code"]),
        )
    )
    return ordered


def _normalize_language(language: str | None) -> str:
    requested = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE
    available_codes = {item["code"] for item in _available_languages()}
    if requested in available_codes:
        return requested
    if DEFAULT_LANGUAGE in available_codes:
        return DEFAULT_LANGUAGE
    return "en"


def _catalog_voice_files(language_family: str) -> List[tuple[str, str]]:
    if not language_family or language_family == "en":
        return []

    downloads: Dict[str, str] = {}
    catalog = _load_piper_catalog() or {}
    for entry in catalog.values():
        if not isinstance(entry, dict):
            continue
        language = entry.get("language") or {}
        family = _registered_language_family(language)
        if family != language_family:
            continue
        files = entry.get("files") or {}
        for rel_path in files.keys():
            if not isinstance(rel_path, str):
                continue
            if not (rel_path.endswith(".onnx") or rel_path.endswith(".onnx.json")):
                continue
            downloads[Path(rel_path).name] = f"{PIPER_VOICES_ROOT_URL}/{rel_path}?download=true"

    return sorted(downloads.items(), key=lambda item: item[0])


def _download_to_path(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = Request(url, headers={"User-Agent": "microWakeWord-Trainer/1.0"})
    with urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp_path.replace(dest_path)


def _ensure_non_english_language_voices(language_family: str, log) -> Dict[str, int]:
    downloads = _catalog_voice_files(language_family)
    local_voices = sorted(PIPER_VOICES_DIR.glob(f"{language_family}_*.onnx")) if PIPER_VOICES_DIR.exists() else []
    if not downloads:
        if local_voices:
            log(f"===== Piper Voices ({language_family}) =====")
            log(f"→ Using {len(local_voices)} installed voice(s) for language '{language_family}'")
            return {
                "downloaded_files": 0,
                "existing_files": len(local_voices),
                "voices": len(local_voices),
            }
        raise RuntimeError(
            f"No Piper ONNX voices found for language '{language_family}' in the upstream catalog."
        )

    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_files = 0
    existing_files = 0
    voice_names = sorted(name for name, _ in downloads if name.endswith(".onnx"))

    log(f"===== Piper Voices ({language_family}) =====")
    log(f"→ Ensuring {len(voice_names)} voice(s) for language '{language_family}'")

    for file_name, url in downloads:
        dest_path = PIPER_VOICES_DIR / file_name
        if dest_path.exists() and dest_path.stat().st_size > 0:
            existing_files += 1
            continue
        log(f"→ Downloading {file_name}")
        _download_to_path(url, dest_path)
        downloaded_files += 1

    log(
        f"✓ Piper voices ready for '{language_family}' "
        f"({downloaded_files} file(s) downloaded, {existing_files} already present)"
    )
    return {
        "downloaded_files": downloaded_files,
        "existing_files": existing_files,
        "voices": len(voice_names),
    }


def _find_ffmpeg() -> str | None:
    candidates = [
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _inspect_wav_bytes(data: bytes) -> Dict[str, Any] | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = (frames / rate) if rate else 0.0
            return {
                "container": "wav",
                "sample_rate": rate,
                "channels": wf.getnchannels(),
                "sample_width_bits": wf.getsampwidth() * 8,
                "compression": wf.getcomptype(),
                "frames": frames,
                "duration_s": round(duration, 3),
            }
    except Exception:
        return None


def _is_target_wav(info: Dict[str, Any] | None) -> bool:
    return bool(
        info
        and info.get("container") == "wav"
        and info.get("sample_rate") == TARGET_SAMPLE_RATE
        and info.get("channels") == TARGET_CHANNELS
        and info.get("sample_width_bits") == TARGET_SAMPLE_WIDTH_BYTES * 8
        and info.get("compression") == "NONE"
        and info.get("frames", 0) > 0
    )


def _next_personal_sample_name(original_name: str) -> str:
    return _next_directory_sample_name(PERSONAL_DIR, "sample", original_name)


def _next_negative_sample_name(original_name: str) -> str:
    return _next_directory_sample_name(NEGATIVE_DIR, "negative", original_name)


def _next_captured_sample_name(original_name: str) -> str:
    return _next_directory_sample_name(CAPTURED_DIR, "captured", original_name)


def _next_directory_sample_name(directory: Path, prefix: str, original_name: str) -> str:
    current = _list_audio_samples(directory)
    next_index = 1
    for name in current:
        match = re.match(rf"{re.escape(prefix)}_(\d{{4}})", name)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)

    stem = safe_name(Path(original_name or "sample").stem)
    suffix = f"_{stem[:32]}" if stem and stem != "wakeword" else ""
    return f"{prefix}_{next_index:04d}{suffix}.wav"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _parse_probability_history(value: Any) -> List[int]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    history: List[int] = []
    for raw_value in raw_values:
        parsed = _parse_int(raw_value)
        if parsed is not None:
            history.append(parsed)
    return history


def _audio_sidecar_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".json")


def _load_sidecar_json(audio_path: Path) -> Dict[str, Any]:
    sidecar = _audio_sidecar_path(audio_path)
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_sidecar_json(audio_path: Path, payload: Dict[str, Any]):
    _audio_sidecar_path(audio_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _remove_audio_with_sidecar(audio_path: Path):
    if audio_path.exists():
        audio_path.unlink()
    sidecar = _audio_sidecar_path(audio_path)
    if sidecar.exists():
        sidecar.unlink()


def _resolve_audio_path(directory: Path, file_name: str) -> Path:
    candidate = Path(file_name or "").name
    if not candidate or candidate != (file_name or "") or not candidate.endswith(".wav"):
        raise FileNotFoundError("Invalid audio file name.")
    path = (directory / candidate).resolve()
    if path.parent != directory.resolve() or not path.exists():
        raise FileNotFoundError("Audio file not found.")
    return path


def _format_hint_from_filename(original_name: str) -> Dict[str, Any]:
    suffix = (Path(original_name or "").suffix or "").lower().lstrip(".")
    return {
        "container": suffix or "unknown",
        "sample_rate": None,
        "channels": None,
        "sample_width_bits": None,
        "compression": None,
        "frames": None,
        "duration_s": None,
    }


def _normalize_audio_to_target_wav(data: bytes, original_name: str) -> bytes:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required to convert uploads that are not already 16 kHz mono 16-bit PCM WAV."
        )

    suffix = (Path(original_name or "").suffix or ".audio")
    with tempfile.TemporaryDirectory(prefix="mww_upload_") as tmpdir:
        src_path = Path(tmpdir) / f"source{suffix}"
        dst_path = Path(tmpdir) / "normalized.wav"
        src_path.write_bytes(data)

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src_path),
            "-vn",
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not dst_path.exists():
            err = (proc.stderr or proc.stdout or "ffmpeg conversion failed").strip()
            raise RuntimeError(err.splitlines()[-1] if err else "ffmpeg conversion failed")

        return dst_path.read_bytes()


def _boost_target_wav_bytes(
    data: bytes,
    *,
    target_peak_ratio: float = 0.88,
    target_rms_ratio: float | None = None,
    max_gain_ratio: float = 10.0,
    min_gain_ratio: float = 1.25,
    profile: str | None = None,
) -> tuple[bytes, Dict[str, Any]]:
    info = _inspect_wav_bytes(data) or {}
    if not _is_target_wav(info):
        return data, {"applied": False, "reason": "not_target_wav"}

    with wave.open(io.BytesIO(data), "rb") as wf:
        raw_frames = wf.readframes(wf.getnframes())

    if not raw_frames:
        return data, {"applied": False, "reason": "empty"}

    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()

    peak = max(abs(sample) for sample in samples) if samples else 0
    if peak <= 0:
        return data, {"applied": False, "reason": "silent", "peak_ratio": 0.0}

    peak_ratio = peak / 32767.0
    rms_ratio = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5 / 32767.0
    desired_peak = max(0.05, min(target_peak_ratio, 0.98))
    peak_limited_gain = desired_peak / peak_ratio
    target_gain = peak_limited_gain
    if target_rms_ratio is not None and rms_ratio > 0:
        target_gain = min(target_rms_ratio / rms_ratio, peak_limited_gain)
    gain_ratio = min(max_gain_ratio, target_gain)

    if gain_ratio < min_gain_ratio:
        return data, {
            "applied": False,
            "reason": "already_loud_enough",
            "peak_ratio": round(peak_ratio, 4),
            "rms_ratio": round(rms_ratio, 4),
            "gain_ratio": round(gain_ratio, 3),
            "gain_db": round(20.0 * log10(max(gain_ratio, 1e-9)), 2),
            "profile": profile or "",
        }

    boosted = array("h", (max(-32768, min(32767, int(round(sample * gain_ratio)))) for sample in samples))
    if sys.byteorder != "little":
        boosted.byteswap()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_SAMPLE_WIDTH_BYTES)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(boosted.tobytes())

    return buf.getvalue(), {
        "applied": True,
        "peak_ratio": round(peak_ratio, 4),
        "rms_ratio": round(rms_ratio, 4),
        "gain_ratio": round(gain_ratio, 3),
        "gain_db": round(20.0 * log10(max(gain_ratio, 1e-9)), 2),
        "profile": profile or "",
    }


def _build_audio_result_message(*, converted: bool, postprocess_info: Dict[str, Any] | None = None) -> str:
    message = (
        "Converted to 16 kHz mono 16-bit PCM WAV"
        if converted
        else "Already in the correct 16 kHz mono 16-bit PCM WAV format"
    )
    if postprocess_info and postprocess_info.get("applied"):
        message += f"; boosted {postprocess_info['gain_db']} dB for clearer captured playback"
    return message


def _ensure_captured_playback_ready(audio_path: Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    existing_postprocess = metadata.get("postprocess")
    if isinstance(existing_postprocess, dict) and existing_postprocess.get("profile") == CAPTURE_GAIN_PROFILE:
        return metadata

    with SAMPLES_LOCK:
        data = audio_path.read_bytes()
        final_bytes, postprocess_info = _boost_target_wav_bytes(
            data,
            target_peak_ratio=0.88,
            target_rms_ratio=0.06,
            max_gain_ratio=220.0,
            profile=CAPTURE_GAIN_PROFILE,
        )
        if postprocess_info.get("applied"):
            audio_path.write_bytes(final_bytes)
        if isinstance(existing_postprocess, dict):
            try:
                previous_gain = float(existing_postprocess.get("gain_ratio") or 1.0)
            except Exception:
                previous_gain = 1.0
            current_gain = float(postprocess_info.get("gain_ratio") or 1.0)
            total_gain = previous_gain * current_gain
            if previous_gain != 1.0:
                postprocess_info["gain_ratio"] = round(total_gain, 3)
                postprocess_info["gain_db"] = round(20.0 * log10(max(total_gain, 1e-9)), 2)
        metadata["postprocess"] = postprocess_info
        metadata["final_format"] = _inspect_wav_bytes(final_bytes) or metadata.get("final_format") or {}
        metadata["message"] = _build_audio_result_message(
            converted=bool(metadata.get("converted")),
            postprocess_info=postprocess_info,
        )
        _write_sidecar_json(audio_path, metadata)

    return metadata


def _save_audio_sample(
    data: bytes,
    original_name: str,
    *,
    target_dir: Path,
    out_name: str,
    postprocess_target_wav: Callable[[bytes], tuple[bytes, Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    if not data:
        raise ValueError("Empty or invalid audio file.")

    original_info = _inspect_wav_bytes(data) or _format_hint_from_filename(original_name)
    normalized = _is_target_wav(original_info)
    final_bytes = data if normalized else _normalize_audio_to_target_wav(data, original_name)
    postprocess_info: Dict[str, Any] = {"applied": False}
    if postprocess_target_wav is not None:
        final_bytes, postprocess_info = postprocess_target_wav(final_bytes)
    final_info = _inspect_wav_bytes(final_bytes)

    if not _is_target_wav(final_info):
        raise ValueError("Uploaded audio could not be normalized to 16 kHz mono 16-bit PCM WAV.")

    with SAMPLES_LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
        final_name = out_name
        out_path = target_dir / final_name
        out_path.write_bytes(final_bytes)

    return {
        "saved_as": final_name,
        "converted": not normalized,
        "postprocess": postprocess_info,
        "original_name": original_name or final_name,
        "detected_format": original_info,
        "final_format": final_info,
        "message": _build_audio_result_message(
            converted=not normalized,
            postprocess_info=postprocess_info,
        ),
    }


def _save_personal_sample(data: bytes, original_name: str, out_name: str | None = None) -> Dict[str, Any]:
    return _save_audio_sample(
        data,
        original_name,
        target_dir=PERSONAL_DIR,
        out_name=out_name or _next_personal_sample_name(original_name),
    )


def _save_captured_sample(data: bytes, original_name: str, out_name: str | None = None) -> Dict[str, Any]:
    return _save_audio_sample(
        data,
        original_name,
        target_dir=CAPTURED_DIR,
        out_name=out_name or _next_captured_sample_name(original_name),
        postprocess_target_wav=lambda wav_data: _boost_target_wav_bytes(
            wav_data,
            target_peak_ratio=0.88,
            target_rms_ratio=0.06,
            max_gain_ratio=220.0,
            profile=CAPTURE_GAIN_PROFILE,
        ),
    )


def _pcm_s16le_to_wav_bytes(
    pcm_data: bytes,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    sample_width_bytes: int = TARGET_SAMPLE_WIDTH_BYTES,
) -> bytes:
    if not pcm_data:
        raise ValueError("Captured audio payload was empty.")

    if sample_width_bytes <= 0:
        raise ValueError("Invalid sample width for PCM conversion.")

    frame_width = channels * sample_width_bytes
    if frame_width <= 0 or (len(pcm_data) % frame_width) != 0:
        raise ValueError("Captured PCM payload does not align to whole audio frames.")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width_bytes)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buf.getvalue()


def _captured_item_from_path(audio_path: Path) -> Dict[str, Any]:
    meta = _ensure_captured_playback_ready(audio_path, _load_sidecar_json(audio_path))
    stat = audio_path.stat()
    event_type = str(meta.get("event_type") or "captured").strip() or "captured"
    final_format = meta.get("final_format") or _inspect_wav_bytes(audio_path.read_bytes()) or {}
    return {
        "saved_as": audio_path.name,
        "original_name": meta.get("original_name") or audio_path.name,
        "source_device": meta.get("source_device") or "",
        "wake_word": meta.get("wake_word") or "",
        "event_type": event_type,
        "capture_label": str(meta.get("capture_label") or event_type.replace("_", " ").title()),
        "received_at": meta.get("received_at") or datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "captured_at": meta.get("captured_at") or "",
        "converted": bool(meta.get("converted")),
        "blocked_by_vad": bool(meta.get("blocked_by_vad")),
        "max_probability": meta.get("max_probability"),
        "average_probability": meta.get("average_probability"),
        "probability_cutoff": meta.get("probability_cutoff"),
        "peak_probability_cutoff": meta.get("peak_probability_cutoff"),
        "active_window_count": meta.get("active_window_count"),
        "min_active_windows": meta.get("min_active_windows"),
        "rise_score": meta.get("rise_score"),
        "vad_max_probability": meta.get("vad_max_probability"),
        "vad_average_probability": meta.get("vad_average_probability"),
        "detection_profile": meta.get("detection_profile") or "",
        "probability_history": meta.get("probability_history") or [],
        "detected_format": meta.get("detected_format") or {},
        "final_format": final_format,
        "postprocess": meta.get("postprocess") or {},
        "message": meta.get("message") or "",
        "notes": meta.get("notes") or "",
        "review_status": meta.get("review_status") or "pending",
        "transcript": meta.get("transcript") or "",
        "transcribed_at": meta.get("transcribed_at") or "",
        "auto_review_status": meta.get("auto_review_status") or "",
        "auto_review_reason": meta.get("auto_review_reason") or "",
        "auto_review_error": meta.get("auto_review_error") or "",
        "size_bytes": stat.st_size,
        "audio_url": f"/api/audio/captured/{audio_path.name}",
    }


def _list_captured_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
    for audio_path in sorted(CAPTURED_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(_captured_item_from_path(audio_path))
        except Exception:
            continue
    return items


def _sample_item_from_path(audio_path: Path, bucket: str) -> Dict[str, Any]:
    meta = _load_sidecar_json(audio_path)
    stat = audio_path.stat()
    final_format = meta.get("final_format") or meta.get("detected_format") or _inspect_wav_bytes(audio_path.read_bytes()) or {}
    return {
        "bucket": bucket,
        "saved_as": audio_path.name,
        "original_name": meta.get("original_name") or audio_path.name,
        "wake_word": meta.get("wake_word") or "",
        "event_type": meta.get("event_type") or "",
        "review_status": meta.get("review_status") or "",
        "received_at": meta.get("received_at") or "",
        "reviewed_at": meta.get("reviewed_at") or "",
        "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "converted": bool(meta.get("converted")),
        "trimmed": bool(meta.get("trimmed")),
        "source_file": meta.get("source_file") or "",
        "final_format": final_format,
        "message": meta.get("message") or "",
        "transcript": meta.get("transcript") or "",
        "transcribed_at": meta.get("transcribed_at") or "",
        "auto_negative": bool(meta.get("auto_negative")),
        "auto_positive": bool(meta.get("auto_positive")),
        "auto_review_reason": meta.get("auto_review_reason") or "",
        "size_bytes": stat.st_size,
        "audio_url": f"/api/audio/{bucket}/{audio_path.name}",
    }


def _list_sample_items(directory: Path, bucket: str) -> List[Dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for audio_path in sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(_sample_item_from_path(audio_path, bucket))
        except Exception:
            continue
    # Untrimmed first (stable sort preserves mtime order within each group).
    items.sort(key=lambda x: x.get("trimmed", False))
    return items

def _samples_payload() -> Dict[str, Any]:
    takes = _sync_personal_samples_state()
    personal_items = _list_sample_items(PERSONAL_DIR, "personal")
    negative_items = _list_sample_items(NEGATIVE_DIR, "negative")
    return {
        "ok": True,
        "personal": personal_items,
        "negative": negative_items,
        "personal_count": len(personal_items),
        "negative_count": len(negative_items),
        "takes_received": len(takes),
    }


def _move_captured_audio(file_name: str, target_dir: Path, *, target_prefix: str, review_status: str) -> Dict[str, Any]:
    with SAMPLES_LOCK:
        src_path = _resolve_audio_path(CAPTURED_DIR, file_name)
        metadata = _load_sidecar_json(src_path)
        original_name = str(metadata.get("original_name") or src_path.name)
        if target_prefix == "sample":
            target_name = _next_personal_sample_name(original_name)
        else:
            target_name = _next_negative_sample_name(original_name)

        target_dir.mkdir(parents=True, exist_ok=True)
        dst_path = target_dir / target_name
        src_path.replace(dst_path)

        metadata["review_status"] = review_status
        metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["saved_as"] = target_name
        _write_sidecar_json(dst_path, metadata)

        stale_sidecar = _audio_sidecar_path(src_path)
        if stale_sidecar.exists():
            stale_sidecar.unlink()

    takes = _sync_personal_samples_state()
    return {
        "saved_as": target_name,
        "captured_remaining": len(_list_captured_sample_names()),
        "negative_count": len(_list_negative_samples()),
        "takes_received": len(takes),
    }


def _append_train_log(line: str):
    line = (line or "").rstrip("\n")
    with STATE_LOCK:
        buf: List[str] = STATE["training"]["log_lines"]
        buf.append(line)
        if len(buf) > 250:
            del buf[: (len(buf) - 250)]


def _clear_training_log():
    log_path = DATA_DIR / "recorder_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("================================================================================\n")
        lf.write("===== New trainer session started =====\n")
        lf.write("================================================================================\n")
        lf.flush()

    with STATE_LOCK:
        STATE["training"]["log_path"] = str(log_path)
        STATE["training"]["log_lines"] = []
        STATE["training"]["last_sent_tail"] = []
        STATE["training"]["last_log_size"] = 0


def _title_from_phrase(raw_phrase: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", raw_phrase or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.title() if s else ""


def _run_streamed(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    header: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    if header:
        _append_train_log(header)

    _append_train_log("→ " + " ".join(cmd))

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n" + ("=" * 80) + "\n")
        if header:
            lf.write(header + "\n")
        lf.write("→ " + " ".join(cmd) + "\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            _append_train_log(line)

        return proc.wait()


def _ensure_training_venv(log_path: Path) -> None:
    activate = DATA_DIR / ".venv" / "bin" / "activate"
    if activate.exists():
        _append_train_log("✅ Training venv found (skipping setup_python_venv)")
        return

    setup = CLI_DIR / "setup_python_venv"
    if not setup.exists():
        raise RuntimeError(f"Missing setup_python_venv at: {setup}")

    rc = _run_streamed(
        ["bash", "-lc", f"cd '{DATA_DIR}' && '{setup}' --data-dir='{DATA_DIR}'"],
        cwd=DATA_DIR,
        log_path=log_path,
        header="===== Ensuring Python venv (/data/.venv) =====",
    )

    if rc != 0:
        raise RuntimeError(f"setup_python_venv failed (exit_code={rc})")

    if not activate.exists():
        raise RuntimeError(f"setup_python_venv finished, but {activate} is still missing")


def _ensure_training_datasets(log_path: Path) -> None:
    setup = CLI_DIR / "setup_training_datasets"
    if not setup.exists():
        raise RuntimeError(f"Missing setup_training_datasets at: {setup}")

    cleanup_arch = "true" if DATASET_CLEANUP_ARCHIVES else "false"
    cleanup_inter = "true" if DATASET_CLEANUP_INTERMEDIATE else "false"

    cmd = [
        "bash",
        "-lc",
        (
            f"cd '{DATA_DIR}' && "
            f"'{setup}' "
            f"--cleanup-archives='{cleanup_arch}' "
            f"--cleanup-intermediate-files='{cleanup_inter}' "
            f"--data-dir='{DATA_DIR}'"
        ),
    ]

    rc = _run_streamed(
        cmd,
        cwd=DATA_DIR,
        log_path=log_path,
        header="===== Ensuring training datasets (setup_training_datasets) =====",
    )

    if rc != 0:
        raise RuntimeError(f"setup_training_datasets failed (exit_code={rc})")


def _read_tail_lines(log_path: Path, max_lines: int) -> List[str]:
    if not log_path.exists():
        return []

    try:
        size = log_path.stat().st_size
        start = max(0, size - TRAIN_LOG_MAX_BYTES)
        with open(log_path, "rb") as f:
            f.seek(start)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return lines
        return lines[-max_lines:]
    except Exception:
        return []


def _compute_new_lines(prev_tail: List[str], new_tail: List[str]) -> List[str]:
    if not prev_tail:
        return new_tail

    max_k = min(len(prev_tail), len(new_tail))
    for k in range(max_k, 0, -1):
        if prev_tail[-k:] == new_tail[:k]:
            return new_tail[k:]

    return new_tail


def _find_latest_output_pair(output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not output_dir.exists():
        return (None, None)

    tflites = sorted(output_dir.rglob("*.tflite"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tflites:
        return (None, None)

    tfl = tflites[0]
    js = tfl.with_suffix(".json")
    if js.exists():
        return (tfl, js)

    jsons = sorted(output_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (tfl, jsons[0] if jsons else None)


def _deep_replace_strings(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_deep_replace_strings(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_replace_strings(v, old, new) for k, v in obj.items()}
    return obj


def _normalize_output_artifacts(safe_word: str, log_path: Path) -> None:
    output_root = DATA_DIR / "output"
    tfl, js = _find_latest_output_pair(output_root)

    if not tfl:
        _append_train_log(f"⚠️ No .tflite found in {output_root}")
        return

    new_tfl = tfl.parent / f"{safe_word}.tflite"
    new_js = tfl.parent / f"{safe_word}.json"
    old_tfl_name = tfl.name

    if tfl.resolve() != new_tfl.resolve():
        if new_tfl.exists():
            backup = new_tfl.with_name(f"{new_tfl.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.tflite")
            shutil.move(str(new_tfl), str(backup))
            _append_train_log(f"↪️ Backed up existing {new_tfl.name} → {backup.name}")
        shutil.move(str(tfl), str(new_tfl))
        _append_train_log(f"✅ Renamed model: {old_tfl_name} → {new_tfl.name}")

    if js and js.exists():
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            data = None

        if js.resolve() != new_js.resolve():
            if new_js.exists():
                backup = new_js.with_name(f"{new_js.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.json")
                shutil.move(str(new_js), str(backup))
                _append_train_log(f"↪️ Backed up existing {new_js.name} → {backup.name}")
            shutil.move(str(js), str(new_js))
            _append_train_log(f"✅ Renamed metadata: {js.name} → {new_js.name}")

        if data is not None:
            patched = _deep_replace_strings(data, old_tfl_name, new_tfl.name)
            for key in ("model", "model_file", "model_filename", "tflite", "tflite_file", "tflite_filename"):
                if isinstance(patched, dict) and key in patched and isinstance(patched[key], str):
                    patched[key] = new_tfl.name
            new_js.write_text(json.dumps(patched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _append_train_log(f"✅ Patched JSON to reference: {new_tfl.name}")
    else:
        _append_train_log("⚠️ No .json found to patch (model renamed only)")

    _sync_trained_wake_word_artifacts()
    _append_train_log(f"✅ Trained wake words synced to {TRAINED_WAKE_WORDS_DIR}")


def _run_training_background(
    safe_word: str,
    language: str,
    allow_no_personal: bool,
    auto_run: bool = False,
):
    language = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE
    rc = 999
    with STATE_LOCK:
        raw_phrase = STATE.get("raw_phrase") or ""

    wake_word_title = _title_from_phrase(raw_phrase)

    with STATE_LOCK:
        STATE["training"]["running"] = True
        STATE["training"]["exit_code"] = None
        STATE["training"]["log_lines"] = []
        STATE["training"]["safe_word"] = safe_word
        STATE["training"]["last_sent_tail"] = []
        STATE["training"]["last_log_size"] = 0
        log_path = Path(str(DATA_DIR / "recorder_training.log"))
        STATE["training"]["log_path"] = str(log_path)

    _append_train_log("================================================================================")
    _append_train_log("===== Nvidia Docker Training Run =====")
    _append_train_log("================================================================================")

    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n" + ("=" * 80) + "\n")
            lf.write("===== Nvidia Docker Training Run =====\n")
            lf.write(("=" * 80) + "\n")
            lf.flush()
    except Exception:
        pass

    try:
        _ensure_training_venv(log_path)
        _ensure_training_datasets(log_path)
        if language != "en":
            _ensure_non_english_language_voices(language, _append_train_log)

        if wake_word_title:
            cmd_str = f"{TRAIN_CMD} --language='{language}' '{safe_word}' '{wake_word_title}'"
        else:
            cmd_str = f"{TRAIN_CMD} --language='{language}' '{safe_word}'"

        env = os.environ.copy()
        env["MWW_ALLOW_NO_PERSONAL"] = "true" if allow_no_personal else "false"

        _append_train_log("===== Training (train_wake_word) =====")
        _append_train_log(f"→ Running: {cmd_str}")

        with open(log_path, "a", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd_str],
                cwd=str(DATA_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                _append_train_log(line)

            rc = proc.wait()

        _append_train_log(f"✓ Training finished (exit_code={rc})")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = rc

        if rc == 0:
            _normalize_output_artifacts(safe_word, log_path)

    except Exception as e:
        rc = 999
        _append_train_log(f"✗ Training crashed: {e!r}")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = 999

    finally:
        with STATE_LOCK:
            STATE["training"]["running"] = False

    if auto_run:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["last_train_finished_at"] = _iso_now()
            AUTO_TRAIN_STATE["last_train_exit_code"] = rc
            if rc == 0:
                consumed = int(AUTO_TRAIN_RUNTIME.get("training_pending_consumed") or 0)
                AUTO_TRAIN_STATE["pending_negative_count"] = max(
                    0,
                    int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0) - consumed,
                )
            AUTO_TRAIN_RUNTIME["training_pending_consumed"] = 0
            _save_auto_train_state_locked()
        if rc == 0:
            _append_train_log("→ Asking Tater to refresh the active wake model on connected satellites")
            notify_result = _notify_tater_satellites()
            if notify_result.get("ok"):
                if notify_result.get("skipped"):
                    _append_train_log("→ Satellite refresh skipped (disabled in Auto Training)")
                else:
                    count = notify_result.get("count")
                    suffix = f" ({count} connected)" if count is not None else ""
                    _append_train_log(f"✓ Tater satellite refresh requested{suffix}")
            else:
                _append_train_log(f"✗ Tater satellite refresh failed: {notify_result.get('error')}")


# -------------------- Routes --------------------
@app.on_event("startup")
def start_auto_train_worker_event():
    _start_auto_train_worker()


@app.on_event("shutdown")
def stop_auto_train_worker_event():
    _stop_auto_train_worker()


@app.get("/api/auto_train")
def auto_train_status(request: Request):
    payload = _auto_train_status_payload()
    payload["ok"] = True
    payload["advertised_base_url"] = _advertised_base_url(request)
    payload["stt_backend"] = "faster-whisper"
    return payload


@app.put("/api/auto_train")
def update_auto_train(payload: Dict[str, Any] = None):
    incoming = dict(payload or {})
    with AUTO_TRAIN_LOCK:
        previous = dict(AUTO_TRAIN_CONFIG)
        if incoming.pop("clear_tater_api_token", False):
            incoming["tater_api_token"] = ""
        elif not str(incoming.get("tater_api_token") or "").strip():
            incoming.pop("tater_api_token", None)
        try:
            normalized = _normalize_auto_train_config(incoming, base=previous)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        if normalized["enabled"] and not normalized["wake_phrase"]:
            return JSONResponse(
                {"ok": False, "error": "Enter the wake phrase before enabling Auto Training."},
                status_code=400,
            )
        AUTO_TRAIN_CONFIG.clear()
        AUTO_TRAIN_CONFIG.update(normalized)
        _save_auto_train_config_locked()
        schedule_changed = (
            previous.get("enabled") != normalized.get("enabled")
            or previous.get("schedule_hours") != normalized.get("schedule_hours")
        )
        if schedule_changed or not AUTO_TRAIN_STATE.get("next_run_at"):
            _schedule_next_auto_run_locked()
    if normalized["enabled"]:
        queued = _queue_pending_auto_reviews()
        AUTO_TRAIN_WAKE_EVENT.set()
    else:
        queued = 0
    return {"ok": True, "queued": queued, **_auto_train_status_payload()}


@app.post("/api/auto_train/action")
def auto_train_action(payload: Dict[str, Any] = None):
    action = str((payload or {}).get("action") or "").strip().lower()
    if action == "review_now":
        with AUTO_TRAIN_LOCK:
            if not AUTO_TRAIN_CONFIG.get("enabled"):
                return JSONResponse({"ok": False, "error": "Enable Auto Training first."}, status_code=400)
        queued = _queue_pending_auto_reviews(force=True)
        AUTO_TRAIN_WAKE_EVENT.set()
        return {"ok": True, "queued": queued, **_auto_train_status_payload()}
    if action == "train_now":
        result = _start_auto_training()
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return {**result, **_auto_train_status_payload()}
    if action == "notify_now":
        result = _notify_tater_satellites()
        if not result.get("ok"):
            return JSONResponse(result, status_code=502)
        return {**result, **_auto_train_status_payload()}
    return JSONResponse({"ok": False, "error": "Unknown Auto Training action."}, status_code=400)



@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h3>Missing UI</h3><p>Create <code>static/index.html</code>.</p>",
            status_code=500,
        )
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/api/start_session")
def start_session(payload: Dict[str, Any]):
    raw = (payload.get("phrase") or "").strip()
    if not raw:
        return JSONResponse({"ok": False, "error": "phrase is required"}, status_code=400)

    safe = safe_name(raw)

    speakers_total = int(payload.get("speakers_total") or SPEAKERS_TOTAL_DEFAULT)
    takes_per_speaker = int(payload.get("takes_per_speaker") or TAKES_PER_SPEAKER_DEFAULT)
    language = _normalize_language(payload.get("language"))
    available_languages = _available_languages()

    speakers_total = max(1, min(10, speakers_total))
    takes_per_speaker = max(1, min(50, takes_per_speaker))

    with STATE_LOCK:
        STATE["raw_phrase"] = raw
        STATE["safe_word"] = safe
        STATE["language"] = language
        STATE["speakers_total"] = speakers_total
        STATE["takes_per_speaker"] = takes_per_speaker
        # do not interrupt training if running
    takes = _sync_personal_samples_state()

    _clear_training_log()

    return {
        "ok": True,
        "raw_phrase": raw,
        "safe_word": safe,
        "language": language,
        "speakers_total": speakers_total,
        "takes_per_speaker": takes_per_speaker,
        "takes_total": speakers_total * takes_per_speaker,
        "takes_received": len(takes),
        "takes": takes,
        "available_languages": available_languages,
        "personal_dir": str(PERSONAL_DIR),
        "data_dir": str(DATA_DIR),
    }


@app.get("/api/session")
def get_session():
    takes = _sync_personal_samples_state()
    available_languages = _available_languages()
    with STATE_LOCK:
        current_language = _normalize_language(STATE["language"])
        STATE["language"] = current_language
        return {
            "ok": True,
            "raw_phrase": STATE["raw_phrase"],
            "safe_word": STATE["safe_word"],
            "language": current_language,
            "speakers_total": STATE["speakers_total"],
            "takes_per_speaker": STATE["takes_per_speaker"],
            "takes_received": len(takes),
            "takes": list(takes),
            "training": dict(STATE["training"]),
            "available_languages": available_languages,
        }


@app.post("/api/upload_take")
async def upload_take(
    speaker_index: int = Form(...),
    take_index: int = Form(...),
    file: UploadFile = File(...),
):
    with STATE_LOCK:
        safe_word = STATE["safe_word"]
        speakers_total = int(STATE["speakers_total"])
        takes_per_speaker = int(STATE["takes_per_speaker"])

    if not safe_word:
        return JSONResponse({"ok": False, "error": "No active session. Call /api/start_session first."}, status_code=400)

    if speaker_index < 1 or speaker_index > speakers_total:
        return JSONResponse({"ok": False, "error": f"speaker_index must be 1..{speakers_total}"}, status_code=400)

    if take_index < 1 or take_index > takes_per_speaker:
        return JSONResponse({"ok": False, "error": f"take_index must be 1..{takes_per_speaker}"}, status_code=400)

    out_name = f"speaker{speaker_index:02d}_take{take_index:02d}.wav"

    data = await file.read()
    try:
        result = _save_personal_sample(data, file.filename or out_name, out_name=out_name)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    takes = _sync_personal_samples_state()
    return {"ok": True, **result, "takes_received": len(takes)}


@app.post("/api/upload_personal_sample")
async def upload_personal_sample(file: UploadFile = File(...)):
    with STATE_LOCK:
        safe_word = STATE["safe_word"]

    if not safe_word:
        return JSONResponse({"ok": False, "error": "No active session. Call /api/start_session first."}, status_code=400)

    data = await file.read()
    try:
        result = _save_personal_sample(data, file.filename or "sample")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    takes = _sync_personal_samples_state()
    return {"ok": True, **result, "takes_received": len(takes)}


@app.post("/api/upload_captured_audio")
async def upload_captured_audio(
    file: UploadFile = File(...),
    source_device: str | None = Form(None),
    wake_word: str | None = Form(None),
    event_type: str | None = Form(None),
    captured_at: str | None = Form(None),
    blocked_by_vad: str | None = Form(None),
    max_probability: str | None = Form(None),
    average_probability: str | None = Form(None),
    notes: str | None = Form(None),
    metadata_json: str | None = Form(None),
):
    data = await file.read()
    try:
        result = _save_captured_sample(data, file.filename or "captured")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    extra_meta: Dict[str, Any] = {}
    if metadata_json:
        try:
            parsed = json.loads(metadata_json)
            if isinstance(parsed, dict):
                extra_meta = parsed
        except Exception:
            return JSONResponse({"ok": False, "error": "metadata_json must be a JSON object"}, status_code=400)

    with STATE_LOCK:
        current_safe_word = STATE.get("safe_word")

    audio_path = CAPTURED_DIR / result["saved_as"]
    sidecar = {
        **extra_meta,
        "saved_as": result["saved_as"],
        "original_name": result["original_name"],
        "source_device": source_device or extra_meta.get("source_device") or "",
        "wake_word": wake_word or extra_meta.get("wake_word") or current_safe_word or "",
        "event_type": (event_type or extra_meta.get("event_type") or "captured").strip() or "captured",
        "capture_label": extra_meta.get("capture_label") or "",
        "captured_at": captured_at or extra_meta.get("captured_at") or "",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "blocked_by_vad": _parse_bool(extra_meta.get("blocked_by_vad") if blocked_by_vad is None else blocked_by_vad),
        "max_probability": _parse_float(extra_meta.get("max_probability") if max_probability is None else max_probability),
        "average_probability": _parse_float(
            extra_meta.get("average_probability") if average_probability is None else average_probability
        ),
        "probability_cutoff": _parse_int(extra_meta.get("probability_cutoff")),
        "peak_probability_cutoff": _parse_int(extra_meta.get("peak_probability_cutoff")),
        "active_window_count": _parse_int(extra_meta.get("active_window_count")),
        "min_active_windows": _parse_int(extra_meta.get("min_active_windows")),
        "rise_score": _parse_int(extra_meta.get("rise_score")),
        "vad_max_probability": _parse_int(extra_meta.get("vad_max_probability")),
        "vad_average_probability": _parse_int(extra_meta.get("vad_average_probability")),
        "detection_profile": str(extra_meta.get("detection_profile") or "").strip(),
        "probability_history": _parse_probability_history(extra_meta.get("probability_history")),
        "notes": notes or extra_meta.get("notes") or "",
        "converted": result["converted"],
        "detected_format": result["detected_format"],
        "final_format": result["final_format"],
        "postprocess": result["postprocess"],
        "message": result["message"],
        "review_status": "pending",
    }
    _write_sidecar_json(audio_path, sidecar)
    with AUTO_TRAIN_LOCK:
        auto_review_config = dict(AUTO_TRAIN_CONFIG)
    if auto_review_config.get("enabled") and _captured_event_is_auto_reviewable(sidecar, auto_review_config):
        _queue_auto_review(audio_path.name)

    return {
        "ok": True,
        "item": _captured_item_from_path(audio_path),
        "captured_count": len(_list_captured_sample_names()),
    }


@app.post("/api/upload_captured_audio_raw")
async def upload_captured_audio_raw(
    request: Request,
    x_audio_format: str | None = Header(default=None),
    x_original_name: str | None = Header(default=None),
    x_source_device: str | None = Header(default=None),
    x_wake_word: str | None = Header(default=None),
    x_event_type: str | None = Header(default=None),
    x_captured_at: str | None = Header(default=None),
    x_blocked_by_vad: str | None = Header(default=None),
    x_max_probability: str | None = Header(default=None),
    x_average_probability: str | None = Header(default=None),
    x_probability_cutoff: str | None = Header(default=None),
    x_peak_probability_cutoff: str | None = Header(default=None),
    x_active_windows: str | None = Header(default=None),
    x_min_active_windows: str | None = Header(default=None),
    x_rise_score: str | None = Header(default=None),
    x_vad_max_probability: str | None = Header(default=None),
    x_vad_average_probability: str | None = Header(default=None),
    x_detection_profile: str | None = Header(default=None),
    x_probability_history: str | None = Header(default=None),
    x_notes: str | None = Header(default=None),
):
    raw_data = await request.body()
    audio_format = (x_audio_format or "wav").strip().lower()

    try:
        if audio_format == "pcm_s16le":
            data = _pcm_s16le_to_wav_bytes(raw_data)
            original_name = x_original_name or "captured.raw.wav"
        elif audio_format in {"wav", "audio/wav", "audio/x-wav"}:
            data = raw_data
            original_name = x_original_name or "captured.wav"
        else:
            raise ValueError(f"Unsupported x-audio-format '{audio_format}'.")

        result = _save_captured_sample(data, original_name)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    with STATE_LOCK:
        current_safe_word = STATE.get("safe_word")

    audio_path = CAPTURED_DIR / result["saved_as"]
    sidecar = {
        "saved_as": result["saved_as"],
        "original_name": result["original_name"],
        "source_device": x_source_device or "",
        "wake_word": x_wake_word or current_safe_word or "",
        "event_type": (x_event_type or "captured").strip() or "captured",
        "capture_label": "",
        "captured_at": x_captured_at or "",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "blocked_by_vad": _parse_bool(x_blocked_by_vad),
        "max_probability": _parse_float(x_max_probability),
        "average_probability": _parse_float(x_average_probability),
        "probability_cutoff": _parse_int(x_probability_cutoff),
        "peak_probability_cutoff": _parse_int(x_peak_probability_cutoff),
        "active_window_count": _parse_int(x_active_windows),
        "min_active_windows": _parse_int(x_min_active_windows),
        "rise_score": _parse_int(x_rise_score),
        "vad_max_probability": _parse_int(x_vad_max_probability),
        "vad_average_probability": _parse_int(x_vad_average_probability),
        "detection_profile": (x_detection_profile or "").strip(),
        "probability_history": _parse_probability_history(x_probability_history),
        "notes": x_notes or "",
        "converted": result["converted"],
        "detected_format": result["detected_format"],
        "final_format": result["final_format"],
        "postprocess": result["postprocess"],
        "message": result["message"],
        "review_status": "pending",
    }
    _write_sidecar_json(audio_path, sidecar)
    with AUTO_TRAIN_LOCK:
        auto_review_config = dict(AUTO_TRAIN_CONFIG)
    if auto_review_config.get("enabled") and _captured_event_is_auto_reviewable(sidecar, auto_review_config):
        _queue_auto_review(audio_path.name)

    return {
        "ok": True,
        "item": _captured_item_from_path(audio_path),
        "captured_count": len(_list_captured_sample_names()),
    }


@app.get("/api/captured_audio")
def captured_audio():
    takes = _sync_personal_samples_state()
    items = _list_captured_items()
    samples = _samples_payload()
    return {
        "ok": True,
        "items": items,
        "captured_count": len(items),
        "negative_count": samples["negative_count"],
        "personal_count": len(takes),
    }


@app.get("/api/samples")
def samples():
    return _samples_payload()


@app.get("/api/audio/{bucket}/{file_name}")
def audio_file(bucket: str, file_name: str):
    bucket_map = {
        "captured": CAPTURED_DIR,
        "personal": PERSONAL_DIR,
        "negative": NEGATIVE_DIR,
    }
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown audio bucket."}, status_code=404)
    try:
        path = _resolve_audio_path(directory, file_name)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    if bucket == "captured":
        _ensure_captured_playback_ready(path, _load_sidecar_json(path))
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.delete("/api/samples/{bucket}/{file_name}")
def delete_sample(bucket: str, file_name: str):
    bucket_map = {
        "personal": PERSONAL_DIR,
        "negative": NEGATIVE_DIR,
    }
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)
    try:
        path = _resolve_audio_path(directory, file_name)
        metadata = _load_sidecar_json(path)
        _remove_audio_with_sidecar(path)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    if bucket == "negative" and metadata.get("auto_negative"):
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["pending_negative_count"] = max(
                0,
                int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0) - 1,
            )
            _save_auto_train_state_locked()
    return {"ok": True, "deleted_bucket": bucket, "deleted_file": file_name, "message": f"Deleted {file_name}"}


@app.post("/api/samples/{bucket}/{file_name}/vad")
def vad_segments(bucket: str, file_name: str):
    bucket_map = {"personal": PERSONAL_DIR, "negative": NEGATIVE_DIR}
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)
    try:
        path = _resolve_audio_path(directory, file_name)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)

    wav_bytes = path.read_bytes()
    try:
        all_segments = _detect_speech_segments(wav_bytes)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"VAD failed: {str(e)}"}, status_code=500)

    # Only return the first segment longer than 250 ms. Add deterministic
    # padding so VAD guides trimming without clipping quiet wake-word edges.
    filtered = [s for s in all_segments if (s["end"] - s["start"]) >= 0.25]
    if not filtered:
        return {"ok": True, "file_name": file_name, "segments": [], "segment_count": 0}
    seg = filtered[0]
    info = _inspect_wav_bytes(wav_bytes) or {}
    duration_s = float(info.get("duration_s") or 0.0)
    start = max(0.0, round(seg["start"] - VAD_SELECTION_PAD_START_S, 3))
    end = round(seg["end"] + VAD_SELECTION_PAD_END_S, 3)
    if duration_s > 0:
        end = min(duration_s, end)
    if end <= start:
        end = start + 0.001
    segment = {"start": start, "end": end}
    return {"ok": True, "file_name": file_name, "segments": [segment], "segment_count": 1}


@app.post("/api/samples/trim")
async def trim_sample_upload(
    file: UploadFile = File(...),
    bucket: str = Form(...),
    source_file: str = Form(...),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
):
    bucket_map = {"personal": PERSONAL_DIR, "negative": NEGATIVE_DIR}
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)

    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "Empty audio file."}, status_code=400)

    info = _inspect_wav_bytes(data)
    if not info:
        try:
            data = _normalize_audio_to_target_wav(data, file.filename or "trimmed.wav")
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Audio normalization failed: {e}"}, status_code=400)
    elif not _is_target_wav(info):
        try:
            data = _normalize_audio_to_target_wav(data, file.filename or "trimmed.wav")
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Audio normalization failed: {e}"}, status_code=400)

    try:
        orig_path = _resolve_audio_path(directory, source_file)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)

    TRIM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    backup_name = f"{ts}_{source_file}"
    backup_path = TRIM_HISTORY_DIR / backup_name
    shutil.copy2(orig_path, backup_path)

    orig_sidecar = _audio_sidecar_path(orig_path)
    if orig_sidecar.exists():
        shutil.copy2(orig_sidecar, _audio_sidecar_path(backup_path))

    orig_path.write_bytes(data)

    old_sidecar = _load_sidecar_json(orig_path)
    sidecar = {
        **old_sidecar,
        "trimmed": True,
        "source_file": source_file,
        "source_bucket": bucket,
        "trim_start_s": float(start_time) if start_time else None,
        "trim_end_s": float(end_time) if end_time else None,
        "undo_backup_file": backup_name,
    }
    _write_sidecar_json(orig_path, sidecar)

    updated_item = _sample_item_from_path(orig_path, bucket)
    updated_item["trimmed"] = True
    updated_item["source_file"] = source_file
    return {"ok": True, "updated_sample": updated_item, "message": f"Trimmed {source_file}"}


@app.post("/api/samples/revert")
def revert_trim(
    bucket: str = Form(...),
    file_name: str = Form(...),
):
    bucket_map = {"personal": PERSONAL_DIR, "negative": NEGATIVE_DIR}
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)

    try:
        file_path = _resolve_audio_path(directory, file_name)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)

    sidecar = _load_sidecar_json(file_path)
    backup_name = sidecar.get("undo_backup_file")
    if not backup_name:
        return JSONResponse({"ok": False, "error": "No trim backup found for this sample."}, status_code=400)

    backup_path = TRIM_HISTORY_DIR / backup_name
    if not backup_path.exists():
        return JSONResponse({"ok": False, "error": "Trim backup file missing."}, status_code=404)

    shutil.copy2(backup_path, file_path)
    backup_sidecar = _audio_sidecar_path(backup_path)
    if backup_sidecar.exists():
        shutil.copy2(backup_sidecar, _audio_sidecar_path(file_path))

    backup_path.unlink()
    if backup_sidecar.exists():
        backup_sidecar.unlink()

    updated_item = _sample_item_from_path(file_path, bucket)
    return {"ok": True, "updated_sample": updated_item, "message": f"Reverted {file_name}"}


@app.post("/api/captured_audio/{file_name}/approve_personal")
def approve_captured_audio_to_personal(file_name: str):
    try:
        result = _move_captured_audio(file_name, PERSONAL_DIR, target_prefix="sample", review_status="approved_personal")
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, **result}


@app.post("/api/captured_audio/{file_name}/mark_negative")
def mark_captured_audio_negative(file_name: str):
    try:
        result = _move_captured_audio(file_name, NEGATIVE_DIR, target_prefix="negative", review_status="approved_negative")
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, **result}


@app.post("/api/captured_audio/{file_name}/discard")
def discard_captured_audio(file_name: str):
    try:
        path = _resolve_audio_path(CAPTURED_DIR, file_name)
        _remove_audio_with_sidecar(path)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    return {"ok": True, "captured_count": len(_list_captured_sample_names())}


@app.get("/api/trained_wake_words/catalog")
def trained_wake_words_catalog(request: Request):
    return {
        "ok": True,
        "base_url": _advertised_base_url(request),
        "wake_words": _list_trained_wake_words(_advertised_base_url(request)),
    }


@app.get("/api/trained_wake_words/{filename}")
def trained_wake_word_artifact(filename: str):
    safe_filename = Path(filename or "").name
    if not safe_filename or Path(safe_filename).suffix.lower() not in {".json", ".tflite"}:
        return JSONResponse({"ok": False, "error": "Unsupported wake word artifact."}, status_code=400)
    _sync_trained_wake_word_artifacts()
    artifact_path = TRAINED_WAKE_WORDS_DIR / safe_filename
    if not artifact_path.exists() or not artifact_path.is_file():
        return JSONResponse({"ok": False, "error": "Wake word artifact not found."}, status_code=404)
    media_type = "application/json" if artifact_path.suffix.lower() == ".json" else "application/octet-stream"
    return FileResponse(str(artifact_path), media_type=media_type, filename=artifact_path.name)


@app.post("/api/train")
def train_now(payload: Dict[str, Any] = None):
    payload = payload or {}
    allow_no_personal = bool(payload.get("allow_no_personal", False))

    with STATE_LOCK:
        safe_word = STATE["safe_word"]
        language = (STATE.get("language") or DEFAULT_LANGUAGE)
        takes_received = int(STATE["takes_received"])
        speakers_total = int(STATE["speakers_total"])
        takes_per_speaker = int(STATE["takes_per_speaker"])
        training_running = bool(STATE["training"]["running"])

    takes_total = speakers_total * takes_per_speaker

    if training_running:
        return JSONResponse({"ok": False, "error": "Training already running"}, status_code=400)

    if not safe_word:
        return JSONResponse({"ok": False, "error": "No active session"}, status_code=400)

    if takes_received == 0 and not allow_no_personal:
        return JSONResponse(
            {
                "ok": False,
                "error": "No personal voice samples uploaded yet.",
                "code": "NO_PERSONAL_SAMPLES",
                "message": "You can train without personal voices, or upload samples first.",
                "takes_total": takes_total,
            },
            status_code=400,
        )

    with STATE_LOCK:
        STATE["training"]["running"] = True
    t = threading.Thread(
        target=_run_training_background,
        args=(safe_word, language, allow_no_personal, False),
        daemon=True,
    )
    t.start()

    return {
        "ok": True,
        "started": True,
        "safe_word": safe_word,
        "personal_samples_used": takes_received > 0,
        "allow_no_personal": allow_no_personal,
    }


@app.get("/api/train_status")
def train_status():
    with STATE_LOCK:
        tr = dict(STATE["training"])
        log_path_str = tr.get("log_path")
        prev_tail = list(STATE["training"].get("last_sent_tail") or [])
        prev_size = int(STATE["training"].get("last_log_size") or 0)

    new_lines: List[str] = []
    full_tail: List[str] = []
    size_now = 0

    if log_path_str:
        p = Path(log_path_str)
        if p.exists():
            try:
                size_now = int(p.stat().st_size)
            except Exception:
                size_now = 0

            if size_now < prev_size:
                prev_tail = []

            full_tail = _read_tail_lines(p, TRAIN_LOG_TAIL_LINES)
            new_lines = _compute_new_lines(prev_tail, full_tail)

    with STATE_LOCK:
        STATE["training"]["last_sent_tail"] = full_tail
        STATE["training"]["last_log_size"] = size_now

    tr["log_text"] = "\n".join(new_lines)
    tr["log_tail_preview"] = "\n".join(full_tail)
    tr["log_lines"] = full_tail
    return {"ok": True, "training": tr}


@app.post("/api/reset_recordings")
def reset_recordings():
    _reset_personal_samples_dir()
    takes = _sync_personal_samples_state()
    return {"ok": True, "takes_received": len(takes), "takes": takes}


@app.post("/api/reset_negative_samples")
def reset_negative_samples():
    _reset_audio_dir(NEGATIVE_DIR)
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_STATE["pending_negative_count"] = 0
        _save_auto_train_state_locked()
    return {"ok": True, "negative_count": len(_list_negative_samples())}
