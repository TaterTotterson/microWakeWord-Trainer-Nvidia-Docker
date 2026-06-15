#!/usr/bin/env python3

# trainer_server.py
import contextlib
import copy
import gzip
import hashlib
import io
import os
import re
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from array import array
from datetime import datetime, timezone
from math import log10
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Tuple
from urllib.parse import quote, urlparse
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

# Firmware build/flash cache lives inside /data so Docker runs can reuse downloads.
FIRMWARE_CACHE_DIR = Path(os.environ.get("FIRMWARE_CACHE_DIR", str(DATA_DIR / ".cache" / "firmware_flasher"))).resolve()
FIRMWARE_DEFAULT_OTA_PORT = int(os.environ.get("ESPHOME_OTA_PORT", "3232"))
FIRMWARE_DISCOVERY_SECONDS = float(os.environ.get("ESPHOME_DISCOVERY_SECONDS", "2.5"))
FIRMWARE_MAX_LOG_LINES = int(os.environ.get("FIRMWARE_MAX_LOG_LINES", "500"))
FIRMWARE_GITHUB_OWNER = os.environ.get("FIRMWARE_GITHUB_OWNER", "TaterTotterson")
FIRMWARE_GITHUB_REPO = os.environ.get("FIRMWARE_GITHUB_REPO", "microWakeWords")
FIRMWARE_GITHUB_REF = os.environ.get("FIRMWARE_GITHUB_REF", "main")
WAKE_SOUND_CATALOG_CACHE_TTL_SECONDS = int(os.environ.get("WAKE_SOUND_CATALOG_CACHE_TTL_SECONDS", "600"))
FIRMWARE_PREBUILT_DIR = FIRMWARE_CACHE_DIR / "prebuilt_firmware"
FIRMWARE_DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("FIRMWARE_DOWNLOAD_TIMEOUT_SECONDS", "120"))
FIRMWARE_JSON_CACHE_TTL_SECONDS = float(os.environ.get("FIRMWARE_JSON_CACHE_TTL_SECONDS", "900"))
FIRMWARE_OTA_BLOCK_SIZE = int(os.environ.get("FIRMWARE_OTA_BLOCK_SIZE", "8192"))
FIRMWARE_PROFILE_FILE = Path(
    os.environ.get("FIRMWARE_PROFILE_FILE", str(FIRMWARE_CACHE_DIR / "profiles.json"))
).resolve()
FIRMWARE_WEB_FLASH_DIR = FIRMWARE_CACHE_DIR / "web_flash"
WAKE_SOUND_MANIFEST_PATHS = ("wake_sound_manifest.json", "wake-sound-manifest.json")
WAKE_SOUND_CATALOG_CACHE: Dict[str, Any] = {"ts": 0.0, "payload": {}}
WAKE_SOUND_CATALOG_LOCK = threading.Lock()
FIRMWARE_JSON_CACHE: Dict[str, Dict[str, Any]] = {}
FIRMWARE_JSON_CACHE_LOCK = threading.Lock()
TRAIN_LOG_TAIL_LINES = int(os.environ.get("REC_TRAIN_LOG_TAIL_LINES", "400"))
TRAIN_LOG_MAX_BYTES = int(os.environ.get("REC_TRAIN_LOG_MAX_BYTES", str(512 * 1024)))

FIRMWARE_TEMPLATE_SPECS = (
    {
        "key": "voicepe",
        "label": "VoicePE",
        "description": "VoicePE satellite prebuilt firmware",
    },
    {
        "key": "satellite1",
        "label": "Sat1",
        "description": "Satellite1 prebuilt firmware",
    },
    {
        "key": "respeaker_lite",
        "label": "ReSpeaker Lite",
        "description": "ReSpeaker Lite prebuilt firmware",
    },
    {
        "key": "koala",
        "label": "Koala Satellite",
        "description": "Koala satellite prebuilt firmware",
    },
    {
        "key": "respeaker_xvf3800",
        "label": "ReSpeaker XVF3800",
        "description": "ReSpeaker XVF3800 prebuilt firmware",
    },
)
FIRMWARE_PREBUILT_LATEST_URL = (
    f"https://raw.githubusercontent.com/{FIRMWARE_GITHUB_OWNER}/{FIRMWARE_GITHUB_REPO}/{FIRMWARE_GITHUB_REF}/prebuilt_firmware/latest.json"
)
FIRMWARE_PREBUILT_TEMPLATE_KEYS = {str(spec.get("key") or "").lower() for spec in FIRMWARE_TEMPLATE_SPECS}

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
PIPER_CATALOG_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "entries": None,
}
FIRMWARE_LOCK = threading.Lock()
FIRMWARE_SESSIONS: Dict[str, Dict[str, Any]] = {}
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])")

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
    """Mirror generated output artifacts into /data/trained_wake_words for firmware flashing."""
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


def _list_trained_wake_words(base_url: str = "") -> List[Dict[str, str]]:
    _sync_trained_wake_word_artifacts()
    base = str(base_url or "").rstrip("/")
    rows: List[Dict[str, str]] = []
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
            }
        )
    return rows


def _request_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


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
        "detected_format": meta.get("detected_format") or {},
        "final_format": final_format,
        "postprocess": meta.get("postprocess") or {},
        "message": meta.get("message") or "",
        "notes": meta.get("notes") or "",
        "review_status": meta.get("review_status") or "pending",
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


def _run_training_background(safe_word: str, language: str, allow_no_personal: bool):
    language = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE
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
        _append_train_log(f"✗ Training crashed: {e!r}")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = 999

    finally:
        with STATE_LOCK:
            STATE["training"]["running"] = False


# -------------------- Firmware flasher --------------------
def _template_default_string(raw_value: Any) -> str:
    if isinstance(raw_value, dict) and raw_value.get("__secret__"):
        return ""
    if isinstance(raw_value, bool):
        return "true" if raw_value else "false"
    if raw_value is None:
        return ""
    return str(raw_value)


def _humanize_key(key: str) -> str:
    token = str(key or "").strip()
    if not token:
        return "Value"
    special = {
        "ha": "HA",
        "ip": "IP",
        "id": "ID",
        "ssid": "SSID",
        "wifi": "Wi-Fi",
        "xmos": "XMOS",
        "fw": "FW",
    }
    return " ".join(special.get(part.lower(), part.capitalize()) for part in token.replace("_", " ").split())


def _firmware_template_spec(template_key: str) -> Dict[str, Any]:
    token = (template_key or "").strip().lower()
    for spec in FIRMWARE_TEMPLATE_SPECS:
        if str(spec.get("key") or "").lower() == token:
            return dict(spec)
    raise ValueError("Unknown firmware template.")


def _firmware_raw_url(path: str) -> str:
    clean = str(path or "").strip().lstrip("/")
    return f"https://raw.githubusercontent.com/{FIRMWARE_GITHUB_OWNER}/{FIRMWARE_GITHUB_REPO}/{FIRMWARE_GITHUB_REF}/{clean}"


def _fetch_text_url(url: str, timeout: float = 20) -> str:
    req = URLRequest(url, headers={"User-Agent": "microWakeWord-Trainer/1.0"})
    with urlopen(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _as_int(value: Any, default: int = 0, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _sanitize_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", _text(value)).strip("._-")
    return (token[:96] or "default").lower()


def _prebuilt_firmware_raw_url(path_or_url: Any) -> str:
    token = _text(path_or_url)
    if not token:
        return ""
    parsed = urlparse(token)
    if parsed.scheme and parsed.netloc:
        return token
    clean = token.lstrip("/")
    quoted = "/".join(quote(part) for part in clean.split("/") if part)
    return f"https://raw.githubusercontent.com/{FIRMWARE_GITHUB_OWNER}/{FIRMWARE_GITHUB_REPO}/{FIRMWARE_GITHUB_REF}/{quoted}"


def _fetch_json_url(url: str, *, timeout: float = 20, force_refresh: bool = False) -> Dict[str, Any]:
    now = time.time()
    with FIRMWARE_JSON_CACHE_LOCK:
        cached = FIRMWARE_JSON_CACHE.get(url)
        if (
            not force_refresh
            and isinstance(cached, dict)
            and isinstance(cached.get("payload"), dict)
            and (now - float(cached.get("ts") or 0.0)) < FIRMWARE_JSON_CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(cached["payload"])

    req = URLRequest(
        url,
        headers={
            "User-Agent": "microWakeWord-Trainer/1.0",
            "Accept": "application/json, */*",
            "Cache-Control": "no-cache" if force_refresh else "max-age=60",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        payload = json.loads(response.read().decode(charset, errors="replace"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Remote JSON did not parse into an object: {url}")
    with FIRMWARE_JSON_CACHE_LOCK:
        FIRMWARE_JSON_CACHE[url] = {"ts": now, "payload": copy.deepcopy(payload)}
    return payload


def _load_prebuilt_firmware_manifest(*, force_refresh: bool = False) -> Dict[str, Any]:
    latest_payload = _fetch_json_url(
        FIRMWARE_PREBUILT_LATEST_URL,
        timeout=20,
        force_refresh=force_refresh,
    )
    manifest_ref = _text(latest_payload.get("manifest"))
    if not manifest_ref:
        raise RuntimeError("Prebuilt firmware latest.json is missing a manifest path.")

    manifest_url = _prebuilt_firmware_raw_url(manifest_ref)
    manifest_payload = _fetch_json_url(manifest_url, timeout=20, force_refresh=force_refresh)
    devices = manifest_payload.get("devices")
    if not isinstance(devices, list):
        raise RuntimeError("Prebuilt firmware manifest is missing its devices list.")

    payload = copy.deepcopy(manifest_payload)
    payload["version"] = _text(manifest_payload.get("version")) or _text(latest_payload.get("version"))
    payload["latest_url"] = FIRMWARE_PREBUILT_LATEST_URL
    payload["manifest_url"] = manifest_url
    payload["manifest_path"] = manifest_ref
    payload["devices_by_key"] = {
        _lower(row.get("key")): dict(row)
        for row in devices
        if isinstance(row, dict) and _text(row.get("key"))
    }
    return payload


def _prebuilt_firmware_info(template_key: Any, *, force_refresh: bool = False) -> Dict[str, Any]:
    key = _lower(template_key)
    if key not in FIRMWARE_PREBUILT_TEMPLATE_KEYS:
        return {"available": False, "template_key": key, "reason": "not_prebuilt"}
    try:
        manifest = _load_prebuilt_firmware_manifest(force_refresh=force_refresh)
    except Exception as exc:
        return {
            "available": False,
            "template_key": key,
            "reason": "manifest_unavailable",
            "error": _text(exc) or exc.__class__.__name__,
        }

    devices_by_key = manifest.get("devices_by_key") if isinstance(manifest.get("devices_by_key"), dict) else {}
    device = devices_by_key.get(key) if isinstance(devices_by_key.get(key), dict) else None
    if not isinstance(device, dict):
        return {
            "available": False,
            "template_key": key,
            "reason": "missing_device",
            "version": _text(manifest.get("version")),
            "manifest_url": _text(manifest.get("manifest_url")),
        }

    artifacts = device.get("artifacts") if isinstance(device.get("artifacts"), dict) else {}
    return {
        "available": bool(artifacts.get("ota") or artifacts.get("factory")),
        "template_key": key,
        "version": _text(manifest.get("version")),
        "manifest_url": _text(manifest.get("manifest_url")),
        "latest_url": _text(manifest.get("latest_url")),
        "device": copy.deepcopy(device),
        "artifacts": copy.deepcopy(artifacts),
    }


def _prebuilt_artifact_ui_summary(prebuilt: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = prebuilt.get("artifacts") if isinstance(prebuilt.get("artifacts"), dict) else {}
    ota_artifact = artifacts.get("ota") if isinstance(artifacts.get("ota"), dict) else None
    return {
        "available": bool(isinstance(ota_artifact, dict) and _text(ota_artifact.get("path"))),
        "version": _text(prebuilt.get("version")),
        "manifest_url": _text(prebuilt.get("manifest_url")),
        "latest_url": _text(prebuilt.get("latest_url")),
        "error": _text(prebuilt.get("error")),
        "artifacts": {
            kind: {
                "kind": _text(row.get("kind") or kind),
                "path": _text(row.get("path")),
                "size_bytes": _as_int(row.get("size_bytes"), 0, minimum=0),
                "sha256": _text(row.get("sha256")),
            }
            for kind, row in artifacts.items()
            if isinstance(row, dict)
        },
    }


def _prebuilt_artifact_meta(prebuilt: Dict[str, Any], kind: str) -> Dict[str, Any]:
    artifacts = prebuilt.get("artifacts") if isinstance(prebuilt.get("artifacts"), dict) else {}
    artifact = artifacts.get(_lower(kind)) if isinstance(artifacts.get(_lower(kind)), dict) else None
    if not isinstance(artifact, dict) or not _text(artifact.get("path")):
        raise RuntimeError(f"No prebuilt {kind} firmware artifact is available for this target.")
    return dict(artifact)


def _prebuilt_cache_path(template_key: Any, version: Any, artifact: Dict[str, Any]) -> Path:
    name = Path(_text(artifact.get("path"))).name or f"{_sanitize_token(template_key)}-{_text(artifact.get('kind')) or 'firmware'}.bin"
    return FIRMWARE_PREBUILT_DIR / _sanitize_token(version or "latest") / _sanitize_token(template_key) / name


def _prebuilt_binary_is_valid(path: Path, artifact: Dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    expected_size = _as_int(artifact.get("size_bytes"), 0, minimum=0)
    if expected_size and int(path.stat().st_size) != expected_size:
        return False
    expected_sha = _lower(artifact.get("sha256"))
    if expected_sha and hashlib.sha256(path.read_bytes()).hexdigest().lower() != expected_sha:
        return False
    return True


def _download_prebuilt_firmware_binary(
    template_key: Any,
    prebuilt: Dict[str, Any],
    kind: str,
    *,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    artifact = _prebuilt_artifact_meta(prebuilt, kind)
    target_path = _prebuilt_cache_path(template_key, prebuilt.get("version"), artifact)
    url = _prebuilt_firmware_raw_url(artifact.get("path"))
    if not url:
        raise RuntimeError("Prebuilt firmware URL is missing.")
    if not force_refresh and _prebuilt_binary_is_valid(target_path, artifact):
        return {"path": target_path, "artifact": artifact, "url": url, "cached": True}

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
    req = URLRequest(
        url,
        headers={
            "User-Agent": "microWakeWord-Trainer/1.0",
            "Accept": "application/octet-stream, */*",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    try:
        with urlopen(req, timeout=FIRMWARE_DOWNLOAD_TIMEOUT_SECONDS) as response:
            tmp_path.write_bytes(response.read())
        if not _prebuilt_binary_is_valid(tmp_path, artifact):
            raise RuntimeError(f"Downloaded prebuilt firmware failed verification: {target_path.name}.")
        tmp_path.replace(target_path)
    except Exception:
        with contextlib.suppress(Exception):
            tmp_path.unlink()
        raise
    return {"path": target_path, "artifact": artifact, "url": url, "cached": False}


def _browser_flash_artifact_id(template_key: Any) -> str:
    return "_".join(
        part
        for part in [
            _sanitize_token(template_key),
            str(int(time.time())),
            uuid.uuid4().hex[:8],
        ]
        if part
    )


def _create_browser_flash_artifact(template_key: Any, prebuilt: Dict[str, Any], binary_path: Path) -> Dict[str, Any]:
    artifact_id = _browser_flash_artifact_id(template_key)
    artifact_dir = FIRMWARE_WEB_FLASH_DIR / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    target_binary_name = "firmware.bin"
    target_binary_path = artifact_dir / target_binary_name
    shutil.copy2(binary_path, target_binary_path)
    return {
        "artifact_id": artifact_id,
        "binary_url": f"/api/firmware/browser_flash/{artifact_id}/{target_binary_name}",
        "binary_name": target_binary_name,
        "template_key": _text(template_key),
        "firmware_version": _text(prebuilt.get("version")),
        "source_binary": str(binary_path),
        "binary_size": int(target_binary_path.stat().st_size),
        "erase_all": True,
        "flash_size": "4MB",
        "flash_mode": "dio",
        "flash_freq": "40m",
    }


def _browser_flash_artifact_path(artifact_id: str, relative_path: str) -> Path:
    artifact = _sanitize_token(artifact_id)
    if not artifact:
        raise KeyError("Browser flash artifact is missing.")
    rel = Path(_text(relative_path))
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise KeyError("Browser flash artifact path is invalid.")
    root = (FIRMWARE_WEB_FLASH_DIR / artifact).resolve()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise KeyError("Browser flash artifact path is invalid.")
    if not target.is_file():
        raise KeyError("Browser flash artifact file was not found.")
    return target


class _NativeOTAError(RuntimeError):
    pass


def _native_ota_check(data: bytes, expected: set[int] | None = None) -> None:
    error_messages = {
        0x80: "Invalid magic byte.",
        0x81: "Device could not prepare flash memory for update.",
        0x82: "OTA authentication failed.",
        0x83: "Writing OTA data to flash failed.",
        0x84: "Finishing OTA update failed.",
        0x85: "Manual reset is required before this OTA update.",
        0x86: "Current flash configuration does not match this firmware.",
        0x87: "New firmware flash configuration does not match this device.",
        0x89: "The OTA partition is too small for this firmware.",
        0x8A: "The OTA partition could not be found. Recover with USB flashing.",
        0x8B: "OTA MD5 mismatch. Retry or recover with USB flashing.",
        0x8D: "Firmware signature verification failed.",
        0x8E: "This OTA type is not supported by the device.",
        0xFF: "Unknown OTA error from device.",
    }
    if not data:
        raise _NativeOTAError("Device closed the OTA connection without responding.")
    code = int(data[0])
    if code in error_messages:
        raise _NativeOTAError(error_messages[code])
    if expected is not None and code not in expected:
        expected_text = ", ".join(f"0x{item:02X}" for item in sorted(expected))
        raise _NativeOTAError(f"Unexpected OTA response 0x{code:02X}; expected {expected_text}.")


def _native_ota_receive(sock: socket.socket, amount: int, label: str, expected: set[int] | None = None) -> bytes:
    data = b""
    while len(data) < amount:
        try:
            chunk = sock.recv(amount - len(data))
        except OSError as exc:
            raise _NativeOTAError(f"OTA receive failed while reading {label}: {exc}") from exc
        if not chunk:
            raise _NativeOTAError(f"OTA connection closed while reading {label}.")
        data += chunk
        if len(data) == 1:
            _native_ota_check(data, expected)
    if len(data) > 1 and expected is not None:
        _native_ota_check(data[:1], expected)
    return data


def _native_ota_send(sock: socket.socket, data: bytes | str | int | List[int], label: str) -> None:
    if isinstance(data, str):
        payload = data.encode("utf-8")
    elif isinstance(data, int):
        payload = bytes([data])
    elif isinstance(data, list):
        payload = bytes(data)
    else:
        payload = data
    try:
        sock.sendall(payload)
    except OSError as exc:
        raise _NativeOTAError(f"OTA send failed while writing {label}: {exc}") from exc


def _native_ota_upload(
    host: str,
    port: int,
    binary_path: Path,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> str:
    if not host:
        raise _NativeOTAError("OTA target host is missing.")
    if not binary_path.is_file():
        raise _NativeOTAError(f"OTA firmware file was not found: {binary_path}.")

    upload_contents = binary_path.read_bytes()
    sock: socket.socket | None = None
    try:
        sock = socket.create_connection((host, int(port or FIRMWARE_DEFAULT_OTA_PORT)), timeout=20.0)
        sock.settimeout(20.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _native_ota_send(sock, bytes([0x6C, 0x26, 0xF7, 0x5C, 0x45]), "magic bytes")
        version_response = _native_ota_receive(sock, 2, "OTA version", {0x00})
        version = int(version_response[1])
        if version not in {1, 2}:
            raise _NativeOTAError(f"Device uses unsupported OTA protocol version {version}.")

        _native_ota_send(sock, 0x01 | 0x04, "client features")
        feature_response = _native_ota_receive(sock, 1, "server features")
        extended_proto = False
        server_features = 0
        first_feature = int(feature_response[0])
        if first_feature == 0x48:
            extended_proto = True
            server_features = int(_native_ota_receive(sock, 1, "server feature flags")[0])
        elif first_feature == 0x46:
            server_features = 0x01

        auth_response = int(_native_ota_receive(sock, 1, "OTA auth", {0x01, 0x02, 0x41})[0])
        if auth_response != 0x41:
            raise _NativeOTAError("Device requested OTA authentication, but Tater prebuilt OTA has no password configured.")

        sock.settimeout(90.0)
        if extended_proto:
            _native_ota_send(sock, 0x00, "OTA app update type")
        if server_features & 0x01:
            upload_contents = gzip.compress(upload_contents, compresslevel=9)

        upload_size = len(upload_contents)
        _native_ota_send(
            sock,
            [
                (upload_size >> 24) & 0xFF,
                (upload_size >> 16) & 0xFF,
                (upload_size >> 8) & 0xFF,
                upload_size & 0xFF,
            ],
            "binary size",
        )
        _native_ota_receive(sock, 1, "update prepare", {0x42})
        _native_ota_send(sock, hashlib.md5(upload_contents).hexdigest(), "binary md5")
        _native_ota_receive(sock, 1, "md5 check", {0x43})
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)

        sent = 0
        last_percent = -1
        while sent < upload_size:
            chunk = upload_contents[sent : sent + FIRMWARE_OTA_BLOCK_SIZE]
            _native_ota_send(sock, chunk, "firmware chunk")
            sent += len(chunk)
            if version >= 2:
                _native_ota_receive(sock, 1, "chunk acknowledgement", {0x47})
            percent = int((sent / upload_size) * 100) if upload_size else 100
            if callable(progress_callback) and (percent >= last_percent + 5 or percent == 100):
                last_percent = percent
                progress_callback(percent, sent, upload_size)

        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _native_ota_receive(sock, 1, "receive result", {0x44})
        _native_ota_receive(sock, 1, "update end", {0x45})
        _native_ota_send(sock, 0x00, "end acknowledgement")
        return host
    except OSError as exc:
        raise _NativeOTAError(f"OTA connection to {host}:{int(port or FIRMWARE_DEFAULT_OTA_PORT)} failed: {exc}") from exc
    finally:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


def _load_firmware_template_text(spec: Dict[str, Any]) -> tuple[str, str]:
    rel_path = str(spec.get("path") or "").strip()
    url = _firmware_raw_url(rel_path)
    try:
        return _fetch_text_url(url, timeout=20), url
    except Exception as exc:
        raise RuntimeError(f"Could not download firmware template from {url}: {exc}") from exc


def _wake_sound_label_from_slug(slug: str) -> str:
    text = str(slug or "").strip()
    if not text:
        return "Wake Sound"
    return re.sub(r"[_\-.]+", " ", text).strip().title() or "Wake Sound"


def _wake_sound_entries_from_manifest(payload: Any) -> List[Dict[str, str]]:
    rows: List[Any] = []
    if isinstance(payload, list):
        rows = list(payload)
    elif isinstance(payload, dict):
        for key in ("entries", "wake_sounds", "sounds", "audio", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = list(candidate)
                break

    entries: List[Dict[str, str]] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(
            row.get("url")
            or row.get("download_url")
            or row.get("audio_url")
            or row.get("sound_url")
            or row.get("wake_sound_url")
            or row.get("wake_word_triggered_sound_file")
            or ""
        ).strip()
        path = str(row.get("path") or "").strip()
        if not url and path:
            url = _firmware_raw_url(path)
        if not url or url in seen:
            continue
        seen.add(url)
        slug = str(row.get("slug") or row.get("name") or row.get("key") or Path(path or url).stem).strip()
        entries.append(
            {
                "value": url,
                "label": str(row.get("label") or row.get("title") or _wake_sound_label_from_slug(slug)).strip(),
            }
        )
    return sorted(entries, key=lambda item: (item["label"].lower(), item["value"]))


def _load_wake_sound_catalog() -> Dict[str, Any]:
    now = time.time()
    with WAKE_SOUND_CATALOG_LOCK:
        cached_ts = float(WAKE_SOUND_CATALOG_CACHE.get("ts") or 0.0)
        cached_payload = WAKE_SOUND_CATALOG_CACHE.get("payload")
        if isinstance(cached_payload, dict) and (now - cached_ts) < WAKE_SOUND_CATALOG_CACHE_TTL_SECONDS:
            return copy.deepcopy(cached_payload)

    warnings: List[str] = []
    for manifest_path in WAKE_SOUND_MANIFEST_PATHS:
        manifest_url = _firmware_raw_url(manifest_path)
        try:
            payload = json.loads(_fetch_text_url(manifest_url, timeout=20))
            entries = _wake_sound_entries_from_manifest(payload)
            if entries:
                catalog = {"entries": entries, "warning": "", "source_label": manifest_url}
                with WAKE_SOUND_CATALOG_LOCK:
                    WAKE_SOUND_CATALOG_CACHE["ts"] = now
                    WAKE_SOUND_CATALOG_CACHE["payload"] = copy.deepcopy(catalog)
                return catalog
        except Exception as exc:
            warnings.append(f"{manifest_path}: {exc}")

    catalog = {
        "entries": [],
        "warning": warnings[0] if warnings else "Wake sound catalog unavailable.",
        "source_label": "",
    }
    with WAKE_SOUND_CATALOG_LOCK:
        WAKE_SOUND_CATALOG_CACHE["ts"] = now
        WAKE_SOUND_CATALOG_CACHE["payload"] = copy.deepcopy(catalog)
    return catalog


def _wake_sound_picker_options(catalog: Dict[str, Any]) -> List[Dict[str, str]]:
    entries = catalog.get("entries") if isinstance(catalog.get("entries"), list) else []
    return [{"value": "__custom__", "label": "Custom URL"}, *[dict(row) for row in entries if isinstance(row, dict)]]


def _extract_substitution_sections(raw_text: str) -> Dict[str, str]:
    section_map: Dict[str, str] = {}
    in_substitutions = False
    current_section = "Firmware"

    for line in raw_text.splitlines():
        if not in_substitutions:
            if re.match(r"^\s*substitutions:\s*$", line):
                in_substitutions = True
            continue
        if line and not line.startswith((" ", "\t")):
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comment = stripped[1:].strip()
            if comment and set(comment) > {"-"} and len(comment) <= 48 and re.search(r"[A-Za-z]", comment):
                current_section = comment.title() if comment.isupper() else comment
            continue
        match = re.match(r"^([A-Za-z0-9_]+)\s*:", stripped)
        if match:
            section_map[match.group(1)] = current_section
    return section_map


def _load_firmware_profiles() -> Dict[str, Dict[str, str]]:
    with contextlib.suppress(Exception):
        if FIRMWARE_PROFILE_FILE.exists():
            parsed = json.loads(FIRMWARE_PROFILE_FILE.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return {
                    str(key): {str(k): str(v) for k, v in value.items()}
                    for key, value in parsed.items()
                    if isinstance(value, dict)
                }
    return {}


def _save_firmware_profile(profile_key: str, values: Dict[str, str]) -> None:
    FIRMWARE_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    profiles = _load_firmware_profiles()
    profiles[profile_key] = {str(key): str(value) for key, value in values.items() if str(key)}
    FIRMWARE_PROFILE_FILE.write_text(json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8")


def _firmware_profile_target(raw_host: Any = "", raw_port: Any = "") -> tuple[str, str]:
    host = str(raw_host or "").strip()
    port = str(raw_port or "").strip()
    if "://" in host:
        parsed = urlparse(host)
        host = parsed.hostname or ""
        if not port and parsed.port:
            port = str(parsed.port)
    host = host.strip().strip("/")
    if host.count(":") == 1 and not port:
        maybe_host, maybe_port = host.rsplit(":", 1)
        if maybe_port.isdigit():
            host = maybe_host
            port = maybe_port
    host = host.strip("[]").strip().lower()
    if not host:
        return "", ""
    with contextlib.suppress(Exception):
        parsed_port = int(port or FIRMWARE_DEFAULT_OTA_PORT)
        if parsed_port == 6053:
            parsed_port = FIRMWARE_DEFAULT_OTA_PORT
        port = str(parsed_port)
    if not port:
        port = str(FIRMWARE_DEFAULT_OTA_PORT)
    return host, port


def _firmware_profile_key_for_target(raw_host: Any = "", raw_port: Any = "") -> str:
    host, port = _firmware_profile_target(raw_host, raw_port)
    return f"device:{host}:{port}" if host else ""


def _firmware_profile_key(template_key: Any = "", raw_host: Any = "", raw_port: Any = "") -> str:
    target_key = _firmware_profile_key_for_target(raw_host, raw_port)
    template = str(template_key or "").strip().lower()
    if target_key and template:
        return f"{target_key}:template:{template}"
    return target_key or template


def _firmware_cache_slug(*parts: Any) -> str:
    raw = "__".join(str(part or "").strip() for part in parts if str(part or "").strip())
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return (slug[:96] or "default").lower()


def _firmware_build_cache_path(
    template_key: str,
    normalized: Dict[str, str],
    host: str,
    port: Any = None,
    identity_key: str = "",
    friendly_key: str = "",
) -> Path:
    normalized_host, normalized_port = _firmware_profile_target(host, port)
    template_slug = _firmware_cache_slug(template_key, "template")
    identity_key = str(identity_key or "").strip()
    friendly_key = str(friendly_key or "").strip()
    device_identity = (
        (normalized.get(identity_key) if identity_key else "")
        or (normalized.get(friendly_key) if friendly_key else "")
        or normalized.get("node_name")
        or normalized.get("device_name")
        or normalized.get("friendly_name")
        or normalized.get("name")
        or normalized_host
        or "device"
    )
    target_slug = _firmware_cache_slug(device_identity, normalized_host, normalized_port)
    return FIRMWARE_CACHE_DIR / "builds" / template_slug / target_slug


def _load_firmware_profile(template_key: str, profile_key: str = "") -> Dict[str, str]:
    profiles = _load_firmware_profiles()
    profile = profiles.get(profile_key) if profile_key else None
    if isinstance(profile, dict):
        return dict(profile)
    if profile_key and ":template:" in profile_key:
        legacy_device_key = profile_key.split(":template:", 1)[0]
        legacy_device = profiles.get(legacy_device_key)
        if isinstance(legacy_device, dict):
            return dict(legacy_device)
    legacy = profiles.get(template_key)
    return dict(legacy) if isinstance(legacy, dict) else {}


def _firmware_profile_values_for_template(profile: Dict[str, Any], substitutions: Dict[str, Any]) -> Dict[str, str]:
    keep_keys = {str(key or "").strip() for key in substitutions.keys()}
    keep_keys.update({"__target_host", "__target_port", "wake_sound_catalog", "wake_word_choice"})
    return {
        key: str(profile.get(key) or "")
        for key in keep_keys
        if key and key in profile
    }


def _normalize_firmware_profile_update(template_key: str, values: Dict[str, Any], profile_key: str = "") -> Dict[str, str]:
    ctx = _load_firmware_template_context(template_key, profile_key)
    spec = ctx["spec"]
    profile = ctx.get("profile") if isinstance(ctx.get("profile"), dict) else {}
    substitutions = ctx["substitutions"]
    normalized = _firmware_profile_values_for_template(profile, substitutions)
    fixed_keys = set(spec.get("fixed_keys") or set())
    identity_key = str(spec.get("identity_key") or "").strip()
    if identity_key:
        fixed_keys.add(identity_key)

    for key in substitutions.keys():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        default = _template_default_string(substitutions.get(key_text))
        if key_text in fixed_keys:
            normalized[key_text] = default
            continue
        if key_text == "wifi_password":
            raw_password = str(values.get(key_text) or "").strip()
            if raw_password:
                normalized[key_text] = raw_password
            elif key_text not in normalized:
                normalized[key_text] = ""
            continue
        if key_text == "hidden_ssid":
            if key_text in values:
                normalized[key_text] = "true" if _parse_bool(values.get(key_text)) else "false"
            elif key_text not in normalized:
                normalized[key_text] = "true" if str(default).lower() == "true" else "false"
            continue
        if key_text == "wake_word_model_url":
            if key_text in values:
                normalized[key_text] = _local_trained_wake_word_url(values.get(key_text))
            elif key_text not in normalized:
                normalized[key_text] = ""
            continue
        if key_text in values:
            normalized[key_text] = str(values.get(key_text) or "").strip()
        elif key_text not in normalized:
            normalized[key_text] = default

    wake_word_choice = str(values.get("wake_word_choice") or "").strip()
    if wake_word_choice:
        normalized["wake_word_choice"] = wake_word_choice
    wake_sound_choice = str(values.get("wake_sound_catalog") or "").strip()
    if wake_sound_choice:
        normalized["wake_sound_catalog"] = wake_sound_choice
        if wake_sound_choice != "__custom__" and "wake_word_triggered_sound_file" in substitutions:
            normalized["wake_word_triggered_sound_file"] = wake_sound_choice

    target_host = str(values.get("__target_host") or "").strip()
    target_port = str(values.get("__target_port") or "").strip()
    if target_port == "6053":
        target_port = str(FIRMWARE_DEFAULT_OTA_PORT)
    if target_host:
        normalized["__target_host"] = target_host
    if target_port:
        normalized["__target_port"] = target_port

    return normalized


def _local_trained_wake_word_url(value: Any) -> str:
    url = str(value or "").strip()
    return url if "/api/trained_wake_words/" in url else ""


def _selected_trained_wake_word(
    trained_wake_words: List[Dict[str, Any]],
    profile: Dict[str, Any],
    substitutions: Dict[str, Any],
) -> Dict[str, Any] | None:
    if not trained_wake_words:
        return None

    saved_choice = str(profile.get("wake_word_choice") or "").strip()
    current_wake_word_name = str(
        profile.get("wake_word_name") or _template_default_string(substitutions.get("wake_word_name"))
    ).strip()
    current_wake_word_url = str(profile.get("wake_word_model_url") or "").strip()

    def match(predicate):
        return next((row for row in trained_wake_words if predicate(row)), None)

    return (
        match(lambda row: str(row.get("key") or "") == saved_choice)
        or match(lambda row: str(row.get("json_url") or "") == current_wake_word_url)
        or match(lambda row: str(row.get("model_url") or "") == current_wake_word_url)
        or match(lambda row: str(row.get("wake_word_name") or "") == current_wake_word_name)
        or trained_wake_words[0]
    )


def _load_firmware_template_context(template_key: str, profile_key: str = "") -> Dict[str, Any]:
    spec = _firmware_template_spec(template_key)
    raw_text, source_label = _load_firmware_template_text(spec)
    parsed = yaml.load(raw_text, Loader=_FirmwareYamlLoader)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Firmware template {spec.get('label') or template_key} did not parse into a YAML mapping.")
    substitutions = parsed.get("substitutions") if isinstance(parsed.get("substitutions"), dict) else {}
    if not substitutions:
        raise RuntimeError(f"Firmware template {spec.get('label') or template_key} has no substitutions.")
    return {
        "spec": spec,
        "raw_text": raw_text,
        "source_label": source_label,
        "template_doc": parsed,
        "substitutions": dict(substitutions),
        "sections": _extract_substitution_sections(raw_text),
        "profile": _load_firmware_profile(str(spec.get("key") or ""), profile_key),
    }


def _firmware_template_fields(template_key: str, base_url: str = "", profile_key: str = "") -> List[Dict[str, Any]]:
    ctx = _load_firmware_template_context(template_key, profile_key)
    spec = ctx["spec"]
    profile = ctx.get("profile") if isinstance(ctx.get("profile"), dict) else {}
    fields: List[Dict[str, Any]] = []
    fixed_keys = set(spec.get("fixed_keys") or set())
    identity_key = str(spec.get("identity_key") or "").strip()
    if identity_key:
        fixed_keys.add(identity_key)
    hidden_keys = {"ha_voice_ip"} | set(spec.get("auto_keys") or set())
    trained_wake_words = _list_trained_wake_words(base_url)
    wake_sound_catalog = _load_wake_sound_catalog()
    selected_wake_word_row = _selected_trained_wake_word(trained_wake_words, profile, ctx["substitutions"])
    selected_wake_word = str(selected_wake_word_row.get("key") or "") if selected_wake_word_row else ""
    wake_picker_added = False
    wake_sound_picker_added = False

    for key, raw_value in ctx["substitutions"].items():
        key_text = str(key or "").strip()
        if not key_text or key_text in hidden_keys:
            continue

        if key_text in {"wake_word_name", "wake_word_model_url"} and not wake_picker_added:
            fields.append(
                {
                    "key": "wake_word_choice",
                    "label": "Trained Wake Word",
                    "type": "wake_word_select",
                    "value": selected_wake_word,
                    "placeholder": "Choose a trained wake word...",
                    "description": (
                        "Select a locally trained wake word to fill the name and model URL below."
                        if trained_wake_words
                        else "No locally trained wake words found yet. Train one first, then return here."
                    ),
                    "options": trained_wake_words,
                    "section": "Micro Wake Word",
                }
            )
            wake_picker_added = True

        if key_text == "wake_word_triggered_sound_file" and not wake_sound_picker_added:
            wake_sound_entries = wake_sound_catalog.get("entries") if isinstance(wake_sound_catalog.get("entries"), list) else []
            current_sound_url = str(profile.get(key_text) or _template_default_string(raw_value) or "").strip()
            saved_sound_choice = str(profile.get("wake_sound_catalog") or "").strip()
            available_sound_urls = {str(row.get("value") or "") for row in wake_sound_entries if isinstance(row, dict)}
            if saved_sound_choice in available_sound_urls or saved_sound_choice == "__custom__":
                picker_value = saved_sound_choice
            else:
                picker_value = current_sound_url if current_sound_url in available_sound_urls else "__custom__"
            description = (
                f"Choose from {len(wake_sound_entries)} prebuilt wake sounds, or leave this on Custom URL and paste your own audio URL below."
                if wake_sound_entries
                else "Prebuilt wake-sound catalog is unavailable right now. You can still paste any custom audio URL below."
            )
            if wake_sound_catalog.get("warning") and not wake_sound_entries:
                description = f"{description} {wake_sound_catalog['warning']}".strip()
            fields.append(
                {
                    "key": "wake_sound_catalog",
                    "label": "Prebuilt Wake Sound",
                    "type": "wake_sound_select",
                    "value": picker_value,
                    "options": _wake_sound_picker_options(wake_sound_catalog),
                    "description": description,
                    "section": "Wake Sound",
                }
            )
            wake_sound_picker_added = True

        default = _template_default_string(raw_value)
        saved = str(profile.get(key_text) or "")
        field_type = "text"
        read_only = key_text in fixed_keys
        value = default if read_only else (saved or default)
        placeholder = ""
        description = ""
        label = _humanize_key(key_text)

        if read_only:
            label = "Device Name" if key_text == identity_key else label
            description = "Locked to the selected firmware YAML."
        elif key_text == "wifi_password":
            field_type = "password"
            value = ""
            placeholder = "Leave blank to keep saved password" if saved else "Required before flashing"
            description = "Required for the first build. Leave blank later to keep the saved value."
        elif key_text == "hidden_ssid":
            field_type = "checkbox"
            value = str(saved or default).lower() == "true"
        elif key_text == "wifi_ssid":
            placeholder = "Your Wi-Fi SSID"
            description = "Required before build + flash."
        elif key_text == "wake_word_model_url":
            value = str(selected_wake_word_row.get("json_url") or "") if selected_wake_word_row else ""
            placeholder = "Train or select a local wake word first"
            description = "Filled from the local trained wake-word picker."
        elif key_text == "wake_word_name":
            if selected_wake_word_row:
                value = str(selected_wake_word_row.get("wake_word_name") or selected_wake_word_row.get("key") or "")
            placeholder = "hey_tater"
        elif key_text == "wake_word_triggered_sound_file":
            placeholder = "https://.../wake-sound.mp3"
            description = "Pick a prebuilt wake sound above or paste any custom audio URL."
        section = ctx["sections"].get(key_text) or "Firmware"
        if key_text == "wake_word_triggered_sound_file":
            section = "Wake Sound"
        elif key_text in {"wake_word_name", "wake_word_model_url"}:
            section = "Micro Wake Word"
        elif key_text.endswith("_sound_file"):
            section = "Sounds"

        fields.append(
            {
                "key": key_text,
                "label": label,
                "type": field_type,
                "value": value,
                "placeholder": placeholder,
                "description": description,
                "read_only": read_only,
                "section": section,
            }
        )
    return fields


def _esphome_pythonpath() -> str:
    existing = os.environ.get("PYTHONPATH", "")
    candidates = []
    env_repo = os.environ.get("ESPHOME_REPO_DIR")
    if env_repo:
        candidates.extend(Path(part).expanduser() for part in env_repo.split(os.pathsep) if part)
    candidates.append(ROOT_DIR.parent / "esphome")

    paths = [str(path) for path in candidates if (path / "esphome").is_dir()]
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def _render_firmware_config(
    template_key: str,
    values: Dict[str, Any],
    host: str,
    session_id: str,
    port: Any = None,
) -> tuple[Path, Dict[str, str], Path]:
    profile_key = _firmware_profile_key(template_key, host, port)
    ctx = _load_firmware_template_context(template_key, profile_key)
    spec = ctx["spec"]
    profile = ctx.get("profile") if isinstance(ctx.get("profile"), dict) else {}
    substitutions = ctx["substitutions"]
    normalized = _firmware_profile_values_for_template(profile, substitutions)
    fixed_keys = set(spec.get("fixed_keys") or set())
    identity_key = str(spec.get("identity_key") or "").strip()
    if identity_key:
        fixed_keys.add(identity_key)

    for key in substitutions.keys():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        raw_value = values.get(key_text)
        if key_text in fixed_keys:
            normalized[key_text] = _template_default_string(substitutions.get(key_text))
        elif key_text == "wifi_password":
            normalized[key_text] = str(raw_value or "").strip() or str(profile.get(key_text) or "")
        elif key_text == "hidden_ssid":
            normalized[key_text] = "true" if bool(raw_value) else "false"
        elif key_text == "ha_voice_ip":
            normalized[key_text] = host
        elif key_text == "wake_word_model_url":
            normalized[key_text] = _local_trained_wake_word_url(raw_value)
        else:
            normalized[key_text] = str(raw_value if raw_value is not None else "").strip() or _template_default_string(
                substitutions.get(key_text)
            )
    wake_sound_choice = str(values.get("wake_sound_catalog") or "").strip()
    if wake_sound_choice and wake_sound_choice != "__custom__" and "wake_word_triggered_sound_file" in substitutions:
        normalized["wake_word_triggered_sound_file"] = wake_sound_choice

    missing = []
    if not normalized.get("wifi_ssid"):
        missing.append("Wi-Fi SSID")
    if not normalized.get("wifi_password"):
        missing.append("Wi-Fi password")
    if not host:
        missing.append("device IP or hostname")
    if "wake_word_model_url" in substitutions and not normalized.get("wake_word_model_url"):
        missing.append("local trained wake word")
    if missing:
        raise RuntimeError(f"Missing required firmware values: {', '.join(missing)}.")

    config = copy.deepcopy(ctx["template_doc"])
    config["substitutions"] = {key: str(normalized.get(key, "")) for key in substitutions.keys()}
    build_path = _firmware_build_cache_path(
        str(spec.get("key") or template_key),
        normalized,
        host,
        port,
        str(spec.get("identity_key") or ""),
        str(spec.get("friendly_key") or ""),
    )
    esphome_block = config.get("esphome") if isinstance(config.get("esphome"), dict) else None
    if isinstance(esphome_block, dict):
        esphome_block["build_path"] = str(build_path)
        config["esphome"] = esphome_block

    session_dir = FIRMWARE_CACHE_DIR / "configs" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    config_path = session_dir / f"{build_path.parent.name}__{build_path.name}.yaml"
    config_path.write_text(yaml.dump(config, Dumper=_FirmwareYamlDumper, sort_keys=False, allow_unicode=True), encoding="utf-8")
    normalized_host, normalized_port = _firmware_profile_target(host, port)
    if normalized_host:
        normalized["__target_host"] = normalized_host
        normalized["__target_port"] = normalized_port
    _save_firmware_profile(profile_key or str(spec.get("key") or template_key), normalized)
    return config_path, normalized, build_path


def _firmware_session_payload(session_id: str) -> Dict[str, Any]:
    with FIRMWARE_LOCK:
        session = FIRMWARE_SESSIONS.get(session_id)
        if not isinstance(session, dict):
            return {"ok": False, "error": "Firmware flash session not found."}
        return {
            "ok": True,
            "session_id": session_id,
            "running": bool(session.get("running")),
            "exit_code": session.get("exit_code"),
            "host": session.get("host"),
            "port": session.get("port"),
            "filename": session.get("filename"),
            "message": session.get("message") or "",
            "log_lines": list(session.get("log_lines") or []),
            "started_at": session.get("started_at"),
            "finished_at": session.get("finished_at"),
        }


def _clean_terminal_text(value: Any) -> str:
    text_value = str(value or "")
    if not text_value:
        return ""
    clean = ANSI_ESCAPE_RE.sub("", text_value).replace("\r", "")
    clean = "".join(ch for ch in clean if ch == "\t" or ord(ch) >= 32)
    return clean.strip()


def _append_firmware_log(session_id: str, line: str):
    clean = _clean_terminal_text(line)
    if not clean:
        return
    with FIRMWARE_LOCK:
        session = FIRMWARE_SESSIONS.get(session_id)
        if not isinstance(session, dict):
            return
        lines: List[str] = session.setdefault("log_lines", [])
        lines.append(clean)
        if len(lines) > FIRMWARE_MAX_LOG_LINES:
            del lines[: len(lines) - FIRMWARE_MAX_LOG_LINES]


def _firmware_runner_env(*, include_esphome_pythonpath: bool = False) -> Dict[str, str]:
    FIRMWARE_HOME_DIR.mkdir(parents=True, exist_ok=True)
    FIRMWARE_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIRMWARE_PLATFORMIO_DIR.mkdir(parents=True, exist_ok=True)
    FIRMWARE_ESPHOME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["HOME"] = str(FIRMWARE_HOME_DIR)
    env["XDG_CACHE_HOME"] = str(FIRMWARE_XDG_CACHE_DIR)
    env["ESPHOME_DATA_DIR"] = str(FIRMWARE_ESPHOME_DATA_DIR)
    env["PLATFORMIO_CORE_DIR"] = str(FIRMWARE_PLATFORMIO_DIR)
    env["PLATFORMIO_CACHE_DIR"] = str(FIRMWARE_PLATFORMIO_DIR / "cache")
    if include_esphome_pythonpath:
        pythonpath = _esphome_pythonpath()
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
    return env


def _normalize_firmware_filename(raw_name: str) -> str:
    name = Path(raw_name or "firmware.bin").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "firmware.bin"


def _parse_flash_target(raw_host: str, raw_port: Any = None) -> tuple[str, int]:
    host_text = (raw_host or "").strip()
    if not host_text:
        raise ValueError("Device IP or hostname is required.")

    if "://" in host_text:
        parsed = urlparse(host_text)
        host_text = parsed.hostname or ""
        if raw_port in (None, "") and parsed.port:
            raw_port = parsed.port

    host_text = host_text.strip().strip("/")
    if "/" in host_text:
        host_text = host_text.split("/", 1)[0].strip()

    if host_text.count(":") == 1:
        maybe_host, maybe_port = host_text.rsplit(":", 1)
        if maybe_port.isdigit():
            host_text = maybe_host.strip("[]")
            raw_port = maybe_port

    host_text = host_text.strip("[]").strip()
    if not host_text:
        raise ValueError("Device IP or hostname is required.")
    if not re.match(r"^[A-Za-z0-9_.:-]+$", host_text):
        raise ValueError("Device host contains unsupported characters.")

    try:
        port = int(raw_port or FIRMWARE_DEFAULT_OTA_PORT)
    except Exception as exc:
        raise ValueError("OTA port must be a number.") from exc
    if port < 1 or port > 65535:
        raise ValueError("OTA port must be between 1 and 65535.")
    return host_text, port


def _run_firmware_flash_background(session_id: str):
    with FIRMWARE_LOCK:
        session = FIRMWARE_SESSIONS.get(session_id)
        if not isinstance(session, dict):
            return
        host = str(session.get("host") or "")
        port = int(session.get("port") or FIRMWARE_DEFAULT_OTA_PORT)
        firmware_path = str(session.get("firmware_path") or "")

    _append_firmware_log(session_id, "===== Firmware Flash Console =====")
    _append_firmware_log(session_id, f"→ Device: {host}:{port}")
    _append_firmware_log(session_id, f"→ OTA image: {Path(firmware_path).name}")

    try:
        with FIRMWARE_LOCK:
            live = FIRMWARE_SESSIONS.get(session_id)
            if isinstance(live, dict):
                live["message"] = "Firmware upload running."

        def progress(percent: int, sent: int, total: int) -> None:
            _append_firmware_log(session_id, f"→ OTA upload progress: {percent}% ({sent}/{total} bytes)")

        uploaded_host = _native_ota_upload(host, port, Path(firmware_path), progress_callback=progress)
        _append_firmware_log(session_id, f"✓ Firmware flash finished to {uploaded_host or host}")
        with FIRMWARE_LOCK:
            live = FIRMWARE_SESSIONS.get(session_id)
            if isinstance(live, dict):
                live["running"] = False
                live["exit_code"] = 0
                live["finished_at"] = datetime.now(timezone.utc).isoformat()
                live["message"] = "Firmware uploaded successfully."
    except Exception as exc:
        _append_firmware_log(session_id, f"✗ Firmware flash crashed: {exc!r}")
        with FIRMWARE_LOCK:
            live = FIRMWARE_SESSIONS.get(session_id)
            if isinstance(live, dict):
                live["running"] = False
                live["exit_code"] = 999
                live["finished_at"] = datetime.now(timezone.utc).isoformat()
                live["message"] = f"Firmware upload crashed: {exc}"


def _run_firmware_build_flash_background(session_id: str):
    with FIRMWARE_LOCK:
        session = FIRMWARE_SESSIONS.get(session_id)
        if not isinstance(session, dict):
            return
        host = str(session.get("host") or "")
        port = int(session.get("port") or FIRMWARE_DEFAULT_OTA_PORT)
        template_key = str(session.get("template_key") or "")
        template_label = str(session.get("template_label") or template_key)

    _append_firmware_log(session_id, "===== Prebuilt Firmware Flash Console =====")
    _append_firmware_log(session_id, f"→ Firmware: {template_label}")
    _append_firmware_log(session_id, f"→ Device: {host}:{port}")
    _append_firmware_log(session_id, "→ Loading latest prebuilt firmware manifest...")

    try:
        prebuilt = _prebuilt_firmware_info(template_key, force_refresh=True)
        if not bool(prebuilt.get("available")):
            raise RuntimeError(_text(prebuilt.get("error")) or "No prebuilt OTA image is available for this firmware target.")
        firmware_version = _text(prebuilt.get("version")) or "latest"
        _append_firmware_log(session_id, f"→ Latest firmware: {firmware_version}")
        _append_firmware_log(session_id, "→ Downloading or reusing verified OTA image...")
        binary = _download_prebuilt_firmware_binary(template_key, prebuilt, "ota")
        cached_text = "cached" if bool(binary.get("cached")) else "downloaded"
        _append_firmware_log(session_id, f"→ OTA image {cached_text}: {Path(binary['path']).name}")
        with FIRMWARE_LOCK:
            live = FIRMWARE_SESSIONS.get(session_id)
            if isinstance(live, dict):
                live["message"] = "Firmware upload running."
                live["filename"] = Path(binary["path"]).name
                live["firmware_version"] = firmware_version

        def progress(percent: int, sent: int, total: int) -> None:
            _append_firmware_log(session_id, f"→ OTA upload progress: {percent}% ({sent}/{total} bytes)")

        uploaded_host = _native_ota_upload(host, port, Path(binary["path"]), progress_callback=progress)
        _append_firmware_log(session_id, f"✓ Prebuilt firmware uploaded successfully to {uploaded_host or host}")
        with FIRMWARE_LOCK:
            live = FIRMWARE_SESSIONS.get(session_id)
            if isinstance(live, dict):
                live["running"] = False
                live["exit_code"] = 0
                live["finished_at"] = datetime.now(timezone.utc).isoformat()
                live["message"] = "Firmware uploaded successfully."
    except Exception as exc:
        _append_firmware_log(session_id, f"✗ Prebuilt firmware flash failed: {exc!r}")
        with FIRMWARE_LOCK:
            live = FIRMWARE_SESSIONS.get(session_id)
            if isinstance(live, dict):
                live["running"] = False
                live["exit_code"] = 999
                live["finished_at"] = datetime.now(timezone.utc).isoformat()
                live["message"] = f"Firmware flash failed: {exc}"


def _dedupe_discovered_devices(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    clean_devices: List[Dict[str, Any]] = []
    for item in devices:
        host = str(item.get("host") or "").strip()
        name = str(item.get("name") or host or "ESPHome device").strip()
        if not host:
            continue
        key = (host.lower(), int(item.get("port") or FIRMWARE_DEFAULT_OTA_PORT))
        if key in seen:
            continue
        seen.add(key)
        clean_devices.append(
            {
                "name": name,
                "host": host,
                "port": int(item.get("port") or FIRMWARE_DEFAULT_OTA_PORT),
                "source": item.get("source") or "mdns",
            }
        )
    return sorted(clean_devices, key=lambda row: (str(row.get("name") or "").lower(), str(row.get("host") or "")))


def _discover_with_zeroconf(timeout_seconds: float) -> List[Dict[str, Any]]:
    try:
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    except Exception:
        return []

    devices: List[Dict[str, Any]] = []

    class Listener(ServiceListener):
        def add_service(self, zeroconf, service_type, name):
            info = zeroconf.get_service_info(service_type, name, timeout=1000)
            if info is None:
                return
            addresses = []
            for raw in getattr(info, "addresses", []) or []:
                try:
                    family = socket.AF_INET6 if len(raw) == 16 else socket.AF_INET
                    addresses.append(socket.inet_ntop(family, raw))
                except Exception:
                    continue
            host = addresses[0] if addresses else str(getattr(info, "server", "") or "").rstrip(".")
            label = name.replace(service_type, "").rstrip(".") or host
            devices.append(
                {
                    "name": label,
                    "host": host,
                    "port": FIRMWARE_DEFAULT_OTA_PORT,
                    "source": "zeroconf",
                }
            )

        def update_service(self, zeroconf, service_type, name):
            self.add_service(zeroconf, service_type, name)

        def remove_service(self, zeroconf, service_type, name):
            return None

    zeroconf = Zeroconf()
    try:
        ServiceBrowser(zeroconf, "_esphomelib._tcp.local.", Listener())
        time.sleep(max(0.5, timeout_seconds))
    finally:
        zeroconf.close()
    return _dedupe_discovered_devices(devices)


def _discover_with_dns_sd(timeout_seconds: float) -> List[Dict[str, Any]]:
    if not shutil.which("dns-sd"):
        return []
    try:
        proc = subprocess.Popen(
            ["dns-sd", "-B", "_esphomelib._tcp", "local"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return []

    try:
        time.sleep(max(0.5, timeout_seconds))
        proc.terminate()
        output, _ = proc.communicate(timeout=1.5)
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()
        output = ""

    devices: List[Dict[str, Any]] = []
    for line in (output or "").splitlines():
        if " Add " not in f" {line} " or "_esphomelib._tcp" not in line:
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        instance = parts[-1].strip()
        if not instance or instance.lower().startswith("local."):
            continue
        hostname = f"{re.sub(r'[^A-Za-z0-9_-]+', '-', instance).strip('-')}.local"
        host = hostname
        try:
            host = socket.gethostbyname(hostname)
        except Exception:
            pass
        devices.append(
            {
                "name": instance,
                "host": host,
                "port": FIRMWARE_DEFAULT_OTA_PORT,
                "source": "dns-sd",
            }
        )
    return _dedupe_discovered_devices(devices)


def _discover_esphome_devices() -> tuple[List[Dict[str, Any]], str]:
    devices = _discover_with_zeroconf(FIRMWARE_DISCOVERY_SECONDS)
    if devices:
        return devices, f"Found {len(devices)} ESPHome device{'' if len(devices) == 1 else 's'} with mDNS."

    devices = _discover_with_dns_sd(FIRMWARE_DISCOVERY_SECONDS)
    if devices:
        return devices, f"Found {len(devices)} ESPHome device{'' if len(devices) == 1 else 's'} with dns-sd."

    return [], "No ESPHome devices were auto-detected. Enter the device IP or hostname manually."


# -------------------- Routes --------------------
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
        "notes": notes or extra_meta.get("notes") or "",
        "converted": result["converted"],
        "detected_format": result["detected_format"],
        "final_format": result["final_format"],
        "postprocess": result["postprocess"],
        "message": result["message"],
        "review_status": "pending",
    }
    _write_sidecar_json(audio_path, sidecar)

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
        "notes": x_notes or "",
        "converted": result["converted"],
        "detected_format": result["detected_format"],
        "final_format": result["final_format"],
        "postprocess": result["postprocess"],
        "message": result["message"],
        "review_status": "pending",
    }
    _write_sidecar_json(audio_path, sidecar)

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
        _remove_audio_with_sidecar(path)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
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


@app.get("/api/firmware/devices")
def firmware_devices():
    devices, message = _discover_esphome_devices()
    return {"ok": True, "devices": devices, "message": message}


@app.get("/api/firmware/templates")
def firmware_templates(request: Request, target_host: str = "", target_port: str = ""):
    templates = []
    base_url = _request_base_url(request)
    wake_words = _list_trained_wake_words(base_url)
    selected_host, selected_port = _firmware_profile_target(target_host, target_port)
    for spec in FIRMWARE_TEMPLATE_SPECS:
        key = str(spec.get("key") or "")
        profile_key = _firmware_profile_key(key, target_host, target_port)
        profile = _load_firmware_profile(key, profile_key)
        row_target_host = selected_host or str(profile.get("__target_host") or "")
        row_target_port = selected_port or str(profile.get("__target_port") or "")
        if row_target_port == "6053":
            row_target_port = str(FIRMWARE_DEFAULT_OTA_PORT)
        prebuilt = _prebuilt_firmware_info(key)
        prebuilt_summary = _prebuilt_artifact_ui_summary(prebuilt)
        row = {
            "value": key,
            "label": str(spec.get("label") or key),
            "description": str(spec.get("description") or ""),
            "source_url": _text(prebuilt.get("manifest_url")),
            "target_host": row_target_host,
            "target_port": row_target_port,
            "fields": [],
            "prebuilt_firmware_available": bool(prebuilt_summary.get("available")),
            "prebuilt_firmware": prebuilt_summary,
            "firmware_version": _text(prebuilt.get("version")),
        }
        templates.append(row)
    active = next((row["value"] for row in templates if row.get("prebuilt_firmware_available")), "")
    return {
        "ok": True,
        "templates": templates,
        "active_template_key": active or (templates[0]["value"] if templates else ""),
        "wake_words": wake_words,
        "warnings": [],
    }


@app.post("/api/firmware/profile")
def firmware_profile(payload: Dict[str, Any]):
    body = payload if isinstance(payload, dict) else {}
    try:
        template_key = str(body.get("template_key") or "").strip()
        _firmware_template_spec(template_key)
        values = body.get("values") if isinstance(body.get("values"), dict) else {}
        profile_key = _firmware_profile_key(template_key, values.get("__target_host"), values.get("__target_port"))
        host, port = _firmware_profile_target(values.get("__target_host"), values.get("__target_port"))
        saved = {}
        if host:
            saved["__target_host"] = host
        if port:
            saved["__target_port"] = port
        _save_firmware_profile(profile_key or template_key, saved)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, "template_key": template_key, "profile_key": profile_key or template_key, "saved_fields": sorted(saved.keys())}


@app.get("/api/trained_wake_words/catalog")
def trained_wake_words_catalog(request: Request):
    return {"ok": True, "wake_words": _list_trained_wake_words(_request_base_url(request))}


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


@app.post("/api/firmware/build_flash")
def firmware_build_flash(payload: Dict[str, Any]):
    body = payload if isinstance(payload, dict) else {}
    try:
        target_host, target_port = _parse_flash_target(str(body.get("host") or ""), body.get("port"))
        template_key = str(body.get("template_key") or "").strip()
        template_spec = _firmware_template_spec(template_key)
        prebuilt = _prebuilt_firmware_info(template_key)
        if not bool(prebuilt.get("available")):
            raise RuntimeError(_text(prebuilt.get("error")) or "No prebuilt OTA image is available for this firmware target.")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    values = body.get("values") if isinstance(body.get("values"), dict) else {}
    with contextlib.suppress(Exception):
        profile_key = _firmware_profile_key(template_key, target_host, target_port)
        _save_firmware_profile(profile_key or template_key, {"__target_host": target_host, "__target_port": str(target_port)})
    session_id = f"fw_{uuid.uuid4().hex}"
    session = {
        "id": session_id,
        "mode": "prebuilt_ota_flash",
        "running": True,
        "exit_code": None,
        "host": target_host,
        "port": target_port,
        "template_key": template_key,
        "template_label": str(template_spec.get("label") or template_key),
        "firmware_version": _text(prebuilt.get("version")),
        "filename": "",
        "values": values,
        "message": "Preparing prebuilt firmware flash.",
        "log_lines": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "pid": None,
    }
    with FIRMWARE_LOCK:
        FIRMWARE_SESSIONS[session_id] = session

    worker = threading.Thread(target=_run_firmware_build_flash_background, args=(session_id,), daemon=True)
    worker.start()
    return _firmware_session_payload(session_id)


@app.post("/api/firmware/browser_flash")
def firmware_browser_flash(payload: Dict[str, Any]):
    body = payload if isinstance(payload, dict) else {}
    try:
        template_key = str(body.get("template_key") or "").strip()
        template_spec = _firmware_template_spec(template_key)
        prebuilt = _prebuilt_firmware_info(template_key, force_refresh=True)
        if not bool(prebuilt.get("available")):
            raise RuntimeError(_text(prebuilt.get("error")) or "No prebuilt firmware image is available for this firmware target.")
        _prebuilt_artifact_meta(prebuilt, "factory")
        binary = _download_prebuilt_firmware_binary(template_key, prebuilt, "factory", force_refresh=True)
        artifact = _create_browser_flash_artifact(template_key, prebuilt, Path(binary["path"]))
        template_label = str(template_spec.get("label") or template_key)
        cached_text = "cached" if bool(binary.get("cached")) else "downloaded"
        return {
            "ok": True,
            "template_key": template_key,
            "template_label": template_label,
            "firmware_version": _text(prebuilt.get("version")),
            "message": f"Prepared prebuilt USB firmware for {template_label}.",
            "entries": [
                f"Using prebuilt {template_label} factory firmware {_text(prebuilt.get('version')) or 'latest'}.",
                f"Factory image {cached_text}: {Path(binary['path']).name}",
                f"Browser flash binary ready: {artifact['binary_name']} ({artifact['binary_size']} bytes)",
            ],
            **artifact,
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/firmware/browser_flash/{artifact_id}/{filename}")
def firmware_browser_flash_binary(artifact_id: str, filename: str):
    try:
        target = _browser_flash_artifact_path(artifact_id, filename)
    except KeyError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    return FileResponse(
        str(target),
        media_type="application/octet-stream",
        filename=target.name,
    )


@app.post("/api/firmware/clean")
def firmware_clean():
    active = []
    with FIRMWARE_LOCK:
        for session in FIRMWARE_SESSIONS.values():
            if isinstance(session, dict) and bool(session.get("running")):
                active.append(str(session.get("host") or session.get("template_key") or "firmware session"))
    if active:
        return JSONResponse({"ok": False, "error": f"Wait for active firmware session(s) to finish: {', '.join(active[:3])}."}, status_code=400)

    removed = []
    for child in ("prebuilt_firmware", "uploads", "web_flash"):
        path = FIRMWARE_CACHE_DIR / child
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(child)
    return {
        "ok": True,
        "removed": removed,
        "message": "Cleaned downloaded firmware images." if removed else "No downloaded firmware images needed cleaning.",
    }


@app.post("/api/firmware/flash")
async def firmware_flash(
    file: UploadFile = File(...),
    host: str = Form(...),
    port: str | None = Form(None),
    password: str | None = Form(None),
):
    try:
        target_host, target_port = _parse_flash_target(host, port)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    original_name = file.filename or "firmware.bin"
    filename = _normalize_firmware_filename(original_name)
    if not filename.lower().endswith(".bin"):
        return JSONResponse({"ok": False, "error": "Firmware upload must be a compiled .bin file."}, status_code=400)

    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "Firmware file is empty."}, status_code=400)

    session_id = f"fw_{uuid.uuid4().hex}"
    session_dir = FIRMWARE_CACHE_DIR / "uploads" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    firmware_path = session_dir / filename
    firmware_path.write_bytes(data)

    session = {
        "id": session_id,
        "running": True,
        "exit_code": None,
        "host": target_host,
        "port": target_port,
        "password": password or "",
        "filename": filename,
        "firmware_path": str(firmware_path),
        "message": "Preparing firmware upload.",
        "log_lines": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "pid": None,
    }
    with FIRMWARE_LOCK:
        FIRMWARE_SESSIONS[session_id] = session

    worker = threading.Thread(target=_run_firmware_flash_background, args=(session_id,), daemon=True)
    worker.start()
    return _firmware_session_payload(session_id)


@app.get("/api/firmware/flash_status/{session_id}")
def firmware_flash_status(session_id: str):
    payload = _firmware_session_payload(session_id)
    if not payload.get("ok"):
        return JSONResponse(payload, status_code=404)
    return payload


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

    t = threading.Thread(target=_run_training_background, args=(safe_word, language, allow_no_personal), daemon=True)
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
    return {"ok": True, "negative_count": len(_list_negative_samples())}
