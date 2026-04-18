# trainer_server.py
import io
import os
import re
import json
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.request import Request, urlopen

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent

# In Docker CLI world, DATA_DIR should be /data
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data")).resolve()

# UI files live next to this script by default
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT_DIR / "static"))).resolve()

# Personal samples MUST land in /data/personal_samples for your CLI pipeline
PERSONAL_DIR = Path(os.environ.get("PERSONAL_DIR", str(DATA_DIR / "personal_samples"))).resolve()

# CLI folder inside repo
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
        str(ROOT_DIR / ".cache" / "piper_voices_catalog.json"),
    )
).resolve()

DATASET_CLEANUP_ARCHIVES = os.environ.get("REC_DATASET_CLEANUP_ARCHIVES", "false").lower() in ("1", "true", "yes", "y")
DATASET_CLEANUP_INTERMEDIATE = os.environ.get("REC_DATASET_CLEANUP_INTERMEDIATE_FILES", "false").lower() in ("1", "true", "yes", "y")

TRAIN_CMD = os.environ.get(
    "TRAIN_CMD",
    f"source '{DATA_DIR}/.venv/bin/activate' && train_wake_word --data-dir '{DATA_DIR}'"
)

DEFAULT_LANGUAGE = os.environ.get("MWW_LANGUAGE", "en")

TAKES_PER_SPEAKER_DEFAULT = int(os.environ.get("REC_TAKES_PER_SPEAKER", "10"))
SPEAKERS_TOTAL_DEFAULT = int(os.environ.get("REC_SPEAKERS_TOTAL", "1"))
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2

# Tail lines shown to UI
TRAIN_LOG_TAIL_LINES = int(os.environ.get("REC_TRAIN_LOG_TAIL_LINES", "400"))
# Safety cap for reads (bytes) to avoid giant file reads
TRAIN_LOG_MAX_BYTES = int(os.environ.get("REC_TRAIN_LOG_MAX_BYTES", str(512 * 1024)))  # 512KB

app = FastAPI(title="microWakeWord Personal Samples")

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def safe_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"^_+|_+$", "", s)
    return s or "wakeword"


STATE: Dict[str, Any] = {
    "raw_phrase": None,
    "safe_word": None,
    "language": DEFAULT_LANGUAGE,

    "speakers_total": SPEAKERS_TOTAL_DEFAULT,
    "takes_per_speaker": TAKES_PER_SPEAKER_DEFAULT,

    "takes_received": 0,
    "takes": [],

    "training": {
        "running": False,
        "exit_code": None,
        "log_lines": [],          # legacy in-memory tail (kept, but not relied on)
        "log_path": None,         # path to recorder_training.log
        "safe_word": None,

        # prevent UI duplication when UI appends:
        "last_sent_tail": [],      # last tail snapshot (list of lines)
        "last_log_size": 0,        # detect truncation
    },
}

STATE_LOCK = threading.Lock()
SAMPLES_LOCK = threading.Lock()
PIPER_CATALOG_LOCK = threading.Lock()
PIPER_CATALOG_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "entries": None,
}


def _reset_personal_samples_dir():
    PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
    for p in PERSONAL_DIR.glob("*.wav"):
        try:
            p.unlink()
        except Exception:
            pass



def _list_personal_samples() -> List[str]:
    PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in PERSONAL_DIR.glob("*.wav"))


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


def _fetch_piper_catalog() -> Optional[Dict[str, Any]]:
    req = Request(
        PIPER_VOICES_INDEX_URL,
        headers={"User-Agent": "microWakeWord-Trainer/1.0"},
    )
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else None


def _read_cached_piper_catalog_file() -> Optional[Dict[str, Any]]:
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


def _load_piper_catalog() -> Optional[Dict[str, Any]]:
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


def _catalog_voice_files(language_family: str) -> List[Tuple[str, str]]:
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


def _find_ffmpeg() -> Optional[str]:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _inspect_wav_bytes(data: bytes) -> Optional[Dict[str, Any]]:
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


def _is_target_wav(info: Optional[Dict[str, Any]]) -> bool:
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
    current = _list_personal_samples()
    next_index = 1
    for name in current:
        match = re.match(r"sample_(\d{4})", name)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)

    stem = safe_name(Path(original_name or "sample").stem)
    suffix = f"_{stem[:32]}" if stem and stem != "wakeword" else ""
    return f"sample_{next_index:04d}{suffix}.wav"


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


def _save_personal_sample(data: bytes, original_name: str, out_name: Optional[str] = None) -> Dict[str, Any]:
    if not data:
        raise ValueError("Empty or invalid audio file.")

    original_info = _inspect_wav_bytes(data) or _format_hint_from_filename(original_name)
    normalized = _is_target_wav(original_info)
    final_bytes = data if normalized else _normalize_audio_to_target_wav(data, original_name)
    final_info = _inspect_wav_bytes(final_bytes)

    if not _is_target_wav(final_info):
        raise ValueError("Uploaded audio could not be normalized to 16 kHz mono 16-bit PCM WAV.")

    with SAMPLES_LOCK:
        PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
        final_name = out_name or _next_personal_sample_name(original_name)
        out_path = PERSONAL_DIR / final_name
        out_path.write_bytes(final_bytes)

    return {
        "saved_as": final_name,
        "converted": not normalized,
        "original_name": original_name or final_name,
        "detected_format": original_info,
        "final_format": final_info,
        "message": (
            "Converted to 16 kHz mono 16-bit PCM WAV"
            if not normalized
            else "Already in the correct 16 kHz mono 16-bit PCM WAV format"
        ),
    }

def _clear_training_log():
    """
    Truncate recorder_training.log for a fresh session.
    """
    log_path = DATA_DIR / "recorder_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("================================================================================\n")
        lf.write("===== New recorder session started =====\n")
        lf.write("================================================================================\n")
        lf.flush()

    with STATE_LOCK:
        STATE["training"]["log_path"] = str(log_path)
        STATE["training"]["log_lines"] = []
        STATE["training"]["last_sent_tail"] = []
        STATE["training"]["last_log_size"] = 0


def _append_train_log(line: str):
    line = (line or "").rstrip("\n")
    with STATE_LOCK:
        buf: List[str] = STATE["training"]["log_lines"]
        buf.append(line)
        if len(buf) > 250:
            del buf[: (len(buf) - 250)]


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
    """
    Read the last N lines, bounded by TRAIN_LOG_MAX_BYTES.
    Returns list of lines (no trailing newlines).
    """
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
    """
    Given previous and current tail snapshots, return only the newly-added lines.
    Works even if the tail window shifts.
    """
    if not prev_tail:
        return new_tail

    # Try to find the largest suffix of prev_tail that matches a prefix of new_tail
    max_k = min(len(prev_tail), len(new_tail))
    for k in range(max_k, 0, -1):
        if prev_tail[-k:] == new_tail[:k]:
            return new_tail[k:]

    # If no overlap, just return full new_tail (probably truncation or big jump)
    return new_tail


# -------------------- output artifact normalization --------------------

def _find_latest_output_pair(output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find the most recently modified .tflite and its matching .json (same basename)
    in output_dir. Falls back to newest .json if an exact match doesn't exist.
    Returns (tflite_path, json_path) or (None, None).
    """
    if not output_dir.exists():
        return (None, None)

    tflites = sorted(output_dir.glob("*.tflite"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tflites:
        return (None, None)

    tfl = tflites[0]
    js = tfl.with_suffix(".json")
    if js.exists():
        return (tfl, js)

    jsons = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (tfl, jsons[0] if jsons else None)


def _deep_replace_strings(obj: Any, old: str, new: str) -> Any:
    """
    Recursively replace occurrences of old in any string values with new.
    """
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_deep_replace_strings(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_replace_strings(v, old, new) for k, v in obj.items()}
    return obj


def _normalize_output_artifacts(safe_word: str, log_path: Path) -> None:
    """
    Rename output artifacts to <safe_word>.tflite / <safe_word>.json
    and patch the JSON so it references the renamed tflite.

    Handles weird trainer names like ____r_.tflite by normalizing post-run.
    """
    out_dir = DATA_DIR / "output"
    tfl, js = _find_latest_output_pair(out_dir)

    if not tfl:
        _append_train_log(f"⚠️ No .tflite found in {out_dir}")
        return

    new_tfl = out_dir / f"{safe_word}.tflite"
    new_js = out_dir / f"{safe_word}.json"
    old_tfl_name = tfl.name

    # Already normalized
    if tfl.name == new_tfl.name and (js and js.name == new_js.name):
        _append_train_log(f"✅ Output names already normalized: {new_tfl.name}")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def backup_if_exists(p: Path, suffix: str) -> None:
        if p.exists():
            bk = out_dir / f"{safe_word}.{ts}.bak{suffix}"
            shutil.move(str(p), str(bk))
            _append_train_log(f"↪️ Backed up existing {p.name} → {bk.name}")

    # Avoid clobbering existing target files (back them up)
    if new_tfl.exists() and new_tfl.resolve() != tfl.resolve():
        backup_if_exists(new_tfl, ".tflite")
    if new_js.exists() and (not js or new_js.resolve() != js.resolve()):
        backup_if_exists(new_js, ".json")

    # Rename tflite
    if tfl.resolve() != new_tfl.resolve():
        new_tfl.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tfl), str(new_tfl))
        _append_train_log(f"✅ Renamed model: {old_tfl_name} → {new_tfl.name}")

    # Rename + patch json if present
    if js and js.exists():
        # Read JSON before move (safer if we want the old name)
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            data = None

        if js.resolve() != new_js.resolve():
            shutil.move(str(js), str(new_js))
            _append_train_log(f"✅ Renamed metadata: {js.name} → {new_js.name}")

        if data is not None:
            patched = _deep_replace_strings(data, old_tfl_name, new_tfl.name)

            # Patch common keys if present
            for key in ("model", "model_file", "model_filename", "tflite", "tflite_file", "tflite_filename"):
                if isinstance(patched, dict) and key in patched and isinstance(patched[key], str):
                    patched[key] = new_tfl.name

            new_js.write_text(json.dumps(patched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _append_train_log(f"✅ Patched JSON to reference: {new_tfl.name}")
    else:
        _append_train_log("⚠️ No .json found to patch (model renamed only)")


# -------------------- training worker --------------------

def _run_training_background(safe_word: str, allow_no_personal: bool):
    with STATE_LOCK:
        raw_phrase = STATE.get("raw_phrase") or ""
        language = STATE.get("language") or DEFAULT_LANGUAGE

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
    _append_train_log("===== Recorder Training Run =====")
    _append_train_log("================================================================================")

    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n" + ("=" * 80) + "\n")
            lf.write("===== Recorder Training Run =====\n")
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

        # Normalize output artifact names on success
        if rc == 0:
            _normalize_output_artifacts(safe_word, log_path)

    except Exception as e:
        _append_train_log(f"✗ Training crashed: {e!r}")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = 999

    finally:
        with STATE_LOCK:
            STATE["training"]["running"] = False


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h3>Missing UI</h3><p>Create <code>static/index.html</code>.</p>",
            status_code=500,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


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

    takes = _sync_personal_samples_state()

    # Always wipe log on start_session (even if same wakeword)
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


@app.post("/api/train")
def train_now(payload: Dict[str, Any] = None):
    payload = payload or {}
    allow_no_personal = bool(payload.get("allow_no_personal", False))

    with STATE_LOCK:
        safe_word = STATE["safe_word"]
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

    t = threading.Thread(target=_run_training_background, args=(safe_word, allow_no_personal), daemon=True)
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
    """
    Return only NEW lines since last poll (prevents UI duplication spam even if UI appends).
    """
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

            # If file was truncated/cleared, reset history
            if size_now < prev_size:
                prev_tail = []

            full_tail = _read_tail_lines(p, TRAIN_LOG_TAIL_LINES)
            new_lines = _compute_new_lines(prev_tail, full_tail)

    # Save snapshot for next poll
    with STATE_LOCK:
        STATE["training"]["last_sent_tail"] = full_tail
        STATE["training"]["last_log_size"] = size_now

    tr["log_text"] = "\n".join(new_lines)  # ONLY new lines
    tr["log_tail_preview"] = "\n".join(full_tail)  # optional: handy for debugging
    tr["log_lines"] = full_tail
    return {"ok": True, "training": tr}


@app.post("/api/reset_recordings")
def reset_recordings():
    _reset_personal_samples_dir()
    takes = _sync_personal_samples_state()
    return {"ok": True, "takes_received": len(takes), "takes": takes}
