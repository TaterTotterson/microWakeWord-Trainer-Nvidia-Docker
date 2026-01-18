# recorder_server.py
import os
import re
import json
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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

DATASET_CLEANUP_ARCHIVES = os.environ.get("REC_DATASET_CLEANUP_ARCHIVES", "false").lower() in ("1", "true", "yes", "y")
DATASET_CLEANUP_INTERMEDIATE = os.environ.get("REC_DATASET_CLEANUP_INTERMEDIATE_FILES", "false").lower() in ("1", "true", "yes", "y")

TRAIN_CMD = os.environ.get(
    "TRAIN_CMD",
    f"source '{DATA_DIR}/.venv/bin/activate' && train_wake_word --data-dir '{DATA_DIR}'"
)

TAKES_PER_SPEAKER_DEFAULT = int(os.environ.get("REC_TAKES_PER_SPEAKER", "10"))
SPEAKERS_TOTAL_DEFAULT = int(os.environ.get("REC_SPEAKERS_TOTAL", "1"))

# Tail lines shown to UI
TRAIN_LOG_TAIL_LINES = int(os.environ.get("REC_TRAIN_LOG_TAIL_LINES", "400"))
# Safety cap for reads (bytes) to avoid giant file reads
TRAIN_LOG_MAX_BYTES = int(os.environ.get("REC_TRAIN_LOG_MAX_BYTES", str(512 * 1024)))  # 512KB

app = FastAPI(title="microWakeWord Personal Recorder")

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


def _reset_personal_samples_dir():
    PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
    for p in PERSONAL_DIR.glob("*.wav"):
        try:
            p.unlink()
        except Exception:
            pass


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

        if wake_word_title:
            cmd_str = f"{TRAIN_CMD} '{safe_word}' '{wake_word_title}'"
        else:
            cmd_str = f"{TRAIN_CMD} '{safe_word}'"

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

    speakers_total = max(1, min(10, speakers_total))
    takes_per_speaker = max(1, min(50, takes_per_speaker))

    with STATE_LOCK:
        STATE["raw_phrase"] = raw
        STATE["safe_word"] = safe
        STATE["speakers_total"] = speakers_total
        STATE["takes_per_speaker"] = takes_per_speaker
        STATE["takes_received"] = 0
        STATE["takes"] = []

    _reset_personal_samples_dir()

    # Always wipe log on start_session (even if same wakeword)
    _clear_training_log()

    return {
        "ok": True,
        "raw_phrase": raw,
        "safe_word": safe,
        "speakers_total": speakers_total,
        "takes_per_speaker": takes_per_speaker,
        "takes_total": speakers_total * takes_per_speaker,
        "personal_dir": str(PERSONAL_DIR),
        "data_dir": str(DATA_DIR),
    }


@app.get("/api/session")
def get_session():
    with STATE_LOCK:
        return {
            "ok": True,
            "raw_phrase": STATE["raw_phrase"],
            "safe_word": STATE["safe_word"],
            "speakers_total": STATE["speakers_total"],
            "takes_per_speaker": STATE["takes_per_speaker"],
            "takes_received": STATE["takes_received"],
            "takes": list(STATE["takes"]),
            "training": dict(STATE["training"]),
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

    PERSONAL_DIR.mkdir(parents=True, exist_ok=True)

    out_name = f"speaker{speaker_index:02d}_take{take_index:02d}.wav"
    out_path = PERSONAL_DIR / out_name

    data = await file.read()
    if not data or len(data) < 44:
        return JSONResponse({"ok": False, "error": "Empty/invalid file"}, status_code=400)

    out_path.write_bytes(data)

    with STATE_LOCK:
        if out_name not in STATE["takes"]:
            STATE["takes"].append(out_name)
            STATE["takes_received"] = len(STATE["takes"])

    return {"ok": True, "saved_as": out_name, "takes_received": STATE["takes_received"]}


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

    min_required = max(1, min(3, takes_total))

    if takes_received == 0 and not allow_no_personal:
        return JSONResponse(
            {
                "ok": False,
                "error": f"No personal voice samples recorded (0/{takes_total}).",
                "code": "NO_PERSONAL_SAMPLES",
                "message": "You can train without personal voices, or record samples first.",
                "takes_total": takes_total,
            },
            status_code=400,
        )

    if 0 < takes_received < min_required:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Not enough takes yet ({takes_received}/{takes_total}).",
                "code": "NOT_ENOUGH_TAKES",
                "min_required": min_required,
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
        "personal_samples_used": takes_received >= min_required,
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
    return {"ok": True, "training": tr}


@app.post("/api/reset_recordings")
def reset_recordings():
    _reset_personal_samples_dir()
    with STATE_LOCK:
        STATE["takes_received"] = 0
        STATE["takes"] = []
    return {"ok": True}