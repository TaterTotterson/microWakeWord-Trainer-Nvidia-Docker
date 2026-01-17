# recorder_server.py
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Query
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

# If you want cleanup defaults for auto dataset setup, set these env vars:
#   REC_DATASET_CLEANUP_ARCHIVES=true/false
#   REC_DATASET_CLEANUP_INTERMEDIATE_FILES=true/false
DATASET_CLEANUP_ARCHIVES = os.environ.get("REC_DATASET_CLEANUP_ARCHIVES", "false").lower() in ("1", "true", "yes", "y")
DATASET_CLEANUP_INTERMEDIATE = os.environ.get("REC_DATASET_CLEANUP_INTERMEDIATE_FILES", "false").lower() in ("1", "true", "yes", "y")

# We want "Start training" to trigger your CLI entrypoint, using the existing venv
# (train_wake_word should be in /data/.venv/bin via setup_python_venv)
TRAIN_CMD = os.environ.get(
    "TRAIN_CMD",
    f"source '{DATA_DIR}/.venv/bin/activate' && train_wake_word --data-dir '{DATA_DIR}'"
)

TAKES_PER_SPEAKER_DEFAULT = int(os.environ.get("REC_TAKES_PER_SPEAKER", "10"))
SPEAKERS_TOTAL_DEFAULT = int(os.environ.get("REC_SPEAKERS_TOTAL", "1"))

# How many lines to show in WebUI (tail)
TRAIN_LOG_TAIL_LINES = int(os.environ.get("REC_TRAIN_LOG_TAIL_LINES", "400"))
# If you prefer bytes-based tailing (fast), keep this non-zero.
TRAIN_LOG_MAX_BYTES = int(os.environ.get("REC_TRAIN_LOG_MAX_BYTES", str(512 * 1024)))  # 512KB

app = FastAPI(title="microWakeWord Personal Recorder")

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

    "speakers_total": SPEAKERS_TOTAL_DEFAULT,
    "takes_per_speaker": TAKES_PER_SPEAKER_DEFAULT,

    "takes_received": 0,
    "takes": [],

    "training": {
        "running": False,
        "exit_code": None,
        "log_lines": [],      # legacy in-memory tail (still maintained)
        "log_path": None,     # path to recorder_training.log
        "safe_word": None,

        # NEW: byte offset for efficient log tailing
        "log_offset": 0,
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


def _append_train_log(line: str):
    line = (line or "").rstrip("\n")
    with STATE_LOCK:
        buf: List[str] = STATE["training"]["log_lines"]
        buf.append(line)
        if len(buf) > 250:
            del buf[: (len(buf) - 250)]


def _title_from_phrase(raw_phrase: str) -> str:
    # Keep it human-friendly for the optional <wake_word_title> argument
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
    """
    Run a command streaming stdout/stderr to both:
      - recorder_training.log (disk)
      - STATE["training"]["log_lines"] (UI) [best-effort]
    Returns process exit code.
    """
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
    """
    Ensure /data/.venv exists by running cli/setup_python_venv if needed.
    """
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
    """
    Always run setup_training_datasets before training.
    The underlying scripts should skip work when already done.
    """
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


def _read_log_tail_by_bytes(log_path: Path, max_bytes: int) -> str:
    """
    Read up to the last max_bytes from a file (UTF-8 best effort).
    """
    if not log_path.exists():
        return ""

    try:
        size = log_path.stat().st_size
        start = max(0, size - max_bytes)
        with open(log_path, "rb") as f:
            f.seek(start)
            data = f.read()
        # If we started in the middle of a line, it's ok; UI will show partial.
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _read_log_tail_by_lines(log_path: Path, max_lines: int) -> str:
    """
    Read last N lines of a file (simple, may be slower on huge files).
    """
    if not log_path.exists():
        return ""
    try:
        # Read by bytes limited first, then line-tail
        raw = _read_log_tail_by_bytes(log_path, TRAIN_LOG_MAX_BYTES)
        if not raw:
            return ""
        lines = raw.splitlines()
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _read_log_since_offset(log_path: Path, offset: int, max_bytes: int = 256 * 1024) -> Tuple[str, int]:
    """
    Read log file incrementally starting from `offset`.
    Returns (new_text, new_offset). Caps bytes read per call.
    """
    if not log_path.exists():
        return ("", offset)

    try:
        size = log_path.stat().st_size
        # If file rotated/truncated, reset offset
        if offset > size:
            offset = 0

        with open(log_path, "rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)

        new_offset = offset + len(data)
        text = data.decode("utf-8", errors="replace")
        return (text, new_offset)
    except Exception:
        return ("", offset)


def _run_training_background(safe_word: str, allow_no_personal: bool):
    with STATE_LOCK:
        raw_phrase = STATE.get("raw_phrase") or ""

    wake_word_title = _title_from_phrase(raw_phrase)

    with STATE_LOCK:
        STATE["training"]["running"] = True
        STATE["training"]["exit_code"] = None
        STATE["training"]["log_lines"] = []
        STATE["training"]["safe_word"] = safe_word
        log_path = Path(str(DATA_DIR / "recorder_training.log"))
        STATE["training"]["log_path"] = str(log_path)
        STATE["training"]["log_offset"] = 0

    # fresh header at the start of a run
    _append_train_log("================================================================================")
    _append_train_log("===== Recorder Training Run =====")
    _append_train_log("================================================================================")

    # Ensure the log exists and starts cleanly with a header separator for this run
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n" + ("=" * 80) + "\n")
            lf.write("===== Recorder Training Run =====\n")
            lf.write(("=" * 80) + "\n")
            lf.flush()
    except Exception:
        pass

    try:
        # 1) Ensure venv (auto-installs)
        _ensure_training_venv(log_path)

        # 2) Ensure datasets (auto-installs / skips if already present)
        _ensure_training_datasets(log_path)

        # 3) Run training
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

    except Exception as e:
        _append_train_log(f"✗ Training crashed: {e!r}")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = 999

    finally:
        with STATE_LOCK:
            STATE["training"]["running"] = False


# -------------------- Routes --------------------
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
        # do not interrupt training if running

    _reset_personal_samples_dir()

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
def train_status(
    offset: int = Query(0, ge=0),
    max_bytes: int = Query(65536, ge=1024, le=262144),
    last_size: int = Query(0, ge=0),
    last_mtime: float = Query(0.0, ge=0.0),
):
    """
    Stream training output from the log file on disk.

    Robust to log overwrite/truncation:
      - UI passes offset + last_size + last_mtime
      - If file shrinks or mtime goes backwards/changes weirdly, reset offset to 0
    """
    with STATE_LOCK:
        tr = dict(STATE["training"])
        log_path_str = tr.get("log_path")

    log_text = ""
    next_offset = offset
    log_size = 0
    log_mtime = 0.0

    if log_path_str:
        p = Path(log_path_str)
        if p.exists():
            try:
                st = p.stat()
                log_size = int(st.st_size)
                log_mtime = float(st.st_mtime)

                # Detect overwrite/truncate/reset:
                # - file shrank
                # - file mtime moved "backwards" (rare) or changed while size reset
                # If anything indicates a reset, restart from beginning.
                if (log_size < last_size) or (last_mtime and log_mtime < last_mtime):
                    offset = 0

                # Clamp offset to current file size
                if offset > log_size:
                    offset = log_size

                # Read incrementally from the file
                with p.open("rb") as f:
                    f.seek(offset)
                    chunk = f.read(max_bytes)

                log_text = chunk.decode("utf-8", errors="replace")
                next_offset = offset + len(chunk)

            except Exception as e:
                log_text = f"\n[log read error: {e!r}]\n"
                next_offset = offset

    tr["log_text"] = log_text
    tr["next_offset"] = next_offset
    tr["log_size"] = log_size
    tr["log_mtime"] = log_mtime

    return {"ok": True, "training": tr}


@app.post("/api/reset_recordings")
def reset_recordings():
    _reset_personal_samples_dir()
    with STATE_LOCK:
        STATE["takes_received"] = 0
        STATE["takes"] = []
    return {"ok": True}