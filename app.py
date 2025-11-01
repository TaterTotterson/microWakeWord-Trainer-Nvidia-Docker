# app.py
import streamlit as st
import sys
import io
import os
import platform
import uuid
import shutil
import re
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Force-add runtime clone to Python path (so imports work after install)
MW_RUNTIME_DIR = Path("/data/microWakeWord")
if MW_RUNTIME_DIR.exists():
    mw_path = str(MW_RUNTIME_DIR)
    if mw_path not in sys.path:
        sys.path.insert(0, mw_path)

st.set_page_config(page_title="microWakeWord Trainer", layout="wide")

# -------------------------------------------------
# session init
# -------------------------------------------------
if "runs" not in st.session_state:
    st.session_state["runs"] = {}

if "last_wakeword" not in st.session_state:
    st.session_state["last_wakeword"] = None

if "console_lines" not in st.session_state:
    st.session_state["console_lines"] = []


# -------------------------------------------------
# console helpers
# -------------------------------------------------
# 1) tqdm-ish
# 2) requests/wget percent lines
# 3) wget-style "234700K .........." lines
PROGRESS_PATTERNS = [
    re.compile(r"^Downloading\s+.+\d+%"),          # "Downloading xxx.zip: 40%"
    re.compile(r"\d+%\|#+"),                       # "46%|####5     | ..."
    re.compile(r"it/s\]$"),                        # "... 57.8MB/s]"
    re.compile(r"^\d+(?:\.\d+)?[KMG]?\s+(?:\.+\s+)+\d+%"),  # wget: "234700K .......... ... 8%"
]

def is_progress_line(line: str) -> bool:
    line = line.strip()
    for pat in PROGRESS_PATTERNS:
        if pat.search(line):
            return True
    return False


def render_console():
    text = "\n".join(st.session_state["console_lines"])
    st.session_state["console_placeholder"].markdown(
        f"<div style='max-height:350px; overflow-y:auto; border:1px solid #444; background:#111; padding:0.5rem; border-radius:0.5rem;'>"
        f"<pre style='white-space:pre-wrap; font-size:0.8rem; line-height:1.1rem; color:#eee;'>{text}</pre>"
        f"</div>",
        unsafe_allow_html=True,
    )


def append_line(line: str):
    # spacing + newest-first
    st.session_state["console_lines"].insert(0, "")
    st.session_state["console_lines"].insert(0, line)
    st.session_state["console_lines"] = st.session_state["console_lines"][:400]
    render_console()


def update_progress_line(line: str):
    if st.session_state["console_lines"]:
        st.session_state["console_lines"][0] = line
    else:
        st.session_state["console_lines"].insert(0, line)
    render_console()


def run_block(fn, step_name: str, title: str = "Running…"):
    class StreamToUI(io.StringIO):
        def write(self, s):
            if not s:
                return 0

            # wget sometimes sends \r ... \r ... \n  → normalize to lines
            s = s.replace("\r", "\n")

            for chunk in s.splitlines():
                chunk = chunk.rstrip("\n")
                if not chunk:
                    continue

                if is_progress_line(chunk):
                    update_progress_line(chunk)
                else:
                    append_line(chunk)

            return len(s)

    out_stream = StreamToUI()
    err_stream = StreamToUI()

    append_line(f"▶ {step_name} – {title}")
    try:
        with redirect_stdout(out_stream), redirect_stderr(err_stream):
            fn()
    except Exception as e:
        append_line(f"[ERROR] {e}")
    append_line(f"✔ {step_name} – {title} done")

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = Path("/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(DATA_DIR)

# =================================================
# WRAPPERS
# =================================================
def cell_prepare_all_once():
    print("=== RUN ONCE: full environment + install + dataset prep ===")
    cell_env_check()
    cell_install_microwakeword()
    cell_dataset_prep_all()
    print("✅ All run-once steps completed.")


def cell_generate_full():
    # read from sidebar settings (with fallbacks)
    num_samples = st.session_state.get("gen_num_samples", 50000)
    wakeword = st.session_state.get("gen_wakeword", "tater")
    piper_model_url = st.session_state.get(
        "gen_piper_model_url",
        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
    )

    print("=== GENERATE: samples → augment → train+export ===")
    cell_generate_samples(
        num_samples=num_samples,
        wakeword=wakeword,
        model_url=piper_model_url,
    )
    cell_setup_augmentations()
    cell_train_export()
    print("✅ Full generate pipeline completed.")

# =================================================
# CELLS (steps)
# =================================================
def cell_env_check():
    print("=== 1. Env / runtime check ===")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Working dir: {os.getcwd()}")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"PyTorch check failed: {e}")

    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        try:
            devs = tf.config.list_physical_devices()
            print(f"TF devices: {devs}")
        except Exception:
            pass
    except Exception as e:
        print(f"TensorFlow check failed: {e}")

    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
        print(f"ONNX providers: {ort.get_all_providers()}")
    except Exception as e:
        print(f"ONNX check failed: {e}")

    cgroup_path = Path("/proc/1/cgroup")
    if cgroup_path.exists():
        print("cgroup info:")
        print(cgroup_path.read_text().strip())

    print("✅ Env check complete")


def cell_install_microwakeword():
    """
    Clone/install microwakeword from OHF-Voice and install the current main.
    No commit pinning, no hot patching.
    If /data/microWakeWord already exists, we just pull latest.
    """
    import sys, subprocess
    from pathlib import Path

    print("=== 2. Check/install microWakeWord ===")
    repo_dir = Path("/data/microWakeWord")
    repo_url = "https://github.com/TaterTotterson/micro-wake-word.git"

    if not repo_dir.exists():
        print(f"📦 cloning {repo_url} → {repo_dir}")
        subprocess.check_call([
            "git", "clone",
            repo_url,
            str(repo_dir),
        ])
    else:
        print("✅ repo exists, updating…")
        # make sure remote is reachable, then pull
        subprocess.check_call(["git", "-C", str(repo_dir), "fetch", "origin"])
        subprocess.check_call(["git", "-C", str(repo_dir), "checkout", "main"])
        subprocess.check_call(["git", "-C", str(repo_dir), "pull", "origin", "main"])

    print("📦 installing microWakeWord (editable) with THIS python:", sys.executable)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--no-build-isolation",
        "-e", str(repo_dir),
    ])

    # make sure /data/microWakeWord is importable right now
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    # show the version/commit we actually ended up with
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
        print(f"✅ microwakeword installed from commit {rev}")
    except Exception:
        pass

    try:
        import microwakeword  # noqa: F401
        print("✅ microWakeWord loaded (post-install) – ready.")
    except ImportError:
        print("❌ still cannot import microwakeword (check sys.path + venv)")

def cell_generate_samples(num_samples=50000, wakeword="tater", model_url=None):
    import sys
    import subprocess
    import shutil
    import urllib.parse
    import re
    import time
    from pathlib import Path
    import streamlit as st

    # -------------------------------------------------
    # tiny helpers to talk to the existing console
    # -------------------------------------------------
    def _console_render():
        if "console_placeholder" in st.session_state:
            st.session_state["console_placeholder"].markdown(
                f"<div style='max-height:350px; overflow-y:auto; border:1px solid #444; background:#111; padding:0.5rem; border-radius:0.5rem;'>"
                f"<pre style='white-space:pre-wrap; font-size:0.8rem; line-height:1.1rem; color:#eee;'>{'\n'.join(st.session_state.get('console_lines', []))}</pre>"
                f"</div>",
                unsafe_allow_html=True,
            )

    def _console_push(line: str):
        st.session_state.setdefault("console_lines", [])
        st.session_state["console_lines"].insert(0, line)
        st.session_state["console_lines"] = st.session_state["console_lines"][:400]
        _console_render()

    def _console_update_top(line: str):
        st.session_state.setdefault("console_lines", [])
        if st.session_state["console_lines"]:
            st.session_state["console_lines"][0] = line
        else:
            st.session_state["console_lines"].insert(0, line)
        _console_render()

    # -------------------------------------------------
    # 1) clean previous artifacts
    # -------------------------------------------------
    CLEAN_PATHS = [
        "training_parameters.yaml",
        "trained_models",
        "generated_augmented_features",
        "generated_samples",
    ]

    for p in CLEAN_PATHS:
        path = Path(p)
        if path.exists():
            _console_push(f"🧹 removing {path} …")
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    _console_push("✅ environment cleaned, starting new sample generation …")

    REPO_URL = "https://github.com/rhasspy/piper-sample-generator"
    REPO_DIR = Path("/data") / "piper-sample-generator"
    OUT_DIR = Path("/data") / "generated_samples"

    if not model_url:
        model_url = (
            "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/"
            "en_US-libritts_r-medium.pt"
        )

    # we always want a fresh output dir too
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # remember wakeword for export step
    st.session_state["last_wakeword"] = wakeword

    parsed = urllib.parse.urlparse(model_url)
    model_name = Path(parsed.path).name
    MODELS_DIR = REPO_DIR / "models"
    MODEL_PATH = MODELS_DIR / model_name

    # -------------------------------------------------
    # run_logged that can detect "Batch X/500"
    # -------------------------------------------------
    def run_logged(cmd, check=True, detect_batches=False):
        _console_push("→ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        batch_re = re.compile(r"Batch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)

        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue

            if detect_batches:
                m = batch_re.search(line)
                if m:
                    cur, total = m.group(1), m.group(2)
                    _console_update_top(f"🎤 Generating samples… batch {cur}/{total}")
                else:
                    # non-batch lines still get logged once
                    _console_push(line)
            else:
                _console_push(line)

            # let streamlit repaint
            time.sleep(0.01)

        rc = proc.wait()
        if check and rc != 0:
            raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
        return rc

    # -------------------------------------------------
    # 2) clone / update repo
    # -------------------------------------------------
    def safe_clone():
        if REPO_DIR.exists() and not (REPO_DIR / ".git").exists():
            _console_push("⚠️ found partial clone, removing …")
            shutil.rmtree(REPO_DIR, ignore_errors=True)
        if not REPO_DIR.exists():
            _console_push(f"📦 cloning {REPO_URL} → {REPO_DIR}")
            run_logged(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
        else:
            _console_push("✅ piper-sample-generator exists, pulling latest…")
            # ignore failure here, offline is fine
            run_logged(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)

    def ensure_model():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
            _console_push(f"⬇️ downloading model {model_name} …")
            run_logged(["wget", "-O", str(MODEL_PATH), model_url])
            if MODEL_PATH.stat().st_size < 100 * 1024:
                raise RuntimeError("Downloaded model looks too small.")
        _console_push(f"✅ model ready: {MODEL_PATH}")

    def ensure_deps():
        deps = [
            "piper-tts>=1.2.0",
            "piper-phonemize-cross==1.2.1",
            "soundfile",
            "numpy",
            "onnxruntime-gpu>=1.16.0",
        ]
        _console_push("📦 installing piper deps (safe to re-run)…")
        run_logged([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
        run_logged([sys.executable, "-m", "pip", "install", *deps], check=True)

    _console_push(f"🎤 generating {num_samples} samples for wakeword '{wakeword}'")
    _console_push(f"🗣 model: {model_name}")

    safe_clone()
    ensure_deps()
    ensure_model()

    gen_script = REPO_DIR / "generate_samples.py"
    if not gen_script.exists():
        raise FileNotFoundError(f"Missing generator script: {gen_script}")

    cmd = [
        sys.executable,
        str(gen_script),
        wakeword,
        "--model", str(MODEL_PATH),
        "--max-samples", str(num_samples),
        "--batch-size", "100",
        "--output-dir", str(OUT_DIR),
    ]
    # ✅ this is the key bit: detect_batches=True
    run_logged(cmd, detect_batches=True)

    _console_push(f"✅ done — samples in {OUT_DIR}")

def cell_dataset_prep_all():
    """
    3. Prep datasets

    IMPORTANT: we avoid os.system(...) so that Streamlit actually sees the output.
    Everything goes through run_cmd_logged(...) which streams lines to stdout.
    """
    import numpy as np
    import scipy.io.wavfile
    import soundfile as sf
    import librosa
    from tqdm import tqdm
    import zipfile
    import requests
    import subprocess
    from pathlib import Path
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    def run_cmd_logged(cmd, cwd=None):
        """Run a shell cmd and stream its output to Python stdout (so Streamlit sees it)."""
        print("→", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                print(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
        return rc

    def write_wav(dst: Path, data: np.ndarray, sr: int):
        dst.parent.mkdir(parents=True, exist_ok=True)
        x = np.clip(data, -1.0, 1.0)
        scipy.io.wavfile.write(dst, sr, (x * 32767).astype(np.int16))

    # ---------------- MIT RIR ----------------
    print("=== MIT RIR ===")
    rir_out = Path("mit_rirs")
    rir_out.mkdir(exist_ok=True)

    have_rir = any(rir_out.rglob("*.wav"))
    if not have_rir:
        try:
            print("⬇️ Downloading MIT RIR ZIP …")
            zip_url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
            zip_path = rir_out.parent / "MIT_RIR_Audio.zip"
            if not zip_path.exists():
                run_cmd_logged(["wget", "-q", "-O", str(zip_path), zip_url])
            print("📦 Unzipping…")
            run_cmd_logged(["unzip", "-q", "-o", str(zip_path), "-d", str(rir_out)])

            normalized = 0
            bad = []
            wavs = list(rir_out.rglob("*.wav"))
            for p in tqdm(wavs, desc="Normalize MIT RIR → 16k mono"):
                try:
                    a, sr = sf.read(p, always_2d=False)
                    if a is None or a.size == 0:
                        raise ValueError("empty audio")
                    if a.ndim > 1:
                        a = a[:, 0]
                    if sr != 16000:
                        a, _ = librosa.load(p, sr=16000, mono=True)
                    write_wav(p, a, 16000)
                    normalized += 1
                except Exception as e:
                    bad.append(f"{p}:{e}")
            if bad:
                (rir_out / "mit_rir_corrupted_files.log").write_text("\n".join(bad))
            print(f"✅ MIT RIR ready ({normalized} files, {len(bad)} failed)")
        except Exception as e:
            print(f"❌ MIT RIR failed: {e}")
    else:
        print("✅ mit_rirs exists; skipping.")

    # ---------------- AudioSet ----------------
    print("\n=== AudioSet subset (pinned FLAC .tar → 16k mono) ===")
    audioset_dir = Path("audioset"); audioset_dir.mkdir(exist_ok=True)
    audioset_out = Path("audioset_16k"); audioset_out.mkdir(exist_ok=True)

    already_done_audioset = any(audioset_out.rglob("*.wav"))
    if not already_done_audioset:
        # sometimes HF layout changes; we try a couple
        REV_CANDIDATES = [
            "6762f044d1c88619c7f2006486036192128fb07e",
            "0049167e89f259a010c3f070fe3666d9e5242836",
            "ceb9eaaa7844c9ad7351e659c84a572e376ad06d",
        ]
        TAR_PATTERNS = [
            "data/bal_train0{idx}.tar",
            "data/bal_train/bal_train0{idx}.tar",
        ]

        def find_working_rev():
            for rev in REV_CANDIDATES:
                for pat in TAR_PATTERNS:
                    test_url = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/{rev}/{pat.format(idx=0)}"
                    # use wget --spider but stream it
                    try:
                        run_cmd_logged(["wget", "-q", "--spider", test_url])
                        return rev, pat
                    except Exception:
                        continue
            return None, None

        rev, pattern = find_working_rev()
        if rev is None:
            print("❌ Could not locate a revision with FLAC tarballs.")
        else:
            print(f"📌 Using revision: {rev}")
            print(f"🗂️ Tar layout pattern: {pattern}")

            # this can be slow; now it will show progress
            for i in range(10):
                rel = pattern.format(idx=i)
                link = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/{rev}/{rel}"
                fname = rel.split("/")[-1]
                out_tar = audioset_dir / fname
                if not out_tar.exists():
                    print(f"⬇️ {fname}")
                    try:
                        run_cmd_logged(["wget", "-O", str(out_tar), link])
                        print(f"📦 Extract {fname}")
                        run_cmd_logged(["tar", "-xf", str(out_tar), "-C", str(audioset_dir)])
                    except Exception as e:
                        print(f"⚠️ {fname} failed: {e}")
                        if out_tar.exists():
                            out_tar.unlink(missing_ok=True)

            flacs = list(audioset_dir.rglob("*.flac"))
            print(f"🔎 FLAC files: {len(flacs)}")
            audioset_bad = []
            ok = 0
            for p in tqdm(flacs, desc="AudioSet→WAV (resample 16k mono)"):
                try:
                    y, _ = librosa.load(p, sr=16000, mono=True)
                    if y.size == 0:
                        raise ValueError("empty audio")
                    write_wav(audioset_out / (p.stem + ".wav"), y, 16000)
                    ok += 1
                except Exception as e:
                    audioset_bad.append(f"{p}:{e}")
            if audioset_bad:
                (audioset_out / "audioset_corrupted_files.log").write_text("\n".join(audioset_bad))
            print(f"✅ AudioSet complete ({ok} ok, {len(audioset_bad)} failed)")
    else:
        print("✅ AudioSet already prepared; skipping.")

    # -----------------------------
    # FMA xsmall (resample to 16 kHz mono)
    # -----------------------------
    print("\n=== FMA xsmall ===")
    fma_zip_dir = Path("fma"); fma_zip_dir.mkdir(exist_ok=True)
    fma_out = Path("fma_16k"); fma_out.mkdir(exist_ok=True)

    zipname = "fma_xs.zip"
    zipurl  = f"https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/{zipname}"
    zipout  = fma_zip_dir / zipname

    # only download/unzip if we don't already have WAVs
    if not any(fma_out.rglob("*.wav")):
        if (not zipout.exists()) or (zipout.stat().st_size < 100_000):
            print(f"⬇️ downloading {zipurl}")
            rc = os.system(f"wget -q -O '{zipout}' '{zipurl}'")
            if rc != 0 or (not zipout.exists()) or (zipout.stat().st_size < 100_000):
                raise RuntimeError("❌ FMA zip download failed or is too small")

        print("📦 Unzipping FMA …")
        rc = os.system(f"cd fma && unzip -q '{zipname}'")
        if rc != 0:
            raise RuntimeError(f"❌ unzip failed for {zipname}")

        mp3s = list(Path("fma/fma_small").rglob("*.mp3"))
        print(f"🎵 FMA mp3 count: {len(mp3s)}")
        corrupt = []
        for p in tqdm(mp3s, desc="FMA→16k WAV"):
            try:
                y, sr = librosa.load(p, sr=16000, mono=True)
                if y.size == 0:
                    raise ValueError("empty audio")
                write_wav(fma_out / (p.stem + ".wav"), y, 16000)
            except Exception as e:
                corrupt.append(f"{p}:{e}")
        if corrupt:
            Path("fma_corrupted_files.log").write_text("\n".join(corrupt))
        print("✅ FMA complete!")
    else:
        print("✅ FMA already prepared; skipping.")

    # ---------------- Negative datasets (HF) ----------------
    print("\n=== Negative datasets (HF) ===")
    neg_out = Path("./negative_datasets")
    neg_out.mkdir(exist_ok=True)

    link_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    files = [
        ("dinner_party", "dinner_party.zip"),
        ("dinner_party_eval", "dinner_party_eval.zip"),
        ("no_speech", "no_speech.zip"),
        ("speech", "speech.zip"),
    ]

    def download_file(url, out_path: Path, label: str):
        r = requests.get(url, stream=True)
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total) * 100 if total else 0
                print(f"Downloading {label}: {pct:0.0f}% ({downloaded // (1024*1024)}MB/{total // (1024*1024)}MB)")

    for folder_name, zip_name in files:
        target_dir = neg_out / folder_name
        zip_path = neg_out / zip_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"✅ {folder_name} already extracted; skipping.")
            continue

        if not zip_path.exists():
            print(f"⬇️ downloading {zip_name} …")
            try:
                download_file(link_root + zip_name, zip_path, zip_name)
            except Exception as e:
                print(f"⚠️ error downloading {zip_name}: {e}")
                continue

        print(f"📦 Extracting {zip_name} …")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(target_dir)
        except Exception as e:
            print(f"⚠️ error extracting {zip_name}: {e}")

    print("✅ Dataset prep fully complete!")

    try:
        import yaml
        base_cfg = {
            "window_step_ms": 10,
            "train_dir": "trained_models/wakeword",
        }
        with open("training_parameters.yaml", "w") as f:
            yaml.dump(base_cfg, f)
        print("✅ Wrote training_parameters.yaml (base)")
    except Exception as e:
        print(f"⚠️ Could not write training_parameters.yaml: {e}")


def cell_setup_augmentations():
    from pathlib import Path

    # these are from microwakeword
    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap
    import yaml

    # make sure the input stuff from steps 3 + 4 exists
    impulse_paths = ["mit_rirs"]
    background_paths = ["fma_16k", "audioset_16k"]
    for p in impulse_paths + background_paths:
        if not Path(p).exists():
            raise RuntimeError(f"❌ Missing directory: {p}. Run '3. Prep datasets' first.")

    gen_dir = Path("./generated_samples")
    if not gen_dir.exists():
        raise RuntimeError("❌ No generated_samples/ folder. Run '4. Generate samples' first.")

    # upstream-style Clips
    clips = Clips(
        input_directory=str(gen_dir),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    # your augmentation settings
    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.05,
            "PitchShift": 0.15,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.7,
            "Gain": 0.8,
            "RIR": 0.7,
        },
        impulse_paths=impulse_paths,
        background_paths=background_paths,
        background_min_snr_db=5,
        background_max_snr_db=10,
        min_jitter_s=0.2,
        max_jitter_s=0.3,
    )

    out_root = Path("generated_augmented_features")
    out_root.mkdir(exist_ok=True)

    # three splits, like author
    for split in ["training", "validation", "testing"]:
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"🧪 Processing {split} …")

        slide_frames = 10
        repeat = 2
        split_name = "train"
        if split == "validation":
            split_name = "validation"
            repeat = 1
        elif split == "testing":
            split_name = "test"
            repeat = 1
            slide_frames = 1  # streaming style

        spectros = SpectrogramGeneration(
            clips=clips,
            augmenter=augmenter,
            slide_frames=slide_frames,
            step_ms=10,
        )

        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectros.spectrogram_generator(
                split=split_name,
                repeat=repeat,
            ),
            batch_size=100,
            verbose=True,
        )

    # write YAML for training step
    cfg = {
        "window_step_ms": 10,
        "train_dir": "trained_models/wakeword",
        "features": [
            {
                "features_dir": "generated_augmented_features",
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/speech",
                "sampling_weight": 12.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/dinner_party",
                "sampling_weight": 12.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/no_speech",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/dinner_party_eval",
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [40000],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 16,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    with open("training_parameters.yaml", "w") as f:
        yaml.dump(cfg, f)
    print("✅ Wrote training_parameters.yaml")
    print("✅ Augmentation + feature generation complete (upstream style).")

def cell_train_export():
    import os, sys, gc, shutil, json, subprocess
    from pathlib import Path
    import yaml

    print("=== 6. Train + export ===")

    if "tensorflow" not in sys.modules:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    import tensorflow as tf

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    gc.collect()

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision policy:", mixed_precision.global_policy())
    except Exception as e:
        print("Mixed precision not enabled:", e)

    cfg = {
        "window_step_ms": 10,
        "train_dir": "trained_models/wakeword",
        "features": [
            {
                "features_dir": "generated_augmented_features",
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/speech",
                "sampling_weight": 12.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/dinner_party",
                "sampling_weight": 12.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/no_speech",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": "negative_datasets/dinner_party_eval",
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [40000],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 16,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }
    with open("training_parameters.yaml", "w") as f:
        yaml.dump(cfg, f)
    print("✅ Wrote training_parameters.yaml")

    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        "/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda")
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")

    PYTHONPATH = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"/data/microWakeWord:{PYTHONPATH}"

    cmd = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        "--training_config=training_parameters.yaml",
        "--train", "1",
        "--restore_checkpoint", "1",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        "mixednet",
        "--pointwise_filters", "64,64,64,64",
        "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
        "--residual_connection", "0,0,0,0",
        "--first_conv_filters", "32",
        "--first_conv_kernel_size", "5",
        "--stride", "2",
    ]
    print("Running trainer: This can take awhile!")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"❌ Training process failed with exit code {proc.returncode}")

    wake_word = st.session_state.get("last_wakeword") or "tater"
    src_tflite = Path("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")
    if not src_tflite.exists():
        raise RuntimeError(f"❌ Expected exported tflite not found at {src_tflite}")

    dst_tflite = Path(f"{wake_word}.tflite")
    shutil.copy(src_tflite, dst_tflite)
    print(f"✅ Copied model to {dst_tflite}")

    json_payload = {
        "type": "micro",
        "wake_word": wake_word,
        "author": "Tater Totterson",
        "website": "https://github.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker.git",
        "model": dst_tflite.name,
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": 0.97,
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
    }
    dst_json = Path(f"{wake_word}.json")
    dst_json.write_text(json.dumps(json_payload, indent=2))
    print(f"✅ Wrote metadata to {dst_json}")

    st.session_state["last_export"] = {
        "wakeword": wake_word,
        "tflite": str(dst_tflite),
        "json": str(dst_json),
    }
    print("✅ Export info saved to session_state.")


# =================================================
# UI
# =================================================
st.title("🥔 microWakeWord Trainer (Docker)")

with st.sidebar:
    if "active_group" not in st.session_state:
        st.session_state["active_group"] = "run_once"

    st.markdown("### Trainer mode")
    col_tab1, col_tab2 = st.columns(2)
    if col_tab1.button("Run Once", use_container_width=True):
        st.session_state["active_group"] = "run_once"
    if col_tab2.button("Generate", use_container_width=True):
        st.session_state["active_group"] = "generate"

    st.markdown("---")

    if st.session_state["active_group"] == "run_once":
        st.markdown("**🧰 Prepare Data (Run Once)**")
        st.write("Runs environment check, installs microWakeWord, and prepares datasets.")
        if st.button("Run full prepare", use_container_width=True, key="btn_full_prepare"):
            run_block(cell_prepare_all_once, "Prepare Data (Run Once)")
    else:
        st.markdown("**🎤 Generate microWakeWord**")
        st.write("Adjust settings, then run the full pipeline.")

        # --- settings (just stored in session) ---
        gen_wake = st.text_input(
            "Wakeword",
            value=st.session_state.get("gen_wakeword", "tater"),
            key="gen_wakeword",
        )
        gen_num = st.number_input(
            "Number of samples",
            min_value=1,
            max_value=100_000,
            value=st.session_state.get("gen_num_samples", 50_000),
            step=1_000,
            key="gen_num_samples",
        )
        model_choice = st.selectbox(
            "Piper voice model",
            [
                "en_US-libritts_r-medium.pt",
                "de_DE-mls-medium.pt",
                "fr_FR-mls-medium.pt",
                "nl_NL-mls-medium.pt",
            ],
            index=0,
            key="gen_piper_model_choice",
        )
        MODEL_URLS = {
            "en_US-libritts_r-medium.pt": "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
            "de_DE-mls-medium.pt":        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/de_DE-mls-medium.pt",
            "fr_FR-mls-medium.pt":        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/fr_FR-mls-medium.pt",
            "nl_NL-mls-medium.pt":        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/nl_NL-mls-medium.pt",
        }
        st.session_state["gen_piper_model_url"] = MODEL_URLS[model_choice]

        st.markdown("---")
        if st.button("Run full generate", use_container_width=True, key="btn_full_generate"):
            run_block(cell_generate_full, "Generate microWakeWord")

console_ph = st.empty()
st.session_state["console_placeholder"] = console_ph
render_console()

st.markdown("---")
st.subheader("📦 Latest export")

last_export = st.session_state.get("last_export")
if last_export:
    ww = last_export.get("wakeword", "wakeword")
    tflite_path = last_export.get("tflite")
    json_path = last_export.get("json")

    st.write(f"Your last trained wake word: **{ww}**")

    # Show actual paths (helps with volume mounts)
    st.code(
        "\n".join(
            p for p in [tflite_path, json_path] if p
        ),
    )

    # download model
    if tflite_path and os.path.exists(tflite_path):
        with open(tflite_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download model ({os.path.basename(tflite_path)})",
                data=f,
                file_name=os.path.basename(tflite_path),
                mime="application/octet-stream",
            )
    else:
        st.caption("Model file not found on disk — run Generate again.")

    # download json
    if json_path and os.path.exists(json_path):
        with open(json_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download metadata ({os.path.basename(json_path)})",
                data=f,
                file_name=os.path.basename(json_path),
                mime="application/json",
            )
    else:
        st.caption("Metadata file not found on disk — run Generate again.")
else:
    st.info("Train once with “Generate microWakeWord” to see export files here.")