"""NanoWakeWord CNN/GRU training stage for the Tater trainers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from wake_bundle import publish_nano_wake_word

NWW_TRAINER_REPOSITORY = "https://github.com/TaterTotterson/nanoWakeWord-Trainer.git"
NWW_TRAINER_REVISION = "30cad9c1947f1a6e894b6d716e543603fa360522"
NWW_ARCHITECTURES = ("cnn", "gru")
NWW_TRAINER_COMPATIBILITY_MARKER = "NWW_FEATURE_DEVICE"

def normalize_train_nanowakeword(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}

def _valid_trainer_root(path: Path) -> bool:
    launcher = path / "train_nanowakeword.sh"
    trainer = path / "scripts" / "train_nanowakeword.py"
    if not launcher.is_file() or not trainer.is_file():
        return False
    try:
        trainer_source = trainer.read_text(encoding="utf-8")
        return (
            "--positive-tts-dir" in launcher.read_text(encoding="utf-8")
            and "NWW_POSITIVE_TTS_DIR" in trainer_source
            and NWW_TRAINER_COMPATIBILITY_MARKER in trainer_source
        )
    except OSError:
        return False

def ensure_nanowakeword_trainer(data_dir: Path, *, environ: Mapping[str, str] | None = None, log=lambda _line: None) -> Path:
    env = os.environ if environ is None else environ
    override = str(env.get("TATER_NWW_TRAINER_DIR") or "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not _valid_trainer_root(root):
            raise RuntimeError(f"TATER_NWW_TRAINER_DIR is not a usable trainer checkout: {root}")
        return root
    root = Path(env.get("TATER_NWW_TRAINER_CACHE") or (Path(data_dir).resolve() / "tools" / "nanowakeword-trainer")).expanduser().resolve()
    marker = root / ".tater-trainer-revision"
    if _valid_trainer_root(root):
        try:
            if marker.read_text(encoding="utf-8").strip() == NWW_TRAINER_REVISION:
                return root
        except OSError:
            pass
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="nanowakeword-trainer-", dir=str(root.parent)))
    checkout = temporary_root / "checkout"
    try:
        log("→ Downloading the pinned Tater NanoWakeWord trainer (first NanoWakeWord run only)")
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", NWW_TRAINER_REPOSITORY, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", NWW_TRAINER_REVISION], check=True)
        (checkout / marker.name).write_text(NWW_TRAINER_REVISION + "\n", encoding="utf-8")
        if not _valid_trainer_root(checkout):
            raise RuntimeError("NanoWakeWord trainer checkout is incomplete")
        backup = root.with_name(root.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if root.exists():
            root.rename(backup)
        checkout.rename(root)
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return root

def prepare_nanowakeword_stage(*, phrase: str, safe_word: str, architecture: str, data_dir: Path, generated_positive_dir: Path, personal_dir: Path, negative_dir: Path, generated_negative_phrases: Sequence[str] = (), background_dirs: Sequence[Path] = (), rir_dirs: Sequence[Path] = (), environ: Mapping[str, str] | None = None, log=lambda _line: None) -> tuple[list[str], Path, dict[str, str], Path]:
    architecture = str(architecture).strip().lower()
    if architecture not in NWW_ARCHITECTURES:
        raise ValueError(f"Unsupported NanoWakeWord architecture: {architecture}")
    generated = Path(generated_positive_dir).resolve()
    if not any(generated.glob("*.wav")):
        raise RuntimeError(f"Shared generated TTS corpus is empty: {generated}")
    env = dict(os.environ if environ is None else environ)
    trainer_root = ensure_nanowakeword_trainer(data_dir, environ=env, log=log)
    nww_root = Path(data_dir).resolve() / "nanowakeword"
    staging_dir = nww_root / "publish-staging" / safe_word / architecture
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_root = nww_root / "output" / architecture
    if env.get("MWW_BLACKWELL_IMAGE") == "1":
        env.setdefault("NWW_TORCH_CUDA", "cu128")
        env.setdefault("NWW_TORCH_VERSION", "2.8.0")
    elif not env.get("NWW_FORCE_CPU"):
        env.setdefault("NWW_TORCH_CUDA", "cu126")
        env.setdefault("NWW_TORCH_VERSION", "2.8.0")
    env.setdefault("NWW_FEATURE_DEVICE", "cpu" if env.get("NWW_FORCE_CPU") else "auto")
    env.update({
        "NWW_DATA_DIR": str(nww_root), "NWW_OUTPUT_ROOT": str(output_root),
        "NWW_EXPORT_DIR": str(staging_dir), "NWW_POSITIVE_TTS_DIR": str(generated),
        "NWW_PERSONAL_DIR": str(Path(personal_dir).resolve()), "NWW_NEGATIVE_DIR": str(Path(negative_dir).resolve()),
    })
    cmd = ["bash", str(trainer_root / "train_nanowakeword.sh"), phrase, "--model-type", architecture, "--output-root", str(output_root), "--export-dir", str(staging_dir), "--positive-tts-dir", str(generated), "--positive-dir", str(Path(personal_dir).resolve()), "--negative-dir", str(Path(negative_dir).resolve())]
    for value in generated_negative_phrases:
        if str(value).strip():
            cmd.extend(["--custom-negative-phrase", str(value).strip()])
    for directory in background_dirs:
        cmd.extend(["--background-dir", str(Path(directory).resolve())])
    for directory in rir_dirs:
        cmd.extend(["--rir-dir", str(Path(directory).resolve())])
    return cmd, trainer_root, env, staging_dir

def publish_nanowakeword_artifacts(*, staging_dir: Path, trained_dir: Path, safe_word: str, phrase: str, architecture: str) -> Path:
    staging = Path(staging_dir).resolve()
    source_model = staging / f"{safe_word}.onnx"
    source_metadata = staging / f"{safe_word}.json"
    if not source_model.is_file() or not source_metadata.is_file():
        raise RuntimeError("NanoWakeWord did not produce its required ONNX and metadata artifacts")
    try:
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("NanoWakeWord produced invalid metadata") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("NanoWakeWord produced invalid metadata")
    validation = metadata.get("validation")
    quality = metadata.get("quality")
    if isinstance(quality, dict):
        validation = quality.get("validation")
        validation_passed = (
            quality.get("passed") is True
            and isinstance(validation, dict)
            and validation.get("passed") is True
        )
    else:
        # Retain compatibility with NanoWakeWord trainer revisions that wrote
        # the streaming report directly under the legacy validation key.
        validation_passed = isinstance(validation, dict) and validation.get("passed") is True
    if not validation_passed:
        raise RuntimeError(
            "NanoWakeWord failed streaming validation; the existing published model was preserved"
        )
    destination = Path(trained_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / f"{safe_word}.nww.{architecture}.onnx"
    temporary = model_path.with_name(model_path.name + ".tmp")
    shutil.copy2(source_model, temporary)
    temporary.replace(model_path)
    for local_only in ("config", "feature_bank_dir", "artifacts"):
        metadata.pop(local_only, None)
    recommended_threshold = float(metadata.get("recommended_threshold", 0.90))
    recommended_patience = int(metadata.get("recommended_patience", 2))
    return publish_nano_wake_word(
        trained_dir=destination,
        safe_word=safe_word,
        phrase=phrase,
        architecture=architecture,
        metadata=metadata,
        model_path=model_path,
        recommended_threshold=recommended_threshold,
        recommended_patience=recommended_patience,
    )
