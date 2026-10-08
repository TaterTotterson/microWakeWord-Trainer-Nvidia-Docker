"""Shared openWakeWord stage used after a successful microWakeWord build.

The stage deliberately publishes differently named artifacts so the existing
microWakeWord JSON/TFLite URLs remain stable for Tater and ESPHome clients.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


OWW_TRAINER_REPOSITORY = "https://github.com/TaterTotterson/openWakeWord-Trainer.git"
OWW_TRAINER_REVISION = "5dceecd783a381ea59c06a4574266d3fcd54d49e"
OWW_BUNDLE_SUFFIX = ".wake-bundle.json"
OWW_METADATA_SUFFIX = ".oww.json"


def normalize_train_openwakeword(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _valid_trainer_root(path: Path) -> bool:
    return (path / "train_openwakeword.sh").is_file() and (
        path / "scripts" / "train_openwakeword.py"
    ).is_file()


def ensure_openwakeword_trainer(
    data_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
    log=lambda _line: None,
) -> Path:
    env = os.environ if environ is None else environ
    override = str(env.get("TATER_OWW_TRAINER_DIR") or "").strip()
    if override:
        root = Path(override).expanduser().resolve()
        if not _valid_trainer_root(root):
            raise RuntimeError(f"TATER_OWW_TRAINER_DIR is not a usable trainer checkout: {root}")
        return root

    root = Path(
        env.get("TATER_OWW_TRAINER_CACHE")
        or (Path(data_dir).resolve() / "tools" / "openwakeword-trainer")
    ).expanduser().resolve()
    revision_file = root / ".tater-pinned-revision"
    if _valid_trainer_root(root):
        try:
            if revision_file.read_text(encoding="utf-8").strip() == OWW_TRAINER_REVISION:
                return root
        except OSError:
            pass

    root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="openwakeword-trainer-", dir=str(root.parent)))
    checkout = temp_root / "checkout"
    try:
        log("→ Downloading the pinned Tater openWakeWord trainer (first dual-model run only)")
        subprocess.run(
            ["git", "clone", "--quiet", "--no-checkout", OWW_TRAINER_REPOSITORY, str(checkout)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", OWW_TRAINER_REVISION],
            check=True,
        )
        revision_file_in_checkout = checkout / revision_file.name
        revision_file_in_checkout.write_text(OWW_TRAINER_REVISION + "\n", encoding="utf-8")
        if not _valid_trainer_root(checkout):
            raise RuntimeError("Pinned openWakeWord trainer checkout is incomplete")

        backup = root.with_name(root.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        if root.exists():
            root.rename(backup)
        checkout.rename(root)
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return root


def prepare_openwakeword_stage(
    *,
    phrase: str,
    safe_word: str,
    data_dir: Path,
    personal_dir: Path,
    negative_dir: Path,
    trained_dir: Path,
    environ: Mapping[str, str] | None = None,
    log=lambda _line: None,
) -> tuple[list[str], Path, dict[str, str], Path]:
    base_env = dict(os.environ if environ is None else environ)
    trainer_root = ensure_openwakeword_trainer(data_dir, environ=base_env, log=log)
    oww_root = Path(data_dir).resolve() / "openwakeword"
    staging_dir = oww_root / "publish-staging" / safe_word
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    env = dict(base_env)
    env.update(
        {
            "OWW_DATA_DIR": str(oww_root),
            "OWW_ASSET_DIR": str(oww_root / "assets"),
            "OWW_OUTPUT_ROOT": str(oww_root / "output"),
            "OWW_EXPORT_DIR": str(staging_dir),
            "OWW_PERSONAL_DIR": str(Path(personal_dir).resolve()),
            "OWW_NEGATIVE_DIR": str(Path(negative_dir).resolve()),
            "OWW_TRAINED_DIR": str(staging_dir),
        }
    )
    if env.get("MWW_BLACKWELL_IMAGE") == "1":
        env.setdefault("OWW_TORCH_CUDA", "cu128")
        env.setdefault("OWW_TORCH_VERSION", "2.7.1")
    elif os.name != "darwin" and not env.get("OWW_FORCE_CPU"):
        env.setdefault("OWW_TORCH_CUDA", "cu124")

    cmd = [
        "bash",
        str(trainer_root / "train_openwakeword.sh"),
        phrase,
        "--model-name",
        safe_word,
        "--positive-dir",
        str(Path(personal_dir).resolve()),
        "--negative-dir",
        str(Path(negative_dir).resolve()),
        "--output-root",
        str(oww_root / "output"),
        "--export-dir",
        str(staging_dir),
        "--data-dir",
        str(oww_root / "assets"),
    ]
    if any(Path(personal_dir).glob("*.wav")) and any(Path(negative_dir).glob("*.wav")):
        cmd.append("--train-verifier")
    Path(trained_dir).mkdir(parents=True, exist_ok=True)
    return cmd, trainer_root, env, staging_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _oww_runtime_policy(calibration: Mapping[str, Any]) -> tuple[float, int, float, int]:
    standalone_threshold = float(calibration.get("recommended_threshold", 0.95))
    standalone_patience = int(calibration.get("recommended_patience", 3))
    confirmation_threshold = calibration.get("recommended_confirmation_threshold")
    confirmation_patience = int(
        calibration.get("recommended_confirmation_patience", standalone_patience)
    )

    # Upgrade calibration produced before a separate dual-mode field existed.
    # Preserve its automatically selected strict standalone threshold and use
    # the saved score table to derive only the confirmation policy.
    metrics = calibration.get("threshold_metrics")
    if confirmation_threshold is None and isinstance(metrics, list):
        confirmation_candidates = [
            row
            for row in metrics
            if isinstance(row, dict)
            and float(row.get("positive_recall") or 0.0) >= 0.80
        ]
        if confirmation_candidates:
            confirmation_threshold = max(
                float(row.get("threshold") or 0.0) for row in confirmation_candidates
            )

    return (
        max(0.01, min(1.0, standalone_threshold)),
        max(1, min(20, standalone_patience)),
        max(0.01, min(1.0, float(confirmation_threshold or 0.90))),
        max(1, min(20, confirmation_patience)),
    )


def publish_openwakeword_artifacts(
    *,
    staging_dir: Path,
    trained_dir: Path,
    safe_word: str,
    phrase: str,
) -> Path:
    staging = Path(staging_dir).resolve()
    destination = Path(trained_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    source_metadata = staging / f"{safe_word}.json"
    source_onnx = staging / f"{safe_word}.onnx"
    if not source_onnx.is_file() or not source_metadata.is_file():
        raise RuntimeError("openWakeWord did not produce its required ONNX and metadata artifacts")

    mappings: list[tuple[Path, Path]] = [
        (source_onnx, destination / f"{safe_word}.oww.onnx"),
        (source_metadata, destination / f"{safe_word}{OWW_METADATA_SUFFIX}"),
    ]
    optional = (
        (staging / f"{safe_word}.onnx.data", destination / f"{safe_word}.oww.onnx.data"),
        (staging / f"{safe_word}_verifier.pkl", destination / f"{safe_word}.oww.verifier.pkl"),
    )
    mappings.extend(pair for pair in optional if pair[0].is_file())
    for source, target in optional:
        if not source.is_file() and target.exists():
            target.unlink()

    for source, target in mappings:
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)

    oww_metadata_path = destination / f"{safe_word}{OWW_METADATA_SUFFIX}"
    oww_metadata = _read_json(source_metadata)
    oww_artifacts = {
        target.suffix.lstrip(".") or target.name: {
            "file": target.name,
            "sha256": _sha256(target),
            "size_bytes": target.stat().st_size,
        }
        for source, target in mappings
        if source != source_metadata
    }
    oww_metadata.update(
        {
            "type": "open_wake_word",
            "model_name": safe_word,
            "phrase": phrase,
            "artifacts": oww_artifacts,
            "trainer_revision": OWW_TRAINER_REVISION,
        }
    )
    temporary_metadata = oww_metadata_path.with_name(oww_metadata_path.name + ".tmp")
    temporary_metadata.write_text(json.dumps(oww_metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(oww_metadata_path)

    mww_metadata_path = destination / f"{safe_word}.json"
    mww_metadata = _read_json(mww_metadata_path)
    mww_model_name = Path(str(mww_metadata.get("model") or f"{safe_word}.tflite")).name
    mww_model_path = destination / mww_model_name
    if not mww_metadata_path.is_file() or not mww_model_path.is_file():
        raise RuntimeError("microWakeWord artifacts disappeared before bundle publication")

    calibration = oww_metadata.get("calibration")
    if not isinstance(calibration, dict):
        calibration = {}
    threshold, patience, confirmation_threshold, confirmation_patience = _oww_runtime_policy(calibration)
    bundle = {
        "schema_version": 1,
        "type": "tater_wake_word_bundle",
        "wake_word": str(mww_metadata.get("wake_word") or phrase),
        "key": safe_word,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "micro_wake_word": {
            "manifest": mww_metadata_path.name,
            "model": mww_model_path.name,
            "manifest_sha256": _sha256(mww_metadata_path),
            "model_sha256": _sha256(mww_model_path),
        },
        "open_wake_word": {
            "metadata": oww_metadata_path.name,
            "metadata_sha256": _sha256(oww_metadata_path),
            "artifacts": oww_artifacts,
            "recommended_threshold": threshold,
            "recommended_patience": patience,
            "recommended_confirmation_threshold": confirmation_threshold,
            "recommended_confirmation_patience": confirmation_patience,
        },
    }
    bundle_path = destination / f"{safe_word}{OWW_BUNDLE_SUFFIX}"
    temporary_bundle = bundle_path.with_name(bundle_path.name + ".tmp")
    temporary_bundle.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    temporary_bundle.replace(bundle_path)
    return bundle_path


def bundle_catalog_fields(trained_dir: Path, safe_word: str, base_url: str = "") -> dict[str, Any]:
    destination = Path(trained_dir).resolve()
    bundle_path = destination / f"{safe_word}{OWW_BUNDLE_SUFFIX}"
    if not bundle_path.is_file():
        return {"dual_model": False}
    bundle = _read_json(bundle_path)
    oww = bundle.get("open_wake_word") if isinstance(bundle.get("open_wake_word"), dict) else {}
    prefix = str(base_url or "").rstrip("/")

    def url_for(filename: Any) -> str:
        from urllib.parse import quote

        name = Path(str(filename or "")).name
        if not name:
            return ""
        path = f"/api/trained_wake_words/{quote(name)}"
        return f"{prefix}{path}" if prefix else path

    artifacts = oww.get("artifacts") if isinstance(oww.get("artifacts"), dict) else {}
    onnx = next(
        (row for row in artifacts.values() if isinstance(row, dict) and str(row.get("file", "")).endswith(".onnx")),
        {},
    )
    micro = bundle.get("micro_wake_word") if isinstance(bundle.get("micro_wake_word"), dict) else {}
    if not onnx or not str(micro.get("manifest") or "").endswith(".json") or not str(micro.get("model") or "").endswith(".tflite"):
        return {"dual_model": False}
    return {
        "dual_model": True,
        "bundle_url": url_for(bundle_path.name),
        "openwakeword_metadata_url": url_for(oww.get("metadata")),
        "openwakeword_model_url": url_for(onnx.get("file")),
        "openwakeword_threshold": oww.get("recommended_threshold"),
        "openwakeword_patience": oww.get("recommended_patience"),
        "openwakeword_confirmation_threshold": oww.get("recommended_confirmation_threshold"),
        "openwakeword_confirmation_patience": oww.get("recommended_confirmation_patience"),
    }
