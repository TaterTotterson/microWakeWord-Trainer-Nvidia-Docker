"""Shared openWakeWord stage used after a successful microWakeWord build.

The stage deliberately publishes differently named artifacts so the existing
microWakeWord JSON/TFLite URLs remain stable for Tater and ESPHome clients.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from wake_bundle import (
    BUNDLE_SUFFIX,
    bundle_catalog_fields as canonical_bundle_catalog_fields,
    publish_open_wake_word,
    sha256_file,
)


OWW_TRAINER_REPOSITORY = "https://github.com/TaterTotterson/openWakeWord-Trainer.git"
OWW_TRAINER_REVISION = "e6f3eba4d421c8a53f489b4f7bdda7b189ccb8d3"
OWW_BUNDLE_SUFFIX = BUNDLE_SUFFIX
OWW_METADATA_SUFFIX = ".oww.json"
OWW_FEATURE_CACHE_POLICY = "cpu-onnx-v1"
OWW_GENERATED_FEATURE_FILES = (
    "positive_features_train.npy",
    "negative_features_train.npy",
    "positive_features_test.npy",
    "negative_features_test.npy",
)


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


def _apply_feature_cache_policy(oww_root: Path, safe_word: str, *, log=lambda _line: None) -> None:
    """Invalidate legacy CUDA embeddings once so training matches CPU ONNX runtime."""
    feature_dir = Path(oww_root) / "output" / safe_word / safe_word
    marker = feature_dir / ".tater-feature-cache-policy"
    try:
        if marker.read_text(encoding="utf-8").strip() == OWW_FEATURE_CACHE_POLICY:
            return
    except OSError:
        pass

    removed = 0
    for name in OWW_GENERATED_FEATURE_FILES:
        feature_path = feature_dir / name
        try:
            feature_path.unlink()
            removed += 1
        except FileNotFoundError:
            pass

    feature_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(OWW_FEATURE_CACHE_POLICY + "\n", encoding="utf-8")
    if removed:
        log(
            "→ Rebuilding cached openWakeWord embeddings once with CPU ONNX "
            "to match satellite inference"
        )


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
    _apply_feature_cache_policy(oww_root, safe_word, log=log)

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
            "OWW_FEATURE_DEVICE": "cpu",
        }
    )
    # Match the validated Apple class balance while CUDA remains dedicated to
    # model optimization. Feature extraction stays on CPU ONNX above so its
    # embeddings match calibration and satellite inference.
    env.setdefault("OWW_DEFAULT_NEGATIVE_BATCH", "256")
    env.setdefault("OWW_DEFAULT_VALIDATION_BATCH", "2048")
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

    oww_metadata = _read_json(source_metadata)
    calibration = oww_metadata.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("deployment_ready") is not True:
        raise RuntimeError(
            "openWakeWord failed deployment calibration; the existing published model was preserved"
        )

    mappings: list[tuple[Path, Path]] = [
        (source_onnx, destination / f"{safe_word}.oww.onnx"),
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

    oww_artifacts = {
        target.suffix.lstrip(".") or target.name: {
            "file": target.name,
            "sha256": sha256_file(target),
            "size_bytes": target.stat().st_size,
        }
        for _source, target in mappings
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
    threshold, patience, confirmation_threshold, confirmation_patience = _oww_runtime_policy(calibration)
    return publish_open_wake_word(
        trained_dir=destination,
        safe_word=safe_word,
        phrase=phrase,
        metadata=oww_metadata,
        artifacts=oww_artifacts,
        recommended_threshold=threshold,
        recommended_patience=patience,
        recommended_confirmation_threshold=confirmation_threshold,
        recommended_confirmation_patience=confirmation_patience,
    )


def bundle_catalog_fields(trained_dir: Path, safe_word: str, base_url: str = "") -> dict[str, Any]:
    return canonical_bundle_catalog_fields(trained_dir, safe_word, base_url)
