"""Read immutable training artifacts and safely select a published version.

Training-run snapshots remain untouched. Selecting a version only replaces the
published artifacts; recordings, generated speech, and feature caches are not
part of this operation. Callers must hold the trainer's publishing/run lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from wake_bundle import esphome_manifest, read_json


ARTIFACT_SUFFIXES = {".json", ".tflite", ".onnx", ".data", ".pkl"}


def _segment(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+\-]*", value or "") or value in {".", ".."}:
        raise ValueError("Invalid wake-word or version id.")
    return value


def word_directory(data_dir: Path, safe_word: str) -> Path:
    root = Path(data_dir).resolve() / "wake_word_library" / "words"
    path = root / _segment(safe_word)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Invalid wake-word directory.")
    return path


def run_directory(data_dir: Path, safe_word: str, run_id: str) -> Path:
    root = word_directory(data_dir, safe_word)
    path = root / "runs" / _segment(run_id)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Invalid version directory.")
    if not (path / "run.json").is_file():
        raise FileNotFoundError("Saved training version not found.")
    return path


def artifact_path(data_dir: Path, safe_word: str, run_id: str, filename: str) -> Path:
    run = run_directory(data_dir, safe_word, run_id)
    if filename != Path(filename).name or "\\" in filename or not filename.startswith(f"{safe_word}."):
        raise ValueError("Invalid version artifact.")
    if Path(filename).suffix.lower() not in ARTIFACT_SUFFIXES:
        raise ValueError("Unsupported version artifact.")
    metadata = read_json(run / "run.json")
    root = run / "artifacts"
    path = root / filename
    if not root.resolve().is_relative_to(run.resolve()):
        raise ValueError("Invalid version artifact directory.")
    if filename not in metadata.get("artifacts", []) or not path.is_file():
        raise FileNotFoundError("Version artifact not found.")
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Invalid version artifact path.")
    return path


def _bundle_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def mark_published(data_dir: Path, trained_dir: Path, safe_word: str, run_id: str) -> None:
    directory = word_directory(data_dir, safe_word)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "published_version.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({
        "run_id": run_id,
        "bundle_sha256": _bundle_digest(Path(trained_dir) / f"{safe_word}.wake-bundle.json"),
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def published_run_id(data_dir: Path, trained_dir: Path, safe_word: str, *, prefer_committed: bool = False) -> str:
    """Return the committed run while training prepares the next publication."""
    directory = word_directory(data_dir, safe_word)
    digest = _bundle_digest(Path(trained_dir) / f"{safe_word}.wake-bundle.json")
    if not digest:
        return ""
    marker = read_json(directory / "published_version.json")
    if marker.get("run_id") and (prefer_committed or marker.get("bundle_sha256") == digest):
        try:
            run_directory(data_dir, safe_word, str(marker.get("run_id") or ""))
            return str(marker["run_id"])
        except (ValueError, FileNotFoundError):
            pass
    # Existing libraries predate the marker. Match a successfully published run
    # without migrating, rewriting, or assuming the latest attempt succeeded.
    for path in sorted((directory / "runs").glob("*/run.json"), reverse=True):
        run = read_json(path)
        outcomes = run.get("outcomes") or {}
        if any(code == 0 for code in outcomes.values()) and _bundle_digest(
            path.parent / "artifacts" / f"{safe_word}.wake-bundle.json"
        ) == digest:
            return str(run.get("run_id") or "")
    return ""


def _references(value: Any) -> dict[str, str]:
    references: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"model", "file", "metadata"} and isinstance(item, str) and item:
                if Path(item).name != item or "\\" in item or ":" in item:
                    raise ValueError("Saved version contains a non-local artifact reference.")
                digest = str(value.get("model_sha256" if key == "model" else "sha256") or "")
                references[item] = digest
            else:
                references.update(_references(item))
    elif isinstance(value, list):
        for item in value:
            references.update(_references(item))
    return references


def validate_version(data_dir: Path, safe_word: str, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run = read_json(run_directory(data_dir, safe_word, run_id) / "run.json")
    if not any(code == 0 for code in (run.get("outcomes") or {}).values()):
        raise ValueError("This run did not publish a model. Choose a completed version.")
    path = artifact_path(data_dir, safe_word, run_id, f"{safe_word}.wake-bundle.json")
    bundle = read_json(path)
    if not bundle or not any(isinstance(bundle.get(key), dict) and bundle[key] for key in (
        "micro_wake_word", "open_wake_word", "nano_wake_word", "wake_boundary",
    )):
        raise ValueError("This saved version has no usable wake-word bundle.")
    for filename, digest in _references(bundle).items():
        artifact = artifact_path(data_dir, safe_word, run_id, filename)
        if digest:
            checksum = hashlib.sha256()
            with artifact.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != digest:
                raise ValueError(f"Saved artifact failed its checksum: {filename}")
    return run, bundle


def version_catalog(data_dir: Path, trained_dir: Path, safe_word: str, runs: list[dict[str, Any]], base_url: str = "", *, prefer_committed: bool = False) -> list[dict[str, Any]]:
    current = published_run_id(data_dir, trained_dir, safe_word, prefer_committed=prefer_committed)
    rows = []
    for index, run in enumerate(runs):
        run_id = str(run.get("run_id") or "")
        prefix = f"{base_url.rstrip('/')}/api/wake_word_library/{quote(safe_word)}/runs/{quote(run_id)}/artifacts"
        try:
            bundle_file = artifact_path(data_dir, safe_word, run_id, f"{safe_word}.wake-bundle.json")
            bundle = read_json(bundle_file)
        except (ValueError, FileNotFoundError):
            bundle = {}
        usable = bool(bundle) and any(code == 0 for code in (run.get("outcomes") or {}).values())
        rows.append({
            **run,
            "version": len(runs) - index,
            "is_current": run_id == current,
            "json_url": f"{prefix}/{quote(safe_word)}.wake-bundle.json" if usable else "",
            "esphome_json_url": f"{prefix}/{quote(safe_word)}.esphome.json" if usable and bundle.get("micro_wake_word") else "",
            "restorable": usable,
            "artifact_urls": {name: f"{prefix}/{quote(name)}" for name in run.get("artifacts", []) if isinstance(name, str) and name.startswith(f"{safe_word}.") and Path(name).name == name and Path(name).suffix.lower() in ARTIFACT_SUFFIXES},
        })
    return rows


def version_json(data_dir: Path, safe_word: str, run_id: str, filename: str) -> dict[str, Any]:
    if filename == f"{safe_word}.esphome.json":
        bundle = read_json(artifact_path(data_dir, safe_word, run_id, f"{safe_word}.wake-bundle.json"))
        micro = bundle.get("micro_wake_word")
        if not isinstance(micro, dict) or not micro:
            raise FileNotFoundError("This version does not contain an ESPHome model.")
        artifact_path(data_dir, safe_word, run_id, str(micro.get("model") or ""))
        return esphome_manifest(micro)
    return read_json(artifact_path(data_dir, safe_word, run_id, filename))


def pin_json_artifacts(value: Any, artifact_base: str) -> Any:
    """Keep every model referenced by a JSON URL on that same immutable run."""
    if isinstance(value, dict):
        return {
            key: f"{artifact_base.rstrip('/')}/{quote(item)}"
            if key in {"model", "file", "metadata"} and isinstance(item, str) and item
            and Path(item).name == item and ":" not in item and "\\" not in item
            else pin_json_artifacts(item, artifact_base)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [pin_json_artifacts(item, artifact_base) for item in value]
    return value


def publish_version(data_dir: Path, trained_dir: Path, safe_word: str, run_id: str) -> dict[str, Any]:
    """Validate everything before replacing files; roll back failed publication."""
    run, _ = validate_version(data_dir, safe_word, run_id)
    target = Path(trained_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".version-", dir=target) as temporary:
        stage = Path(temporary) / "stage"
        backup = Path(temporary) / "backup"
        stage.mkdir()
        backup.mkdir()
        names = []
        for name in run.get("artifacts", []):
            source = artifact_path(data_dir, safe_word, run_id, name)
            shutil.copy2(source, stage / name)
            destination = target / name
            if destination.is_symlink():
                raise ValueError("Published artifacts must not be symbolic links.")
            if destination.exists():
                shutil.copy2(destination, backup / name)
            names.append(name)
        # Publish the manifest last. Keep unreferenced files: removing them is
        # unnecessary and could invalidate an existing consumer's cached JSON.
        names.sort(key=lambda name: name.endswith(".wake-bundle.json"))
        replaced = []
        try:
            for name in names:
                os.replace(stage / name, target / name)
                replaced.append(name)
            mark_published(data_dir, target, safe_word, run_id)
        except Exception:
            for name in reversed(replaced):
                if (backup / name).exists():
                    os.replace(backup / name, target / name)
                else:
                    (target / name).unlink(missing_ok=True)
            raise
    return run
