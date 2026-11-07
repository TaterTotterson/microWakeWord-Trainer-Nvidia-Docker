"""Persistent per-word datasets and immutable training-run history."""

from __future__ import annotations

import json
import os
import hashlib
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LIBRARY_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _library_root(data_dir: Path) -> Path:
    return Path(data_dir).resolve() / "wake_word_library"


def _active_path(data_dir: Path) -> Path:
    return _library_root(data_dir) / "active_word.json"


def active_word(data_dir: Path) -> str:
    return str(_read_json(_active_path(data_dir)).get("safe_word") or "").strip()


def _safe_name(raw: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw or "").strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", normalized)).strip("_")
    return value or f"wakeword_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:8]}"


def _legacy_workspace_profile(data_dir: Path) -> dict[str, str]:
    for path in (data_dir / "work" / "wake_word_samples" / ".generation_manifest.json", data_dir / "generated_samples" / ".generation_manifest.json"):
        manifest = _read_json(path)
        phrase = str(manifest.get("phrase") or "").strip()
        if phrase:
            return {
                "safe_word": _safe_name(phrase),
                "phrase": phrase,
                "language": str(manifest.get("language") or "en"),
                "english_accent": str(manifest.get("english_accent") or "mixed"),
                "tts_mode": str(manifest.get("tts_mode") or "hybrid"),
            }
    return {}


def _profile_path(data_dir: Path, safe_word: str) -> Path:
    return _library_root(data_dir) / "words" / safe_word / "profile.json"


def save_profile(data_dir: Path, *, safe_word: str, phrase: str, language: str, english_accent: str, tts_mode: str) -> dict[str, Any]:
    path = _profile_path(data_dir, safe_word)
    previous = _read_json(path)
    profile = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "safe_word": safe_word,
        "phrase": phrase,
        "language": language,
        "english_accent": english_accent,
        "tts_mode": tts_mode,
        "created_at": previous.get("created_at") or _utc_now(),
        "updated_at": _utc_now(),
    }
    _atomic_json(path, profile)
    return profile


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _files_equal(left: Path, right: Path) -> bool:
    try:
        if not left.is_file() or not right.is_file() or left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_file, right.open("rb") as right_file:
            while True:
                left_chunk = left_file.read(1024 * 1024)
                right_chunk = right_file.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError:
        return False


def _recovered_path(destination: Path, *, paired_suffixes: Sequence[str] = ()) -> Path:
    for index in range(1, 10000):
        candidate = destination.with_name(
            f"{destination.stem}.recovered-{index}{destination.suffix}"
        )
        paired = [candidate.with_suffix(suffix) for suffix in paired_suffixes]
        if not _path_exists(candidate) and not any(_path_exists(path) for path in paired):
            return candidate
    raise RuntimeError(f"Could not allocate a recovery name for {destination}")


def _merge_audio_pair(source: Path, destination: Path) -> None:
    source_sidecar = source.with_suffix(".json")
    destination_sidecar = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not _path_exists(destination) and not _path_exists(destination_sidecar):
        shutil.move(os.fspath(source), os.fspath(destination))
        if _path_exists(source_sidecar):
            shutil.move(os.fspath(source_sidecar), os.fspath(destination_sidecar))
        return

    if _path_exists(destination) and _files_equal(source, destination):
        if not _path_exists(source_sidecar):
            source.unlink()
            return
        if not _path_exists(destination_sidecar):
            source.unlink()
            shutil.move(os.fspath(source_sidecar), os.fspath(destination_sidecar))
            return
        if _files_equal(source_sidecar, destination_sidecar):
            source.unlink()
            source_sidecar.unlink()
            return

    recovered = _recovered_path(destination, paired_suffixes=(".json",))
    shutil.move(os.fspath(source), os.fspath(recovered))
    if _path_exists(source_sidecar):
        shutil.move(os.fspath(source_sidecar), os.fspath(recovered.with_suffix(".json")))


def _merge_asset_path(source: Path, destination: Path) -> None:
    if not _path_exists(source):
        return
    if not _path_exists(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(os.fspath(source), os.fspath(destination))
        return

    if source.is_dir() and not source.is_symlink() and destination.is_dir() and not destination.is_symlink():
        if not any(destination.iterdir()):
            destination.rmdir()
            shutil.move(os.fspath(source), os.fspath(destination))
            return
        if not any(source.iterdir()):
            source.rmdir()
            return

        paired_sidecars = {
            child.with_suffix(".json")
            for child in source.iterdir()
            if child.is_file() and child.suffix.lower() == ".wav"
        }
        for child in sorted(source.iterdir(), key=lambda path: path.name):
            if child in paired_sidecars:
                continue
            target = destination / child.name
            if child.is_file() and child.suffix.lower() == ".wav":
                _merge_audio_pair(child, target)
            else:
                _merge_asset_path(child, target)
        if source.exists() and not any(source.iterdir()):
            source.rmdir()
        return

    if source.is_file() and destination.is_file() and _files_equal(source, destination):
        source.unlink()
        return

    recovered = _recovered_path(destination)
    recovered.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(os.fspath(source), os.fspath(recovered))


def _move_asset(source: Path, destination: Path) -> None:
    _merge_asset_path(source, destination)


def activate_word(data_dir: Path, *, safe_word: str, phrase: str, language: str, english_accent: str, tts_mode: str, workspace_paths: Sequence[str]) -> dict[str, Any]:
    """Switch fixed trainer paths to the requested word without copying caches."""
    root = Path(data_dir).resolve()
    library = _library_root(root)
    previous = active_word(root)
    legacy_profile = _legacy_workspace_profile(root) if not previous else {}
    previous = previous or legacy_profile.get("safe_word", "")
    safe_word = str(safe_word).strip()
    if not safe_word:
        raise ValueError("safe_word is required")
    profile = save_profile(root, safe_word=safe_word, phrase=phrase, language=language, english_accent=english_accent, tts_mode=tts_mode)
    if previous == safe_word:
        _atomic_json(_active_path(root), {"schema_version": LIBRARY_SCHEMA_VERSION, "safe_word": safe_word, "activated_at": _utc_now()})
        return {**profile, "reused": True, "previous_safe_word": previous}
    relative_paths = tuple(dict.fromkeys(str(item).strip("/") for item in workspace_paths if str(item).strip("/")))
    if previous:
        if legacy_profile:
            save_profile(root, **legacy_profile)
        previous_assets = library / "words" / previous / "assets"
        for relative in relative_paths:
            _move_asset(root / relative, previous_assets / relative)
    requested_assets = library / "words" / safe_word / "assets"
    for relative in relative_paths:
        _move_asset(requested_assets / relative, root / relative)
    _atomic_json(_active_path(root), {"schema_version": LIBRARY_SCHEMA_VERSION, "safe_word": safe_word, "activated_at": _utc_now()})
    return {**profile, "reused": False, "previous_safe_word": previous}


def _usage(paths: Iterable[Path]) -> tuple[int, int]:
    total_bytes = 0
    file_count = 0
    for path in paths:
        candidates = [path] if path.is_file() else path.rglob("*") if path.is_dir() else ()
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                total_bytes += candidate.stat().st_size
                file_count += 1
            except OSError:
                pass
    return total_bytes, file_count


def _history(data_dir: Path, safe_word: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs_dir = _library_root(data_dir) / "words" / safe_word / "runs"
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("*/run.json"), reverse=True):
            row = _read_json(path)
            if row:
                rows.append(row)
    return rows


def list_words(data_dir: Path, *, trained_dir: Path, workspace_paths: Sequence[str]) -> list[dict[str, Any]]:
    root = Path(data_dir).resolve()
    library = _library_root(root)
    current = active_word(root)
    safe_words = {path.parent.name for path in (library / "words").glob("*/profile.json")}
    safe_words.update(path.name.removesuffix(".wake-bundle.json") for path in Path(trained_dir).glob("*.wake-bundle.json"))
    if current:
        safe_words.add(current)
    relative_paths = tuple(dict.fromkeys(str(item).strip("/") for item in workspace_paths if str(item).strip("/")))
    rows: list[dict[str, Any]] = []
    for safe_word in sorted(safe_words):
        profile = _read_json(_profile_path(root, safe_word))
        bundle = _read_json(Path(trained_dir) / f"{safe_word}.wake-bundle.json")
        assets_root = root if safe_word == current else library / "words" / safe_word / "assets"
        size_bytes, file_count = _usage(assets_root / relative for relative in relative_paths)
        history = _history(root, safe_word)
        boundary_only_ready = any(path.is_file() for path in (
            assets_root / "work" / "trained_models" / "wakeword" / "training_config.yaml",
            assets_root / "trained_models" / "wakeword" / "training_config.yaml",
        ))
        rows.append({
            "safe_word": safe_word,
            "phrase": profile.get("phrase") or bundle.get("wake_word") or safe_word.replace("_", " "),
            "language": profile.get("language") or "",
            "english_accent": profile.get("english_accent") or "",
            "tts_mode": profile.get("tts_mode") or "",
            "active": safe_word == current,
            "has_published_bundle": bool(bundle),
            "boundary_only_ready": bool(bundle) and boundary_only_ready,
            "dataset_size_bytes": size_bytes,
            "dataset_file_count": file_count,
            "run_count": len(history),
            "latest_run": history[0] if history else None,
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at"),
        })
    return rows


def word_history(data_dir: Path, safe_word: str) -> list[dict[str, Any]]:
    return _history(Path(data_dir).resolve(), safe_word)


def record_run(data_dir: Path, *, trained_dir: Path, summary: Mapping[str, Any], selected_models: Sequence[str], outcomes: Mapping[str, int | None], run_kind: str = "full") -> dict[str, Any]:
    safe_word = str(summary.get("safe_word") or "").strip()
    if not safe_word:
        raise ValueError("training summary does not contain safe_word")
    completed_at = str(summary.get("completed_at") or _utc_now())
    stamp = completed_at.replace(":", "").replace("-", "").replace("+", "_").replace(".", "_")
    run_id = f"{stamp}-{run_kind}"
    run_dir = _library_root(data_dir) / "words" / safe_word / "runs" / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = run_dir.with_name(f"{run_id}-{suffix}")
        suffix += 1
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    artifact_names: list[str] = []
    for source in sorted(Path(trained_dir).glob(f"{safe_word}.*")):
        if source.is_file():
            shutil.copy2(source, artifacts_dir / source.name)
            artifact_names.append(source.name)
    payload = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "run_kind": run_kind,
        "safe_word": safe_word,
        "wake_word": summary.get("wake_word") or safe_word.replace("_", " "),
        "completed_at": completed_at,
        "selected_models": list(selected_models),
        "outcomes": dict(outcomes),
        "summary": dict(summary),
        "artifacts": artifact_names,
    }
    _atomic_json(run_dir / "run.json", payload)
    return payload
