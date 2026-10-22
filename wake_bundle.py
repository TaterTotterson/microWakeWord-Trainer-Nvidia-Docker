"""Canonical Tater wake-word bundle helpers.

The bundle is the only JSON consumed by Tater. It embeds each engine's runtime
metadata while model binaries remain adjacent downloadable artifacts. The
separate ESPHome manifest is intentionally MicroWakeWord-only.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

BUNDLE_SUFFIX = ".wake-bundle.json"
BUNDLE_SCHEMA_VERSION = 3
ESPHOME_MANIFEST_KEYS = ("type", "wake_word", "author", "website", "model", "trained_languages", "version", "micro")

# A wake bundle is a runtime manifest, not a training report. Detailed quality
# diagnostics stay in the trainer workspace so public bundles remain compact
# and never expose machine-local paths.
MICRO_PUBLIC_KEYS = (
    "type", "wake_word", "label", "author", "website", "model",
    "trained_languages", "version", "model_format", "quantization",
    "sample_rate", "micro", "tater_native", "calibration",
    "model_sha256", "model_size_bytes", "manifest", "manifest_sha256",
)
OPEN_PUBLIC_KEYS = (
    "type", "name", "model_name", "phrase", "created_at", "trained_at",
    "synced_at", "trainer_revision", "metadata", "metadata_sha256",
    "artifacts", "recommended_threshold", "recommended_patience",
    "recommended_confirmation_threshold", "recommended_confirmation_patience",
    "confirmation_required",
)
NANO_PUBLIC_KEYS = (
    "type", "name", "phrase", "created_at", "trained_at", "synced_at",
    "model_type", "architecture", "steps", "layer_size",
    "hard_negative_samples", "quality_retries_used", "model",
    "model_sha256", "model_size_bytes", "artifacts",
    "recommended_threshold", "recommended_patience",
    "quality_summary", "validated_existing_candidate", "seed", "loss_bias",
)

def _select_nano_default(variants: Mapping[str, Any], current: str = "") -> str:
    ranked: list[tuple[tuple[float, ...], str]] = []
    for architecture, variant in variants.items():
        if not isinstance(variant, Mapping):
            continue
        summary = variant.get("quality_summary")
        if not isinstance(summary, Mapping):
            continue
        try:
            recall = float(summary["recall"])
            negative_rate = float(summary["negative_rate"])
            patience = int(variant.get("recommended_patience", 2))
        except (KeyError, TypeError, ValueError):
            continue
        preferred = float(recall >= 0.95 and negative_rate <= 0.05)
        utility = recall - negative_rate - 0.01 * patience
        ranked.append(((preferred, utility, -negative_rate, recall, -patience, float(architecture == "cnn")), str(architecture)))
    if ranked:
        return max(ranked)[1]
    if current in variants:
        return current
    return "cnn" if "cnn" in variants else next(iter(variants), "")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}

def esphome_manifest(micro_metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(micro_metadata[key]) for key in ESPHOME_MANIFEST_KEYS if key in micro_metadata}

def bundle_path(trained_dir: Path, safe_word: str) -> Path:
    return Path(trained_dir).resolve() / f"{safe_word}{BUNDLE_SUFFIX}"

def _public_fields(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: copy.deepcopy(metadata[key]) for key in keys if key in metadata}

def _sanitized_existing_bundle(existing: Mapping[str, Any]) -> dict[str, Any]:
    """Remove private diagnostics whenever any part of a bundle is rewritten."""
    source = dict(existing) if existing else {}
    bundle = {
        key: copy.deepcopy(source[key])
        for key in ("schema_version", "type", "wake_word", "key", "created_at", "updated_at")
        if key in source
    }
    micro = source.get("micro_wake_word")
    if isinstance(micro, Mapping):
        bundle["micro_wake_word"] = _public_fields(micro, MICRO_PUBLIC_KEYS)
    oww = source.get("open_wake_word")
    if isinstance(oww, Mapping):
        bundle["open_wake_word"] = _public_fields(oww, OPEN_PUBLIC_KEYS)
    nano = source.get("nano_wake_word")
    if isinstance(nano, Mapping):
        variants = nano.get("variants")
        clean_variants: dict[str, Any] = {}
        if isinstance(variants, Mapping):
            for architecture, variant in variants.items():
                if isinstance(variant, Mapping):
                    clean_variants[str(architecture)] = _public_fields(variant, NANO_PUBLIC_KEYS)
        bundle["nano_wake_word"] = {
            "default_architecture": str(nano.get("default_architecture") or ""),
            "variants": clean_variants,
        }
    return bundle

def _base_bundle(existing: Mapping[str, Any], safe_word: str, phrase: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    bundle = _sanitized_existing_bundle(existing)
    bundle.update({
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "type": "tater_wake_word_bundle",
        "wake_word": phrase or str(bundle.get("wake_word") or safe_word.replace("_", " ")),
        "key": safe_word,
        "created_at": str(bundle.get("created_at") or now),
        "updated_at": now,
    })
    return bundle

def _atomic_write(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path

def load_bundle(trained_dir: Path, safe_word: str) -> dict[str, Any]:
    return read_json(bundle_path(trained_dir, safe_word))

def prune_unreferenced_companion_artifacts(trained_dir: Path, safe_word: str, bundle: Mapping[str, Any]) -> None:
    """Remove obsolete public metadata/DNN files without touching training data."""
    root = Path(trained_dir).resolve()
    oww = bundle.get("open_wake_word")
    referenced = {Path(str(oww.get("metadata") or "")).name if isinstance(oww, Mapping) else ""}
    nano = bundle.get("nano_wake_word")
    variants = nano.get("variants") if isinstance(nano, Mapping) else None
    if isinstance(variants, Mapping):
        for variant in variants.values():
            if isinstance(variant, Mapping):
                referenced.add(Path(str(variant.get("model") or "")).name)
    candidates = {
        f"{safe_word}.oww.json", f"{safe_word}.nww.cnn.json",
        f"{safe_word}.nww.dnn.json", f"{safe_word}.nww.gru.json",
        f"{safe_word}.nww.dnn.onnx",
    }
    for name in candidates - referenced:
        try:
            (root / name).unlink()
        except FileNotFoundError:
            pass

def publish_micro_wake_word(*, trained_dir: Path, safe_word: str, phrase: str, metadata: Mapping[str, Any], model_path: Path) -> Path:
    model = Path(model_path).resolve()
    if not model.is_file():
        raise RuntimeError(f"microWakeWord model not found: {model}")
    section = _public_fields(metadata, MICRO_PUBLIC_KEYS)
    section.update({"model": model.name, "model_sha256": sha256_file(model), "model_size_bytes": model.stat().st_size})
    bundle = _base_bundle(load_bundle(trained_dir, safe_word), safe_word, phrase)
    bundle["micro_wake_word"] = section
    path = _atomic_write(bundle_path(trained_dir, safe_word), bundle)
    prune_unreferenced_companion_artifacts(trained_dir, safe_word, bundle)
    return path

def publish_open_wake_word(*, trained_dir: Path, safe_word: str, phrase: str, metadata: Mapping[str, Any], artifacts: Mapping[str, Any], recommended_threshold: float, recommended_patience: int, recommended_confirmation_threshold: float, recommended_confirmation_patience: int) -> Path:
    section = _public_fields(metadata, OPEN_PUBLIC_KEYS)
    section.update({
        "type": "open_wake_word", "artifacts": copy.deepcopy(dict(artifacts)),
        "recommended_threshold": recommended_threshold, "recommended_patience": recommended_patience,
        "recommended_confirmation_threshold": recommended_confirmation_threshold,
        "recommended_confirmation_patience": recommended_confirmation_patience,
    })
    bundle = _base_bundle(load_bundle(trained_dir, safe_word), safe_word, phrase)
    bundle["open_wake_word"] = section
    path = _atomic_write(bundle_path(trained_dir, safe_word), bundle)
    prune_unreferenced_companion_artifacts(trained_dir, safe_word, bundle)
    return path

def publish_nano_wake_word(*, trained_dir: Path, safe_word: str, phrase: str, architecture: str, metadata: Mapping[str, Any], model_path: Path, recommended_threshold: float = 0.90, recommended_patience: int = 2) -> Path:
    architecture = str(architecture).strip().lower()
    if architecture not in {"cnn", "gru"}:
        raise ValueError(f"Unsupported NanoWakeWord architecture: {architecture}")
    model = Path(model_path).resolve()
    if not model.is_file():
        raise RuntimeError(f"NanoWakeWord model not found: {model}")
    if not 0.0 < float(recommended_threshold) <= 1.0:
        raise ValueError("NanoWakeWord recommended threshold must be between 0 and 1")
    if int(recommended_patience) < 1:
        raise ValueError("NanoWakeWord recommended patience must be at least 1")
    variant = _public_fields(metadata, NANO_PUBLIC_KEYS)
    variant.update({
        "type": "nano_wake_word", "architecture": architecture, "model": model.name,
        "model_sha256": sha256_file(model), "model_size_bytes": model.stat().st_size,
        "recommended_threshold": recommended_threshold, "recommended_patience": recommended_patience,
    })
    bundle = _base_bundle(load_bundle(trained_dir, safe_word), safe_word, phrase)
    nano = bundle.get("nano_wake_word") if isinstance(bundle.get("nano_wake_word"), dict) else {}
    variants = nano.get("variants") if isinstance(nano.get("variants"), dict) else {}
    variants = {key: value for key, value in variants.items() if str(key).strip().lower() in {"cnn", "gru"}}
    variants[architecture] = variant
    nano["variants"] = variants
    nano["default_architecture"] = _select_nano_default(variants, str(nano.get("default_architecture") or ""))
    bundle["nano_wake_word"] = nano
    path = _atomic_write(bundle_path(trained_dir, safe_word), bundle)
    prune_unreferenced_companion_artifacts(trained_dir, safe_word, bundle)
    return path

def migrate_legacy_micro_manifest(trained_dir: Path, manifest_path: Path) -> Path | None:
    legacy_path = Path(manifest_path)
    metadata = read_json(legacy_path)
    safe_word = legacy_path.stem
    model = Path(trained_dir).resolve() / Path(str(metadata.get("model") or f"{safe_word}.tflite")).name
    if not metadata or not model.is_file():
        return None
    migrated = publish_micro_wake_word(trained_dir=trained_dir, safe_word=safe_word, phrase=str(metadata.get("wake_word") or safe_word.replace("_", " ")), metadata=metadata, model_path=model)
    try:
        legacy_path.unlink()
    except OSError:
        pass
    return migrated

def bundle_catalog_fields(trained_dir: Path, safe_word: str, base_url: str = "") -> dict[str, Any]:
    path = bundle_path(trained_dir, safe_word)
    bundle = read_json(path)
    if not bundle:
        return {"available_engines": []}
    prefix = str(base_url or "").rstrip("/")
    def url_for(filename: Any) -> str:
        name = Path(str(filename or "")).name
        if not name:
            return ""
        relative = f"/api/trained_wake_words/{quote(name)}"
        return f"{prefix}{relative}" if prefix else relative
    fields: dict[str, Any] = {"bundle_url": url_for(path.name), "available_engines": [], "dual_model": False}
    micro = bundle.get("micro_wake_word")
    if isinstance(micro, dict) and str(micro.get("model") or "").endswith(".tflite"):
        fields["available_engines"].append("micro_wake_word")
        fields["model_url"] = url_for(micro.get("model"))
    oww = bundle.get("open_wake_word")
    if isinstance(oww, dict):
        artifacts = oww.get("artifacts") if isinstance(oww.get("artifacts"), dict) else {}
        onnx = next((row for row in artifacts.values() if isinstance(row, dict) and str(row.get("file") or "").endswith(".onnx")), {})
        if onnx:
            fields["available_engines"].append("open_wake_word")
            fields.update({
                "openwakeword_model_url": url_for(onnx.get("file")),
                "openwakeword_threshold": oww.get("recommended_threshold"),
                "openwakeword_patience": oww.get("recommended_patience"),
                "openwakeword_confirmation_threshold": oww.get("recommended_confirmation_threshold"),
                "openwakeword_confirmation_patience": oww.get("recommended_confirmation_patience"),
                "openwakeword_confirmation_required": bool(oww.get("confirmation_required")),
            })
    nano = bundle.get("nano_wake_word")
    variants = nano.get("variants") if isinstance(nano, dict) and isinstance(nano.get("variants"), dict) else {}
    nano_urls = {architecture: url_for(variants[architecture].get("model")) for architecture in ("cnn", "gru") if isinstance(variants.get(architecture), dict) and variants[architecture].get("model")}
    if nano_urls:
        fields["available_engines"].append("nano_wake_word")
        fields["nanowakeword_models"] = nano_urls
        fields["nanowakeword_architectures"] = list(nano_urls)
        fields["nanowakeword_default_architecture"] = nano.get("default_architecture")
    fields["dual_model"] = "micro_wake_word" in fields["available_engines"] and "open_wake_word" in fields["available_engines"]
    return fields
