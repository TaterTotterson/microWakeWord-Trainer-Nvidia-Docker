"""Provision the large negative feature bank shared by OWW and NWW."""

from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import json
import os
import struct
import urllib.request
from pathlib import Path
from typing import Callable, Iterable


FEATURE_BANK_FILENAME = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
FEATURE_BANK_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/"
    f"main/{FEATURE_BANK_FILENAME}"
)
FEATURE_BANK_SIZE = 17_280_000_128
FEATURE_BANK_SHA256 = "721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f"
FEATURE_BANK_DTYPE = "<f2"
FEATURE_BANK_SHAPE = (5_625_000, 16, 96)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_header(path: Path) -> tuple[str, tuple[int, ...]]:
    with Path(path).open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("not a NumPy array")
        major, _minor = handle.read(2)
        if major == 1:
            header_size = struct.unpack("<H", handle.read(2))[0]
        elif major in {2, 3}:
            header_size = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"unsupported NumPy format version {major}")
        header = ast.literal_eval(handle.read(header_size).decode("latin1").strip())
    return str(header.get("descr") or ""), tuple(header.get("shape") or ())


def _valid_header(
    path: Path,
    *,
    expected_size: int,
    expected_dtype: str = FEATURE_BANK_DTYPE,
    expected_shape: tuple[int, ...] = FEATURE_BANK_SHAPE,
) -> bool:
    try:
        return (
            path.is_file()
            and path.stat().st_size == expected_size
            and _npy_header(path) == (expected_dtype, expected_shape)
        )
    except (OSError, ValueError, SyntaxError, struct.error):
        return False


def _marker_matches(path: Path, marker: Path, *, expected_sha256: str) -> bool:
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        stat = path.stat()
        return (
            value.get("sha256") == expected_sha256
            and value.get("size") == stat.st_size
            and value.get("mtime_ns") == stat.st_mtime_ns
            and value.get("resolved_path") == str(path.resolve())
        )
    except (OSError, json.JSONDecodeError):
        return False


def _write_marker(path: Path, marker: Path, *, expected_sha256: str) -> None:
    stat = path.stat()
    marker.write_text(
        json.dumps(
            {
                "sha256": expected_sha256,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "resolved_path": str(path.resolve()),
                "shape": list(FEATURE_BANK_SHAPE),
                "dtype": FEATURE_BANK_DTYPE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_shared_negative_feature_bank(
    data_dir: Path,
    *,
    legacy_candidates: Iterable[Path] = (),
    log: Callable[[str], None] = lambda _line: None,
    url: str = FEATURE_BANK_URL,
    expected_size: int = FEATURE_BANK_SIZE,
    expected_sha256: str = FEATURE_BANK_SHA256,
    expected_dtype: str = FEATURE_BANK_DTYPE,
    expected_shape: tuple[int, ...] = FEATURE_BANK_SHAPE,
) -> Path:
    """Return a checked, persistent bank, adopting an old copy without duplicating it."""
    shared_dir = Path(data_dir).resolve() / "shared_feature_banks"
    shared_dir.mkdir(parents=True, exist_ok=True)
    destination = shared_dir / FEATURE_BANK_FILENAME
    marker = shared_dir / f".{FEATURE_BANK_FILENAME}.verified.json"
    lock_path = shared_dir / f".{FEATURE_BANK_FILENAME}.lock"

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if _valid_header(destination, expected_size=expected_size, expected_dtype=expected_dtype, expected_shape=expected_shape):
            if _marker_matches(destination, marker, expected_sha256=expected_sha256):
                log(f"→ Reusing shared negative feature bank: {destination}")
                return destination
            log("→ Verifying shared negative feature bank checksum (one-time check)")
            if sha256_file(destination) == expected_sha256:
                _write_marker(destination, marker, expected_sha256=expected_sha256)
                return destination

        for candidate_value in legacy_candidates:
            candidate = Path(candidate_value).expanduser()
            try:
                if candidate.resolve() == destination.resolve(strict=False):
                    continue
            except OSError:
                continue
            if not _valid_header(candidate, expected_size=expected_size, expected_dtype=expected_dtype, expected_shape=expected_shape):
                continue
            log(f"→ Verifying existing negative feature bank: {candidate}")
            if sha256_file(candidate) != expected_sha256:
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            adopted_by_move = False
            try:
                os.replace(candidate, destination)
                adopted_by_move = True
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                destination.symlink_to(candidate.resolve())
            _write_marker(destination, marker, expected_sha256=expected_sha256)
            action = "Moved" if adopted_by_move else "Linked"
            log(f"→ {action} existing negative feature bank into shared storage without copying 17 GB: {candidate}")
            return destination

        if destination.exists() or destination.is_symlink():
            destination.unlink()
        partial = shared_dir / f".{FEATURE_BANK_FILENAME}.part"
        partial.unlink(missing_ok=True)
        log("→ Downloading the shared OWW/NWW negative feature bank (about 17 GB)")
        try:
            with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
                total = int(response.headers.get("content-length") or expected_size)
                downloaded = 0
                next_report = 10
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    downloaded += len(block)
                    if total:
                        percent = int(downloaded * 100 / total)
                        if percent >= next_report:
                            log(f"  shared feature bank: {percent}%")
                            next_report += 10
            if not _valid_header(partial, expected_size=expected_size, expected_dtype=expected_dtype, expected_shape=expected_shape):
                raise RuntimeError("downloaded feature bank has the wrong size, dtype, or shape")
            if sha256_file(partial) != expected_sha256:
                raise RuntimeError("downloaded feature bank checksum did not match")
            os.replace(partial, destination)
            _write_marker(destination, marker, expected_sha256=expected_sha256)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            raise
