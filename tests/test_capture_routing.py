import io
import tempfile
import unittest
import wave
from pathlib import Path

import trainer_server as trainer
from wake_word_library import activate_word


def silent_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


class CaptureRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name)
        self.original_paths = (
            trainer.DATA_DIR,
            trainer.PERSONAL_DIR,
            trainer.NEGATIVE_DIR,
            trainer.CAPTURED_DIR,
            trainer.UNASSIGNED_CAPTURED_DIR,
            trainer.TRIM_HISTORY_DIR,
            trainer.TRAINED_WAKE_WORDS_DIR,
        )
        trainer.DATA_DIR = self.data
        trainer.PERSONAL_DIR = self.data / "personal_samples"
        trainer.NEGATIVE_DIR = self.data / "negative_samples"
        trainer.CAPTURED_DIR = self.data / "captured_audio"
        trainer.UNASSIGNED_CAPTURED_DIR = self.data / "unassigned_captured_audio"
        trainer.TRIM_HISTORY_DIR = self.data / "trim_history"
        trainer.TRAINED_WAKE_WORDS_DIR = self.data / "trained_wake_words"
        for safe_word, phrase in (("hey_tater", "hey tater"), ("computer", "computer")):
            activate_word(
                self.data,
                safe_word=safe_word,
                phrase=phrase,
                language="en",
                english_accent="mixed",
                tts_mode="hybrid",
                workspace_paths=(),
            )

    def tearDown(self) -> None:
        (
            trainer.DATA_DIR,
            trainer.PERSONAL_DIR,
            trainer.NEGATIVE_DIR,
            trainer.CAPTURED_DIR,
            trainer.UNASSIGNED_CAPTURED_DIR,
            trainer.TRIM_HISTORY_DIR,
            trainer.TRAINED_WAKE_WORDS_DIR,
        ) = self.original_paths
        self.temporary.cleanup()

    def add_capture(self, wake_word: str) -> tuple[str, Path]:
        scope = trainer._capture_scope_for_upload(wake_word)
        directory = trainer._capture_directory(scope)
        result = trainer._save_captured_sample(
            silent_wav_bytes(),
            "satellite.wav",
            target_dir=directory,
        )
        path = directory / result["saved_as"]
        trainer._write_sidecar_json(
            path,
            {
                "wake_word": wake_word,
                "dataset_wake_word": "" if scope == trainer.CAPTURE_UNASSIGNED_SCOPE else scope,
                "assignment_status": "unassigned" if scope == trainer.CAPTURE_UNASSIGNED_SCOPE else "assigned",
                "event_type": "wake_detected",
                "received_at": "2026-11-06T12:00:00+00:00",
            },
        )
        return scope, path

    def test_upload_scope_uses_active_inactive_and_unassigned_destinations(self) -> None:
        self.assertEqual(trainer._capture_scope_for_upload("computer"), "computer")
        self.assertEqual(trainer._capture_directory("computer"), self.data / "captured_audio")
        self.assertEqual(trainer._capture_scope_for_upload("hey tater"), "hey_tater")
        self.assertEqual(
            trainer._capture_directory("hey_tater"),
            self.data / "wake_word_library" / "words" / "hey_tater" / "assets" / "captured_audio",
        )
        self.assertEqual(trainer._capture_scope_for_upload("unknown word"), trainer.CAPTURE_UNASSIGNED_SCOPE)
        self.assertEqual(trainer._capture_directory(trainer.CAPTURE_UNASSIGNED_SCOPE), trainer.UNASSIGNED_CAPTURED_DIR)

    def test_review_keeps_clip_with_its_routed_wake_word(self) -> None:
        scope, path = self.add_capture("hey_tater")
        result = trainer._move_captured_audio(
            path.name,
            capture_word=scope,
            target_word="hey_tater",
            target_bucket="personal_samples",
            target_prefix="sample",
            review_status="approved_personal",
        )
        destination = self.data / "wake_word_library" / "words" / "hey_tater" / "assets" / "personal_samples" / result["saved_as"]
        self.assertTrue(destination.is_file())
        self.assertFalse(path.exists())
        self.assertEqual(trainer._load_sidecar_json(destination)["dataset_wake_word"], "hey_tater")

    def test_unassigned_clip_requires_and_accepts_explicit_saved_word(self) -> None:
        scope, path = self.add_capture("not installed")
        self.assertEqual(scope, trainer.CAPTURE_UNASSIGNED_SCOPE)
        result = trainer._move_captured_audio(
            path.name,
            capture_word=scope,
            target_word="computer",
            target_bucket="negative_samples",
            target_prefix="negative",
            review_status="approved_negative",
        )
        destination = self.data / "negative_samples" / result["saved_as"]
        self.assertTrue(destination.is_file())
        metadata = trainer._load_sidecar_json(destination)
        self.assertEqual(metadata["dataset_wake_word"], "computer")
        self.assertEqual(metadata["assignment_status"], "assigned")

    def test_review_queue_lists_every_saved_word_and_unassigned_clip(self) -> None:
        self.add_capture("computer")
        self.add_capture("hey_tater")
        self.add_capture("not installed")
        items = trainer._list_captured_items()
        self.assertEqual(len(items), 3)
        self.assertEqual(
            {item["capture_scope"] for item in items},
            {"computer", "hey_tater", trainer.CAPTURE_UNASSIGNED_SCOPE},
        )
        unassigned = next(item for item in items if item["capture_scope"] == trainer.CAPTURE_UNASSIGNED_SCOPE)
        self.assertEqual(unassigned["assignment_status"], "unassigned")
        self.assertEqual(unassigned["review_wake_word"], "")

    def test_legacy_active_inbox_is_migrated_using_satellite_metadata(self) -> None:
        trainer.CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
        known = trainer.CAPTURED_DIR / "captured_0001.wav"
        known.write_bytes(silent_wav_bytes())
        trainer._write_sidecar_json(known, {"wake_word": "hey_tater", "original_name": "known.wav"})
        unknown = trainer.CAPTURED_DIR / "captured_0002.wav"
        unknown.write_bytes(silent_wav_bytes())
        trainer._write_sidecar_json(unknown, {"wake_word": "not installed", "original_name": "unknown.wav"})

        self.assertEqual(trainer._route_legacy_captured_audio(), 2)
        routed_known = self.data / "wake_word_library" / "words" / "hey_tater" / "assets" / "captured_audio" / known.name
        routed_unknown = trainer.UNASSIGNED_CAPTURED_DIR / unknown.name
        self.assertTrue(routed_known.is_file())
        self.assertTrue(routed_unknown.is_file())
        self.assertEqual(trainer._load_sidecar_json(routed_known)["dataset_wake_word"], "hey_tater")
        self.assertEqual(trainer._load_sidecar_json(routed_unknown)["assignment_status"], "unassigned")


if __name__ == "__main__":
    unittest.main()
