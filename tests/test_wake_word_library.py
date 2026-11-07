import json
import tempfile
import unittest
from pathlib import Path

from wake_word_library import activate_word, list_words, record_run, word_history


class WakeWordLibraryTests(unittest.TestCase):
    def test_three_words_keep_independent_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            trained = data / "trained_wake_words"
            (data / "personal_samples").mkdir()
            (data / "personal_samples" / "one.wav").write_bytes(b"one")

            def activate(safe_word: str) -> None:
                activate_word(
                    data,
                    safe_word=safe_word,
                    phrase=safe_word.replace("_", " "),
                    language="en",
                    english_accent="mixed",
                    tts_mode="hybrid",
                    workspace_paths=("personal_samples", "work"),
                )

            activate("word_one")
            activate("word_two")
            self.assertFalse((data / "personal_samples" / "one.wav").exists())
            (data / "personal_samples").mkdir()
            (data / "personal_samples" / "two.wav").write_bytes(b"two")
            activate("word_three")
            (data / "personal_samples").mkdir()
            (data / "personal_samples" / "three.wav").write_bytes(b"three")
            activate("word_one")

            self.assertEqual((data / "personal_samples" / "one.wav").read_bytes(), b"one")
            rows = list_words(data, trained_dir=trained, workspace_paths=("personal_samples", "work"))
            self.assertEqual({row["safe_word"] for row in rows}, {"word_one", "word_two", "word_three"})
            self.assertTrue(next(row for row in rows if row["safe_word"] == "word_one")["active"])

    def test_run_history_snapshots_published_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            trained = data / "trained_wake_words"
            trained.mkdir()
            (trained / "hey_tater.tflite").write_bytes(b"model-v1")
            (trained / "hey_tater.wake-bundle.json").write_text(json.dumps({"schema_version": 4, "wake_word": "hey tater"}), encoding="utf-8")
            summary = {"safe_word": "hey_tater", "wake_word": "hey tater", "completed_at": "2026-11-06T20:00:00+00:00", "overall": "Good", "models": []}
            record_run(data, trained_dir=trained, summary=summary, selected_models=["micro"], outcomes={"micro": 0})
            (trained / "hey_tater.tflite").write_bytes(b"model-v2")

            history = word_history(data, "hey_tater")
            self.assertEqual(len(history), 1)
            snapshot = data / "wake_word_library" / "words" / "hey_tater" / "runs" / history[0]["run_id"] / "artifacts" / "hey_tater.tflite"
            self.assertEqual(snapshot.read_bytes(), b"model-v1")


if __name__ == "__main__":
    unittest.main()
