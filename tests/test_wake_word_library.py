import json
import tempfile
import unittest
from pathlib import Path

from wake_word_library import activate_word, list_words, record_run, word_history


class WakeWordLibraryTests(unittest.TestCase):
    def test_saved_word_activation_replaces_empty_working_scaffolding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "personal_samples").mkdir()
            saved = data / "wake_word_library" / "words" / "jarvis" / "assets"
            (saved / "personal_samples").mkdir(parents=True)
            (saved / "personal_samples" / "sample_0001.wav").write_bytes(b"saved-positive")
            (saved / "negative_samples").mkdir()
            (saved / "negative_samples" / "negative_0001.wav").write_bytes(b"saved-negative")

            result = activate_word(
                data,
                safe_word="jarvis",
                phrase="jarvis",
                language="en",
                english_accent="mixed",
                tts_mode="hybrid",
                workspace_paths=("personal_samples", "negative_samples"),
            )

            self.assertFalse(result["reused"])
            self.assertEqual((data / "personal_samples" / "sample_0001.wav").read_bytes(), b"saved-positive")
            self.assertEqual((data / "negative_samples" / "negative_0001.wav").read_bytes(), b"saved-negative")
            self.assertFalse((saved / "personal_samples").exists())
            self.assertFalse((saved / "negative_samples").exists())
            self.assertEqual(
                json.loads((data / "wake_word_library" / "active_word.json").read_text(encoding="utf-8"))["safe_word"],
                "jarvis",
            )

    def test_split_saved_word_data_merges_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            working = data / "personal_samples"
            working.mkdir()
            (working / "sample_0001.wav").write_bytes(b"working-positive")
            (working / "sample_0001.json").write_text('{"source":"working"}', encoding="utf-8")
            (working / "sample_0002.wav").write_bytes(b"same-positive")
            (working / "sample_0002.json").write_text('{"source":"same"}', encoding="utf-8")

            saved = data / "wake_word_library" / "words" / "jarvis" / "assets" / "personal_samples"
            saved.mkdir(parents=True)
            (saved / "sample_0001.wav").write_bytes(b"saved-positive")
            (saved / "sample_0001.json").write_text('{"source":"saved"}', encoding="utf-8")
            (saved / "sample_0002.wav").write_bytes(b"same-positive")
            (saved / "sample_0002.json").write_text('{"source":"same"}', encoding="utf-8")

            activate_word(
                data,
                safe_word="jarvis",
                phrase="jarvis",
                language="en",
                english_accent="mixed",
                tts_mode="hybrid",
                workspace_paths=("personal_samples",),
            )

            audio_paths = sorted(working.glob("*.wav"))
            self.assertEqual(len(audio_paths), 3)
            self.assertEqual(
                {path.read_bytes() for path in audio_paths},
                {b"working-positive", b"saved-positive", b"same-positive"},
            )
            recovered = working / "sample_0001.recovered-1.wav"
            self.assertEqual(recovered.read_bytes(), b"saved-positive")
            self.assertEqual(
                json.loads(recovered.with_suffix(".json").read_text(encoding="utf-8"))["source"],
                "saved",
            )
            self.assertFalse(saved.exists())

            before = {path.name: path.read_bytes() for path in working.iterdir() if path.is_file()}
            repeated = activate_word(
                data,
                safe_word="jarvis",
                phrase="jarvis",
                language="en",
                english_accent="mixed",
                tts_mode="hybrid",
                workspace_paths=("personal_samples",),
            )
            after = {path.name: path.read_bytes() for path in working.iterdir() if path.is_file()}
            self.assertTrue(repeated["reused"])
            self.assertEqual(after, before)

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
