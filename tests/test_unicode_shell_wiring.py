from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class UnicodeShellWiringTests(unittest.TestCase):
    def test_training_scripts_count_positional_array_elements(self) -> None:
        train_script = (REPO_ROOT / "train_wake_word").read_text(encoding="utf-8")
        sample_trainer = (REPO_ROOT / "cli" / "wake_word_sample_trainer").read_text(
            encoding="utf-8"
        )

        self.assertIn("${#POSITIONAL_ARGS[@]}", train_script)
        self.assertIn("${#POSITIONAL_ARGS[@]}", sample_trainer)
        self.assertNotIn("${#POSITIONAL_ARGS}", train_script)
        self.assertNotIn("${#POSITIONAL_ARGS}", sample_trainer)

    def test_language_and_artifact_slug_reach_model_packaging(self) -> None:
        train_script = (REPO_ROOT / "train_wake_word").read_text(encoding="utf-8")
        sample_trainer = (REPO_ROOT / "cli" / "wake_word_sample_trainer").read_text(
            encoding="utf-8"
        )

        self.assertIn('--language="${LANGUAGE}"', train_script)
        self.assertIn('--artifact-slug="${ARTIFACT_SLUG:-}"', train_script)
        self.assertIn("language artifact-slug", sample_trainer)
        self.assertIn("export WAKE_WORD_TITLE LANGUAGE", sample_trainer)
        self.assertIn("ensure_ascii=False", sample_trainer)


if __name__ == "__main__":
    unittest.main()
