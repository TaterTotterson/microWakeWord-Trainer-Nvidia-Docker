from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MicroWakeWordEarlyStoppingConfigTests(unittest.TestCase):
    def test_primary_training_writes_adaptive_early_stopping_config(self) -> None:
        trainer = (ROOT / "cli" / "wake_word_sample_trainer").read_text(
            encoding="utf-8"
        )

        self.assertIn("EARLY_STOPPING_MIN_STEPS=$(( TRAINING_STEPS / 2 ))", trainer)
        self.assertIn(
            "EARLY_STOPPING_PATIENCE_STEPS=$(( TRAINING_STEPS * 15 / 100 ))",
            trainer,
        )
        self.assertIn("early_stopping:\n  enabled: true", trainer)
        self.assertIn('config["early_stopping"] = {"enabled": False}', trainer)


if __name__ == "__main__":
    unittest.main()
