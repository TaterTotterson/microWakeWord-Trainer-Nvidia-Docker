import unittest
from pathlib import Path


class WhamRemovalTests(unittest.TestCase):
    def test_active_training_pipeline_has_no_wham_dependency(self):
        root = Path(__file__).resolve().parents[1]
        active_files = [
            root / "cli" / "setup_training_datasets",
            root / "cli" / "wake_word_sample_augmenter",
        ]

        for path in active_files:
            with self.subTest(path=path.name):
                self.assertNotIn("wham", path.read_text().lower())

        self.assertFalse((root / "cli" / "setup_wham").exists())

    def test_existing_augmented_features_are_rebuilt_once(self):
        root = Path(__file__).resolve().parents[1]
        training_script = (root / "train_wake_word").read_text()

        self.assertIn('AUGMENTATION_CACHE_VERSION="2"', training_script)
        self.assertIn(".augmentation_version", training_script)


if __name__ == "__main__":
    unittest.main()
