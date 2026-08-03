import tempfile
import unittest
from pathlib import Path

import trainer_server as trainer


class DataManagementTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.original_paths = {
            "DATA_DIR": trainer.DATA_DIR,
            "PERSONAL_DIR": trainer.PERSONAL_DIR,
            "CAPTURED_DIR": trainer.CAPTURED_DIR,
            "NEGATIVE_DIR": trainer.NEGATIVE_DIR,
            "TRIM_HISTORY_DIR": trainer.TRIM_HISTORY_DIR,
            "TRAINED_WAKE_WORDS_DIR": trainer.TRAINED_WAKE_WORDS_DIR,
            "AUTO_TRAIN_MODEL_DIR": trainer.AUTO_TRAIN_MODEL_DIR,
            "PIPER_ROOT": trainer.PIPER_ROOT,
            "PIPER_VOICES_DIR": trainer.PIPER_VOICES_DIR,
            "PIPER_CATALOG_CACHE_FILE": trainer.PIPER_CATALOG_CACHE_FILE,
            "OMNIVOICE_CATALOG_CACHE_FILE": trainer.OMNIVOICE_CATALOG_CACHE_FILE,
        }
        trainer.DATA_DIR = root
        trainer.PERSONAL_DIR = root / "personal_samples"
        trainer.CAPTURED_DIR = root / "captured_audio"
        trainer.NEGATIVE_DIR = root / "negative_samples"
        trainer.TRIM_HISTORY_DIR = root / "trim_history"
        trainer.TRAINED_WAKE_WORDS_DIR = root / "trained_wake_words"
        trainer.AUTO_TRAIN_MODEL_DIR = root / "auto_train_models"
        trainer.PIPER_ROOT = root / "tools" / "piper-sample-generator"
        trainer.PIPER_VOICES_DIR = trainer.PIPER_ROOT / "voices"
        trainer.PIPER_CATALOG_CACHE_FILE = root / ".cache" / "piper_voices_catalog.json"
        trainer.OMNIVOICE_CATALOG_CACHE_FILE = root / ".cache" / "omnivoice_languages.json"
        self.original_training_running = trainer.STATE["training"]["running"]
        self.original_review_running = trainer.AUTO_TRAIN_RUNTIME["review_running"]
        trainer.STATE["training"]["running"] = False
        trainer.AUTO_TRAIN_RUNTIME["review_running"] = False

    def tearDown(self):
        for name, value in self.original_paths.items():
            setattr(trainer, name, value)
        trainer.STATE["training"]["running"] = self.original_training_running
        trainer.AUTO_TRAIN_RUNTIME["review_running"] = self.original_review_running
        self.tempdir.cleanup()

    def test_payload_counts_each_managed_item_and_does_not_follow_symlinks(self):
        generated = trainer.DATA_DIR / "work" / "wake_word_samples"
        generated.mkdir(parents=True)
        (generated / "one.wav").write_bytes(b"a" * 128)
        outside = trainer.DATA_DIR / "outside.bin"
        outside.write_bytes(b"b" * 8192)
        (generated / "outside-link").symlink_to(outside)

        payload = trainer._managed_data_payload()
        item = next(row for row in payload["items"] if row["id"] == "generated_samples")

        self.assertEqual(item["file_count"], 2)
        self.assertGreater(item["size_bytes"], 0)
        self.assertEqual(item["location"], "work/wake_word_samples")
        self.assertEqual(payload["total_file_count"], 2)

        deleted = trainer._delete_managed_data_item("generated_samples")
        self.assertFalse(generated.exists())
        self.assertTrue(outside.exists())
        self.assertEqual(deleted["deleted_id"], "generated_samples")

    def test_unknown_ids_and_active_training_are_rejected(self):
        with self.assertRaises(KeyError):
            trainer._delete_managed_data_item("../../not-allowed")

        generated = trainer.DATA_DIR / "work" / "wake_word_samples"
        generated.mkdir(parents=True)
        (generated / "keep.wav").write_bytes(b"keep")
        trainer.STATE["training"]["running"] = True
        with self.assertRaisesRegex(RuntimeError, "Stop training"):
            trainer._delete_managed_data_item("generated_samples")
        self.assertTrue((generated / "keep.wav").exists())


if __name__ == "__main__":
    unittest.main()
