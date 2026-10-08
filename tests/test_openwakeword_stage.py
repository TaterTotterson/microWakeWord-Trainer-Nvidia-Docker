import json
import os
import tempfile
import unittest
from pathlib import Path

from openwakeword_stage import (
    OWW_TRAINER_REVISION,
    bundle_catalog_fields,
    normalize_train_openwakeword,
    prepare_openwakeword_stage,
    publish_openwakeword_artifacts,
    _oww_runtime_policy,
)


class OpenWakeWordStageTests(unittest.TestCase):
    def test_legacy_calibration_table_derives_both_runtime_policies(self):
        self.assertEqual(
            _oww_runtime_policy(
                {
                    "recommended_threshold": 0.99,
                    "recommended_patience": 3,
                    "threshold_metrics": [
                        {"threshold": 0.90, "false_positive_rate": 0.004, "positive_recall": 0.815},
                        {"threshold": 0.92, "false_positive_rate": 0.002, "positive_recall": 0.80},
                        {"threshold": 0.95, "false_positive_rate": 0.002, "positive_recall": 0.715},
                        {"threshold": 0.99, "false_positive_rate": 0.0, "positive_recall": 0.38},
                    ],
                }
            ),
            (0.99, 3, 0.92, 3),
        )

    def test_boolean_setting_defaults_to_dual_model(self):
        self.assertTrue(normalize_train_openwakeword(None))
        self.assertTrue(normalize_train_openwakeword("yes"))
        self.assertFalse(normalize_train_openwakeword("off"))

    def test_prepare_reuses_samples_and_pinned_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer_root = root / "oww-trainer"
            (trainer_root / "scripts").mkdir(parents=True)
            (trainer_root / "train_openwakeword.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (trainer_root / "scripts" / "train_openwakeword.py").write_text("", encoding="utf-8")
            personal = root / "personal"
            negative = root / "negative"
            personal.mkdir()
            negative.mkdir()
            (personal / "positive.wav").write_bytes(b"positive")
            (negative / "false-wake.wav").write_bytes(b"negative")

            cmd, cwd, env, staging = prepare_openwakeword_stage(
                phrase="hey tater",
                safe_word="hey_tater",
                data_dir=root / "data",
                personal_dir=personal,
                negative_dir=negative,
                trained_dir=root / "trained",
                environ={**os.environ, "TATER_OWW_TRAINER_DIR": str(trainer_root)},
            )

            self.assertEqual(cwd, trainer_root.resolve())
            self.assertIn("--train-verifier", cmd)
            self.assertEqual(env["OWW_PERSONAL_DIR"], str(personal.resolve()))
            self.assertEqual(env["OWW_NEGATIVE_DIR"], str(negative.resolve()))
            self.assertEqual(env["OWW_EXPORT_DIR"], str(staging))

    def test_publish_keeps_mww_urls_and_adds_dual_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            trained = root / "trained"
            staging.mkdir()
            trained.mkdir()
            (trained / "hey_tater.tflite").write_bytes(b"mww")
            (trained / "hey_tater.json").write_text(
                json.dumps({"wake_word": "Hey Tater", "model": "hey_tater.tflite"}),
                encoding="utf-8",
            )
            (staging / "hey_tater.onnx").write_bytes(b"oww-onnx")
            (staging / "hey_tater.json").write_text(
                json.dumps(
                    {
                        "calibration": {
                            "recommended_threshold": 0.91,
                            "recommended_patience": 4,
                            "recommended_confirmation_threshold": 0.86,
                            "recommended_confirmation_patience": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )

            bundle_path = publish_openwakeword_artifacts(
                staging_dir=staging,
                trained_dir=trained,
                safe_word="hey_tater",
                phrase="hey tater",
            )

            self.assertEqual((trained / "hey_tater.tflite").read_bytes(), b"mww")
            self.assertEqual((trained / "hey_tater.oww.onnx").read_bytes(), b"oww-onnx")
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["type"], "tater_wake_word_bundle")
            self.assertEqual(bundle["open_wake_word"]["recommended_threshold"], 0.91)
            self.assertEqual(bundle["open_wake_word"]["recommended_patience"], 4)
            self.assertEqual(bundle["open_wake_word"]["recommended_confirmation_threshold"], 0.86)
            self.assertEqual(bundle["open_wake_word"]["recommended_confirmation_patience"], 2)
            metadata = json.loads((trained / "hey_tater.oww.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["trainer_revision"], OWW_TRAINER_REVISION)

            catalog = bundle_catalog_fields(trained, "hey_tater", "http://trainer.test")
            self.assertTrue(catalog["dual_model"])
            self.assertTrue(catalog["bundle_url"].endswith("hey_tater.wake-bundle.json"))
            self.assertTrue(catalog["openwakeword_model_url"].endswith("hey_tater.oww.onnx"))
            self.assertEqual(catalog["openwakeword_confirmation_threshold"], 0.86)

    def test_publish_refuses_missing_onnx_companion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging = root / "staging"
            trained = root / "trained"
            staging.mkdir()
            trained.mkdir()
            (trained / "hey_tater.tflite").write_bytes(b"mww")
            (trained / "hey_tater.json").write_text(
                json.dumps({"wake_word": "Hey Tater", "model": "hey_tater.tflite"}),
                encoding="utf-8",
            )
            (staging / "hey_tater.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "required ONNX and metadata"):
                publish_openwakeword_artifacts(
                    staging_dir=staging,
                    trained_dir=trained,
                    safe_word="hey_tater",
                    phrase="hey tater",
                )


if __name__ == "__main__":
    unittest.main()
