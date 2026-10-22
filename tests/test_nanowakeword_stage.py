import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanowakeword_stage import prepare_nanowakeword_stage, publish_nanowakeword_artifacts


class NanoWakeWordStageTests(unittest.TestCase):
    def setUp(self):
        self.bank_patch = patch(
            "nanowakeword_stage.ensure_shared_negative_feature_bank",
            side_effect=lambda data_dir, **_kwargs: Path(data_dir) / "shared_feature_banks" / "bank.npy",
        )
        self.bank_patch.start()
        self.addCleanup(self.bank_patch.stop)

    @staticmethod
    def write_contract(root: Path) -> None:
        (root / "trainer_contract.json").write_text(json.dumps({
            "api_version": 2,
            "capabilities": ["shared-positive-tts", "multiple-background-dirs", "multiple-rir-dirs", "shared-negative-feature-bank", "quality-patience-5", "validate-existing-candidate", "advisory-quality-publication"],
        }), encoding="utf-8")

    def test_stale_checkout_without_shared_tts_support_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            (checkout / "scripts").mkdir()
            (checkout / "train_nanowakeword.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (checkout / "scripts" / "train_nanowakeword.py").write_text("", encoding="utf-8")
            (checkout / "tts.wav").write_bytes(b"wav")

            with self.assertRaisesRegex(RuntimeError, "not a usable trainer checkout"):
                prepare_nanowakeword_stage(
                    phrase="hey tater",
                    safe_word="hey_tater",
                    architecture="cnn",
                    data_dir=checkout / "data",
                    generated_positive_dir=checkout,
                    personal_dir=checkout,
                    negative_dir=checkout,
                    environ={"TATER_NWW_TRAINER_DIR": str(checkout)},
                )

    def test_stage_reuses_nvidia_generated_tts_and_selects_gru(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "trainer"
            (checkout / "scripts").mkdir(parents=True)
            (checkout / "train_nanowakeword.sh").write_text("#!/bin/sh\n# --positive-tts-dir\n", encoding="utf-8")
            (checkout / "scripts" / "train_nanowakeword.py").write_text(
                "NWW_POSITIVE_TTS_DIR = True\nNWW_FEATURE_DEVICE = 'cpu'\n",
                encoding="utf-8",
            )
            self.write_contract(checkout)
            generated = root / "work" / "wake_word_samples"
            personal = root / "personal"
            negative = root / "negative"
            background = root / "background"
            rir = root / "rir"
            for directory in (generated, personal, negative, background, rir):
                directory.mkdir(parents=True)
            (generated / "tts.wav").write_bytes(b"wav")
            cmd, cwd, env, _staging = prepare_nanowakeword_stage(
                phrase="hey tater", safe_word="hey_tater", architecture="gru",
                data_dir=root / "data", generated_positive_dir=generated,
                personal_dir=personal, negative_dir=negative,
                generated_negative_phrases=["hey later"],
                background_dirs=[background], rir_dirs=[rir],
                environ={"TATER_NWW_TRAINER_DIR": str(checkout), "NWW_FORCE_CPU": "1"},
            )
            self.assertEqual(cwd, checkout.resolve())
            self.assertEqual(cmd[cmd.index("--model-type") + 1], "gru")
            self.assertEqual(env["NWW_POSITIVE_TTS_DIR"], str(generated.resolve()))
            self.assertNotIn("NWW_TORCH_CUDA", env)
            self.assertEqual(env["NWW_FEATURE_DEVICE"], "cpu")
            self.assertEqual(env["NWW_REQUIRE_BROAD_FEATURE_BANK"], "1")
            self.assertIn("shared_feature_banks", env["NWW_FEATURE_BANK_DIR"])
            self.assertIn("hey later", cmd)
            self.assertIn(str(background.resolve()), cmd)
            self.assertIn(str(rir.resolve()), cmd)

            validate_cmd, *_ = prepare_nanowakeword_stage(
                phrase="hey tater", safe_word="hey_tater", architecture="cnn",
                data_dir=root / "data", generated_positive_dir=generated,
                personal_dir=personal, negative_dir=negative,
                environ={"TATER_NWW_TRAINER_DIR": str(checkout), "NWW_FORCE_CPU": "1"},
                validate_existing=True,
            )
            self.assertIn("--validate-existing-candidate", validate_cmd)

    def test_publish_carries_calibrated_streaming_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            trained = root / "trained"
            staging.mkdir()
            (staging / "hey_tater.onnx").write_bytes(b"gru")
            (staging / "hey_tater.json").write_text(
                json.dumps(
                    {
                        "recommended_threshold": 0.99,
                        "recommended_patience": 3,
                        "validation": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            bundle_path = publish_nanowakeword_artifacts(
                staging_dir=staging,
                trained_dir=trained,
                safe_word="hey_tater",
                phrase="hey tater",
                architecture="gru",
            )
            variant = json.loads(bundle_path.read_text(encoding="utf-8"))["nano_wake_word"]["variants"]["gru"]
            self.assertEqual(variant["recommended_threshold"], 0.99)
            self.assertEqual(variant["recommended_patience"], 3)

    def test_publish_accepts_nested_quality_report_from_pinned_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            trained = root / "trained"
            staging.mkdir()
            (staging / "hey_tater.onnx").write_bytes(b"cnn")
            (staging / "hey_tater.json").write_text(
                json.dumps(
                    {
                        "quality": {
                            "passed": True,
                            "validation": {"passed": True},
                            "failures": [],
                        },
                        "recommended_threshold": 0.99,
                        "recommended_patience": 4,
                    }
                ),
                encoding="utf-8",
            )

            bundle_path = publish_nanowakeword_artifacts(
                staging_dir=staging,
                trained_dir=trained,
                safe_word="hey_tater",
                phrase="hey tater",
                architecture="cnn",
            )

            variant = json.loads(bundle_path.read_text(encoding="utf-8"))["nano_wake_word"]["variants"]["cnn"]
            self.assertEqual((trained / "hey_tater.nww.cnn.onnx").read_bytes(), b"cnn")
            self.assertEqual(variant["recommended_threshold"], 0.99)
            self.assertEqual(variant["recommended_patience"], 4)
            self.assertTrue(variant["quality_evaluated"])
            self.assertTrue(variant["quality_passed"])
            self.assertEqual(variant["quality_status"], "passed")

    def test_stage_uses_nanowakeword_compatible_cuda_wheels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout = root / "trainer"
            (checkout / "scripts").mkdir(parents=True)
            (checkout / "train_nanowakeword.sh").write_text(
                "#!/bin/sh\n# --positive-tts-dir\n", encoding="utf-8"
            )
            (checkout / "scripts" / "train_nanowakeword.py").write_text(
                "NWW_POSITIVE_TTS_DIR = True\nNWW_FEATURE_DEVICE = 'cpu'\n",
                encoding="utf-8",
            )
            self.write_contract(checkout)
            generated = root / "generated"
            personal = root / "personal"
            negative = root / "negative"
            for directory in (generated, personal, negative):
                directory.mkdir()
            (generated / "tts.wav").write_bytes(b"wav")

            _cmd, _cwd, env, _staging = prepare_nanowakeword_stage(
                phrase="hey tater",
                safe_word="hey_tater",
                architecture="cnn",
                data_dir=root / "data",
                generated_positive_dir=generated,
                personal_dir=personal,
                negative_dir=negative,
                environ={"TATER_NWW_TRAINER_DIR": str(checkout)},
            )

            self.assertEqual(env["NWW_TORCH_CUDA"], "cu126")
            self.assertEqual(env["NWW_TORCH_VERSION"], "2.8.0")
            self.assertEqual(env["NWW_FEATURE_DEVICE"], "auto")

    def test_publish_allows_failed_validation_with_quality_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            trained = root / "trained"
            staging.mkdir()
            trained.mkdir()
            published = trained / "hey_tater.nww.cnn.onnx"
            published.write_bytes(b"known-good")
            (staging / "hey_tater.onnx").write_bytes(b"new-model")
            (staging / "hey_tater.json").write_text(
                json.dumps({"validation": {"passed": False}}),
                encoding="utf-8",
            )

            bundle_path = publish_nanowakeword_artifacts(
                staging_dir=staging,
                trained_dir=trained,
                safe_word="hey_tater",
                phrase="hey tater",
                architecture="cnn",
            )

            variant = json.loads(bundle_path.read_text(encoding="utf-8"))["nano_wake_word"]["variants"]["cnn"]
            self.assertEqual(published.read_bytes(), b"new-model")
            self.assertTrue(variant["quality_evaluated"])
            self.assertFalse(variant["quality_passed"])
            self.assertEqual(variant["quality_status"], "low_quality")


if __name__ == "__main__":
    unittest.main()
