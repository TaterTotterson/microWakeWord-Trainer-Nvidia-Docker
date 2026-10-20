import json
import tempfile
import unittest
from pathlib import Path

from nanowakeword_stage import prepare_nanowakeword_stage, publish_nanowakeword_artifacts


class NanoWakeWordStageTests(unittest.TestCase):
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
                "NWW_POSITIVE_TTS_DIR = True\nquality_target_negative_rate = 0.05\n",
                encoding="utf-8",
            )
            generated = root / "work" / "wake_word_samples"
            personal = root / "personal"
            negative = root / "negative"
            for directory in (generated, personal, negative):
                directory.mkdir(parents=True)
            (generated / "tts.wav").write_bytes(b"wav")
            cmd, cwd, env, _staging = prepare_nanowakeword_stage(
                phrase="hey tater", safe_word="hey_tater", architecture="gru",
                data_dir=root / "data", generated_positive_dir=generated,
                personal_dir=personal, negative_dir=negative,
                environ={"TATER_NWW_TRAINER_DIR": str(checkout), "NWW_FORCE_CPU": "1"},
            )
            self.assertEqual(cwd, checkout.resolve())
            self.assertEqual(cmd[cmd.index("--model-type") + 1], "gru")
            self.assertEqual(env["NWW_POSITIVE_TTS_DIR"], str(generated.resolve()))
            self.assertNotIn("NWW_TORCH_CUDA", env)

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


if __name__ == "__main__":
    unittest.main()
