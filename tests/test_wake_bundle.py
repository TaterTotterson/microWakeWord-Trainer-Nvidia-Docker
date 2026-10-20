import json
import tempfile
import unittest
from pathlib import Path

from wake_bundle import bundle_catalog_fields, migrate_legacy_micro_manifest, publish_micro_wake_word, publish_nano_wake_word, publish_open_wake_word


class WakeBundleTests(unittest.TestCase):
    def test_public_bundle_omits_private_training_paths_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oww = root / "hey_tater.oww.onnx"
            nww = root / "hey_tater.nww.cnn.onnx"
            oww.write_bytes(b"oww")
            nww.write_bytes(b"nww")
            stale_files = (
                root / "hey_tater.oww.json",
                root / "hey_tater.nww.cnn.json",
                root / "hey_tater.nww.dnn.json",
                root / "hey_tater.nww.dnn.onnx",
            )
            for stale in stale_files:
                stale.write_text("private", encoding="utf-8")
            publish_open_wake_word(
                trained_dir=root, safe_word="hey_tater", phrase="hey tater",
                metadata={
                    "config_path": "/Users/test/private/config.yaml",
                    "quality": {"examples": [{"path": "/Users/test/private/clip.wav"}]},
                    "model_name": "hey_tater",
                },
                artifacts={"onnx": {"file": oww.name, "sha256": "a" * 64, "size_bytes": 3}},
                recommended_threshold=0.92, recommended_patience=3,
                recommended_confirmation_threshold=0.96, recommended_confirmation_patience=3,
            )
            publish_nano_wake_word(
                trained_dir=root, safe_word="hey_tater", phrase="hey tater",
                architecture="cnn",
                metadata={
                    "name": "hey_tater",
                    "quality": {"validation": {"model": "/Volumes/private/model.onnx"}},
                    "feature_bank_dir": "/Volumes/private/features",
                },
                model_path=nww,
            )
            raw = (root / "hey_tater.wake-bundle.json").read_text(encoding="utf-8")
            bundle = json.loads(raw)
            self.assertNotIn("/Users/", raw)
            self.assertNotIn("/Volumes/", raw)
            self.assertNotIn("quality", bundle["nano_wake_word"]["variants"]["cnn"])
            self.assertNotIn("config_path", bundle["open_wake_word"])
            self.assertFalse(any(stale.exists() for stale in stale_files))

    def test_legacy_micro_json_is_removed_after_successful_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "hey_tater.tflite"
            legacy = root / "hey_tater.json"
            model.write_bytes(b"model")
            legacy.write_text(json.dumps({"type": "micro", "wake_word": "hey tater", "model": model.name}), encoding="utf-8")
            migrated = migrate_legacy_micro_manifest(root, legacy)
            self.assertEqual(migrated, (root / "hey_tater.wake-bundle.json").resolve())
            self.assertFalse(legacy.exists())

    def test_mix_and_match_models_share_one_self_contained_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = {
                "mww": root / "hey_tater.tflite",
                "oww": root / "hey_tater.oww.onnx",
                "cnn": root / "hey_tater.nww.cnn.onnx",
                "gru": root / "hey_tater.nww.gru.onnx",
            }
            for path in models.values():
                path.write_bytes(path.name.encode())
            publish_micro_wake_word(trained_dir=root, safe_word="hey_tater", phrase="hey tater", metadata={"type": "micro", "wake_word": "hey tater", "model": models["mww"].name, "micro": {}}, model_path=models["mww"])
            publish_open_wake_word(trained_dir=root, safe_word="hey_tater", phrase="hey tater", metadata={"calibration": {"passed": True}}, artifacts={"onnx": {"file": models["oww"].name}}, recommended_threshold=0.92, recommended_patience=3, recommended_confirmation_threshold=0.88, recommended_confirmation_patience=2)
            bundle_path = root / "hey_tater.wake-bundle.json"
            stale = json.loads(bundle_path.read_text(encoding="utf-8"))
            stale["nano_wake_word"] = {"default_architecture": "dnn", "variants": {"dnn": {"model": "obsolete.onnx"}}}
            bundle_path.write_text(json.dumps(stale), encoding="utf-8")
            for architecture in ("cnn", "gru"):
                publish_nano_wake_word(trained_dir=root, safe_word="hey_tater", phrase="hey tater", architecture=architecture, metadata={"validation": {"passed": True}}, model_path=models[architecture])
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["schema_version"], 3)
            self.assertNotIn("manifest", bundle["micro_wake_word"])
            self.assertEqual(set(bundle["nano_wake_word"]["variants"]), {"cnn", "gru"})
            catalog = bundle_catalog_fields(root, "hey_tater")
            self.assertEqual(catalog["available_engines"], ["micro_wake_word", "open_wake_word", "nano_wake_word"])


if __name__ == "__main__":
    unittest.main()
