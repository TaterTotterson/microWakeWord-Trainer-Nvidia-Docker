import json
import tempfile
import unittest
from pathlib import Path

from cli.package_boundary_model import publish


class BoundaryOnlyPackagingTests(unittest.TestCase):
    def test_preserves_existing_detectors_and_updates_only_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle_path = root / "hey_tater.wake-bundle.json"
            bundle_path.write_text(json.dumps({
                "schema_version": 3,
                "wake_word": "hey tater",
                "micro_wake_word": {"model": "hey_tater.tflite"},
                "nano_wake_word": {"variants": {"cnn": {"model": "hey_tater.nww.cnn.onnx"}}},
            }), encoding="utf-8")
            model = root / "candidate.tflite"
            model.write_bytes(b"TFL3 boundary only")
            calibration = root / "boundary.json"
            calibration.write_text(json.dumps({"probability_cutoff": 0.73, "selected_metrics": {"recall": 0.96}}), encoding="utf-8")

            publish(phrase="hey tater", safe_word="hey_tater", output_dir=root, model=model, calibration=calibration)
            result = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(result["micro_wake_word"]["model"], "hey_tater.tflite")
            self.assertEqual(result["nano_wake_word"]["variants"]["cnn"]["model"], "hey_tater.nww.cnn.onnx")
            self.assertEqual(result["wake_boundary"]["threshold"], 0.73)
            self.assertTrue((root / "hey_tater.boundary.tflite").is_file())


if __name__ == "__main__":
    unittest.main()
