import json
import tempfile
import unittest
from pathlib import Path

from training_summary import build_training_summary, training_summary_log_lines


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TrainingSummaryTests(unittest.TestCase):
    def test_reports_each_selected_engine_and_real_quality_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trained = root / "trained"
            write_json(trained / "hey_tater.wake-bundle.json", {
                "micro_wake_word": {
                    "calibration": {"recall": 0.98, "false_accepts_per_hour": 0.0},
                    "micro": {"probability_cutoff": 0.96, "sliding_window_size": 6},
                }
            })
            write_json(root / "openwakeword/publish-staging/hey_tater/hey_tater.json", {
                "calibration": {
                    "recommended_threshold": 0.85,
                    "recommended_patience": 3,
                    "recommended_metrics": {"positive_recall": 0.96, "false_positives_per_hour": 0.0},
                }
            })
            write_json(root / "nanowakeword/output/cnn/hey_tater/hey_tater.quality.json", {
                "validation": {
                    "positive": {"rate": 0.97},
                    "negative": {"rate": 0.04},
                    "streaming_contract": {"threshold": 0.99, "patience": 4},
                }
            })
            summary = build_training_summary(
                data_dir=root,
                trained_dir=trained,
                safe_word="hey_tater",
                phrase="hey tater",
                selected_models=["micro", "open", "nano_cnn", "nano_gru"],
                outcomes={"micro": 0, "open": 0, "nano_cnn": 0, "nano_gru": 1},
            )

            self.assertEqual(summary["overall"], "Needs attention")
            self.assertEqual([item["rating"] for item in summary["models"]], ["Excellent", "Excellent", "Excellent", "Failed before publish"])
            self.assertIn({"label": "Recall", "value": "97.0%"}, summary["models"][2]["metrics"])
            lines = training_summary_log_lines(summary)
            self.assertTrue(any("NanoWakeWord GRU — Failed before publish" in line for line in lines))

    def test_distinguishes_quality_rejection_and_preserved_published_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "nanowakeword/output/cnn/wake/wake.quality.json", {"passed": False})
            write_json(root / "trained/wake.wake-bundle.json", {
                "nano_wake_word": {"variants": {"cnn": {"model": "wake.nww.cnn.onnx"}}}
            })
            summary = build_training_summary(
                data_dir=root, trained_dir=root / "trained", safe_word="wake", phrase="wake",
                selected_models=["nano_cnn"], outcomes={"nano_cnn": 2},
            )
            model = summary["models"][0]
            self.assertEqual(model["failure_kind"], "quality_rejected")
            self.assertTrue(model["previous_published_model_active"])

    def test_marks_unreached_models_not_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = build_training_summary(
                data_dir=root,
                trained_dir=root / "trained",
                safe_word="jarvis",
                phrase="jarvis",
                selected_models=["open", "nano_cnn"],
                outcomes={"open": 2, "nano_cnn": None},
            )
            self.assertEqual(summary["models"][0]["status"], "failed")
            self.assertEqual(summary["models"][1]["rating"], "Not completed")


if __name__ == "__main__":
    unittest.main()
