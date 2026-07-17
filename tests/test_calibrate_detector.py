import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "cli"
    / "calibrate_detector.py"
)
SPEC = importlib.util.spec_from_file_location("calibrate_detector", SCRIPT_PATH)
calibrate_detector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibrate_detector)


def candidate(cutoff, window, recall, false_accepts_per_hour):
    return {
        "probability_cutoff": cutoff,
        "sliding_window_size": window,
        "recall": recall,
        "false_accepts_per_hour": false_accepts_per_hour,
    }


class CalibrationSelectionTests(unittest.TestCase):
    def test_defaults_are_conservative(self):
        self.assertEqual(calibrate_detector.DEFAULT_WINDOW_SIZES, [5, 6, 7])
        self.assertEqual(calibrate_detector.DEFAULT_CUTOFF_MIN, 0.95)
        self.assertEqual(calibrate_detector.DEFAULT_RECALL_MARGIN, 0.005)

    def test_prefers_zero_false_accepts_within_recall_margin(self):
        candidates = [
            candidate(0.95, 5, 0.99894, 0.103408),
            candidate(0.95, 6, 0.99744, 0.0),
            candidate(0.95, 7, 0.99554, 0.0),
        ]

        best, selected_limit = calibrate_detector._select_best_candidate(
            candidates,
            target_faph=0.25,
            recall_margin=0.005,
        )

        self.assertEqual(best["sliding_window_size"], 6)
        self.assertEqual(best["false_accepts_per_hour"], 0.0)
        self.assertEqual(selected_limit, 0.25)

    def test_does_not_trade_away_recall_beyond_margin(self):
        candidates = [
            candidate(0.95, 5, 0.99, 0.1),
            candidate(0.99, 6, 0.90, 0.0),
        ]

        best, _ = calibrate_detector._select_best_candidate(
            candidates,
            target_faph=0.25,
            recall_margin=0.005,
        )

        self.assertEqual(best["sliding_window_size"], 5)

    def test_uses_strictest_available_false_accept_tier(self):
        candidates = [
            candidate(0.95, 5, 0.99, 0.6),
            candidate(0.99, 6, 0.99, 1.5),
        ]

        best, selected_limit = calibrate_detector._select_best_candidate(
            candidates,
            target_faph=0.25,
            recall_margin=0.005,
        )

        self.assertEqual(best["false_accepts_per_hour"], 0.6)
        self.assertEqual(selected_limit, 0.75)

    def test_rejects_negative_recall_margin(self):
        with self.assertRaises(ValueError):
            calibrate_detector._select_best_candidate(
                [candidate(0.95, 6, 0.99, 0.0)],
                target_faph=0.25,
                recall_margin=-0.001,
            )


if __name__ == "__main__":
    unittest.main()
