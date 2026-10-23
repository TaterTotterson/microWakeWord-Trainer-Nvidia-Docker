from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_mww_candidate", ROOT / "cli" / "select_mww_candidate.py"
)
assert SPEC is not None and SPEC.loader is not None
candidate_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate_module
SPEC.loader.exec_module(candidate_module)
Candidate = candidate_module.Candidate
choose_candidate = candidate_module.choose_candidate


def candidate(name: str, recall: float, false_accepts: float) -> Candidate:
    return Candidate(
        name,
        Path(f"{name}.tflite"),
        Path(f"{name}.json"),
        recall,
        false_accepts,
    )


class MicroWakeWordCandidateSelectionTests(unittest.TestCase):
    def test_corrective_candidate_wins_when_it_reaches_both_targets(self) -> None:
        initial = candidate("initial", 0.895, 0.10)
        corrective = candidate("corrective", 0.91, 0.20)

        selected = choose_candidate(
            initial,
            corrective,
            minimum_recall=0.90,
            target_faph=0.25,
            recall_margin=0.005,
        )

        self.assertEqual(selected.name, "corrective")

    def test_false_accept_target_protects_better_initial_candidate(self) -> None:
        initial = candidate("initial", 0.895, 0.10)
        corrective = candidate("corrective", 0.93, 0.60)

        selected = choose_candidate(
            initial,
            corrective,
            minimum_recall=0.90,
            target_faph=0.25,
            recall_margin=0.005,
        )

        self.assertEqual(selected.name, "initial")

    def test_lower_false_accept_rate_wins_when_passing_recalls_are_close(self) -> None:
        initial = candidate("initial", 0.912, 0.08)
        corrective = candidate("corrective", 0.915, 0.20)

        selected = choose_candidate(
            initial,
            corrective,
            minimum_recall=0.90,
            target_faph=0.25,
            recall_margin=0.005,
        )

        self.assertEqual(selected.name, "initial")


if __name__ == "__main__":
    unittest.main()
