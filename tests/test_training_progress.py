import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_progress import PROGRESS_PREFIX, iter_progress_lines, parse_progress_line, tracked_generator


class TrainingProgressTests(unittest.TestCase):
    def test_stream_reader_yields_terminal_redraws_immediately(self):
        stream = io.StringIO("10it [00:01, 10it/s]\r20it [00:02, 10it/s]\r\nDone\n")
        self.assertEqual(
            list(iter_progress_lines(stream)),
            ["10it [00:01, 10it/s]\n", "20it [00:02, 10it/s]\n", "Done\n"],
        )

    def test_structured_progress_reaches_one_hundred_percent(self):
        emitted = []
        self.assertEqual(
            list(tracked_generator(
                range(2), total=2, stage="features:generated:training",
                label="Generated · Training features", unit="features",
                emit=emitted.append, update_interval_seconds=0,
            )),
            [0, 1],
        )
        progress, transient = parse_progress_line(emitted[-1])
        self.assertTrue(emitted[0].startswith(PROGRESS_PREFIX))
        self.assertTrue(transient)
        self.assertEqual(progress["current"], progress["total"])
        self.assertEqual(progress["current"], 2)
        self.assertEqual(progress["percent"], 100.0)

    def test_parses_tqdm_and_carriage_return_redraws(self):
        progress, transient = parse_progress_line(
            "596229it [06:33, 1465.05it/s]\r596400it [06:33, 1516.43it/s]"
        )
        self.assertTrue(transient)
        self.assertEqual(progress["current"], 596_400)
        determinate, transient = parse_progress_line(
            "Download 50%|█████     | 5.0k/10.0k [00:04<00:04, 1.25kit/s]"
        )
        self.assertTrue(transient)
        self.assertEqual(determinate["percent"], 50.0)
        self.assertEqual(determinate["total"], 10_000)

    def test_collapses_repeated_counts_and_parses_keras_steps(self):
        generated, generated_transient = parse_progress_line("Generated 40/80 samples...")
        keras, keras_transient = parse_progress_line("25/100 [======>] - ETA: 12s")
        attempt, attempt_transient = parse_progress_line("Attempt 1/3: GPU training")
        self.assertTrue(generated_transient)
        self.assertEqual(generated["percent"], 50.0)
        self.assertTrue(keras_transient)
        self.assertEqual(keras["percent"], 25.0)
        self.assertFalse(attempt_transient)
        self.assertIsNone(attempt)
        retry, retry_transient = parse_progress_line(
            "===== Attempt 2/3: GPU training (Blackwell compatibility profile) ====="
        )
        self.assertFalse(retry_transient)
        self.assertEqual(retry["stage"], "model")

    def test_recognizes_nvidia_feature_stage(self):
        progress, transient = parse_progress_line(
            "   Augmenting training (generated hard negatives)"
        )
        self.assertFalse(transient)
        self.assertEqual(progress["stage"], "features")
        self.assertEqual(progress["label"], "Building generated hard negatives · training features")

    def test_parses_microwakeword_step_updates(self):
        progress, transient = parse_progress_line(
            "INFO:absl:Step #500: rate 0.001000, accuracy 97.99%"
        )
        self.assertTrue(transient)
        self.assertEqual(progress["current"], 500)
        self.assertEqual(progress["unit"], "steps")

    def test_reports_early_stop_and_best_checkpoint(self):
        progress, transient = parse_progress_line(
            "INFO:absl:Early stopping at step 28500/40000: no meaningful "
            "validation improvement for 6000 steps; best checkpoint step 22500"
        )

        self.assertFalse(transient)
        self.assertEqual(progress["status"], "stopped_early")
        self.assertEqual(progress["current"], 28500)
        self.assertEqual(progress["total"], 40000)
        self.assertEqual(progress["best_step"], 22500)

    def test_reports_openwakeword_early_stop(self):
        progress, transient = parse_progress_line(
            "Early stopping openWakeWord main sequence at step 46000/50000 "
            "after 5 stale validation checks; best acceptable checkpoint step 42000"
        )

        self.assertFalse(transient)
        self.assertEqual(progress["status"], "stopped_early")
        self.assertEqual(progress["best_step"], 42000)


if __name__ == "__main__":
    unittest.main()
