from __future__ import annotations

import io
import signal
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import trainer_server as trainer


class _FakeTrainingProcess:
    def __init__(self):
        self.pid = 5432
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -signal.SIGTERM
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL


class _CompletedTrainingProcess:
    def __init__(self):
        self.pid = 6543
        self.returncode = 0
        self.stdout = io.StringIO("worker started\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class SessionStopTests(unittest.TestCase):
    def tearDown(self):
        trainer.TRAINING_STOP_EVENT.clear()

    def test_session_stop_terminates_the_process_group_and_allows_another_run(self):
        proc = _FakeTrainingProcess()
        original_process = trainer.TRAINING_PROCESS
        original_thread = trainer.TRAINING_THREAD
        try:
            trainer.TRAINING_PROCESS = proc
            trainer.TRAINING_THREAD = None
            with (
                patch.object(trainer.os, "getpgid", return_value=proc.pid),
                patch.object(trainer.os, "getpgrp", return_value=999),
                patch.object(trainer.os, "killpg") as killpg,
            ):
                self.assertTrue(trainer._stop_current_training(timeout=0.2))
            killpg.assert_called_once_with(proc.pid, signal.SIGTERM)
            self.assertFalse(trainer.TRAINING_STOP_EVENT.is_set())
        finally:
            trainer.TRAINING_PROCESS = original_process
            trainer.TRAINING_THREAD = original_thread

    def test_reserved_running_state_starts_the_background_worker(self):
        original_process = trainer.TRAINING_PROCESS
        original_thread = trainer.TRAINING_THREAD
        original_raw_phrase = trainer.STATE.get("raw_phrase")
        original_training = dict(trainer.STATE["training"])
        try:
            with tempfile.TemporaryDirectory() as directory:
                data_dir = Path(directory)
                process = _CompletedTrainingProcess()
                trainer.TRAINING_PROCESS = None
                trainer.TRAINING_THREAD = threading.current_thread()
                with trainer.STATE_LOCK:
                    trainer.STATE["raw_phrase"] = "hey tater"
                    trainer.STATE["training"]["running"] = True

                with (
                    patch.object(trainer, "DATA_DIR", data_dir),
                    patch.object(trainer, "_ensure_training_venv"),
                    patch.object(trainer, "_ensure_training_datasets"),
                    patch.object(trainer.subprocess, "Popen", return_value=process) as popen,
                    patch.object(trainer, "_normalize_output_artifacts"),
                ):
                    trainer._run_training_background(
                        "hey_tater",
                        "en",
                        True,
                        auto_run=False,
                        tts_mode="modern",
                    )

                popen.assert_called_once()
                log_text = (data_dir / "recorder_training.log").read_text(encoding="utf-8")
                self.assertIn("Nvidia Docker Training Run", log_text)
                self.assertIn("worker started", log_text)
                self.assertFalse(trainer.STATE["training"]["running"])
                self.assertEqual(trainer.STATE["training"]["exit_code"], 0)
                self.assertIsNone(trainer.TRAINING_THREAD)
        finally:
            with trainer.STATE_LOCK:
                trainer.STATE["raw_phrase"] = original_raw_phrase
                trainer.STATE["training"].clear()
                trainer.STATE["training"].update(original_training)
            trainer.TRAINING_PROCESS = original_process
            trainer.TRAINING_THREAD = original_thread


if __name__ == "__main__":
    unittest.main()
