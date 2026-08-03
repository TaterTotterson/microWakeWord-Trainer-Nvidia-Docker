from __future__ import annotations

import signal
import unittest
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


if __name__ == "__main__":
    unittest.main()
