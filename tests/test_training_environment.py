from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trainer_server as trainer


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainingEnvironmentTests(unittest.TestCase):
    def test_blackwell_setup_explicitly_installs_and_checks_tensorboard(self) -> None:
        setup = (REPO_ROOT / "cli" / "setup_blackwell_venv").read_text(encoding="utf-8")
        health_check = (REPO_ROOT / "cli" / "check_training_venv").read_text(encoding="utf-8")
        dockerfile = (REPO_ROOT / "dockerfile.blackwell").read_text(encoding="utf-8")

        self.assertIn("    tensorboard ", setup)
        self.assertIn('"${VENV}/bin/python" "${HEALTH_CHECK}"', setup)
        self.assertIn("import tensorboard", health_check)
        self.assertIn("from tensorboard.summary.v2 import scalar", health_check)
        self.assertIn("from microwakeword.audio.augmentation import Augmentation", health_check)
        self.assertIn("from microwakeword import model_train_eval", health_check)
        self.assertIn("ENV MWW_BLACKWELL_TF=required", dockerfile)

    def test_blackwell_main_venv_explicitly_installs_tensorboard(self) -> None:
        setup = (REPO_ROOT / "cli" / "setup_python_venv").read_text(encoding="utf-8")

        self.assertIn('default_tensorboard_specs=( "tensorboard" )', setup)
        self.assertIn('"${VENV}/bin/python" "${PROGDIR}/check_training_venv"', setup)
        self.assertIn('pip_install --no-deps -e "${MWW}"', setup)
        for dependency in (
            "audiomentations",
            "audio_metadata",
            "datasets",
            "mmap_ninja",
            "pymicro-features",
            "webrtcvad-wheels",
        ):
            self.assertIn(dependency, setup)

    def test_healthy_existing_training_venv_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / ".venv" / "bin").mkdir(parents=True)
            (data_dir / ".venv" / "bin" / "activate").touch()
            python = data_dir / ".venv" / "bin" / "python"
            python.touch()

            with (
                patch.object(trainer, "DATA_DIR", data_dir),
                patch.object(trainer, "_training_venv_health", return_value=(True, "")),
                patch.object(trainer, "_run_streamed") as run_streamed,
                patch.object(trainer, "_append_train_log"),
            ):
                trainer._ensure_training_venv(data_dir / "training.log")

            run_streamed.assert_not_called()

    def test_incomplete_existing_training_venv_is_repaired_and_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            cli_dir = data_dir / "cli"
            cli_dir.mkdir()
            (cli_dir / "setup_python_venv").touch()
            (data_dir / ".venv" / "bin").mkdir(parents=True)
            (data_dir / ".venv" / "bin" / "activate").touch()
            python = data_dir / ".venv" / "bin" / "python"
            python.touch()

            with (
                patch.object(trainer, "DATA_DIR", data_dir),
                patch.object(trainer, "CLI_DIR", cli_dir),
                patch.object(
                    trainer,
                    "_training_venv_health",
                    side_effect=[(False, "No module named 'tensorboard'"), (True, "")],
                ) as health,
                patch.object(trainer, "_run_streamed", return_value=0) as run_streamed,
                patch.object(trainer, "_append_train_log") as append_log,
            ):
                trainer._ensure_training_venv(data_dir / "training.log")

            self.assertEqual(health.call_count, 2)
            run_streamed.assert_called_once()
            self.assertTrue(
                any("will be repaired" in call.args[0] for call in append_log.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
