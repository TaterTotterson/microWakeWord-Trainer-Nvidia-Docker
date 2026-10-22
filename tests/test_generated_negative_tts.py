import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cli" / "generate_negative_tts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("nvidia_generated_negative_tts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GeneratedNegativeTtsTests(unittest.TestCase):
    def test_builds_deduplicated_cache_and_excludes_target_phrase(self):
        module = load_module()
        module.MAX_TOTAL_SAMPLES = 8
        module.MAX_SAMPLES_PER_PHRASE = 4
        module.MIN_SAMPLES_PER_PHRASE = 2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = root / "generator.py"
            generator.write_text("# fake generator\n", encoding="utf-8")
            output = root / "generated_negative_samples"

            def fake_run(command, check):
                self.assertTrue(check)
                destination = Path(command[command.index("--output-dir") + 1])
                sample_count = int(command[command.index("--samples") + 1])
                destination.mkdir(parents=True, exist_ok=True)
                for index in range(sample_count):
                    (destination / f"{index}.wav").write_bytes(b"RIFF")

            argv = [
                str(SCRIPT), "--phrases-json", json.dumps(["hey later", "Hey Later", "hey tater", "okay tater"]),
                "--target-phrase", "hey tater", "--root-dir", str(output),
                "--generator-script", str(generator), "--data-dir", str(root),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(module.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(module.main(), 0)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(len(list((output / "combined").glob("*.wav"))), 8)
                self.assertEqual(module.main(), 0)
                self.assertEqual(run.call_count, 2)

            manifest = json.loads((output / module.MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["signature"]["phrases"], ["hey later", "okay tater"])

            clear_argv = argv[:]
            clear_argv[2] = "[]"
            with mock.patch.object(sys, "argv", clear_argv):
                self.assertEqual(module.main(), 0)
            self.assertFalse((output / "combined").exists())


if __name__ == "__main__":
    unittest.main()
