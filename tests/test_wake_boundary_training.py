import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

try:
    from cli import prepare_wake_boundary_data
except ModuleNotFoundError:
    prepare_wake_boundary_data = None

try:
    import yaml
    from cli import write_boundary_training_yaml
except ModuleNotFoundError:
    yaml = None
    write_boundary_training_yaml = None


def write_tone(path: Path, frequency: float) -> None:
    samples = array(
        "h",
        (
            round(9000 * math.sin(2 * math.pi * frequency * index / 16000))
            for index in range(6400)
        ),
    )
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(samples.tobytes())


class WakeBoundaryTrainingTests(unittest.TestCase):
    @unittest.skipIf(
        prepare_wake_boundary_data is None,
        "NumPy is installed by the trainer environment",
    )
    def test_builds_deterministic_wake_plus_tail_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positives = root / "positives"
            positives.mkdir()
            for index, frequency in enumerate((180.0, 260.0, 340.0)):
                write_tone(positives / f"wake-{index}.wav", frequency)

            negatives = root / "boundary_negative_samples"
            first = prepare_wake_boundary_data.build([positives], negatives, 2)
            manifest_mtime = (negatives / ".generation_manifest.json").stat().st_mtime_ns
            second = prepare_wake_boundary_data.build([positives], negatives, 2)

            self.assertEqual(first["identity"], second["identity"])
            self.assertEqual(first["count"], 6)
            self.assertEqual(
                (negatives / ".generation_manifest.json").stat().st_mtime_ns,
                manifest_mtime,
            )
            self.assertEqual(len(list(negatives.glob("*.wav"))), 6)

    @unittest.skipIf(yaml is None, "PyYAML is installed by the trainer environment")
    def test_derives_endpoint_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "primary.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "features": [
                            {
                                "features_dir": "positive_features",
                                "truth": True,
                                "type": "mmap",
                            }
                        ],
                        "training_steps": [40000],
                        "early_stopping": {
                            "enabled": True,
                            "min_steps": 20000,
                            "patience_steps": 6000,
                            "min_delta": 0.001,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "boundary.yaml"
            features = root / "boundary_negative_features"
            write_boundary_training_yaml.build(
                source, output, root / "trained_models/wake_boundary", features, 20000
            )
            config = yaml.safe_load(output.read_text(encoding="utf-8"))
            boundary = config["features"][1]
            self.assertFalse(boundary["truth"])
            self.assertEqual(boundary["truncation_strategy"], "truncate_start")
            self.assertEqual(boundary["features_dir"], str(features.resolve()))
            self.assertEqual(config["training_steps"], [20000])
            self.assertEqual(config["negative_class_weight"], [12])
            self.assertEqual(
                config["early_stopping"],
                {
                    "enabled": True,
                    "min_steps": 12000,
                    "patience_steps": 4000,
                    "min_delta": 0.001,
                },
            )


if __name__ == "__main__":
    unittest.main()
