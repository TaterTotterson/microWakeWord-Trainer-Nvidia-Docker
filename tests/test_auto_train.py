import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import trainer_server as trainer


def silent_wav_bytes(duration_s: float = 0.25) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * int(16000 * duration_s))
    return output.getvalue()


class AutoTrainTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.original_paths = (
            trainer.CAPTURED_DIR,
            trainer.NEGATIVE_DIR,
            trainer.PERSONAL_DIR,
            trainer.AUTO_TRAIN_CONFIG_FILE,
            trainer.AUTO_TRAIN_STATE_FILE,
        )
        trainer.CAPTURED_DIR = root / "captured_audio"
        trainer.NEGATIVE_DIR = root / "negative_samples"
        trainer.PERSONAL_DIR = root / "personal_samples"
        trainer.AUTO_TRAIN_CONFIG_FILE = root / "auto_train_config.json"
        trainer.AUTO_TRAIN_STATE_FILE = root / "auto_train_state.json"
        for directory in (trainer.CAPTURED_DIR, trainer.NEGATIVE_DIR, trainer.PERSONAL_DIR):
            directory.mkdir(parents=True)

        self.original_config = dict(trainer.AUTO_TRAIN_CONFIG)
        self.original_state = dict(trainer.AUTO_TRAIN_STATE)
        trainer.AUTO_TRAIN_CONFIG.clear()
        trainer.AUTO_TRAIN_CONFIG.update(
            trainer._normalize_auto_train_config(
                {
                    "enabled": True,
                    "wake_phrase": "hey tater",
                    "language": "en",
                    "tater_url": "http://127.0.0.1:8501",
                    "stt_device": "auto",
                    "stt_compute_type": "auto",
                }
            )
        )
        trainer.AUTO_TRAIN_STATE.clear()
        trainer.AUTO_TRAIN_STATE.update(trainer.AUTO_TRAIN_DEFAULT_STATE)

    def tearDown(self):
        (
            trainer.CAPTURED_DIR,
            trainer.NEGATIVE_DIR,
            trainer.PERSONAL_DIR,
            trainer.AUTO_TRAIN_CONFIG_FILE,
            trainer.AUTO_TRAIN_STATE_FILE,
        ) = self.original_paths
        trainer.AUTO_TRAIN_CONFIG.clear()
        trainer.AUTO_TRAIN_CONFIG.update(self.original_config)
        trainer.AUTO_TRAIN_STATE.clear()
        trainer.AUTO_TRAIN_STATE.update(self.original_state)
        self.tempdir.cleanup()

    def add_capture(self, name: str = "wake.wav", wake_word: str = "hey_tater") -> Path:
        audio_path = trainer.CAPTURED_DIR / name
        audio_path.write_bytes(silent_wav_bytes())
        trainer._write_sidecar_json(
            audio_path,
            {
                "original_name": name,
                "wake_word": wake_word,
                "event_type": "wake_detected",
                "review_status": "pending",
            },
        )
        return audio_path

    def test_phrase_matching_normalizes_case_punctuation_and_underscores(self):
        self.assertTrue(trainer._transcript_contains_wake_phrase("Okay, HEY TATER!", "hey_tater"))
        self.assertFalse(trainer._transcript_contains_wake_phrase("Turn on the television", "hey tater"))

    def test_phrase_miss_moves_wake_trigger_to_negative_samples(self):
        self.add_capture()
        with patch.object(trainer, "_transcribe_capture_with_faster_whisper", return_value="turn on the kitchen lights"):
            trainer._auto_review_capture("wake.wav")

        self.assertFalse((trainer.CAPTURED_DIR / "wake.wav").exists())
        negatives = list(trainer.NEGATIVE_DIR.glob("*.wav"))
        self.assertEqual(len(negatives), 1)
        metadata = trainer._load_sidecar_json(negatives[0])
        self.assertTrue(metadata["auto_negative"])
        self.assertEqual(metadata["review_status"], "auto_approved_negative")
        self.assertEqual(metadata["transcript"], "turn on the kitchen lights")
        self.assertEqual(trainer.AUTO_TRAIN_STATE["pending_negative_count"], 1)

    def test_matching_phrase_stays_in_manual_review_inbox(self):
        audio_path = self.add_capture()
        with patch.object(trainer, "_transcribe_capture_with_faster_whisper", return_value="hey tater turn on the lights"):
            trainer._auto_review_capture("wake.wav")

        self.assertTrue(audio_path.exists())
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        metadata = trainer._load_sidecar_json(audio_path)
        self.assertEqual(metadata["auto_review_status"], "wake_phrase_detected")
        self.assertEqual(trainer.AUTO_TRAIN_STATE["pending_negative_count"], 0)

    def test_capture_for_another_wake_word_is_not_transcribed(self):
        audio_path = self.add_capture(wake_word="computer")
        with patch.object(trainer, "_transcribe_capture_with_faster_whisper") as transcribe:
            trainer._auto_review_capture("wake.wav")

        transcribe.assert_not_called()
        self.assertTrue(audio_path.exists())
        metadata = trainer._load_sidecar_json(audio_path)
        self.assertEqual(metadata["auto_review_status"], "different_wake_phrase")

    def test_due_schedule_starts_training_after_minimum_negatives(self):
        trainer.AUTO_TRAIN_CONFIG["schedule_hours"] = 24
        trainer.AUTO_TRAIN_CONFIG["minimum_new_negatives"] = 3
        trainer.AUTO_TRAIN_STATE["pending_negative_count"] = 3
        trainer.AUTO_TRAIN_STATE["next_run_at"] = "2000-01-01T00:00:00+00:00"
        with patch.object(trainer, "_start_auto_training", return_value={"ok": True, "started": True}) as start:
            trainer._maybe_run_scheduled_auto_training()

        start.assert_called_once_with()
        self.assertTrue(trainer.AUTO_TRAIN_STATE["next_run_at"])

    def test_tater_refresh_repushes_settings_with_selector_and_token(self):
        trainer.AUTO_TRAIN_CONFIG.update(
            {
                "notify_satellites": True,
                "tater_url": "http://127.0.0.1:8501",
                "tater_selector": "kitchen-sat",
                "tater_api_token": "secret-token",
            }
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"push":{"count":2}}'

        with patch.object(trainer, "urlopen", return_value=Response()) as open_url:
            result = trainer._notify_tater_satellites()

        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 2)
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8501/api/tater/satellite/v1/settings")
        self.assertEqual(request.get_header("X-tater-token"), "secret-token")
        self.assertEqual(json.loads(request.data), {"selector": "kitchen-sat", "settings": {}})

    def test_advertised_url_uses_non_loopback_browser_host(self):
        request = SimpleNamespace(
            base_url="http://192.168.1.50:8789/",
            url=SimpleNamespace(hostname="192.168.1.50", scheme="http", port=8789),
        )
        self.assertEqual(trainer._advertised_base_url(request), "http://192.168.1.50:8789")

    def test_advertised_url_replaces_localhost_with_discovered_lan_host(self):
        request = SimpleNamespace(
            base_url="http://127.0.0.1:8789/",
            url=SimpleNamespace(hostname="127.0.0.1", scheme="http", port=8789),
        )
        with patch.object(trainer, "_discover_lan_ipv4", return_value="192.168.1.60"):
            self.assertEqual(trainer._advertised_base_url(request), "http://192.168.1.60:8789")

    def test_configured_public_url_takes_precedence(self):
        trainer.AUTO_TRAIN_CONFIG["advertised_base_url"] = "http://trainer.local:8789"
        request = SimpleNamespace(
            base_url="http://127.0.0.1:8789/",
            url=SimpleNamespace(hostname="127.0.0.1", scheme="http", port=8789),
        )
        self.assertEqual(trainer._advertised_base_url(request), "http://trainer.local:8789")

    def test_faster_whisper_auto_runtime_prefers_cuda_and_float16(self):
        fake_ctranslate2 = SimpleNamespace(get_cuda_device_count=lambda: 1)
        with patch.dict(sys.modules, {"ctranslate2": fake_ctranslate2}):
            self.assertEqual(
                trainer._resolve_faster_whisper_runtime("auto", "auto"),
                ("cuda", "float16"),
            )

    def test_faster_whisper_auto_runtime_falls_back_to_cpu_int8(self):
        fake_ctranslate2 = SimpleNamespace(get_cuda_device_count=lambda: 0)
        with patch.dict(sys.modules, {"ctranslate2": fake_ctranslate2}):
            self.assertEqual(
                trainer._resolve_faster_whisper_runtime("auto", "auto"),
                ("cpu", "int8"),
            )

    def test_faster_whisper_transcription_joins_segments_and_records_runtime(self):
        fake_model = SimpleNamespace()
        fake_model.transcribe = Mock(
            return_value=(
                iter([SimpleNamespace(text=" turn on "), SimpleNamespace(text="the lights ")]),
                SimpleNamespace(),
            )
        )
        with (
            patch.object(trainer, "_resolve_faster_whisper_runtime", return_value=("cuda", "float16")),
            patch.object(trainer, "_load_faster_whisper_model", return_value=fake_model),
        ):
            transcript = trainer._transcribe_capture_with_faster_whisper(
                Path("wake.wav"),
                model="small.en",
                language="en",
            )

        self.assertEqual(transcript, "turn on the lights")
        fake_model.transcribe.assert_called_once_with(
            "wake.wav",
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
        )
        self.assertEqual(trainer.AUTO_TRAIN_STATE["last_stt_device"], "cuda")
        self.assertEqual(trainer.AUTO_TRAIN_STATE["last_stt_compute_type"], "float16")


if __name__ == "__main__":
    unittest.main()
