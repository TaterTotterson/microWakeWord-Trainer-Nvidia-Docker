import io
import json
import queue
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
    def clear_review_queue(self):
        while True:
            try:
                trainer.AUTO_TRAIN_REVIEW_QUEUE.get_nowait()
            except queue.Empty:
                break
            else:
                trainer.AUTO_TRAIN_REVIEW_QUEUE.task_done()
        trainer.AUTO_TRAIN_QUEUED_FILES.clear()

    def setUp(self):
        self.clear_review_queue()
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.original_paths = (
            trainer.CAPTURED_DIR,
            trainer.NEGATIVE_DIR,
            trainer.PERSONAL_DIR,
            trainer.AUTO_TRAIN_CONFIG_FILE,
            trainer.AUTO_TRAIN_STATE_FILE,
            trainer.AUTO_TRAIN_MODEL_DIR,
        )
        trainer.CAPTURED_DIR = root / "captured_audio"
        trainer.NEGATIVE_DIR = root / "negative_samples"
        trainer.PERSONAL_DIR = root / "personal_samples"
        trainer.AUTO_TRAIN_CONFIG_FILE = root / "auto_train_config.json"
        trainer.AUTO_TRAIN_STATE_FILE = root / "auto_train_state.json"
        trainer.AUTO_TRAIN_MODEL_DIR = root / "auto_train_models"
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
            trainer.AUTO_TRAIN_MODEL_DIR,
        ) = self.original_paths
        trainer.AUTO_TRAIN_CONFIG.clear()
        trainer.AUTO_TRAIN_CONFIG.update(self.original_config)
        trainer.AUTO_TRAIN_STATE.clear()
        trainer.AUTO_TRAIN_STATE.update(self.original_state)
        self.clear_review_queue()
        self.tempdir.cleanup()

    def add_capture(
        self,
        name: str = "wake.wav",
        wake_word: str = "hey_tater",
        event_type: str = "wake_detected",
        blocked_by_vad: bool = False,
    ) -> Path:
        audio_path = trainer.CAPTURED_DIR / name
        audio_path.write_bytes(silent_wav_bytes())
        trainer._write_sidecar_json(
            audio_path,
            {
                "original_name": name,
                "wake_word": wake_word,
                "event_type": event_type,
                "blocked_by_vad": blocked_by_vad,
                "review_status": "pending",
            },
        )
        return audio_path

    def test_phrase_matching_normalizes_case_punctuation_and_underscores(self):
        self.assertTrue(trainer._transcript_contains_wake_phrase("Okay, HEY TATER!", "hey_tater"))
        self.assertFalse(trainer._transcript_contains_wake_phrase("Turn on the television", "hey tater"))

    def test_phrase_similarity_recognizes_real_short_clip_mishearings(self):
        for transcript in ("Hey, haters.", "Hate hater.", "Hey Ganger.", "Hey, gator."):
            with self.subTest(transcript=transcript):
                self.assertGreaterEqual(
                    trainer._wake_phrase_similarity(transcript, "hey tater"),
                    trainer.WAKE_PHRASE_GUIDANCE_MIN_SIMILARITY,
                )

        for transcript in ("turn on the lights", "what is the weather", "play some music"):
            with self.subTest(transcript=transcript):
                self.assertLess(
                    trainer._wake_phrase_similarity(transcript, "hey tater"),
                    trainer.WAKE_PHRASE_GUIDANCE_MIN_SIMILARITY,
                )

    def test_stt_engine_selection_uses_managed_models(self):
        config = trainer._normalize_auto_train_config(
            {
                "stt_engine": "parakeet-onnx",
                "stt_model": "user/should-not-be-used",
                "stt_device": "cpu",
                "stt_compute_type": "float32",
            }
        )

        self.assertEqual(config["stt_engine"], trainer.STT_ENGINE_PARAKEET_ONNX)
        self.assertNotIn("stt_model", config)
        self.assertNotIn("stt_device", config)
        self.assertNotIn("stt_compute_type", config)
        self.assertEqual(
            trainer._managed_stt_model(config["stt_engine"], "en"),
            trainer.DEFAULT_PARAKEET_ONNX_MODEL,
        )
        self.assertEqual(
            trainer._managed_stt_model(trainer.STT_ENGINE_FASTER_WHISPER, "de"),
            trainer.DEFAULT_FASTER_WHISPER_MULTILINGUAL_MODEL,
        )

    def test_stt_router_supports_both_nvidia_engines(self):
        audio_path = Path("wake.wav")
        with (
            patch.object(trainer, "_transcribe_capture_with_faster_whisper", return_value="faster") as faster,
            patch.object(trainer, "_transcribe_capture_with_parakeet", return_value="parakeet") as parakeet,
        ):
            self.assertEqual(
                trainer._transcribe_capture(
                    audio_path,
                    engine=trainer.STT_ENGINE_FASTER_WHISPER,
                    language="en",
                ),
                "faster",
            )
            self.assertEqual(
                trainer._transcribe_capture(
                    audio_path,
                    engine=trainer.STT_ENGINE_PARAKEET_ONNX,
                    language="en",
                ),
                "parakeet",
            )

        faster.assert_called_once()
        parakeet.assert_called_once()

    def test_guided_faster_whisper_uses_dynamic_wake_phrase(self):
        fake_model = SimpleNamespace(
            transcribe=Mock(
                return_value=(
                    iter([SimpleNamespace(text=" hello "), SimpleNamespace(text="potato ")]),
                    SimpleNamespace(),
                )
            )
        )
        with (
            patch.object(
                trainer,
                "_resolve_faster_whisper_runtime",
                return_value=("cuda", "float16"),
            ),
            patch.object(trainer, "_load_faster_whisper_model", return_value=fake_model),
        ):
            transcript = trainer._transcribe_capture_with_faster_whisper_guided(
                Path("wake.wav"),
                model="small.en",
                language="en",
                wake_phrase="Hello_Potato",
            )

        self.assertEqual(transcript, "hello potato")
        _, kwargs = fake_model.transcribe.call_args
        self.assertEqual(kwargs["hotwords"], "hello potato")
        self.assertIn("hello potato", kwargs["initial_prompt"])
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertEqual(kwargs["best_of"], 5)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertFalse(kwargs["condition_on_previous_text"])

    def test_parakeet_loader_prefers_cuda_then_cpu(self):
        fake_model = object()
        fake_onnx_asr = SimpleNamespace(load_model=Mock(return_value=fake_model))
        fake_huggingface_hub = SimpleNamespace(
            snapshot_download=Mock(return_value=str(trainer.AUTO_TRAIN_MODEL_DIR))
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "onnx_asr": fake_onnx_asr,
                    "huggingface_hub": fake_huggingface_hub,
                },
            ),
            patch.object(
                trainer,
                "_parakeet_onnx_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
        ):
            with trainer.PARAKEET_ONNX_MODEL_LOCK:
                trainer.PARAKEET_ONNX_MODEL_CACHE.clear()
            loaded = trainer._load_parakeet_onnx_model()

        self.assertIs(loaded, fake_model)
        fake_huggingface_hub.snapshot_download.assert_called_once_with(
            repo_id=trainer.DEFAULT_PARAKEET_ONNX_REPO,
            local_dir=str(trainer.AUTO_TRAIN_MODEL_DIR),
            allow_patterns=[
                "config.json",
                "vocab.txt",
                "encoder-model.int8.onnx",
                "encoder-model.int8.onnx.data",
                "decoder_joint-model.int8.onnx",
                "decoder_joint-model.int8.onnx.data",
            ],
        )
        fake_onnx_asr.load_model.assert_called_once_with(
            trainer.DEFAULT_PARAKEET_ONNX_MODEL,
            str(trainer.AUTO_TRAIN_MODEL_DIR),
            quantization="int8",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_parakeet_loader_reuses_complete_snapshot_offline(self):
        fake_model = object()
        fake_onnx_asr = SimpleNamespace(load_model=Mock(return_value=fake_model))
        fake_huggingface_hub = SimpleNamespace(snapshot_download=Mock())
        trainer.AUTO_TRAIN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for filename in (
            "config.json",
            "vocab.txt",
            "encoder-model.int8.onnx",
            "decoder_joint-model.int8.onnx",
        ):
            (trainer.AUTO_TRAIN_MODEL_DIR / filename).touch()
        with (
            patch.dict(
                sys.modules,
                {
                    "onnx_asr": fake_onnx_asr,
                    "huggingface_hub": fake_huggingface_hub,
                },
            ),
            patch.object(
                trainer,
                "_parakeet_onnx_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
        ):
            with trainer.PARAKEET_ONNX_MODEL_LOCK:
                trainer.PARAKEET_ONNX_MODEL_CACHE.clear()
            loaded = trainer._load_parakeet_onnx_model()

        self.assertIs(loaded, fake_model)
        fake_huggingface_hub.snapshot_download.assert_not_called()
        fake_onnx_asr.load_model.assert_called_once_with(
            trainer.DEFAULT_PARAKEET_ONNX_MODEL,
            str(trainer.AUTO_TRAIN_MODEL_DIR),
            quantization="int8",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_ui_exposes_engine_selector_without_manual_runtime_fields(self):
        source = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "TrainerApp.vue").read_text(
            encoding="utf-8"
        )
        self.assertIn('v-model="trainer.autoForm.stt_engine"', source)
        self.assertNotIn('trainer.autoForm.stt_model', source)
        self.assertNotIn('trainer.autoForm.stt_device', source)
        self.assertNotIn('trainer.autoForm.stt_compute_type', source)
        self.assertIn("Guided wake check", source)

    def test_phrase_miss_moves_wake_trigger_to_negative_samples(self):
        self.add_capture()
        with patch.object(trainer, "_transcribe_capture", return_value="turn on the kitchen lights"):
            trainer._auto_review_capture("wake.wav")

        self.assertFalse((trainer.CAPTURED_DIR / "wake.wav").exists())
        negatives = list(trainer.NEGATIVE_DIR.glob("*.wav"))
        self.assertEqual(len(negatives), 1)
        metadata = trainer._load_sidecar_json(negatives[0])
        self.assertTrue(metadata["auto_negative"])
        self.assertEqual(metadata["review_status"], "auto_approved_negative")
        self.assertEqual(metadata["transcript"], "turn on the kitchen lights")
        self.assertEqual(metadata["auto_review_stt_engine"], "faster_whisper")
        self.assertEqual(metadata["auto_review_stt_model"], "small.en")
        sample_item = trainer._sample_item_from_path(negatives[0], "negative")
        self.assertEqual(sample_item["transcript"], "turn on the kitchen lights")
        self.assertEqual(sample_item["auto_review_stt_engine"], "faster_whisper")
        self.assertEqual(sample_item["auto_review_stt_model"], "small.en")
        self.assertEqual(trainer.AUTO_TRAIN_STATE["pending_negative_count"], 1)

    def test_matching_phrase_stays_in_manual_review_inbox(self):
        audio_path = self.add_capture()
        with patch.object(trainer, "_transcribe_capture", return_value="hey tater turn on the lights"):
            trainer._auto_review_capture("wake.wav")

        self.assertTrue(audio_path.exists())
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        metadata = trainer._load_sidecar_json(audio_path)
        self.assertEqual(metadata["auto_review_status"], "wake_phrase_detected")
        self.assertEqual(trainer.AUTO_TRAIN_STATE["pending_negative_count"], 0)

    def test_close_transcript_uses_guided_faster_whisper_confirmation(self):
        audio_path = self.add_capture()
        with (
            patch.object(trainer, "_transcribe_capture", return_value="Hey, haters."),
            patch.object(
                trainer,
                "_transcribe_capture_with_faster_whisper_guided",
                return_value="Hey Tater",
            ) as guided,
        ):
            trainer._auto_review_capture("wake.wav")

        self.assertTrue(audio_path.exists())
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        metadata = trainer._load_sidecar_json(audio_path)
        self.assertEqual(metadata["auto_review_status"], "wake_phrase_detected")
        self.assertEqual(metadata["transcript"], "Hey, haters.")
        self.assertEqual(metadata["auto_review_guided_transcript"], "Hey Tater")
        self.assertEqual(metadata["auto_review_match_method"], "guided_close_match")
        self.assertGreaterEqual(
            metadata["auto_review_phrase_similarity"],
            trainer.WAKE_PHRASE_GUIDANCE_MIN_SIMILARITY,
        )
        guided.assert_called_once()
        guided_args, guided_kwargs = guided.call_args
        self.assertEqual(guided_args[0].resolve(), audio_path.resolve())
        self.assertEqual(
            guided_kwargs,
            {
                "model": "small.en",
                "language": "en",
                "wake_phrase": "hey tater",
            },
        )

    def test_unconfirmed_close_transcript_stays_for_manual_review(self):
        audio_path = self.add_capture()
        with (
            patch.object(trainer, "_transcribe_capture", return_value="Hate hater."),
            patch.object(
                trainer,
                "_transcribe_capture_with_faster_whisper_guided",
                return_value="Hate hater.",
            ),
        ):
            trainer._auto_review_capture("wake.wav")

        self.assertTrue(audio_path.exists())
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        metadata = trainer._load_sidecar_json(audio_path)
        self.assertEqual(metadata["auto_review_status"], "wake_phrase_ambiguous")
        self.assertEqual(metadata["transcript"], "Hate hater.")
        self.assertEqual(metadata["auto_review_guided_transcript"], "Hate hater.")
        self.assertEqual(trainer.AUTO_TRAIN_STATE["pending_negative_count"], 0)

        self.assertEqual(trainer._queue_pending_auto_reviews(), 0)
        self.assertEqual(trainer._queue_pending_auto_reviews(force=True), 1)

    def test_close_parakeet_transcript_stays_for_manual_review(self):
        audio_path = self.add_capture()
        trainer.AUTO_TRAIN_CONFIG["stt_engine"] = trainer.STT_ENGINE_PARAKEET_ONNX
        with (
            patch.object(trainer, "_transcribe_capture", return_value="Hey Ganger."),
            patch.object(trainer, "_transcribe_capture_with_faster_whisper_guided") as guided,
        ):
            trainer._auto_review_capture("wake.wav")

        guided.assert_not_called()
        self.assertTrue(audio_path.exists())
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        metadata = trainer._load_sidecar_json(audio_path)
        self.assertEqual(metadata["auto_review_status"], "wake_phrase_ambiguous")
        self.assertEqual(metadata["auto_review_stt_engine"], "parakeet_onnx")

    def test_matching_phrase_is_deleted_when_cleanup_is_enabled(self):
        audio_path = self.add_capture()
        trainer.AUTO_TRAIN_CONFIG["delete_confirmed_wakes"] = True
        with patch.object(
            trainer,
            "_transcribe_capture",
            return_value="hey tater turn on the lights",
        ):
            trainer._auto_review_capture("wake.wav")

        self.assertFalse(audio_path.exists())
        self.assertFalse(audio_path.with_suffix(".json").exists())
        self.assertFalse(list(trainer.PERSONAL_DIR.glob("*.wav")))
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        self.assertEqual(trainer.AUTO_TRAIN_STATE["last_review_result"], "deleted_confirmed_wake")

    def test_cleanup_processes_previously_confirmed_wake_without_retranscribing(self):
        audio_path = self.add_capture()
        metadata = trainer._load_sidecar_json(audio_path)
        metadata.update(
            {
                "auto_review_status": "wake_phrase_detected",
                "transcript": "hey tater",
            }
        )
        trainer._write_sidecar_json(audio_path, metadata)
        trainer.AUTO_TRAIN_CONFIG["delete_confirmed_wakes"] = True

        self.assertEqual(trainer._queue_pending_auto_reviews(), 1)
        with patch.object(trainer, "_transcribe_capture") as transcribe:
            trainer._auto_review_capture("wake.wav")

        transcribe.assert_not_called()
        self.assertFalse(audio_path.exists())
        self.assertEqual(trainer.AUTO_TRAIN_STATE["last_review_transcript"], "hey tater")

    def test_close_miss_is_not_transcribed_by_default(self):
        audio_path = self.add_capture(event_type="close_miss")
        with patch.object(trainer, "_transcribe_capture") as transcribe:
            trainer._auto_review_capture("wake.wav")

        transcribe.assert_not_called()
        self.assertTrue(audio_path.exists())
        self.assertFalse(trainer._load_sidecar_json(audio_path).get("auto_review_status"))

    def test_existing_close_miss_is_queued_when_promotion_is_enabled(self):
        self.add_capture(event_type="close_miss")
        self.assertEqual(trainer._queue_pending_auto_reviews(), 0)

        trainer.AUTO_TRAIN_CONFIG["promote_close_misses"] = True
        self.assertEqual(trainer._queue_pending_auto_reviews(), 1)

    def test_close_miss_with_phrase_is_promoted_when_enabled(self):
        self.add_capture(event_type="close_miss")
        trainer.AUTO_TRAIN_CONFIG["promote_close_misses"] = True
        with patch.object(trainer, "_transcribe_capture", return_value="hey tater"):
            trainer._auto_review_capture("wake.wav")

        self.assertFalse((trainer.CAPTURED_DIR / "wake.wav").exists())
        positives = list(trainer.PERSONAL_DIR.glob("*.wav"))
        self.assertEqual(len(positives), 1)
        metadata = trainer._load_sidecar_json(positives[0])
        self.assertTrue(metadata["auto_positive"])
        self.assertEqual(metadata["review_status"], "auto_approved_personal")
        self.assertEqual(metadata["transcript"], "hey tater")
        sample_item = trainer._sample_item_from_path(positives[0], "personal")
        self.assertEqual(sample_item["transcript"], "hey tater")
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        self.assertEqual(trainer.AUTO_TRAIN_STATE["pending_negative_count"], 0)

    def test_guided_stt_remains_visible_after_positive_auto_sort(self):
        self.add_capture(event_type="close_miss")
        trainer.AUTO_TRAIN_CONFIG["promote_close_misses"] = True
        with (
            patch.object(trainer, "_transcribe_capture", return_value="Hey, haters."),
            patch.object(
                trainer,
                "_transcribe_capture_with_faster_whisper_guided",
                return_value="Hey Tater",
            ),
        ):
            trainer._auto_review_capture("wake.wav")

        positives = list(trainer.PERSONAL_DIR.glob("*.wav"))
        self.assertEqual(len(positives), 1)
        sample_item = trainer._sample_item_from_path(positives[0], "personal")
        self.assertEqual(sample_item["transcript"], "Hey, haters.")
        self.assertEqual(sample_item["auto_review_guided_transcript"], "Hey Tater")

    def test_close_miss_without_phrase_stays_in_inbox(self):
        audio_path = self.add_capture(event_type="close_miss")
        trainer.AUTO_TRAIN_CONFIG["promote_close_misses"] = True
        with patch.object(
            trainer,
            "_transcribe_capture",
            return_value="turn on the lights",
        ):
            trainer._auto_review_capture("wake.wav")

        self.assertTrue(audio_path.exists())
        self.assertFalse(list(trainer.PERSONAL_DIR.glob("*.wav")))
        self.assertFalse(list(trainer.NEGATIVE_DIR.glob("*.wav")))
        metadata = trainer._load_sidecar_json(audio_path)
        self.assertEqual(metadata["auto_review_status"], "close_miss_phrase_not_detected")

    def test_vad_blocked_close_miss_is_never_transcribed(self):
        audio_path = self.add_capture(event_type="close_miss", blocked_by_vad=True)
        trainer.AUTO_TRAIN_CONFIG["promote_close_misses"] = True
        with patch.object(trainer, "_transcribe_capture") as transcribe:
            trainer._auto_review_capture("wake.wav")

        transcribe.assert_not_called()
        self.assertTrue(audio_path.exists())
        self.assertFalse(trainer._load_sidecar_json(audio_path).get("auto_review_status"))

    def test_capture_for_another_wake_word_is_not_transcribed(self):
        audio_path = self.add_capture(wake_word="computer")
        with patch.object(trainer, "_transcribe_capture") as transcribe:
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

    def test_tater_notification_sets_new_word_globally_with_token(self):
        trainer.AUTO_TRAIN_CONFIG.update(
            {
                "notify_satellites": True,
                "tater_url": "http://127.0.0.1:8501",
                "tater_link_token": "secret-token",
            }
        )

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"push":{"count":4}}'

        trained_word = {
            "key": "hey_tater",
            "wake_word": "Hey Tater",
            "json_url": "http://10.4.20.210:8789/api/trained_wake_words/hey_tater.json",
        }
        with (
            patch.object(trainer, "_advertised_base_url", return_value="http://10.4.20.210:8789"),
            patch.object(trainer, "_list_trained_wake_words", return_value=[trained_word]) as catalog,
            patch.object(trainer, "urlopen", return_value=Response()) as open_url,
        ):
            result = trainer._notify_tater_satellites("hey_tater")

        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["wake_word"], "Hey Tater")
        self.assertEqual(result["wake_word_url"], trained_word["json_url"])
        catalog.assert_called_once_with("http://10.4.20.210:8789")
        self.assertEqual(open_url.call_count, 1)
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8501/api/tater/satellite/v1/trainer/wake-word")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("X-tater-trainer-token"), "secret-token")
        self.assertEqual(
            json.loads(request.data),
            {
                "wake_word_name": "hey_tater",
                "wake_word_url": trained_word["json_url"],
            },
        )

    def test_trained_word_catalog_keeps_url_alias_for_json_package(self):
        with tempfile.TemporaryDirectory() as directory:
            trained_dir = Path(directory)
            (trained_dir / "hey_tater.tflite").write_bytes(b"model")
            (trained_dir / "hey_tater.json").write_text(
                json.dumps({"wake_word": "hey tater", "model": "hey_tater.tflite"}),
                encoding="utf-8",
            )
            with (
                patch.object(trainer, "TRAINED_WAKE_WORDS_DIR", trained_dir),
                patch.object(trainer, "_sync_trained_wake_word_artifacts"),
            ):
                rows = trainer._list_trained_wake_words("http://10.4.20.210:8789")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["url"], rows[0]["json_url"])
        self.assertTrue(rows[0]["json_url"].endswith("/api/trained_wake_words/hey_tater.json"))
        self.assertTrue(
            rows[0]["esphome_json_url"].endswith(
                "/api/trained_wake_words/hey_tater.esphome.json"
            )
        )

    def test_esphome_manifest_route_removes_tater_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            trained_dir = Path(directory)
            (trained_dir / "hey_tater.tflite").write_bytes(b"model")
            metadata = {
                "type": "micro",
                "wake_word": "hey tater",
                "label": "Hey Tater",
                "author": "Tater Totterson",
                "website": "https://example.com",
                "model": "hey_tater.tflite",
                "trained_languages": ["en"],
                "version": 2,
                "model_format": "tflite_stream_state_internal_quant",
                "quantization": "int8",
                "sample_rate": 16000,
                "micro": {
                    "probability_cutoff": 0.97,
                    "sliding_window_size": 5,
                    "feature_step_size": 10,
                    "tensor_arena_size": 30000,
                    "minimum_esphome_version": "2024.7.0",
                },
                "tater_native": {"format_version": 1},
                "calibration": {"recall": 0.99},
            }
            (trained_dir / "hey_tater.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            with (
                patch.object(trainer, "TRAINED_WAKE_WORDS_DIR", trained_dir),
                patch.object(trainer, "_sync_trained_wake_word_artifacts"),
            ):
                response = trainer.trained_wake_word_artifact(
                    "hey_tater.esphome.json"
                )

        payload = json.loads(response.body)
        self.assertEqual(set(payload), set(trainer.ESPHOME_MANIFEST_KEYS))
        self.assertEqual(payload["micro"], metadata["micro"])
        self.assertNotIn("label", payload)
        self.assertNotIn("tater_native", payload)
        self.assertNotIn("calibration", payload)

    def test_tater_notification_fails_when_trained_word_is_missing(self):
        trainer.AUTO_TRAIN_CONFIG["tater_link_token"] = "secret-token"
        with (
            patch.object(trainer, "_advertised_base_url", return_value="http://10.4.20.210:8789"),
            patch.object(trainer, "_list_trained_wake_words", return_value=[]),
            patch.object(trainer, "urlopen") as open_url,
        ):
            result = trainer._notify_tater_satellites("missing_word")

        self.assertFalse(result["ok"])
        self.assertIn("missing_word", result["error"])
        open_url.assert_not_called()

    def test_tater_notification_requires_secure_link(self):
        trainer.AUTO_TRAIN_CONFIG["tater_link_token"] = ""
        with patch.object(trainer, "urlopen") as open_url:
            result = trainer._notify_tater_satellites("hey_tater")

        self.assertFalse(result["ok"])
        self.assertIn("not linked", result["error"])
        open_url.assert_not_called()

    def test_claim_tater_link_uses_tater_code_and_keeps_token_private(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, *_args):
                return json.dumps(
                    {
                        "ok": True,
                        "token": "a" * 43,
                        "tater_name": "Tater",
                        "linked_at": "2026-07-24T12:00:00+00:00",
                    }
                ).encode("utf-8")

        with (
            patch.object(trainer, "_advertised_base_url", return_value="http://10.4.20.210:8789"),
            patch.object(trainer, "urlopen", return_value=Response()) as open_url,
        ):
            result = trainer._claim_tater_link("http://127.0.0.1:8501", "ABCD-EFGH")

        self.assertTrue(result["linked"])
        self.assertEqual(trainer.AUTO_TRAIN_CONFIG["tater_link_token"], "a" * 43)
        self.assertNotIn("tater_link_token", trainer._public_auto_train_config())
        request = open_url.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8501/api/tater/satellite/v1/trainer/link/claim",
        )
        payload = json.loads(request.data)
        self.assertEqual(payload["pairing_code"], "ABCDEFGH")
        self.assertEqual(payload["publish_base_url"], "http://10.4.20.210:8789")
        self.assertTrue(payload["trainer_id"])

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

    def test_train_status_reads_and_increments_training_log_tail(self):
        log_path = Path(self.tempdir.name) / "training.log"
        log_path.write_text("first\nsecond\nthird\n", encoding="utf-8")
        with trainer.STATE_LOCK:
            original_training = dict(trainer.STATE["training"])
            trainer.STATE["training"].update(
                {
                    "log_path": str(log_path),
                    "last_sent_tail": [],
                    "last_log_size": 0,
                }
            )

        try:
            with (
                patch.object(trainer, "TRAIN_LOG_TAIL_LINES", 2),
                patch.object(trainer, "TRAIN_LOG_MAX_BYTES", 1024),
            ):
                first_status = trainer.train_status()
                self.assertEqual(first_status["training"]["log_lines"], ["second", "third"])
                self.assertEqual(first_status["training"]["log_text"], "second\nthird")

                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write("fourth\n")

                next_status = trainer.train_status()
                self.assertEqual(next_status["training"]["log_lines"], ["third", "fourth"])
                self.assertEqual(next_status["training"]["log_text"], "fourth")
        finally:
            with trainer.STATE_LOCK:
                trainer.STATE["training"].clear()
                trainer.STATE["training"].update(original_training)


if __name__ == "__main__":
    unittest.main()
