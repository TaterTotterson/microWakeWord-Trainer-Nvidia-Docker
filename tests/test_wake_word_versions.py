import hashlib
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from fastapi.testclient import TestClient

import trainer_server as trainer
from wake_bundle import publish_micro_wake_word
from wake_word_library import record_run
from wake_word_versions import artifact_path, publish_version, published_run_id, validate_version


class WakeWordVersionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Path(self.temporary.name)
        self.trained = self.data / "trained_wake_words"
        self.trained.mkdir()
        self.patches = [
            patch.object(trainer, "DATA_DIR", self.data),
            patch.object(trainer, "TRAINED_WAKE_WORDS_DIR", self.trained),
            patch.object(trainer, "_advertised_base_url", lambda request=None: "http://testserver"),
            patch.object(trainer, "_sync_trained_wake_word_artifacts", lambda: None),
            patch.dict(trainer.STATE, {"training": {"running": False}}),
        ]
        if hasattr(trainer, "TRAINING_LOCK_FILE"):
            self.patches.append(patch.object(trainer, "TRAINING_LOCK_FILE", self.data / "training.lock"))
        for item in self.patches:
            item.start()
        self.client = TestClient(trainer.app)
        self.first = self.make_version(b"first-model", "2026-11-10T10:00:00+00:00")
        self.second = self.make_version(b"second-model", "2026-11-11T10:00:00+00:00")

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def make_version(self, binary, completed_at, code=0):
        if code == 0:
            model = self.trained / "hey_tater.tflite"
            model.write_bytes(binary)
            publish_micro_wake_word(
                trained_dir=self.trained,
                safe_word="hey_tater",
                phrase="hey tater",
                metadata={"type": "micro", "wake_word": "hey tater", "model": model.name, "micro": {"probability_cutoff": .9}},
                model_path=model,
            )
        return record_run(
            self.data,
            trained_dir=self.trained,
            summary={"safe_word": "hey_tater", "wake_word": "hey tater", "completed_at": completed_at, "overall": "Good" if code == 0 else "Needs attention", "models": []},
            selected_models=["micro"], outcomes={"micro": code},
        )

    def history(self):
        response = self.client.get("/api/wake_word_library/hey_tater/history")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["runs"]

    def test_history_exposes_immutable_bundle_and_esphome_urls(self):
        rows = self.history()
        self.assertEqual([row["version"] for row in rows], [2, 1])
        self.assertTrue(rows[0]["is_current"])
        self.assertFalse(rows[1]["is_current"])
        for field in ("json_url", "esphome_json_url"):
            response = self.client.get(rows[1][field])
            self.assertEqual(response.status_code, 200, response.text)
            document = response.json()
            micro = document["micro_wake_word"] if field == "json_url" else document
            self.assertIn(self.first["run_id"], micro["model"])
            self.assertEqual(self.client.get(micro["model"]).content, b"first-model")
            self.assertIn("immutable", response.headers["cache-control"])

    def test_selecting_an_older_version_preserves_datasets_and_other_snapshots(self):
        recordings = self.data / "personal_samples"
        recordings.mkdir()
        sentinel = recordings / "sample.wav"
        sentinel.write_bytes(b"personal-recording")
        response = self.client.post(f'/api/wake_word_library/hey_tater/runs/{self.first["run_id"]}/publish')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((self.trained / "hey_tater.tflite").read_bytes(), b"first-model")
        self.assertEqual(sentinel.read_bytes(), b"personal-recording")
        self.assertEqual(artifact_path(self.data, "hey_tater", self.second["run_id"], "hey_tater.tflite").read_bytes(), b"second-model")
        self.assertEqual(published_run_id(self.data, self.trained, "hey_tater"), self.first["run_id"])
        self.assertEqual(len(self.history()), 2)
        stable = self.client.get("/api/trained_wake_words/hey_tater.esphome.json")
        self.assertEqual(self.client.get(stable.json()["model"]).content, b"first-model")

    def test_current_json_stays_on_committed_version_during_a_new_run(self):
        model = self.trained / "hey_tater.tflite"
        model.write_bytes(b"not-yet-committed")
        publish_micro_wake_word(trained_dir=self.trained, safe_word="hey_tater", phrase="hey tater", metadata={"type": "micro", "model": model.name}, model_path=model)
        trainer.STATE["training"]["running"] = True
        stable = self.client.get("/api/trained_wake_words/hey_tater.wake-bundle.json")
        self.assertEqual(stable.status_code, 200)
        url = stable.json()["micro_wake_word"]["model"]
        self.assertIn(self.second["run_id"], url)
        self.assertEqual(self.client.get(url).content, b"second-model")
        rows = self.history()
        self.assertTrue(rows[0]["is_current"])
        library = self.client.get("/api/wake_word_library")
        self.assertEqual(library.status_code, 200, library.text)
        self.assertEqual(library.json()["words"][0]["published_run_id"], self.second["run_id"])
        catalog = self.client.get("/api/trained_wake_words/catalog").json()["wake_words"][0]
        self.assertIn(self.second["run_id"], catalog["model_url"])

    def test_direct_cli_publication_still_updates_current_links(self):
        model = self.trained / "hey_tater.tflite"
        model.write_bytes(b"direct-cli-publication")
        publish_micro_wake_word(trained_dir=self.trained, safe_word="hey_tater", phrase="hey tater", metadata={"type": "micro", "model": model.name}, model_path=model)
        self.assertEqual(published_run_id(self.data, self.trained, "hey_tater"), "")
        self.assertFalse(any(row["is_current"] for row in self.history()))
        catalog = self.client.get("/api/trained_wake_words/catalog").json()["wake_words"][0]
        self.assertEqual(self.client.get(catalog["model_url"]).content, b"direct-cli-publication")
        manifest = self.client.get("/api/trained_wake_words/hey_tater.esphome.json").json()
        self.assertEqual(manifest["model"], "hey_tater.tflite")

    def test_auto_training_commits_snapshot_before_notifying_tater(self):
        class CompletedProcess:
            pid = 6543
            returncode = 0
            stdout = io.StringIO("training complete\n")
            def wait(self, timeout=None):
                return 0
        notified = []
        def notify(safe_word):
            rows = self.history()
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[0]["is_current"])
            self.assertNotEqual(rows[0]["run_id"], self.second["run_id"])
            notified.append(safe_word)
            return {"ok": True, "count": 0}
        options = {"auto_run": True, "train_microwakeword": False, "train_boundary_only": True}
        if "allow_no_personal" in inspect.signature(trainer._run_training_background).parameters:
            options["allow_no_personal"] = True
        with (
            patch.object(trainer.subprocess, "Popen", return_value=CompletedProcess()),
            patch.object(trainer, "_ensure_training_venv", return_value=None, create=True),
            patch.object(trainer, "_ensure_training_datasets", return_value=None, create=True),
            patch.object(trainer, "_notify_tater_satellites", side_effect=notify),
            patch.object(trainer, "_save_auto_train_state_locked"),
            patch.object(trainer, "TRAINING_PROCESS", None),
            patch.object(trainer, "TRAINING_THREAD", None),
            patch.dict(trainer.STATE, {"raw_phrase": "hey tater"}),
            patch.dict(trainer.AUTO_TRAIN_STATE),
            patch.dict(trainer.AUTO_TRAIN_RUNTIME),
        ):
            trainer._run_training_background("hey_tater", "en", **options)
        self.assertEqual(notified, ["hey_tater"])

    def test_failed_run_does_not_replace_current_or_offer_a_publish_action(self):
        failed = self.make_version(b"unused", "2026-11-12T10:00:00+00:00", code=1)
        rows = self.history()
        self.assertEqual(rows[0]["run_id"], failed["run_id"])
        self.assertFalse(rows[0]["restorable"])
        self.assertEqual(rows[0]["json_url"], "")
        self.assertTrue(rows[1]["is_current"])
        response = self.client.post(f'/api/wake_word_library/hey_tater/runs/{failed["run_id"]}/publish')
        self.assertEqual(response.status_code, 400)
        self.assertEqual((self.trained / "hey_tater.tflite").read_bytes(), b"second-model")

    def test_missing_or_corrupted_models_are_rejected_before_any_file_changes(self):
        snapshot = artifact_path(self.data, "hey_tater", self.first["run_id"], "hey_tater.tflite")
        snapshot.write_bytes(b"corrupted")
        response = self.client.post(f'/api/wake_word_library/hey_tater/runs/{self.first["run_id"]}/publish')
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("checksum", response.json()["error"])
        self.assertEqual((self.trained / "hey_tater.tflite").read_bytes(), b"second-model")
        snapshot.unlink()
        with self.assertRaises(FileNotFoundError):
            validate_version(self.data, "hey_tater", self.first["run_id"])

    def test_failed_publication_rolls_back_every_replaced_file(self):
        original_replace = os.replace
        failed = False
        def fail_manifest_once(source, target):
            nonlocal failed
            if not failed and Path(source).parent.name == "stage" and str(target).endswith(".wake-bundle.json"):
                failed = True
                raise OSError("simulated disk failure")
            return original_replace(source, target)
        before = {path.name: path.read_bytes() for path in self.trained.iterdir() if path.is_file()}
        with patch("wake_word_versions.os.replace", side_effect=fail_manifest_once):
            with self.assertRaises(OSError):
                publish_version(self.data, self.trained, "hey_tater", self.first["run_id"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.trained.iterdir() if path.is_file()})
        self.assertEqual(published_run_id(self.data, self.trained, "hey_tater"), self.second["run_id"])

    def test_publish_is_blocked_during_training(self):
        trainer.STATE["training"]["running"] = True
        response = self.client.post(f'/api/wake_word_library/hey_tater/runs/{self.first["run_id"]}/publish')
        self.assertEqual(response.status_code, 409)
        self.assertEqual((self.trained / "hey_tater.tflite").read_bytes(), b"second-model")

    def test_path_traversal_and_symlink_escapes_are_rejected(self):
        for word, run_id, filename in (("../hey_tater", self.first["run_id"], "hey_tater.tflite"), ("hey_tater", "..", "hey_tater.tflite"), ("hey_tater", self.first["run_id"], "../private.json")):
            with self.assertRaises(ValueError):
                artifact_path(self.data, word, run_id, filename)
        snapshot = artifact_path(self.data, "hey_tater", self.first["run_id"], "hey_tater.tflite")
        snapshot.unlink()
        snapshot.symlink_to(self.trained / "hey_tater.tflite")
        with self.assertRaises(ValueError):
            artifact_path(self.data, "hey_tater", self.first["run_id"], "hey_tater.tflite")


if __name__ == "__main__":
    unittest.main()
