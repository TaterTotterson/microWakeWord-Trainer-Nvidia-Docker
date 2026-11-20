import json
import os
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.responses import JSONResponse
import trainer_server as trainer
from wake_word_library import activate_word, list_words


class LiveUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_sends_initial_state_training_changes_and_revisions(self):
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        with patch.dict(trainer.STATE, {"training": {"running": False, "log_lines": [], "exit_code": None}}), patch.object(trainer.asyncio, "sleep", new=AsyncMock()):
            stream = trainer._stream_live_ui(request)
            first = await anext(stream)
            initial = json.loads(first.split("data: ", 1)[1])
            self.assertIn("event: trainer\n", first)
            with trainer.STATE_LOCK:
                trainer.STATE["training"]["running"] = True
                trainer.STATE["training"]["log_lines"].append("working")
            changed = json.loads((await anext(stream)).split("data: ", 1)[1])
            self.assertTrue(changed["training"]["running"])
            self.assertEqual(changed["training"]["log_lines"], ["working"])
            self.assertEqual(initial["training"]["log_lines"], [])
            trainer._invalidate_live_ui()
            invalidated = json.loads((await anext(stream)).split("data: ", 1)[1])
            self.assertNotEqual(initial["revision"], invalidated["revision"])
            await stream.aclose()

    async def test_idle_heartbeat_and_disconnection(self):
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=False))
        with patch.object(trainer.asyncio, "sleep", new=AsyncMock()):
            stream = trainer._stream_live_ui(request)
            await anext(stream)
            self.assertEqual(await anext(stream), "event: heartbeat\ndata: {}\n\n")
            request.is_disconnected.return_value = True
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)

    async def test_stream_headers_disable_proxy_buffering_and_caching(self):
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        response = await trainer.live_ui_events(request)
        self.assertEqual(response.media_type, "text/event-stream")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        self.assertEqual(response.headers["cache-control"], "no-store")

    async def test_successful_api_mutations_invalidate_all_tabs(self):
        for method, status, changed in [("POST", 200, True), ("PUT", 200, True), ("DELETE", 200, True), ("GET", 200, False), ("POST", 400, False)]:
            with self.subTest(method=method, status=status):
                revision = trainer._UI_LIVE_REVISION
                response = JSONResponse({"ok": status == 200}, status_code=status)
                request = SimpleNamespace(url=SimpleNamespace(path="/api/captured_audio"), method=method)
                result = await trainer.live_ui_response(request, AsyncMock(return_value=response))
                self.assertEqual(trainer._UI_LIVE_REVISION > revision, changed)
                self.assertEqual(result.headers["cache-control"], "no-store")

    async def test_versioned_downloads_keep_immutable_cache_headers(self):
        response = JSONResponse({}, headers={"Cache-Control": "public, max-age=31536000, immutable"})
        request = SimpleNamespace(url=SimpleNamespace(path="/api/wake_word_library/word/runs/run/bundle"), method="GET")
        result = await trainer.live_ui_response(request, AsyncMock(return_value=response))
        self.assertIn("immutable", result.headers["cache-control"])

    async def test_lightweight_library_does_not_walk_feature_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            activate_word(root, safe_word="hey_tater", phrase="hey tater", language="en", english_accent="mixed", tts_mode="hybrid", workspace_paths=())
            with patch("wake_word_library._usage", side_effect=AssertionError("storage scan")):
                words = list_words(root, trained_dir=root / "published", workspace_paths=("generated_features",), include_usage=False)
            self.assertEqual(words[0]["safe_word"], "hey_tater")
            self.assertNotIn("dataset_size_bytes", words[0])
            with patch.object(trainer, "list_words", return_value=[]) as catalog:
                trainer.wake_word_library_catalog(include_usage=False)
                self.assertFalse(catalog.call_args.kwargs["include_usage"])


class LiveUiShutdownTests(unittest.TestCase):
    def test_open_event_stream_does_not_hold_server_shutdown_open(self):
        # Match the launchers' grace period while leaving an EventSource connected.
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.close()
            source = """
import trainer_server as trainer
import uvicorn
trainer.app.router.on_startup.clear()
trainer.app.router.on_shutdown.clear()
uvicorn.run(trainer.app, host="127.0.0.1", port=PORT, timeout_graceful_shutdown=2, log_level="critical")
""".replace("PORT", str(port))
            environment = {**os.environ, "DATA_DIR": directory, "WAKEWORD_TRAINER_DATA_DIR": directory, "WAKEWORD_TRAINER_SUPPORT_DIR": directory}
            process = subprocess.Popen([sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1], env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stream = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        stream = urlopen(f"http://127.0.0.1:{port}/api/events", timeout=1)
                        break
                    except OSError:
                        if process.poll() is not None:
                            self.fail("Preview server exited before accepting connections")
                        time.sleep(0.05)
                self.assertIsNotNone(stream, "Preview server did not start")
                self.assertEqual(stream.readline(), b"event: trainer\n")
                process.terminate()
                process.wait(timeout=6)
                self.assertIsNotNone(process.returncode)
            finally:
                if stream is not None:
                    stream.close()
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
