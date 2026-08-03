from __future__ import annotations

import json
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class VueTrainerUiTests(unittest.TestCase):
    def test_frontend_uses_typed_vue_and_vite(self) -> None:
        package = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(package["dependencies"]["vue"], "3.5.40")
        self.assertIn("vue-tsc --noEmit", package["scripts"]["build"])
        self.assertIn("vite build", package["scripts"]["build"])

        config = (REPO_ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
        self.assertIn('"../static/ui"', config)
        self.assertIn('fileName: () => "trainer-ui.js"', config)

    def test_static_shell_loads_prebuilt_bundle(self) -> None:
        index = (REPO_ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="trainer-app"', index)
        self.assertIn('/static/ui/trainer-ui.css', index)
        self.assertIn('/static/ui/trainer-ui.js', index)
        self.assertNotIn("fonts.googleapis.com", index)

        self.assertGreater((REPO_ROOT / "static" / "ui" / "trainer-ui.js").stat().st_size, 100_000)
        self.assertGreater((REPO_ROOT / "static" / "ui" / "trainer-ui.css").stat().st_size, 10_000)

    def test_theme_uses_tater_orange_and_neutral_greys(self) -> None:
        styles = (REPO_ROOT / "frontend" / "src" / "trainer.css").read_text(encoding="utf-8")
        self.assertIn("--orange: #ff9134", styles)
        self.assertIn("--surface: rgba(29, 29, 31, .9)", styles)
        for old_blue in ("#070b15", "#11192b", "#5db6ff", "#8d75ff", "#7fc7ff", "#78caff"):
            self.assertNotIn(old_blue, styles)

    def test_reactive_ui_keeps_trainer_workflows(self) -> None:
        app = (REPO_ROOT / "frontend" / "src" / "TrainerApp.vue").read_text(encoding="utf-8")
        store = (REPO_ROOT / "frontend" / "src" / "trainerStore.ts").read_text(encoding="utf-8")
        trim = (REPO_ROOT / "frontend" / "src" / "components" / "AudioTrimModal.vue").read_text(encoding="utf-8")

        for workflow in (
            "startSession",
            "stopSession",
            "startTraining",
            "saveAuto",
            "runAutoAction",
            "reviewCaptured",
            "uploadSelectedFiles",
            "copyWakeWord",
            "deleteManagedData",
        ):
            self.assertIn(workflow, app)

        for endpoint in (
            "/api/start_session",
            "/api/stop_session",
            "/api/upload_personal_sample",
            "/api/captured_audio",
            "/api/auto_train",
            "/api/train_status",
            "/api/trained_wake_words/catalog",
            "/api/data",
        ):
            self.assertIn(endpoint, store)

        self.assertIn("OfflineAudioContext", trim)
        self.assertIn("/api/samples/trim", trim)
        self.assertIn(':disabled="Boolean(trainer.session.safe_word)', app)
        self.assertIn('{ id: "data", label: "Data"', app)

    def test_training_console_pauses_follow_mode_when_scrolled_up(self) -> None:
        app = (REPO_ROOT / "frontend" / "src" / "TrainerApp.vue").read_text(encoding="utf-8")

        self.assertIn("const consoleFollowing = ref(true)", app)
        self.assertIn("distanceFromBottom <= 32", app)
        self.assertIn('if (!consoleFollowing.value) return', app)
        self.assertIn('@scroll.passive="onConsoleScroll"', app)
        self.assertIn("Jump to latest", app)

    def test_wake_word_card_uses_explicit_json_catalog_url(self) -> None:
        app = (REPO_ROOT / "frontend" / "src" / "TrainerApp.vue").read_text(encoding="utf-8")
        types = (REPO_ROOT / "frontend" / "src" / "types.ts").read_text(encoding="utf-8")

        self.assertIn("item.json_url || item.url || item.jsonUrl", app)
        self.assertIn("copyWakeWord(wordJsonUrl(word))", app)
        self.assertNotIn("copyWakeWord(word.url)", app)
        self.assertIn("json_url?: string", types)

    def test_runtime_packaging_uses_bundle_without_node(self) -> None:
        dockerfiles = [REPO_ROOT / "dockerfile", REPO_ROOT / "dockerfile.blackwell"]
        for dockerfile in dockerfiles:
            if not dockerfile.exists():
                continue
            source = dockerfile.read_text(encoding="utf-8")
            self.assertIn("COPY --chown=root:root static/ /root/mww-scripts/static/", source)
            self.assertNotIn("npm install", source)

        macos_builder = REPO_ROOT / "macos" / "WakeWordTrainer" / "scripts" / "build_app.sh"
        if macos_builder.exists():
            source = macos_builder.read_text(encoding="utf-8")
            self.assertIn("--exclude='frontend/node_modules/'", source)
            self.assertNotIn("--exclude='static/'", source)


if __name__ == "__main__":
    unittest.main()
