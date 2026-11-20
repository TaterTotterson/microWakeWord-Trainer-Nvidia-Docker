import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, writeFile, rm } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import "vue";
import { harness, flush } from "./testSupport.mjs";

async function setup(t) {
  const env = harness();
  t.mock.method(Date, "now", env.now);
  const initial = { running: false, exit_code: null, log_lines: [] };
  const backend = {
    session: { raw_phrase: "hey tater", safe_word: "hey_tater", language: "en", english_accent: "mixed", tts_mode: "hybrid", training: initial },
    training: initial,
    samples: { personal: [], negative: [], personal_count: 0, negative_count: 0 },
    captures: { items: [{ saved_as: "clip.wav", capture_scope: "hey_tater", review_wake_word: "hey_tater" }], captured_count: 1 },
    auto: { config: { enabled: false, wake_phrase: "hey tater", schedule_hours: 24 }, state: {} },
    words: [{ safe_word: "hey_tater", phrase: "hey tater", run_count: 1, published_run_id: "old", latest_run: { run_id: "old" }, dataset_size_bytes: 123 }],
    runs: [{ run_id: "old", safe_word: "hey_tater", is_current: true, completed_at: "2026-11-18" }],
    catalog: [],
  };
  const calls = [];
  env.host.confirm = () => true;
  globalThis.window = env.host;
  globalThis.document = env.page;
  t.mock.method(globalThis, "fetch", async (path, options) => {
    calls.push({ path, options });
    let payload = {};
    if (path === "/api/session") payload = backend.session;
    else if (path === "/api/train_status") payload = { training: backend.training };
    else if (path === "/api/samples") payload = backend.samples;
    else if (path === "/api/captured_audio") payload = backend.captures;
    else if (path === "/api/auto_train") payload = backend.auto;
    else if (path.startsWith("/api/wake_word_library?")) payload = { words: backend.words.map(({ dataset_size_bytes, ...word }) => word) };
    else if (path === "/api/wake_word_library") payload = { words: backend.words };
    else if (path.endsWith("/history")) payload = { runs: backend.runs };
    else if (path === "/api/trained_wake_words/catalog") payload = { wake_words: backend.catalog };
    else throw Error(`Unexpected API request: ${path}`);
    return new Response(JSON.stringify(payload), { headers: { "Content-Type": "application/json" } });
  });
  const directory = await mkdtemp(fileURLToPath(new URL(".compiled-", import.meta.url)));
  for (const name of ["api", "liveUpdates", "trainerStore"]) {
    const input = await readFile(new URL(`../src/${name}.ts`, import.meta.url), "utf8");
    const output = ts.transpileModule(input, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
    await writeFile(`${directory}/${name}.js`, output.replace(/from "(\.\/[^".]+)"/g, 'from "$1.js"'));
  }
  const store = await import(pathToFileURL(`${directory}/trainerStore.js`).href);
  t.after(async () => { store.disposeTrainer(); await rm(directory, { recursive: true, force: true }); delete globalThis.window; delete globalThis.document; });
  await store.initializeTrainer(); await flush();
  const source = env.streams[0]; source.open(); await flush();
  return { ...env, backend, calls, store, source };
}

test("satellite captures, recordings, saved versions and external training update without navigation", async t => {
  const { backend, store, source, calls } = await setup(t);
  backend.training = { running: true, exit_code: null, log_lines: ["Training from another tab"], safe_word: "hey_tater", progress: { stage: "features", label: "Features", percent: 35 } };
  backend.samples.personal_count = 2;
  backend.captures.items.push({ saved_as: "sat.wav", capture_scope: "", review_wake_word: "" }); backend.captures.captured_count = 2;
  backend.words[0].run_count = 2; backend.words[0].published_run_id = "new"; backend.words[0].latest_run = { run_id: "new" };
  backend.runs.unshift({ run_id: "new", safe_word: "hey_tater", is_current: true, completed_at: "2026-11-19" });
  source.event("trainer", { training: backend.training, revision: "server:1" }); await flush();
  assert.equal(store.trainer.activeView, "home");
  assert.equal(store.trainer.training.progress.percent, 35);
  assert.equal(store.trainer.captured.captured_count, 2);
  assert.equal(store.personalCount.value, 2);
  assert.equal(store.trainer.modelHistory.length, 2);
  assert.equal(store.trainer.wakeWordLibrary[0].dataset_size_bytes, 123, "lightweight refresh retains measured storage");
  assert.ok(calls.some(call => call.path === "/api/wake_word_library?include_usage=false"));
  assert.ok(calls.every(call => call.options.cache === "no-store"));
});

test("live refresh preserves setup drafts, auto settings and clip assignments", async t => {
  const { store, source, backend } = await setup(t);
  store.trainer.phrase = "a new draft"; store.trainer.language = "de";
  store.trainer.autoForm.schedule_hours = 72;
  store.trainer.captured.items[0].review_wake_word = "another_word";
  backend.auto.config.schedule_hours = 48; backend.auto.config.enabled = true;
  backend.session.raw_phrase = "changed elsewhere";
  source.event("trainer", { training: backend.training, revision: "server:2" }); await flush();
  assert.equal(store.trainer.phrase, "a new draft"); assert.equal(store.trainer.language, "de");
  assert.equal(store.trainer.session.raw_phrase, "changed elsewhere");
  assert.equal(store.trainer.autoForm.schedule_hours, 72);
  assert.equal(store.trainer.autoForm.enabled, true, "untouched settings follow the server");
  assert.equal(store.trainer.captured.items[0].review_wake_word, "another_word");
  source.event("trainer", { training: backend.training, revision: "server:3" }); await flush();
  assert.equal(store.trainer.captured.items[0].review_wake_word, "another_word", "assignment remains a draft across repeated refreshes");
});

test("completion refreshes results once and stale session status cannot undo streamed progress", async t => {
  const { store, source, backend, calls } = await setup(t);
  source.event("trainer", { training: { running: true, exit_code: null, log_lines: ["working"] }, revision: "server:1" }); await flush();
  assert.equal(store.trainer.training.running, true, "the older /api/session training field is ignored during live sync");
  const historyCalls = calls.filter(call => call.path.endsWith("/history")).length;
  source.event("trainer", { training: { running: false, exit_code: 0, log_lines: ["done"] }, revision: "server:1" }); await flush();
  assert.match(store.trainer.toast.message, /finished successfully/);
  const serial = store.trainer.toast.serial;
  source.event("trainer", { training: { running: false, exit_code: 0, log_lines: ["done"] }, revision: "server:1" }); await flush();
  assert.equal(store.trainer.toast.serial, serial);
  assert.equal(calls.filter(call => call.path.endsWith("/history")).length, historyCalls, "unchanged histories are reused");
});

test("polling continues detecting new jobs and inbox data when SSE is unavailable", async t => {
  const env = await setup(t);
  env.source.fail(); await flush();
  env.backend.training = { running: true, exit_code: null, log_lines: ["scheduled job"] };
  env.backend.captures.captured_count = 3;
  await env.advance(6500);
  assert.equal(env.store.trainer.training.running, true);
  assert.equal(env.store.trainer.captured.captured_count, 3);
  assert.equal(env.store.trainer.liveState, "reconnecting");
});
