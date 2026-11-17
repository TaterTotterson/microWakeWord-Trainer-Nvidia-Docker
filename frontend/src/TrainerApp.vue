<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import AudioTrimModal from "./components/AudioTrimModal.vue";
import TrainerWelcome from "./components/TrainerWelcome.vue";
import WordWorkspace from "./components/WordWorkspace.vue";
import UiIcon from "./components/UiIcon.vue";
import {
  autoLinked, captureTone, claimTater, clearSamples, deleteManagedData, describeFormat,
  disposeTrainer, ensureSupportedTtsMode, formatBytes, formatTimestamp, hasConsole, initializeTrainer,
  isBusy, itemAudioUrl, macDataFolderPickerAvailable, negativeCount, notify, personalCount, previewPhrase, refreshAuto,
  refreshCaptured, refreshManagedData, refreshSamples, refreshWakeWordLibrary, refreshWakeWords, removeSample, revertSample, reviewCaptured,
  retryOpenWakeWord, revalidateNanoWakeWord, runAutoAction, saveAuto, selectFiles, selectedSamples, startBoundaryTraining, startSession, startTraining, stopSession, sttEngines,
  trainer, ttsRoute, unlinkTater, uploadSelectedFiles, chooseDataDirectory, useSavedWord,
} from "./trainerStore";
import type { AudioItem, ManagedDataItem, SampleBucket, ViewName, WakeWordLibraryItem } from "./types";

const uploadInput = ref<HTMLInputElement | null>(null);
const consoleLog = ref<HTMLElement | null>(null);
const consoleFollowing = ref(true);
const linkUrl = ref("");
const linkCode = ref("");
const linkComplete = ref(false);
const mascotUrl = "/static/images/tater-wake-word-trainer.png";
const trainerVersion = `v${__TRAINER_VERSION__}`;
const pageSize = 50;
const isApple = __TRAINER_PLATFORM__ === "apple";
const tabs: Array<{ id: ViewName; label: string; icon: string }> = [
  { id: "home", label: "Start", icon: "home" },
  { id: "firmware", label: "Wake words", icon: "wave" },
  { id: "captured", label: "Captures", icon: "inbox" },
  { id: "auto", label: "Auto training", icon: "refresh" },
];
const selectedWord = computed(() => trainer.wakeWordLibrary.find(word => word.safe_word === trainer.selectedWord));
const selectedModelNames = computed(() => [
  ...(trainer.trainMicroWakeWord ? ["MicroWakeWord"] : []),
  ...(trainer.trainOpenWakeWord ? ["openWakeWord"] : []),
  ...(trainer.trainNanoWakeWordCnn ? ["NanoWakeWord CNN"] : []),
  ...(trainer.trainNanoWakeWordGru ? ["NanoWakeWord GRU"] : []),
]);
const currentTrainingWord = computed(() => {
  const safe = String(trainer.training.safe_word || trainer.training.summary?.safe_word || "");
  const saved = trainer.wakeWordLibrary.find(word => word.safe_word === safe)?.phrase;
  return (trainer.training.running ? saved || trainer.session.raw_phrase : trainer.training.summary?.wake_word || saved)
    || safe || trainer.session.raw_phrase || "Wake word";
});


const pagedSamples = computed(() => {
  const page = trainer.samplePage[trainer.sampleBucket];
  return selectedSamples.value.slice(page * pageSize, (page + 1) * pageSize);
});
const samplePages = computed(() => Math.max(1, Math.ceil(selectedSamples.value.length / pageSize)));
const autoState = computed(() => trainer.auto.state || {});
const autoRuntime = computed(() => trainer.auto.runtime || {});
const autoAudit = computed(() => {
  const state = autoState.value;
  const rows: string[] = [];
  if (state.last_review_result) rows.push(`Last review: ${String(state.last_review_result).replaceAll("_", " ")}`);
  if (state.last_review_file) rows.push(String(state.last_review_file));
  if (state.last_review_transcript) rows.push(`STT: “${state.last_review_transcript}”`);
  if (state.last_review_error) rows.push(`Error: ${state.last_review_error}`);
  if (state.last_stt_engine) rows.push(`STT engine: ${String(state.last_stt_engine).replaceAll("_", " ")}`);
  if (state.last_notify_at) rows.push(state.last_notify_error ? `Publish failed: ${state.last_notify_error}` : `Wake word published ${formatTimestamp(state.last_notify_at)}`);
  return rows.join(" · ") || "No automatic review has run yet.";
});
const trainingStatus = computed(() => {
  if (trainer.training.running) return { text: "Training running", tone: "warning" };
  if (trainer.training.exit_code === 0) return { text: "Training finished", tone: "success" };
  if (trainer.training.exit_code !== null) return { text: `Exit ${trainer.training.exit_code}`, tone: "error" };
  return { text: "Not started", tone: "neutral" };
});
const autoStatus = computed(() => {
  if (autoRuntime.value.review_running) return { text: `Transcribing ${autoRuntime.value.review_file || "wake"}`, tone: "warning" };
  if (trainer.training.running && trainer.auto.config?.enabled) return { text: "Training running", tone: "warning" };
  if (trainer.auto.config?.enabled) return { text: "Enabled", tone: "success" };
  return { text: "Disabled", tone: "neutral" };
});
const consoleLines = computed(() => trainer.training.log_lines?.length ? trainer.training.log_lines : ["No training output yet."]);
const trainingProgress = computed(() => trainer.training.progress || null);
const trainingProgressStage = computed(() => {
  const stage = String(trainingProgress.value?.stage || "pipeline").split(":", 1)[0];
  return ({
    starting: "Getting ready", datasets: "Training data", tts: "Speech generation",
    features: "Feature building", model: "Model training", boundary: "Wake boundary",
    calibration: "Calibration", package: "Packaging", openwakeword: "openWakeWord",
    nanowakeword: "NanoWakeWord", complete: "Finished", failed: "Needs attention",
    worker: "Pipeline progress", pipeline: "Live pipeline",
  } as Record<string, string>)[stage] || stage.replaceAll("-", " ");
});
const trainingProgressPercent = computed(() => {
  const raw = trainingProgress.value?.percent;
  if (raw === null || raw === undefined) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
});
const trainingProgressFacts = computed(() => {
  const progress = trainingProgress.value;
  if (!progress) return [] as string[];
  const facts: string[] = [];
  const current = Number(progress.current);
  const total = Number(progress.total);
  const unit = String(progress.unit || "items");
  if (Number.isFinite(current) && Number.isFinite(total) && total > 0) {
    facts.push(`${Math.round(current).toLocaleString()} / ${Math.round(total).toLocaleString()} ${unit}`);
  } else if (Number.isFinite(current) && current > 0) {
    facts.push(`${Math.round(current).toLocaleString()} ${unit}`);
  }
  if (progress.elapsed) facts.push(`Elapsed ${progress.elapsed}`);
  if (progress.eta) facts.push(`About ${progress.eta} left`);
  if (progress.rate_text) facts.push(String(progress.rate_text));
  else if (Number(progress.rate) > 0) facts.push(`${Number(progress.rate).toLocaleString()} ${unit}/sec`);
  return facts;
});
const retryableOpenWakeWord = computed(() => {
  const active = String(trainer.session.safe_word || "");
  const unfinished = trainer.wakeWords.filter((word) => !word.dual_model);
  return unfinished.find((word) => String(word.key || "") === active) || unfinished[0];
});
const retryableNanoWakeWord = computed(() => {
  const models = Array.isArray(trainer.training.summary?.models) ? trainer.training.summary.models : [];
  return models.find((model) => model.failure_kind === "quality_rejected" && String(model.id || "").startsWith("nano_"));
});
const activeLibraryWord = computed(() => trainer.wakeWordLibrary.find((word) => word.safe_word === trainer.session.safe_word));
const dataCategories = computed(() => {
  const groups = new Map<string, ManagedDataItem[]>();
  for (const item of trainer.managedData.items || []) {
    const rows = groups.get(item.category) || [];
    rows.push(item);
    groups.set(item.category, rows);
  }
  return Array.from(groups, ([name, items]) => ({ name, items }));
});
const manualNanoSelected = computed(() => trainer.trainNanoWakeWordCnn || trainer.trainNanoWakeWordGru);
const autoNanoSelected = computed(() => (
  trainer.autoForm.train_nanowakeword_cnn || trainer.autoForm.train_nanowakeword_gru
));
const manualSelectedEngines = computed(() => [
  ...(trainer.trainMicroWakeWord ? ["MWW"] : []),
  ...(trainer.trainOpenWakeWord ? ["OWW"] : []),
  ...(manualNanoSelected.value ? ["NWW"] : []),
]);
const autoSelectedEngines = computed(() => [
  ...(trainer.autoForm.train_microwakeword ? ["MWW"] : []),
  ...(trainer.autoForm.train_openwakeword ? ["OWW"] : []),
  ...(autoNanoSelected.value ? ["NWW"] : []),
]);

watch([() => trainer.language, () => trainer.ttsMode], ensureSupportedTtsMode);
watch(() => trainer.toast.serial, () => window.setTimeout(() => { trainer.toast.message = ""; }, 4500));
watch(consoleLines, async () => {
  if (!consoleFollowing.value) return;
  await nextTick();
  if (consoleFollowing.value && consoleLog.value) {
    consoleLog.value.scrollTop = consoleLog.value.scrollHeight;
  }
});
watch(() => trainer.activeView, async (view) => {
  await nextTick();
  window.scrollTo(0, 0);
  if (view === "training") scrollConsoleToBottom();
});
watch(() => trainer.consoleOpen, (open) => {
  if (open) { trainer.activeView = "training"; trainer.consoleOpen = false; }
});

onMounted(() => {
  void initializeTrainer();
  document.addEventListener("keydown", onKeydown);
});
onBeforeUnmount(() => {
  disposeTrainer();
  document.removeEventListener("keydown", onKeydown);
});

function onKeydown(event: KeyboardEvent): void {
  if (event.key !== "Escape") return;
  trainer.taterLinkOpen = false;
  trainer.trimItem = null;
}
function onConsoleScroll(): void {
  const element = consoleLog.value;
  if (!element) return;
  const distanceFromBottom = element.scrollHeight - element.clientHeight - element.scrollTop;
  consoleFollowing.value = distanceFromBottom <= 32;
}
function scrollConsoleToBottom(): void {
  const element = consoleLog.value;
  if (!element) return;
  consoleFollowing.value = true;
  element.scrollTop = element.scrollHeight;
}
function changeView(view: ViewName): void {
  trainer.activeView = view;
  document.querySelectorAll("details.workspace-more[open]").forEach(menu => menu.removeAttribute("open"));
  const run = view === "home" ? refreshWakeWordLibrary(true) : view === "auto" ? refreshAuto(false)
    : view === "captured" ? refreshCaptured()
      : view === "samples" ? refreshSamples()
        : view === "firmware" ? Promise.all([refreshWakeWords(), refreshWakeWordLibrary()])
          : view === "data" ? refreshManagedData()
          : Promise.resolve();
  void run.catch((error) => notify(error instanceof Error ? error.message : "Refresh failed.", "error"));
}
function setBucket(bucket: SampleBucket): void { trainer.sampleBucket = bucket; }
function openTrim(item: AudioItem, bucket: SampleBucket): void { trainer.trimBucket = bucket; trainer.trimItem = item; }
function toggleManualNano(): void {
  if (manualNanoSelected.value) {
    trainer.trainNanoWakeWordCnn = false;
    trainer.trainNanoWakeWordGru = false;
  } else {
    trainer.trainNanoWakeWordCnn = true;
  }
}
function toggleAutoNano(): void {
  if (autoNanoSelected.value) {
    trainer.autoForm.train_nanowakeword_cnn = false;
    trainer.autoForm.train_nanowakeword_gru = false;
  } else {
    trainer.autoForm.train_nanowakeword_cnn = true;
  }
}
function generatedPhraseCount(value: string): number {
  return new Set(
    String(value || "")
      .split(/[\n,]+/)
      .map((phrase) => phrase.trim().toLocaleLowerCase())
      .filter(Boolean),
  ).size;
}
function openLink(): void {
  linkUrl.value = trainer.autoForm.tater_url || "http://127.0.0.1:8501";
  linkCode.value = "";
  linkComplete.value = false;
  trainer.taterLinkOpen = true;
  void nextTick(() => (document.querySelector("#pairing-code") as HTMLInputElement | null)?.focus());
}
function formatLinkCode(): void {
  const raw = linkCode.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
  linkCode.value = raw.length > 4 ? `${raw.slice(0, 4)}-${raw.slice(4)}` : raw;
}
async function submitLink(): Promise<void> {
  if (!linkUrl.value.trim() || !linkCode.value.trim()) { notify("Tater address and pairing code are required.", "warning"); return; }
  linkComplete.value = await claimTater(linkUrl.value, linkCode.value);
}
function metaRows(item: AudioItem): string[] {
  const rows: string[] = [];
  if (item.source_device) rows.push(String(item.source_device));
  if (item.wake_word) rows.push(String(item.wake_word));
  if (item.max_probability !== null && item.max_probability !== undefined) rows.push(`max ${item.max_probability}`);
  if (item.average_probability !== null && item.average_probability !== undefined) rows.push(`avg ${item.average_probability}`);
  if (item.detection_profile) rows.push(`profile ${String(item.detection_profile).replaceAll("_", " ")}`);
  if (item.auto_review_status) rows.push(`auto ${String(item.auto_review_status).replaceAll("_", " ")}`);
  if (item.vad_max_probability !== null && item.vad_max_probability !== undefined) rows.push(`VAD ${item.vad_max_probability}`);
  return rows;
}
function sampleSubtitle(item: AudioItem): string {
  const rows = [];
  if (item.original_name && item.original_name !== item.saved_as) rows.push(`From ${item.original_name}`);
  const timestamp = formatTimestamp(item.reviewed_at || item.received_at || item.created_at);
  if (timestamp) rows.push(`Saved ${timestamp}`);
  if (item.message) rows.push(String(item.message));
  if (item.auto_negative) rows.push("Auto-reviewed false positive");
  if (item.auto_positive) rows.push("Auto-promoted close miss");
  return rows.join(" · ") || "Training sample";
}
function consoleTone(line: string): string {
  const value = line.trim().toLowerCase();
  if (/^(✓|✅)|success|finished/.test(value)) return "success";
  if (/^(✗|❌)|error|failed|traceback/.test(value)) return "error";
  if (/^(⚠|warning)/.test(value)) return "warning";
  if (/^={4,}|^-----|^=====/.test(value)) return "heading";
  return "";
}

function openWord(word: WakeWordLibraryItem): void {
  trainer.selectedWord = word.safe_word;
  trainer.activeView = "word";
}
async function createWord(): Promise<void> {
  if (trainer.training.running || isBusy("session")) { notify("Wait for the active training run before creating another word.", "warning"); return; }
  if (trainer.session.safe_word) { await stopSession(); if (trainer.session.safe_word) return; }
  trainer.phrase = "";
  trainer.selectedWord = "";
  trainer.wizardRetrain = false;
  trainer.wizardStep = 0;
  trainer.activeView = "trainer";
}
async function prepareWord(word: WakeWordLibraryItem): Promise<boolean> {
  if (trainer.session.safe_word === word.safe_word) return true;
  if (trainer.training.running) { notify("Wait for training before changing the recording workspace.", "warning"); return false; }
  if (trainer.session.safe_word) { await stopSession(); if (trainer.session.safe_word) return false; }
  await useSavedWord(word);
  return trainer.session.safe_word === word.safe_word;
}
async function retrainWord(word: WakeWordLibraryItem): Promise<void> {
  if (!(await prepareWord(word))) return;
  trainer.selectedWord = word.safe_word;
  trainer.wizardRetrain = true;
  trainer.wizardStep = 2;
  trainer.activeView = "trainer";
}
async function manageWordRecordings(word: WakeWordLibraryItem): Promise<void> {
  if (!(await prepareWord(word))) return;
  trainer.selectedWord = word.safe_word;
  changeView("samples");
}
async function continuePhrase(): Promise<void> {
  if (!trainer.session.safe_word) {
    const existing = trainer.wakeWordLibrary.find(word => word.phrase.trim().toLocaleLowerCase() === trainer.phrase.trim().toLocaleLowerCase());
    if (existing) { openWord(existing); notify("This phrase is already saved. Open its results or train a new version.", "warning"); return; }
    await startSession();
    if (!trainer.session.safe_word) return;
  }
  await refreshSamples(true);
  trainer.wizardStep = 1;
}
async function endSetup(): Promise<void> { await stopSession(); if (!trainer.session.safe_word) trainer.wizardStep = 0; }
function showCompletedWord(): void {
  const safe = String(trainer.training.safe_word || trainer.training.summary?.safe_word || trainer.session.safe_word || "");
  const word = trainer.wakeWordLibrary.find(item => item.safe_word === safe);
  if (word) openWord(word); else changeView("firmware");
}
function stageLabel(value: string): string {
  const key = value.split(":", 1)[0];
  return ({starting:"Getting ready",datasets:"Training data",tts:"Speech generation",features:"Feature building",model:"Model training",boundary:"Wake boundary",calibration:"Calibration",package:"JSON & models",openwakeword:"openWakeWord",nanowakeword:"NanoWakeWord",complete:"Finished",failed:"Needs attention"} as Record<string,string>)[key] || key.replaceAll("-", " ");
}
</script>
<template>
  <div class="app-shell trainer-shell">
    <header class="workspace-topbar">
      <div class="workspace-brand"><span class="tater-wordmark">Tater</span><div><h1>Wake Word Trainer</h1><div class="local-trainer-status"><span>{{ isApple ? 'Apple Silicon · local trainer' : 'NVIDIA · Docker / CUDA' }}</span><small>{{ trainerVersion }}</small></div></div></div>
      <nav class="workspace-nav" aria-label="Trainer areas">
        <button v-for="tab in tabs" :key="tab.id" type="button" :aria-current="trainer.activeView === tab.id || (tab.id === 'firmware' && trainer.activeView === 'word') ? 'page' : undefined" @click="changeView(tab.id)"><UiIcon :name="tab.icon" />{{ tab.label }}<span v-if="tab.id === 'captured' && trainer.captured.captured_count" class="nav-count">{{ trainer.captured.captured_count }}</span></button>
        <button v-if="hasConsole" type="button" :aria-current="trainer.activeView === 'training' ? 'page' : undefined" @click="changeView('training')"><UiIcon name="activity" />Training</button>
        <details class="workspace-more"><summary>More</summary><div><button v-if="trainer.session.safe_word" type="button" @click="changeView('trainer')">Resume setup · {{ trainer.session.raw_phrase }}</button><button type="button" @click="changeView('samples')">Samples</button><button type="button" @click="changeView('data')">Data & storage</button><button type="button" @click="openLink">{{ autoLinked ? 'Tater connection' : 'Link Tater' }}</button></div></details>
      </nav>
    </header>
    <div v-if="trainer.training.running && trainer.activeView !== 'training'" class="training-status-strip"><span class="row"><UiIcon name="activity" />Training “{{ currentTrainingWord }}” · {{ trainingProgressStage }}<template v-if="trainingProgressPercent !== null"> · {{ trainingProgressPercent.toFixed(1) }}%</template></span><button type="button" @click="changeView('training')">Open training<UiIcon name="external" /></button></div>
    <main class="main-content workspace-main">
      <div v-if="!trainer.initialized" class="loading-panel" role="status"><span class="spinner" /><strong>Connecting to the trainer…</strong></div>
      <template v-else>
        <template v-if="trainer.activeView === 'trainer'">
          <button type="button" class="workspace-back" @click="trainer.activeView = 'home'"><UiIcon name="back" />Back to start</button>
          <header class="workspace-heading"><div><span class="eyebrow">{{ trainer.wizardRetrain ? 'Improve an existing word' : 'New wake word' }}</span><h2>{{ trainer.wizardRetrain ? `Train a new version of “${trainer.phrase}”` : 'Create a wake word' }}</h2><p>Phrase → recordings → models → review.</p></div><span v-if="trainer.session.safe_word" class="pill">{{ trainer.session.raw_phrase }}</span></header>
          <div class="wizard-steps" aria-label="Setup progress"><button v-for="(step, index) in ['Phrase', 'Recordings', 'Models', 'Review']" :key="step" type="button" :aria-current="trainer.wizardStep === index ? 'step' : undefined" :disabled="index > 0 && !trainer.session.safe_word || index === 3 && !selectedModelNames.length" @click="trainer.wizardStep = index"><b>{{ index < trainer.wizardStep ? '✓' : index + 1 }}</b>{{ step }}</button></div>
        </template>
        <TrainerWelcome v-if="trainer.activeView === 'home'" @create="createWord" @open="openWord" />
        <TrainerWelcome v-else-if="trainer.activeView === 'firmware' || (trainer.activeView === 'word' && !selectedWord)" library @create="createWord" @open="openWord" />
        <WordWorkspace v-else-if="trainer.activeView === 'word' && selectedWord" :word="selectedWord" @retrain="retrainWord" @recordings="manageWordRecordings" />
        <template v-else-if="trainer.activeView === 'training'">
          <button type="button" class="workspace-back" @click="showCompletedWord"><UiIcon name="back" />Back to wake word</button>
          <header class="workspace-heading"><div><span class="eyebrow">Training workspace</span><h2>{{ trainer.training.running ? 'Training' : 'Training results for' }} “{{ currentTrainingWord }}”</h2><p>{{ trainer.training.running ? 'The current published version and its JSON links stay available while this run trains.' : 'Review the report, view saved results, or train another version.' }}</p></div><span class="pill" :class="trainingStatus.tone">{{ trainingStatus.text }}</span></header>
          <div class="training-workspace-grid">
            <section class="panel training-stages-panel">
              <header class="workspace-section-title"><div><h3>{{ trainingProgressStage }}</h3><p>{{ trainingProgress?.label || (trainer.training.running ? 'Waiting for training output…' : 'No stage details available.') }}</p></div></header>
              <div v-if="trainingProgress" class="console-progress" :class="{ indeterminate: trainingProgressPercent === null && !trainingProgress.status, complete: trainingProgress.status === 'complete', failed: trainingProgress.status === 'failed' }"><div class="console-progress-head"><div><span>Current stage progress</span><strong>{{ trainingProgress.label || 'Working…' }}</strong></div><b>{{ trainingProgress.status === 'failed' ? 'Failed' : trainingProgress.status === 'complete' ? 'Done' : trainingProgressPercent === null ? 'Working' : `${trainingProgressPercent.toFixed(1)}%` }}</b></div><div class="console-progress-track" role="progressbar" aria-label="Current training stage progress" :aria-valuenow="trainingProgressPercent ?? undefined" aria-valuemin="0" aria-valuemax="100"><i :style="{ width: trainingProgress.status ? '100%' : trainingProgressPercent === null ? '38%' : `${trainingProgressPercent}%` }" /></div><div v-if="trainingProgressFacts.length" class="console-progress-facts"><span v-for="fact in trainingProgressFacts" :key="fact">{{ fact }}</span></div></div>
              <div class="training-stage-list"><article v-for="(stage, index) in trainer.trainingStages" :key="`${index}-${stage.stage}`" :class="stage.status"><UiIcon :name="stage.status === 'complete' ? 'check' : stage.status === 'running' ? 'activity' : 'circle'" /><div><strong>{{ stageLabel(stage.stage) }}</strong><small>{{ stage.label }}</small></div><span class="pill" :class="stage.status === 'complete' ? 'success' : stage.status === 'failed' ? 'error' : ''">{{ stage.status === 'complete' ? 'Done' : stage.status === 'failed' ? 'Needs attention' : stage.status === 'running' ? 'Running' : stage.status }}</span></article><p v-if="!trainer.trainingStages.length" class="workspace-note">Stage details appear as the trainer reports them.</p></div>
            </section>
            <aside class="stack"><section class="panel"><h3>This training run</h3><dl class="training-run-summary"><div><dt>Wake word</dt><dd>{{ currentTrainingWord }}</dd></div><div><dt>Trainer</dt><dd>{{ isApple ? 'Apple Silicon · local trainer' : 'NVIDIA · Docker / CUDA' }}</dd></div><div><dt>Recordings</dt><dd>{{ personalCount }} personal · {{ negativeCount }} negatives</dd></div><div v-if="trainer.trainingTargets.length"><dt>Requested models</dt><dd>{{ trainer.trainingTargets.join(', ') }}</dd></div><div><dt>Published JSON</dt><dd>Existing versions remain available</dd></div></dl></section><section class="panel"><h3>When it is ready</h3><p class="workspace-note">Validation results, a saved version, and its JSON links appear on the word’s page.</p><div class="row"><span class="pill">Bundle JSON</span><span class="pill">ESPHome JSON with MicroWakeWord</span></div></section></aside>
          </div>
          <section class="panel inline-training-log"><header class="workspace-section-title"><div><h3>Training log</h3><p>Training continues while you browse other pages.</p></div><button v-if="!consoleFollowing" type="button" class="console-follow" @click="scrollConsoleToBottom">Jump to latest</button></header><pre ref="consoleLog" class="console-log" @scroll.passive="onConsoleScroll"><span v-for="(line, index) in consoleLines" :key="`${index}-${line}`" :class="consoleTone(line)">{{ line }}</span></pre></section>
            <section v-if="!trainer.training.running && trainer.training.summary?.models?.length" class="training-summary-card">
              <header><div class="summary-mark" :class="trainer.training.summary.tone">{{ trainer.training.summary.tone === 'success' ? '✓' : trainer.training.summary.tone === 'error' ? '×' : '!' }}</div><div><span class="eyebrow">Last completed run</span><h4>{{ trainer.training.summary.wake_word }}</h4><p>{{ trainer.training.summary.message }}<template v-if="trainer.training.summary.completed_at"> · {{ formatTimestamp(trainer.training.summary.completed_at) }}</template></p></div><span class="pill" :class="trainer.training.summary.tone">{{ trainer.training.summary.overall }}</span></header>
              <div class="training-summary-grid">
                <article v-for="model in trainer.training.summary.models" :key="model.id" :class="model.tone">
                  <header><div><span>{{ model.id.replaceAll('_', ' ') }}</span><strong>{{ model.label }}</strong></div><b>{{ model.rating }}</b></header>
                  <p>{{ model.message }}</p>
                  <div v-if="model.metrics?.length" class="summary-metrics"><span v-for="metric in model.metrics" :key="`${metric.label}-${metric.value}`"><small>{{ metric.label }}</small><strong>{{ metric.value }}</strong></span></div>
                </article>
              </div>
            </section>
          <div class="wizard-actions"><button v-if="trainer.training.running" type="button" :disabled="isBusy('session')" @click="stopSession"><UiIcon name="stop" />Stop training</button><button v-else-if="trainer.session.safe_word" type="button" :disabled="isBusy('session')" @click="endSetup">End setup session</button><span v-else class="workspace-note">Every saved version stays in your library.</span><button type="button" class="primary" @click="showCompletedWord">{{ trainer.training.running ? 'View current word' : 'View results & JSON' }}<UiIcon name="next" /></button></div>
        </template>
<template v-else-if="trainer.activeView === 'trainer' && trainer.wizardStep !== 1"><template v-if="trainer.wizardStep === 0">          <section class="panel">
            <header class="panel-head"><div class="number">1</div><div><h3>Phrase + voice</h3><p>The phrase and voice route lock while a session is active.</p></div><span class="pill" :class="trainer.session.safe_word ? 'success' : ''">{{ trainer.session.safe_word ? `Session · ${trainer.session.safe_word}` : "No session" }}</span></header>
            <div class="form-grid phrase-form">
              <label class="field wide"><span>Wake phrase</span><input v-model="trainer.phrase" type="text" placeholder='e.g. "hey tater"' :disabled="Boolean(trainer.session.safe_word) || isBusy('session')" @keyup.enter="continuePhrase" /></label>
              <label class="field"><span>Language</span><select v-model="trainer.language" :disabled="Boolean(trainer.session.safe_word) || isBusy('session')"><option v-for="item in trainer.languages" :key="item.code" :value="item.code">{{ item.label }}</option></select><small>{{ ttsRoute }}</small></label>
              <label v-if="trainer.language === 'en' && trainer.ttsMode !== 'piper'" class="field"><span>English accent emphasis</span><select v-model="trainer.englishAccent" :disabled="Boolean(trainer.session.safe_word) || isBusy('session')"><option v-for="accent in trainer.englishAccents" :key="accent.code" :value="accent.code">{{ accent.label }}</option></select><small>Qwen emphasizes this accent; MOSS carries it through accepted references. OmniVoice and Piper keep broad English coverage.</small></label>
              <label class="field"><span>TTS source</span><select v-model="trainer.ttsMode" :disabled="Boolean(trainer.session.safe_word) || isBusy('session')">
                <option value="hybrid" :disabled="!trainer.languages.find((item) => item.code === trainer.language)?.engines?.includes('piper')">Four-provider ensemble · recommended</option>
                <option value="modern" :disabled="!trainer.languages.find((item) => item.code === trainer.language)?.engines?.some((engine) => engine !== 'piper')">Modern only · no Piper</option>
                <option value="piper" :disabled="!trainer.languages.find((item) => item.code === trainer.language)?.engines?.includes('piper')">Piper only · legacy</option>
              </select><small>Models download once and stay cached.</small></label>
            </div>
            <div class="row form-actions"><button v-if="!trainer.session.safe_word" type="button" class="button primary" :disabled="isBusy('session') || !trainer.phrase.trim()" @click="continuePhrase">{{ isBusy('session') ? "Starting…" : "Continue" }}</button><button v-else type="button" class="button danger" :disabled="isBusy('session')" @click="endSetup">{{ isBusy('session') ? "Stopping…" : (trainer.training.running ? "Stop training + end session" : "End setup session") }}</button><button v-if="trainer.session.safe_word" type="button" class="primary" @click="continuePhrase">Continue to recordings</button><button type="button" :disabled="!trainer.phrase.trim()" @click="previewPhrase">System preview</button></div>
          </section>
</template><template v-else-if="trainer.wizardStep === 2">          <section class="panel">
            <header class="panel-head"><div class="number">3</div><div><h3>Train wake word</h3><p>Personal positives and reviewed false-wake negatives are automatically included.</p></div><span class="pill" :class="trainingStatus.tone">{{ trainingStatus.text }}</span></header>
            <div class="stats"><article><span>Positive samples</span><strong>{{ personalCount }}</strong></article><article><span>Negative samples</span><strong>{{ negativeCount }}</strong></article><article><span>Training format</span><strong class="format-value">16 kHz · mono · WAV</strong></article></div>
            <div class="model-picker">
              <div class="model-choice-grid" role="group" aria-label="Wake-word models to train">
                <button type="button" class="model-choice" :class="{ active: trainer.trainMicroWakeWord }" :aria-pressed="trainer.trainMicroWakeWord" @click="trainer.trainMicroWakeWord = !trainer.trainMicroWakeWord"><i>MWW</i><span><strong>MicroWakeWord</strong><small>TFLite for ESP32 satellites and ESPHome.</small></span><b>✓</b></button>
                <button type="button" class="model-choice" :class="{ active: trainer.trainOpenWakeWord }" :aria-pressed="trainer.trainOpenWakeWord" @click="trainer.trainOpenWakeWord = !trainer.trainOpenWakeWord"><i>OWW</i><span><strong>openWakeWord</strong><small>ONNX for satellites with OWW support.</small></span><b>✓</b></button>
                <button type="button" class="model-choice" :class="{ active: manualNanoSelected }" :aria-pressed="manualNanoSelected" @click="toggleManualNano"><i>NWW</i><span><strong>NanoWakeWord</strong><small>ONNX with selectable CNN and GRU builds.</small></span><b>✓</b></button>
              </div>
              <div v-if="manualNanoSelected" class="nano-architecture-panel">
                <div class="nano-architecture-heading"><div><strong>NanoWakeWord architectures</strong><small>Select either architecture or train both.</small></div><span>{{ trainer.trainNanoWakeWordCnn && trainer.trainNanoWakeWordGru ? "2 selected" : "1 selected" }}</span></div>
                <div class="nano-architecture-grid" role="group" aria-label="NanoWakeWord architectures">
                  <button type="button" class="architecture-choice" :class="{ active: trainer.trainNanoWakeWordCnn }" :aria-pressed="trainer.trainNanoWakeWordCnn" @click="trainer.trainNanoWakeWordCnn = !trainer.trainNanoWakeWordCnn"><i>CNN</i><span><strong>CNN</strong><small>Recommended default with stronger recall.</small></span><b>✓</b></button>
                  <button type="button" class="architecture-choice" :class="{ active: trainer.trainNanoWakeWordGru }" :aria-pressed="trainer.trainNanoWakeWordGru" @click="trainer.trainNanoWakeWordGru = !trainer.trainNanoWakeWordGru"><i>GRU</i><span><strong>GRU</strong><small>Recurrent model tuned for fewer false triggers.</small></span><b>✓</b></button>
                </div>
              </div>
              <section v-if="manualSelectedEngines.length" class="hard-negative-card">
                <header class="hard-negative-head"><div class="hard-negative-icon">−</div><div><span class="eyebrow">Optional precision tuning</span><strong>Hard negative training</strong><small>Teach the selected models which similar-sounding phrases must not wake the assistant.</small></div><b>{{ generatedPhraseCount(trainer.generatedNegativePhrases) ? `${generatedPhraseCount(trainer.generatedNegativePhrases)} phrase${generatedPhraseCount(trainer.generatedNegativePhrases) === 1 ? "" : "s"}` : "Optional" }}</b></header>
                <div class="hard-negative-engines"><span>Applied to</span><i v-for="engine in ['MWW', 'OWW', 'NWW']" :key="engine" :class="{ active: manualSelectedEngines.includes(engine) }">{{ engine }}</i></div>
                <label class="field"><span>Phrases to reject</span><textarea v-model="trainer.generatedNegativePhrases" rows="4" placeholder="hey later&#10;hey waiter&#10;okay tater"></textarea><small>One phrase per line. Each selected engine synthesizes and validates these near-misses alongside reviewed recordings and its broad negative data.</small></label>
                <footer><span>Unchanged phrases reuse their cached audio on later runs.</span><strong>The actual wake phrase is ignored automatically.</strong></footer>
              </section>
            </div>
            <div class="wizard-actions"><button type="button" @click="trainer.wizardStep = 1">Back to recordings</button><button type="button" class="primary" :disabled="!trainer.session.safe_word || !selectedModelNames.length || trainer.training.running" @click="trainer.wizardStep = 3">Review training<UiIcon name="next" /></button></div>
            <details class="workspace-advanced"><summary>Additional training actions</summary><div class="row"><button type="button" class="boundary-button" :disabled="!activeLibraryWord?.boundary_only_ready || trainer.training.running || isBusy('boundary-start')" @click="startBoundaryTraining">{{ isBusy('boundary-start') ? "Starting boundary…" : "Retrain boundary only" }}</button><button v-if="isApple && retryableOpenWakeWord" type="button" :disabled="trainer.training.running || isBusy('oww-retry')" @click="retryOpenWakeWord(String(retryableOpenWakeWord.key || ''))">{{ isBusy('oww-retry') ? "Starting OWW…" : "Retry OWW only" }}</button><button v-if="retryableNanoWakeWord" type="button" :disabled="trainer.training.running || isBusy('nww-revalidate')" @click="revalidateNanoWakeWord(String(retryableNanoWakeWord.id || '').replace('nano_', ''), String(trainer.session.safe_word || ''))">{{ isBusy('nww-revalidate') ? "Checking NWW…" : `Revalidate ${String(retryableNanoWakeWord.label || 'NWW')} candidate` }}</button></div></details>
          </section>
</template><template v-else><section class="panel"><h3>Ready to train “{{ trainer.phrase }}”</h3><dl class="workspace-summary"><dt>Language</dt><dd>{{ trainer.languages.find(item => item.code === trainer.language)?.label || trainer.language }}</dd><dt>Voice generation</dt><dd>{{ ttsRoute }}</dd><dt>English accent</dt><dd>{{ trainer.englishAccents.find(item => item.code === trainer.englishAccent)?.label || trainer.englishAccent }}</dd><dt>Recordings</dt><dd>{{ personalCount }} personal · {{ negativeCount }} reviewed negatives · generated voices included</dd><dt>Models</dt><dd>{{ selectedModelNames.join(', ') }}</dd><dt>Phrases to reject</dt><dd>{{ generatedPhraseCount(trainer.generatedNegativePhrases) }} configured</dd><dt>Your results</dt><dd>Saved validation report, version history, bundle JSON, and ESPHome JSON when MicroWakeWord is included.</dd></dl><p class="workspace-note">Earlier saved versions and their JSON links stay available throughout training.</p><div class="wizard-actions"><button type="button" @click="trainer.wizardStep = 2">Back to models</button><button type="button" class="primary" :disabled="trainer.training.running || isBusy('training-start') || !trainer.session.safe_word" @click="startTraining"><UiIcon name="activity" />Start training</button></div></section></template></template>        <template v-else-if="trainer.activeView === 'auto'">
          <section class="hero auto-hero"><div><span class="eyebrow">False-positive loop</span><h2>Auto Training</h2><p>Transcribe captures, sort negatives, recover close misses, retrain on schedule, and publish through Tater.</p></div><span class="pill hero-pill" :class="autoStatus.tone">{{ autoStatus.text }}</span></section>
          <section class="panel">
            <header class="panel-head"><div class="number">1</div><div><h3>Review rules</h3><p>Conservative local STT keeps uncertain clips in the manual inbox.</p></div></header>
            <div class="toggle-list">
              <label><input v-model="trainer.autoForm.enabled" type="checkbox" /><span><strong>Enable Auto Training</strong><small>Queue eligible wake triggers for local transcription.</small></span></label>
              <label><input v-model="trainer.autoForm.delete_confirmed_wakes" type="checkbox" /><span><strong>Delete confirmed good wakes</strong><small>Remove normal triggers when STT confirms the phrase.</small></span></label>
              <label><input v-model="trainer.autoForm.promote_close_misses" type="checkbox" /><span><strong>Promote confirmed close misses</strong><small>Move verified close misses into positive samples.</small></span></label>
            </div>
            <div class="form-grid">
              <label class="field"><span>Wake phrase</span><input v-model="trainer.autoForm.wake_phrase" type="text" /></label>
              <label class="field"><span>STT language</span><input v-model="trainer.autoForm.language" type="text" /></label>
              <label v-if="String(trainer.autoForm.language).toLowerCase().startsWith('en')" class="field"><span>English accent emphasis</span><select v-model="trainer.autoForm.english_accent"><option v-for="accent in trainer.englishAccents" :key="accent.code" :value="accent.code">{{ accent.label }}</option></select><small>Used when Auto Training needs to regenerate English TTS.</small></label>
              <label class="field wide"><span>STT engine</span><select v-model="trainer.autoForm.stt_engine"><option v-for="engine in sttEngines" :key="engine.id || engine.value" :value="engine.id || engine.value">{{ engine.label || engine.name || engine.id }}</option></select><small>{{ sttEngines.find((row) => (row.id || row.value) === trainer.autoForm.stt_engine)?.description || "Runs locally on this trainer." }}</small></label>
              <label class="field"><span>Minimum transcript characters</span><input v-model.number="trainer.autoForm.minimum_transcript_chars" min="1" max="100" type="number" /></label>
            </div>
          </section>
          <section class="panel">
            <header class="panel-head"><div class="number">2</div><div><h3>Training schedule</h3><p>A run starts only after enough newly reviewed negatives accumulate.</p></div></header>
            <div class="form-grid">
              <label class="field"><span>Run training</span><select v-model.number="trainer.autoForm.schedule_hours"><option :value="0">Manually only</option><option :value="6">Every 6 hours</option><option :value="12">Every 12 hours</option><option :value="24">Every day</option><option :value="48">Every 2 days</option><option :value="168">Every week</option></select></label>
              <label class="field"><span>Minimum new negatives</span><input v-model.number="trainer.autoForm.minimum_new_negatives" min="1" max="10000" type="number" /></label>
            </div>
            <div class="model-picker">
              <div class="model-choice-grid" role="group" aria-label="Auto Training wake-word models">
                <button type="button" class="model-choice" :class="{ active: trainer.autoForm.train_microwakeword }" :aria-pressed="trainer.autoForm.train_microwakeword" @click="trainer.autoForm.train_microwakeword = !trainer.autoForm.train_microwakeword"><i>MWW</i><span><strong>MicroWakeWord</strong><small>Refresh the ESP32 and ESPHome model.</small></span><b>✓</b></button>
                <button type="button" class="model-choice" :class="{ active: trainer.autoForm.train_openwakeword }" :aria-pressed="trainer.autoForm.train_openwakeword" @click="trainer.autoForm.train_openwakeword = !trainer.autoForm.train_openwakeword"><i>OWW</i><span><strong>openWakeWord</strong><small>Refresh the openWakeWord ONNX model.</small></span><b>✓</b></button>
                <button type="button" class="model-choice" :class="{ active: autoNanoSelected }" :aria-pressed="autoNanoSelected" @click="toggleAutoNano"><i>NWW</i><span><strong>NanoWakeWord</strong><small>Refresh selected CNN and GRU models.</small></span><b>✓</b></button>
              </div>
              <div v-if="autoNanoSelected" class="nano-architecture-panel">
                <div class="nano-architecture-heading"><div><strong>NanoWakeWord architectures</strong><small>Select either architecture or retrain both.</small></div><span>{{ trainer.autoForm.train_nanowakeword_cnn && trainer.autoForm.train_nanowakeword_gru ? "2 selected" : "1 selected" }}</span></div>
                <div class="nano-architecture-grid" role="group" aria-label="Auto Training NanoWakeWord architectures">
                  <button type="button" class="architecture-choice" :class="{ active: trainer.autoForm.train_nanowakeword_cnn }" :aria-pressed="trainer.autoForm.train_nanowakeword_cnn" @click="trainer.autoForm.train_nanowakeword_cnn = !trainer.autoForm.train_nanowakeword_cnn"><i>CNN</i><span><strong>CNN</strong><small>Recommended default with stronger recall.</small></span><b>✓</b></button>
                  <button type="button" class="architecture-choice" :class="{ active: trainer.autoForm.train_nanowakeword_gru }" :aria-pressed="trainer.autoForm.train_nanowakeword_gru" @click="trainer.autoForm.train_nanowakeword_gru = !trainer.autoForm.train_nanowakeword_gru"><i>GRU</i><span><strong>GRU</strong><small>Recurrent model tuned for fewer false triggers.</small></span><b>✓</b></button>
                </div>
              </div>
              <section v-if="autoSelectedEngines.length" class="hard-negative-card">
                <header class="hard-negative-head"><div class="hard-negative-icon">−</div><div><span class="eyebrow">Optional precision tuning</span><strong>Hard negative training</strong><small>Keep near-miss protection consistent whenever Auto Training refreshes a model.</small></div><b>{{ generatedPhraseCount(trainer.autoForm.generated_negative_phrases) ? `${generatedPhraseCount(trainer.autoForm.generated_negative_phrases)} phrase${generatedPhraseCount(trainer.autoForm.generated_negative_phrases) === 1 ? "" : "s"}` : "Optional" }}</b></header>
                <div class="hard-negative-engines"><span>Applied to</span><i v-for="engine in ['MWW', 'OWW', 'NWW']" :key="engine" :class="{ active: autoSelectedEngines.includes(engine) }">{{ engine }}</i></div>
                <label class="field"><span>Phrases to reject</span><textarea v-model="trainer.autoForm.generated_negative_phrases" rows="4" placeholder="hey later&#10;hey waiter&#10;okay tater"></textarea><small>One phrase per line. Saved with the schedule and synthesized for every selected model.</small></label>
                <footer><span>Reviewed false wakes still receive the strongest model weight.</span><strong>The actual wake phrase is ignored automatically.</strong></footer>
              </section>
            </div>
            <section v-if="!trainer.training.running && trainer.training.summary?.models?.length" class="training-summary-card">
              <header><div class="summary-mark" :class="trainer.training.summary.tone">{{ trainer.training.summary.tone === 'success' ? '✓' : trainer.training.summary.tone === 'error' ? '×' : '!' }}</div><div><span class="eyebrow">Last completed run</span><h4>{{ trainer.training.summary.wake_word }}</h4><p>{{ trainer.training.summary.message }}<template v-if="trainer.training.summary.completed_at"> · {{ formatTimestamp(trainer.training.summary.completed_at) }}</template></p></div><span class="pill" :class="trainer.training.summary.tone">{{ trainer.training.summary.overall }}</span></header>
              <div class="training-summary-grid"><article v-for="model in trainer.training.summary.models" :key="`auto-${model.id}`" :class="model.tone"><header><div><span>{{ model.id.replaceAll('_', ' ') }}</span><strong>{{ model.label }}</strong></div><b>{{ model.rating }}</b></header><p>{{ model.message }}</p><div v-if="model.metrics?.length" class="summary-metrics"><span v-for="metric in model.metrics" :key="`${metric.label}-${metric.value}`"><small>{{ metric.label }}</small><strong>{{ metric.value }}</strong></span></div></article></div>
            </section>
            <div class="stats"><article><span>Pending negatives</span><strong>{{ Number(autoState.pending_negative_count || 0) }}</strong></article><article><span>Next check</span><strong class="format-value">{{ autoState.next_run_at ? formatTimestamp(autoState.next_run_at) : "Manual" }}</strong></article><article><span>Last training</span><strong class="format-value">{{ autoState.last_train_finished_at ? formatTimestamp(autoState.last_train_finished_at) : "Never" }}</strong></article></div>
          </section>
          <section class="panel">
            <header class="panel-head"><div class="number">3</div><div><h3>Publish to Tater</h3><p>Securely activate successful models across every connected satellite.</p></div></header>
            <div class="form-grid"><label class="field wide"><span>Trainer public URL</span><input v-model="trainer.autoForm.advertised_base_url" type="text" placeholder="Auto-detect LAN address" /><small>{{ trainer.autoForm.advertised_base_url ? `Configured: ${trainer.autoForm.advertised_base_url}` : `Detected: ${trainer.auto.advertised_base_url || "unavailable"}` }}</small></label><label class="field wide"><span>Tater URL</span><input v-model="trainer.autoForm.tater_url" type="text" /></label></div>
            <div class="link-row"><span class="pill" :class="autoLinked ? 'success' : 'warning'">{{ autoLinked ? `Linked${trainer.auto.trainer_link?.tater_name ? ` · ${trainer.auto.trainer_link.tater_name}` : ''}` : "Not linked" }}</span><button type="button" class="button primary" :disabled="isBusy('auto')" @click="openLink">{{ autoLinked ? "Relink Tater" : "Link Tater" }}</button><button v-if="autoLinked" type="button" class="button danger" :disabled="isBusy('auto')" @click="unlinkTater">Unlink</button></div>
            <div class="toggle-list compact"><label><input v-model="trainer.autoForm.notify_satellites" type="checkbox" /><span><strong>Activate after successful training</strong><small>Tater applies the new word globally.</small></span></label></div>
          </section>
          <section class="panel action-panel"><div class="action-grid"><button type="button" class="button primary" :disabled="isBusy('auto')" @click="saveAuto">Save Auto Training</button><button type="button" :disabled="isBusy('auto')" @click="runAutoAction('review_now')">Review inbox now</button><button type="button" :disabled="isBusy('auto') || trainer.training.running" @click="runAutoAction('train_now')">Train now</button><button type="button" :disabled="isBusy('auto') || !autoLinked" @click="runAutoAction('notify_now')">Publish current word</button></div><p class="audit">{{ autoAudit }}</p></section>
        </template>
        <template v-else-if="trainer.activeView === 'captured'">
          <section class="hero capture-hero"><div><span class="eyebrow">Capture review</span><h2>Captured Audio</h2><p>Listen to clips from your satellites and turn every real-world event into a better model.</p></div><span class="pill hero-pill" :class="trainer.captured.captured_count ? 'warning' : ''">{{ trainer.captured.captured_count ? `${trainer.captured.captured_count} waiting` : "Inbox idle" }}</span></section>
          <section class="panel"><header class="panel-head"><div class="number">1</div><div><h3>Review queue</h3><p>Approve good phrases, keep false positives as negatives, or discard noise.</p></div><button type="button" :disabled="isBusy('captured')" @click="refreshCaptured()">{{ isBusy('captured') ? "Refreshing…" : "Refresh inbox" }}</button></header><div class="stats"><article><span>Inbox</span><strong>{{ trainer.captured.captured_count }}</strong></article><article><span>All-word negatives</span><strong>{{ trainer.captured.negative_count }}</strong></article><article><span>All-word positives</span><strong>{{ trainer.captured.personal_count }}</strong></article></div></section>
          <section class="panel"><header class="panel-head"><div class="number">2</div><div><h3>Listen + sort</h3><p>Metadata remains visible so borderline detections are easy to understand.</p></div></header>
            <div v-if="!trainer.captured.items?.length" class="empty-state">No captured audio yet. Clips sent by satellites will appear here.</div>
            <div v-else class="audio-list"><article v-for="item in trainer.captured.items" :key="item.capture_id || item.saved_as" class="audio-card">
              <header><div><strong>{{ item.original_name || item.saved_as }}</strong><small>{{ formatTimestamp(item.captured_at || item.received_at) }} {{ item.message || "" }}</small></div><span class="pill" :class="captureTone(item).tone">{{ captureTone(item).label }}</span></header>
              <div v-if="metaRows(item).length" class="meta-row"><span v-for="row in metaRows(item)" :key="row">{{ row }}</span></div>
              <div class="capture-assignment" :class="{ unassigned: !item.review_wake_word }"><label><span>Train this clip for</span><select v-model="item.review_wake_word"><option value="">Choose a saved wake word…</option><option v-for="word in trainer.wakeWordLibrary" :key="word.safe_word" :value="word.safe_word">{{ word.phrase }} · {{ word.safe_word }}</option></select></label><span class="pill" :class="item.review_wake_word ? 'success' : 'warning'">{{ item.review_wake_word ? `Routed to ${item.review_wake_word}` : "Unassigned" }}</span></div>
              <div v-if="item.transcript" class="transcript"><b>STT</b> {{ item.transcript }}</div><div v-if="item.auto_review_guided_transcript" class="transcript"><b>Guided wake check</b> {{ item.auto_review_guided_transcript }}</div>
              <audio controls preload="none" :src="itemAudioUrl(item, 'captured')" />
              <footer><span>{{ item.saved_as }} · {{ describeFormat(item.final_format) }}</span><div><button type="button" :disabled="isBusy('review') || !item.review_wake_word" @click="reviewCaptured(item, 'approve_personal')">Add positive</button><button type="button" :disabled="isBusy('review') || !item.review_wake_word" @click="reviewCaptured(item, 'mark_negative')">Mark negative</button><button type="button" class="button danger ghost" :disabled="isBusy('review')" @click="reviewCaptured(item, 'discard')">Discard</button></div></footer>
            </article></div>
          </section>
        </template>
        <template v-else-if="trainer.activeView === 'samples' || (trainer.activeView === 'trainer' && trainer.wizardStep === 1)">
          <section v-if="trainer.activeView === 'samples'" class="hero samples-hero"><div><span class="eyebrow">Sample library</span><h2>Current Training Samples</h2><p>Audit positives and negatives, trim recordings precisely, and import seed audio.</p></div><span class="pill hero-pill">{{ personalCount + negativeCount }} total</span></section>
          <section class="panel">
            <header class="panel-head sample-head"><div class="number">1</div><div><h3>Saved samples</h3><p>Personal clips are positives. Negative clips are false wakes and hard negatives.</p></div><div class="segment-control"><button type="button" :class="{ active: trainer.sampleBucket === 'personal' }" @click="setBucket('personal')">Personal <b>{{ personalCount }}</b></button><button type="button" :class="{ active: trainer.sampleBucket === 'negative' }" @click="setBucket('negative')">Negative <b>{{ negativeCount }}</b></button></div></header>
            <div class="row toolbar"><button type="button" :disabled="isBusy('samples')" @click="refreshSamples()">Refresh</button><button type="button" class="button danger ghost" :disabled="isBusy('review') || personalCount === 0" @click="clearSamples('personal')">Clear positives</button><button type="button" class="button danger ghost" :disabled="isBusy('review') || negativeCount === 0" @click="clearSamples('negative')">Clear negatives</button></div>
            <div v-if="!selectedSamples.length" class="empty-state">No {{ trainer.sampleBucket }} samples saved yet.</div>
            <div v-else class="audio-list compact-list"><article v-for="item in pagedSamples" :key="item.saved_as" class="audio-card">
              <header><div><strong>{{ item.saved_as }}</strong><small>{{ sampleSubtitle(item) }}</small></div><div class="row"><span v-if="item.trimmed" class="pill warning">Trimmed</span><span class="pill" :class="trainer.sampleBucket === 'personal' ? 'success' : 'error'">{{ trainer.sampleBucket === "personal" ? "Positive" : "Negative" }}</span></div></header>
              <div v-if="item.transcript" class="transcript"><b>STT</b> {{ item.transcript }}</div><div v-if="item.auto_review_guided_transcript" class="transcript"><b>Guided wake check</b> {{ item.auto_review_guided_transcript }}</div>
              <audio controls preload="none" :src="itemAudioUrl(item, trainer.sampleBucket)" />
              <footer><span>{{ describeFormat(item.final_format) }}</span><div><button type="button" @click="openTrim(item, trainer.sampleBucket)">Trim</button><button v-if="item.trimmed" type="button" @click="revertSample(item, trainer.sampleBucket)">Revert</button><button type="button" class="button danger ghost" :disabled="isBusy('review')" @click="removeSample(item, trainer.sampleBucket)">Remove</button></div></footer>
            </article></div>
            <div v-if="samplePages > 1" class="pagination"><button type="button" :disabled="trainer.samplePage[trainer.sampleBucket] === 0" @click="trainer.samplePage[trainer.sampleBucket]--">Previous</button><span>Page {{ trainer.samplePage[trainer.sampleBucket] + 1 }} of {{ samplePages }}</span><button type="button" :disabled="trainer.samplePage[trainer.sampleBucket] >= samplePages - 1" @click="trainer.samplePage[trainer.sampleBucket]++">Next</button></div>
          </section>
          <section class="panel">
            <header class="panel-head"><div class="number">2</div><div><h3>Manual sample import</h3><p>Optional seed recordings are normalized to the trainer’s required WAV format.</p></div></header>
            <label class="dropzone"><input ref="uploadInput" type="file" multiple accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.webm,.opus" @change="selectFiles" /><span><strong>Choose one or many audio files</strong><small>WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, and WEBM</small></span><b>{{ trainer.selectedFiles.length ? `${trainer.selectedFiles.length} selected` : "Browse" }}</b></label>
            <button type="button" class="button primary" :disabled="!trainer.session.safe_word || !trainer.selectedFiles.length || isBusy('upload')" @click="uploadSelectedFiles(uploadInput)">{{ isBusy('upload') ? "Uploading…" : "Upload selected samples" }}</button>
            <div class="progress-card"><div><strong>{{ trainer.uploadLabel }}</strong><span>{{ trainer.uploadProgress }}%</span></div><div class="progress-track"><i :style="{ width: `${trainer.uploadProgress}%` }" /></div><small>{{ trainer.uploadDetail }}</small></div>
          </section>
<div v-if="trainer.activeView === 'trainer'" class="wizard-actions"><button type="button" @click="trainer.wizardStep = 0">Back to phrase</button><button type="button" class="primary" @click="trainer.wizardStep = 2">Continue to models<UiIcon name="next" /></button></div>
        </template>
        <template v-else-if="trainer.activeView === 'data'">
          <section class="hero data-hero"><div><span class="eyebrow">Local storage</span><h2>Data Management</h2><p>See exactly what the trainer has downloaded, generated, recorded, and produced.</p></div><span class="pill hero-pill">{{ formatBytes(trainer.managedData.total_size_bytes) }} total</span></section>
          <section v-if="isApple" class="panel storage-location-panel">
            <header class="panel-head"><div class="number storage-icon">↗</div><div><h3>Data folder</h3><p>Put recordings, datasets, model downloads, caches, training environments, and results on any Mac-compatible drive.</p></div><button v-if="macDataFolderPickerAvailable" type="button" :disabled="trainer.training.running || isBusy('data') || isBusy('data-delete')" @click="chooseDataDirectory">Change Folder…</button></header>
            <div class="storage-location"><div><span>Current location</span><code>{{ trainer.managedData.data_directory || "Loading…" }}</code></div><span v-if="macDataFolderPickerAvailable" class="pill success">Mac app</span><span v-else class="pill neutral">Open the Mac app to change</span></div>
            <p class="storage-hint">Changing this folder safely stops the local trainer, copies and verifies the data, then restarts it. Keep an external drive connected whenever you use the trainer.</p>
            <p v-if="trainer.training.running" class="data-warning">Stop the active training session before changing the data folder.</p>
          </section>
          <section class="panel">
            <header class="panel-head"><div class="number">i</div><div><h3>Trainer storage</h3><p>Deleting an item is permanent. Required downloads and generated caches will be rebuilt the next time training needs them.</p></div><button type="button" :disabled="isBusy('data') || isBusy('data-delete')" @click="refreshManagedData()">{{ isBusy('data') ? "Scanning…" : "Refresh sizes" }}</button></header>
            <div class="stats"><article><span>Space used</span><strong class="format-value">{{ formatBytes(trainer.managedData.total_size_bytes) }}</strong></article><article><span>Files</span><strong>{{ Number(trainer.managedData.total_file_count || 0).toLocaleString() }}</strong></article><article><span>Individual items</span><strong>{{ trainer.managedData.items.length }}</strong></article></div>
            <p v-if="trainer.training.running" class="data-warning">Stop the active training session before deleting data.</p>
          </section>
          <section v-for="(group, groupIndex) in dataCategories" :key="group.name" class="panel data-panel">
            <header class="panel-head"><div class="number">{{ groupIndex + 1 }}</div><div><h3>{{ group.name }}</h3><p>{{ group.items.length }} separately managed item{{ group.items.length === 1 ? "" : "s" }}</p></div></header>
            <div class="data-list"><article v-for="item in group.items" :key="item.id" class="data-row" :class="{ empty: !item.file_count }">
              <div class="data-copy"><div class="data-title"><strong>{{ item.label }}</strong><code>{{ item.location }}</code></div><small>{{ item.description }}</small><span v-if="item.rebuild_note" class="data-note">{{ item.rebuild_note }}</span></div>
              <div class="data-usage"><strong>{{ formatBytes(item.size_bytes) }}</strong><span>{{ Number(item.file_count || 0).toLocaleString() }} file{{ item.file_count === 1 ? "" : "s" }}</span></div>
              <button type="button" class="button danger ghost" :disabled="!item.file_count || trainer.training.running || isBusy('data') || isBusy('data-delete')" @click="deleteManagedData(item)">{{ isBusy('data-delete') ? "Please wait…" : "Delete" }}</button>
            </article></div>
          </section>
          <section v-if="!isBusy('data') && !trainer.managedData.items.length" class="panel empty-state">No managed trainer data was found.</section>
        </template>

      </template>
    </main>
    <footer v-if="trainer.initialized" class="workspace-footer"><span class="row"><UiIcon name="link" />{{ autoLinked ? 'Tater linked' : 'Local trainer' }}<button type="button" class="text-button" @click="openLink">{{ autoLinked ? 'Connection' : 'Link Tater' }}</button></span><span>Recordings, results, and every saved version stay with the word.</span></footer>
    <Teleport to="body">
      <div v-if="trainer.taterLinkOpen" class="modal-backdrop" @click.self="trainer.taterLinkOpen = false">
        <section class="modal link-modal" role="dialog" aria-modal="true" aria-label="Link Tater"><header class="modal-head"><div><span class="eyebrow">Secure pairing</span><h2>{{ linkComplete ? "Tater linked" : "Link Tater" }}</h2><p>{{ linkComplete ? "This trainer can securely publish wake-word updates." : "Enter the short-lived code shown in Tater Voice Settings." }}</p></div><button type="button" @click="trainer.taterLinkOpen = false">Close</button></header>
          <div v-if="linkComplete" class="link-success"><i>✓</i><strong>Successfully linked{{ trainer.auto.trainer_link?.tater_name ? ` to ${trainer.auto.trainer_link.tater_name}` : "" }}</strong><span>The private link key is stored locally and is never displayed.</span></div>
          <div v-else class="stack"><label class="field"><span>Tater address</span><input v-model="linkUrl" type="text" /></label><label class="field"><span>Tater pairing code</span><input id="pairing-code" v-model="linkCode" class="pairing-code" maxlength="9" placeholder="ABCD-EFGH" autocomplete="off" @input="formatLinkCode" /></label><small>In Tater, open Voice Settings → Wake Word Trainer → Link Trainer.</small><button type="button" class="button primary" :disabled="isBusy('link')" @click="submitLink">{{ isBusy('link') ? "Linking securely…" : "Link Tater" }}</button></div>
        </section>
      </div>
    </Teleport>
    <AudioTrimModal />
    <Transition name="toast"><div v-if="trainer.toast.message" class="toast" :class="trainer.toast.tone" role="status">{{ trainer.toast.message }}</div></Transition>

  </div>
</template>
