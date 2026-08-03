<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import AudioTrimModal from "./components/AudioTrimModal.vue";
import type { JsonRecord } from "./api";
import {
  autoLinked, captureTone, claimTater, clearSamples, copyWakeWord, deleteManagedData, describeFormat,
  disposeTrainer, ensureSupportedTtsMode, formatBytes, formatTimestamp, hasConsole, initializeTrainer,
  isBusy, itemAudioUrl, negativeCount, notify, personalCount, previewPhrase, refreshAuto,
  refreshCaptured, refreshManagedData, refreshSamples, refreshWakeWords, removeSample, revertSample, reviewCaptured,
  runAutoAction, saveAuto, selectFiles, selectedSamples, startSession, startTraining, stopSession, sttEngines,
  trainer, ttsRoute, unlinkTater, uploadSelectedFiles,
} from "./trainerStore";
import type { AudioItem, ManagedDataItem, SampleBucket, ViewName } from "./types";

const uploadInput = ref<HTMLInputElement | null>(null);
const consoleLog = ref<HTMLElement | null>(null);
const consoleFollowing = ref(true);
const linkUrl = ref("");
const linkCode = ref("");
const linkComplete = ref(false);
const mascotUrl = "/static/images/tater-wake-word-trainer.png";
const pageSize = 50;
const tabs: Array<{ id: ViewName; label: string; short: string }> = [
  { id: "trainer", label: "Trainer", short: "Train" },
  { id: "auto", label: "Auto Training", short: "Auto" },
  { id: "firmware", label: "Wake Words", short: "Words" },
  { id: "captured", label: "Captured Audio", short: "Inbox" },
  { id: "samples", label: "Samples", short: "Samples" },
  { id: "data", label: "Data", short: "Data" },
];

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
const dataCategories = computed(() => {
  const groups = new Map<string, ManagedDataItem[]>();
  for (const item of trainer.managedData.items || []) {
    const rows = groups.get(item.category) || [];
    rows.push(item);
    groups.set(item.category, rows);
  }
  return Array.from(groups, ([name, items]) => ({ name, items }));
});

watch(() => trainer.language, ensureSupportedTtsMode);
watch(() => trainer.toast.serial, () => window.setTimeout(() => { trainer.toast.message = ""; }, 4500));
watch(consoleLines, async () => {
  if (!consoleFollowing.value) return;
  await nextTick();
  if (consoleFollowing.value && consoleLog.value) {
    consoleLog.value.scrollTop = consoleLog.value.scrollHeight;
  }
});
watch(() => trainer.consoleOpen, async (isOpen) => {
  if (!isOpen) return;
  consoleFollowing.value = true;
  await nextTick();
  scrollConsoleToBottom();
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
  trainer.consoleOpen = false;
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
  const run = view === "auto" ? refreshAuto(false)
    : view === "captured" ? refreshCaptured()
      : view === "samples" ? refreshSamples()
        : view === "firmware" ? refreshWakeWords()
          : view === "data" ? refreshManagedData()
          : Promise.resolve();
  void run.catch((error) => notify(error instanceof Error ? error.message : "Refresh failed.", "error"));
}
function setBucket(bucket: SampleBucket): void { trainer.sampleBucket = bucket; }
function openTrim(item: AudioItem, bucket: SampleBucket): void { trainer.trimBucket = bucket; trainer.trimItem = item; }
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
function wordJsonUrl(item: JsonRecord): string { return String(item.json_url || item.url || item.jsonUrl || ""); }
function wordModelUrl(item: JsonRecord): string { return String(item.model_url || item.modelUrl || ""); }
function consoleTone(line: string): string {
  const value = line.trim().toLowerCase();
  if (/^(✓|✅)|success|finished/.test(value)) return "success";
  if (/^(✗|❌)|error|failed|traceback/.test(value)) return "error";
  if (/^(⚠|warning)/.test(value)) return "warning";
  if (/^={4,}|^-----|^=====/.test(value)) return "heading";
  return "";
}
</script>

<template>
  <div class="app-shell">
    <div class="ambient ambient-one" aria-hidden="true" /><div class="ambient ambient-two" aria-hidden="true" />
    <header class="app-header">
      <div class="brand"><div class="brand-mark" aria-hidden="true"><img :src="mascotUrl" alt="" /></div><div><span class="eyebrow">Tater tools</span><h1>Wake Word Studio</h1><p>Generate voices, curate real recordings, train, and publish.</p></div></div>
      <div class="header-status"><span class="live-dot"><i />Local trainer</span><span v-if="trainer.session.safe_word" class="session-chip">{{ trainer.session.safe_word }} · {{ trainer.language }}</span></div>
    </header>
    <nav class="tabs" aria-label="Trainer areas">
      <button v-for="tab in tabs" :key="tab.id" type="button" :class="{ active: trainer.activeView === tab.id }" @click="changeView(tab.id)"><span class="tab-full">{{ tab.label }}</span><span class="tab-short">{{ tab.short }}</span><b v-if="tab.id === 'captured' && trainer.captured.captured_count">{{ trainer.captured.captured_count }}</b></button>
    </nav>
    <main class="main-content">
      <div v-if="!trainer.initialized" class="loading-panel"><span class="spinner" /><strong>Connecting to the local trainer…</strong></div>
      <template v-else>
        <template v-if="trainer.activeView === 'trainer'">
          <section class="hero training-hero">
            <div><span class="eyebrow">Training studio</span><h2>Build a personal wake word</h2><p>Choose a multilingual voice route, check your real samples, then follow the model pipeline live.</p></div>
            <div class="step-row"><span><b>1</b> Phrase</span><span><b>2</b> Samples</span><span><b>3</b> Train</span></div>
          </section>
          <section class="panel">
            <header class="panel-head"><div class="number">1</div><div><h3>Phrase + voice</h3><p>The phrase and voice route lock while a session is active.</p></div><span class="pill" :class="trainer.session.safe_word ? 'success' : ''">{{ trainer.session.safe_word ? `Session · ${trainer.session.safe_word}` : "No session" }}</span></header>
            <div class="form-grid phrase-form">
              <label class="field wide"><span>Wake phrase</span><input v-model="trainer.phrase" type="text" placeholder='e.g. "hey tater"' :disabled="Boolean(trainer.session.safe_word) || isBusy('session')" @keyup.enter="startSession" /></label>
              <label class="field"><span>Language</span><select v-model="trainer.language" :disabled="Boolean(trainer.session.safe_word) || isBusy('session')"><option v-for="item in trainer.languages" :key="item.code" :value="item.code">{{ item.label }}</option></select><small>{{ ttsRoute }}</small></label>
              <label class="field"><span>TTS source</span><select v-model="trainer.ttsMode" :disabled="Boolean(trainer.session.safe_word) || isBusy('session')">
                <option value="hybrid" :disabled="!trainer.languages.find((item) => item.code === trainer.language)?.engines?.includes('piper')">Four-provider ensemble · recommended</option>
                <option value="modern" :disabled="!trainer.languages.find((item) => item.code === trainer.language)?.engines?.some((engine) => engine !== 'piper')">Modern only · no Piper</option>
                <option value="piper" :disabled="!trainer.languages.find((item) => item.code === trainer.language)?.engines?.includes('piper')">Piper only · legacy</option>
              </select><small>Models download once and stay cached.</small></label>
            </div>
            <div class="row form-actions"><button v-if="!trainer.session.safe_word" type="button" class="button primary" :disabled="isBusy('session') || !trainer.phrase.trim()" @click="startSession">{{ isBusy('session') ? "Starting…" : "Start session" }}</button><button v-else type="button" class="button danger" :disabled="isBusy('session')" @click="stopSession">{{ isBusy('session') ? "Stopping…" : (trainer.training.running ? "Stop session + training" : "Stop session") }}</button><button type="button" :disabled="!trainer.phrase.trim()" @click="previewPhrase">System preview</button></div>
          </section>
          <section class="panel">
            <header class="panel-head"><div class="number">2</div><div><h3>Train wake word</h3><p>Personal positives and reviewed false-wake negatives are automatically included.</p></div><span class="pill" :class="trainingStatus.tone">{{ trainingStatus.text }}</span></header>
            <div class="stats"><article><span>Positive samples</span><strong>{{ personalCount }}</strong></article><article><span>Negative samples</span><strong>{{ negativeCount }}</strong></article><article><span>Training format</span><strong class="format-value">16 kHz · mono · WAV</strong></article></div>
            <div class="train-action"><button type="button" class="button primary large" :disabled="!trainer.session.safe_word || trainer.training.running || isBusy('training-start')" @click="startTraining">{{ trainer.training.running ? "Training in progress" : "Start training" }}</button></div>
            <footer class="panel-footer"><span>Training opens the console automatically and continues if the window is closed.</span><button type="button" :disabled="!hasConsole" @click="trainer.consoleOpen = true">Open console</button></footer>
          </section>
        </template>
        <template v-else-if="trainer.activeView === 'auto'">
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
          <section class="panel"><header class="panel-head"><div class="number">1</div><div><h3>Review queue</h3><p>Approve good phrases, keep false positives as negatives, or discard noise.</p></div><button type="button" :disabled="isBusy('captured')" @click="refreshCaptured()">{{ isBusy('captured') ? "Refreshing…" : "Refresh inbox" }}</button></header><div class="stats"><article><span>Inbox</span><strong>{{ trainer.captured.captured_count }}</strong></article><article><span>Reviewed negatives</span><strong>{{ negativeCount }}</strong></article><article><span>Personal samples</span><strong>{{ personalCount }}</strong></article></div></section>
          <section class="panel"><header class="panel-head"><div class="number">2</div><div><h3>Listen + sort</h3><p>Metadata remains visible so borderline detections are easy to understand.</p></div></header>
            <div v-if="!trainer.captured.items?.length" class="empty-state">No captured audio yet. Clips sent by satellites will appear here.</div>
            <div v-else class="audio-list"><article v-for="item in trainer.captured.items" :key="item.saved_as" class="audio-card">
              <header><div><strong>{{ item.original_name || item.saved_as }}</strong><small>{{ formatTimestamp(item.captured_at || item.received_at) }} {{ item.message || "" }}</small></div><span class="pill" :class="captureTone(item).tone">{{ captureTone(item).label }}</span></header>
              <div v-if="metaRows(item).length" class="meta-row"><span v-for="row in metaRows(item)" :key="row">{{ row }}</span></div>
              <div v-if="item.transcript" class="transcript"><b>STT</b> {{ item.transcript }}</div><div v-if="item.auto_review_guided_transcript" class="transcript"><b>Guided wake check</b> {{ item.auto_review_guided_transcript }}</div>
              <audio controls preload="none" :src="itemAudioUrl(item, 'captured')" />
              <footer><span>{{ item.saved_as }} · {{ describeFormat(item.final_format) }}</span><div><button type="button" :disabled="isBusy('review')" @click="reviewCaptured(item, 'approve_personal')">Add positive</button><button type="button" :disabled="isBusy('review')" @click="reviewCaptured(item, 'mark_negative')">Mark negative</button><button type="button" class="button danger ghost" :disabled="isBusy('review')" @click="reviewCaptured(item, 'discard')">Discard</button></div></footer>
            </article></div>
          </section>
        </template>
        <template v-else-if="trainer.activeView === 'samples'">
          <section class="hero samples-hero"><div><span class="eyebrow">Sample library</span><h2>Current Training Samples</h2><p>Audit positives and negatives, trim recordings precisely, and import seed audio.</p></div><span class="pill hero-pill">{{ personalCount + negativeCount }} total</span></section>
          <section class="panel">
            <header class="panel-head sample-head"><div class="number">1</div><div><h3>Saved samples</h3><p>Personal clips are positives. Negative clips are false wakes and hard negatives.</p></div><div class="segment-control"><button type="button" :class="{ active: trainer.sampleBucket === 'personal' }" @click="setBucket('personal')">Personal <b>{{ personalCount }}</b></button><button type="button" :class="{ active: trainer.sampleBucket === 'negative' }" @click="setBucket('negative')">Negative <b>{{ negativeCount }}</b></button></div></header>
            <div class="row toolbar"><button type="button" :disabled="isBusy('samples')" @click="refreshSamples()">Refresh</button><button type="button" class="button danger ghost" :disabled="isBusy('review') || personalCount === 0" @click="clearSamples('personal')">Clear positives</button><button type="button" class="button danger ghost" :disabled="isBusy('review') || negativeCount === 0" @click="clearSamples('negative')">Clear negatives</button></div>
            <div v-if="!selectedSamples.length" class="empty-state">No {{ trainer.sampleBucket }} samples saved yet.</div>
            <div v-else class="audio-list compact-list"><article v-for="item in pagedSamples" :key="item.saved_as" class="audio-card">
              <header><div><strong>{{ item.saved_as }}</strong><small>{{ sampleSubtitle(item) }}</small></div><div class="row"><span v-if="item.trimmed" class="pill warning">Trimmed</span><span class="pill" :class="trainer.sampleBucket === 'personal' ? 'success' : 'error'">{{ trainer.sampleBucket === "personal" ? "Positive" : "Negative" }}</span></div></header>
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
        </template>
        <template v-else-if="trainer.activeView === 'data'">
          <section class="hero data-hero"><div><span class="eyebrow">Local storage</span><h2>Data Management</h2><p>See exactly what the trainer has downloaded, generated, recorded, and produced.</p></div><span class="pill hero-pill">{{ formatBytes(trainer.managedData.total_size_bytes) }} total</span></section>
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
        <template v-else-if="trainer.activeView === 'firmware'">
          <section class="hero firmware-hero"><div><span class="eyebrow">Wake-word catalog</span><h2>Trained Wake Words</h2><p>Copy a local JSON package URL into Tater to switch every native satellite live.</p></div><span class="pill hero-pill" :class="trainer.wakeWords.length ? 'success' : 'warning'">{{ trainer.wakeWords.length ? `${trainer.wakeWords.length} trained` : "Catalog empty" }}</span></section>
          <div class="native-notice"><strong>Tater Native</strong><span>These packages include model metadata and a direct model URL for live satellite updates.</span></div>
          <section class="panel"><header class="panel-head"><div class="number">v1</div><div><h3>Published model URLs</h3><p>URLs stay local and are refreshed after each successful run.</p></div><button type="button" :disabled="isBusy('firmware')" @click="refreshWakeWords()">Refresh</button></header>
            <div v-if="!trainer.wakeWords.length" class="empty-state">Train a wake word and its package will appear here.</div>
            <div v-else class="word-list"><article v-for="word in trainer.wakeWords" :key="word.key || wordJsonUrl(word)"><div><strong>{{ word.label || word.name || "Trained wake word" }}</strong><a v-if="wordJsonUrl(word)" :href="wordJsonUrl(word)" target="_blank" rel="noreferrer">JSON · {{ wordJsonUrl(word) }}</a><span v-else class="muted">JSON package URL unavailable</span><a v-if="wordModelUrl(word)" :href="wordModelUrl(word)" target="_blank" rel="noreferrer">Model · {{ wordModelUrl(word) }}</a><div class="meta-row"><span v-if="word.language">{{ word.language }}</span><span v-if="word.trained_at">{{ formatTimestamp(word.trained_at) }}</span><span v-if="word.recall !== undefined">recall {{ word.recall }}</span></div></div><button type="button" :disabled="!wordJsonUrl(word)" @click="copyWakeWord(wordJsonUrl(word))">Copy URL</button></article></div>
          </section>
        </template>
      </template>
    </main>
    <Teleport to="body">
      <div v-if="trainer.consoleOpen" class="modal-backdrop console-backdrop" @click.self="trainer.consoleOpen = false">
        <section class="modal console-modal" role="dialog" aria-modal="true" aria-label="Training console"><header class="modal-head"><div><span class="eyebrow">Live pipeline</span><h2>Training Console</h2><p>Closing this window does not interrupt training.</p></div><div class="row console-actions"><button v-if="!consoleFollowing" type="button" class="console-follow" @click="scrollConsoleToBottom">Jump to latest</button><span class="pill" :class="trainingStatus.tone">{{ trainingStatus.text }}</span><button type="button" @click="trainer.consoleOpen = false">Close</button></div></header><pre ref="consoleLog" class="console-log" @scroll.passive="onConsoleScroll"><span v-for="(line, index) in consoleLines" :key="`${index}-${line}`" :class="consoleTone(line)">{{ line }}</span></pre></section>
      </div>
    </Teleport>
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
