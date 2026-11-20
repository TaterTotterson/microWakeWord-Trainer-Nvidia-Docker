import { computed, reactive } from "vue";
import { getJson, postJson, putJson, request, type JsonRecord } from "./api";
import { createLiveUpdates, type LiveState } from "./liveUpdates";
import type {
  AccentOption,
  AudioItem,
  AutoTrainForm,
  AutoTrainPayload,
  CapturedPayload,
  LanguageOption,
  ManagedDataItem,
  ManagedDataPayload,
  SampleBucket,
  SamplesPayload,
  SessionPayload,
  ToastState,
  TrainingState,
  TrainingRun,
  ViewName,
  WakeWordLibraryItem,
  WakeWordItem,
} from "./types";

declare global {
  interface Window {
    __wakeWordTrainerMac?: { dataFolderPicker?: boolean };
    webkit?: {
      messageHandlers?: {
        wakeWordTrainerStorage?: { postMessage: (message: unknown) => void };
      };
    };
  }
}

const emptyTraining = (): TrainingState => ({ running: false, exit_code: null, log_lines: [], progress: null });
const emptySamples = (): SamplesPayload => ({ personal: [], negative: [], personal_count: 0, negative_count: 0 });
const emptyCaptured = (): CapturedPayload => ({ items: [], captured_count: 0, personal_count: 0, negative_count: 0 });
const emptyManagedData = (): ManagedDataPayload => ({ items: [], total_size_bytes: 0, total_file_count: 0 });

const defaultAutoForm = (): AutoTrainForm => ({
  enabled: false,
  wake_phrase: "",
  language: "en",
  english_accent: "mixed",
  stt_engine: "faster_whisper",
  minimum_transcript_chars: 2,
  delete_confirmed_wakes: false,
  promote_close_misses: false,
  schedule_hours: 24,
  minimum_new_negatives: 3,
  advertised_base_url: "",
  tater_url: "http://127.0.0.1:8501",
  notify_satellites: true,
  train_microwakeword: true,
  train_openwakeword: false,
  train_nanowakeword_cnn: false,
  train_nanowakeword_gru: false,
  generated_negative_phrases: "",
});

export const trainer = reactive({
  activeView: "home" as ViewName,
  selectedWord: "",
  wizardStep: 0,
  wizardRetrain: false,
  trainingStages: [] as Array<{ stage: string; label: string; status: string }>,
  trainingTargets: [] as string[],
  initialized: false,
  liveState: "connecting" as LiveState,
  busy: new Set<string>(),
  phrase: "",
  language: "en",
  englishAccent: "mixed",
  ttsMode: "hybrid",
  trainMicroWakeWord: true,
  trainOpenWakeWord: false,
  trainNanoWakeWordCnn: false,
  trainNanoWakeWordGru: false,
  generatedNegativePhrases: "",
  languages: [{ code: "en", label: "English (en)", engines: ["omnivoice"] }] as LanguageOption[],
  englishAccents: [
    { code: "mixed", label: "Mixed English" },
    { code: "australian", label: "Australian" },
    { code: "american", label: "American" },
    { code: "british", label: "British" },
    { code: "canadian", label: "Canadian" },
    { code: "irish", label: "Irish" },
    { code: "scottish", label: "Scottish" },
    { code: "new_zealand", label: "New Zealand" },
    { code: "indian", label: "Indian" },
    { code: "south_african", label: "South African" },
  ] as AccentOption[],
  session: {} as SessionPayload,
  samples: emptySamples(),
  captured: emptyCaptured(),
  training: emptyTraining(),
  auto: {} as AutoTrainPayload,
  autoForm: defaultAutoForm(),
  wakeWords: [] as WakeWordItem[],
  wakeWordLibrary: [] as WakeWordLibraryItem[],
  modelHistory: [] as TrainingRun[],
  managedData: emptyManagedData(),
  selectedFiles: [] as File[],
  sampleBucket: "personal" as SampleBucket,
  samplePage: { personal: 0, negative: 0 },
  uploadProgress: 0,
  uploadLabel: "No upload in progress",
  uploadDetail: "Choose files and upload when you are ready.",
  consoleOpen: false,
  taterLinkOpen: false,
  trimItem: null as AudioItem | null,
  trimBucket: "personal" as SampleBucket,
  toast: { message: "", tone: "success", serial: 0 } as ToastState,
});

export const macDataFolderPickerAvailable = Boolean(
  window.__wakeWordTrainerMac?.dataFolderPicker
  && window.webkit?.messageHandlers?.wakeWordTrainerStorage,
);

let liveUpdates: ReturnType<typeof createLiveUpdates> | null = null;
let autoBaseline = defaultAutoForm();
const captureAssignments = new Map<string, unknown>();
const historySignatures = new Map<string, string>();
let lastDataRefresh = 0;

export const personalCount = computed(() => Number(trainer.samples.personal_count ?? trainer.session.takes_received ?? 0));
export const negativeCount = computed(() => Number(trainer.samples.negative_count ?? trainer.captured.negative_count ?? 0));
export const currentLanguage = computed<LanguageOption>(() =>
  trainer.languages.find((item) => item.code === trainer.language) || trainer.languages[0],
);
export const ttsRoute = computed(() => {
  const engines = currentLanguage.value?.engines?.length ? currentLanguage.value.engines : ["omnivoice"];
  const selected = trainer.ttsMode === "piper"
    ? engines.filter((engine) => engine === "piper")
    : trainer.ttsMode === "hybrid"
      ? engines
      : engines.filter((engine) => engine !== "piper");
  const labels: Record<string, string> = { omnivoice: "OmniVoice", qwen3: "Qwen3", moss: "MOSS", piper: "Piper" };
  const quality = trainer.ttsMode === "piper" ? "Legacy" : titleCase(currentLanguage.value?.quality || "experimental");
  return `${selected.map((engine) => labels[engine] || engine).join(" + ") || "Unavailable"} · ${quality}`;
});
export const hasConsole = computed(() => Boolean(
  trainer.training.running || trainer.training.exit_code !== null || trainer.training.log_lines?.length,
));
export const selectedSamples = computed(() => trainer.samples[trainer.sampleBucket] || []);
export const autoLinked = computed(() => Boolean(trainer.auto.trainer_link?.linked));
export const sttEngines = computed<JsonRecord[]>(() => {
  const rows = trainer.auto.stt_engines;
  return Array.isArray(rows) && rows.length
    ? rows
    : [{ id: "faster_whisper", label: "Faster Whisper" }, { id: "parakeet_onnx", label: "Parakeet ONNX" }];
});

function titleCase(value: unknown): string {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function isBusy(name?: string): boolean {
  return name ? trainer.busy.has(name) : trainer.busy.size > 0;
}

function setBusy(name: string, active: boolean): void {
  if (active) trainer.busy.add(name);
  else trainer.busy.delete(name);
}

export function notify(message: unknown, tone: ToastState["tone"] = "success"): void {
  trainer.toast = { message: String(message || ""), tone, serial: trainer.toast.serial + 1 };
}

function reportError(error: unknown, fallback: string): void {
  notify(error instanceof Error ? error.message : fallback, "error");
}

function applyTraining(payload: TrainingState): void {
  trainer.training = payload;
  const progress = payload.progress;
  if (progress?.stage) {
    const previous = trainer.trainingStages.at(-1);
    if (!previous || previous.stage !== progress.stage) {
      if (previous && previous.status === "running") previous.status = progress.status === "failed" ? "failed" : "complete";
      trainer.trainingStages.push({ stage: progress.stage, label: progress.label, status: progress.status || (payload.running ? "running" : payload.exit_code === 0 ? "complete" : "failed") });
      if (trainer.trainingStages.length > 40) trainer.trainingStages.shift();
    } else {
      previous.label = progress.label;
      previous.status = progress.status || (payload.running ? "running" : payload.exit_code === 0 ? "complete" : "failed");
    }
  }
}

function applySession(payload: SessionPayload, populate = true, includeTraining = true): void {
  const previous = trainer.session;
  trainer.session = payload || {};
  if (Array.isArray(payload.available_languages) && payload.available_languages.length) {
    trainer.languages = payload.available_languages;
  }
  if (Array.isArray(payload.available_english_accents) && payload.available_english_accents.length) {
    trainer.englishAccents = payload.available_english_accents;
  }
  if (payload.raw_phrase && (populate || trainer.phrase === (previous.raw_phrase || ""))) trainer.phrase = payload.raw_phrase;
  if (payload.language && (populate || trainer.language === (previous.language || "en"))) trainer.language = payload.language;
  if (payload.english_accent && (populate || trainer.englishAccent === (previous.english_accent || "mixed"))) trainer.englishAccent = payload.english_accent;
  if (payload.tts_mode && (populate || trainer.ttsMode === (previous.tts_mode || "hybrid"))) trainer.ttsMode = payload.tts_mode;
  if (populate && includeTraining && payload.training) applyTraining(payload.training);
}

export async function refreshSession(populate = true): Promise<SessionPayload> {
  const payload = await getJson<SessionPayload>("/api/session");
  applySession(payload, populate, !liveUpdates);
  return payload;
}

export async function startSession(): Promise<void> {
  if (!trainer.phrase.trim()) {
    notify("Enter a wake phrase first.", "warning");
    return;
  }
  setBusy("session", true);
  try {
    const payload = await postJson<SessionPayload>("/api/start_session", {
      phrase: trainer.phrase.trim(),
      language: trainer.language,
      english_accent: trainer.englishAccent,
      tts_mode: trainer.ttsMode,
    });
    applySession(payload);
    await refreshWakeWordLibrary(true);
    notify(`Session ${payload.safe_word || "started"} is ready.`);
  } catch (error) {
    reportError(error, "Session failed to start.");
  } finally {
    setBusy("session", false);
  }
}

export async function stopSession(): Promise<void> {
  const wasTraining = Boolean(trainer.training.running);
  if (wasTraining && !window.confirm("Training is running. Stop training cleanly and end this session?")) {
    return;
  }
  setBusy("session", true);
  try {
    const payload = await postJson<SessionPayload>("/api/stop_session");
    applySession(payload);
    notify(wasTraining ? "Training stopped cleanly and the session ended." : "Session ended. You can edit the wake phrase now.");
  } catch (error) {
    if (wasTraining) beginTrainingPoll();
    reportError(error, "Session could not be stopped.");
  } finally {
    setBusy("session", false);
  }
}

export function previewPhrase(): void {
  if (!trainer.phrase.trim() || !("speechSynthesis" in window)) return;
  const utterance = new SpeechSynthesisUtterance(trainer.phrase.trim());
  utterance.lang = trainer.language;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

export function ensureSupportedTtsMode(): void {
  const engines = currentLanguage.value?.engines || [];
  const modern = engines.some((engine) => engine !== "piper");
  const piper = engines.includes("piper");
  if (trainer.ttsMode === "modern" && !modern) trainer.ttsMode = "piper";
  if (trainer.ttsMode === "hybrid" && !(modern && piper)) trainer.ttsMode = modern ? "modern" : "piper";
  if (trainer.ttsMode === "piper" && !piper) trainer.ttsMode = "modern";
  if (trainer.language !== "en" || trainer.ttsMode === "piper") trainer.englishAccent = "mixed";
}

export async function refreshSamples(quiet = false): Promise<SamplesPayload> {
  if (!quiet) setBusy("samples", true);
  try {
    const payload = await getJson<SamplesPayload>("/api/samples");
    trainer.samples = { ...emptySamples(), ...payload };
    for (const bucket of ["personal", "negative"] as const) {
      const lastPage = Math.max(0, Math.ceil((trainer.samples[bucket]?.length || 0) / 50) - 1);
      trainer.samplePage[bucket] = Math.min(trainer.samplePage[bucket], lastPage);
    }
    return payload;
  } finally {
    if (!quiet) setBusy("samples", false);
  }
}

export async function refreshCaptured(quiet = false): Promise<CapturedPayload> {
  if (!quiet) setBusy("captured", true);
  try {
    const payload = await getJson<CapturedPayload>("/api/captured_audio");
    const current = new Map(trainer.captured.items.map(item => [`${item.capture_scope || ""}/${item.saved_as}`, item]));
    const nextAssignments = new Map<string, unknown>();
    for (const item of payload.items || []) {
      const key = `${item.capture_scope || ""}/${item.saved_as}`;
      const local = current.get(key);
      nextAssignments.set(key, item.review_wake_word);
      if (local && captureAssignments.has(key) && local.review_wake_word !== captureAssignments.get(key)) {
        item.review_wake_word = local.review_wake_word;
      }
    }
    captureAssignments.clear();
    nextAssignments.forEach((value, key) => captureAssignments.set(key, value));
    trainer.captured = { ...emptyCaptured(), ...payload };
    return payload;
  } finally {
    if (!quiet) setBusy("captured", false);
  }
}

export function selectFiles(event: Event): void {
  const input = event.target as HTMLInputElement;
  trainer.selectedFiles = Array.from(input.files || []);
}

function uploadOne(file: File, index: number, total: number): Promise<JsonRecord> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const data = new FormData();
    data.append("file", file, file.name);
    xhr.open("POST", "/api/upload_personal_sample");
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      trainer.uploadProgress = Math.round(((index + event.loaded / event.total) / total) * 100);
      trainer.uploadLabel = `Uploading ${file.name} (${index + 1}/${total})`;
      trainer.uploadDetail = "Sending and normalizing the recording.";
    };
    xhr.onload = () => {
      const body = xhr.response || {};
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.error || `Upload failed for ${file.name}`));
    };
    xhr.onerror = () => reject(new Error(`Upload failed for ${file.name}`));
    xhr.send(data);
  });
}

export async function uploadSelectedFiles(input?: HTMLInputElement | null): Promise<void> {
  if (!trainer.session.safe_word) {
    notify("Start a trainer session before uploading samples.", "warning");
    return;
  }
  if (!trainer.selectedFiles.length) return;
  setBusy("upload", true);
  try {
    const files = [...trainer.selectedFiles];
    for (let index = 0; index < files.length; index += 1) await uploadOne(files[index], index, files.length);
    trainer.uploadProgress = 100;
    trainer.uploadLabel = "Upload complete";
    trainer.uploadDetail = `${files.length} sample${files.length === 1 ? "" : "s"} saved in the required training format.`;
    trainer.selectedFiles = [];
    if (input) input.value = "";
    await Promise.all([refreshSession(), refreshSamples(true)]);
    notify("Personal samples uploaded.");
  } catch (error) {
    trainer.uploadProgress = 0;
    reportError(error, "Sample upload failed.");
  } finally {
    setBusy("upload", false);
  }
}

export async function reviewCaptured(item: AudioItem, action: "approve_personal" | "mark_negative" | "discard"): Promise<void> {
  if (action === "discard" && !window.confirm(`Discard ${item.saved_as} from the captured-audio inbox?`)) return;
  const captureWord = String(item.capture_scope || "").trim();
  const targetWord = String(item.review_wake_word || item.dataset_wake_word || "").trim();
  if (action !== "discard" && !targetWord) {
    notify("Choose the saved wake word this clip should train first.", "warning");
    return;
  }
  const query = new URLSearchParams();
  if (captureWord) query.set("capture_word", captureWord);
  if (action !== "discard") query.set("target_word", targetWord);
  setBusy("review", true);
  try {
    await postJson(`/api/captured_audio/${encodeURIComponent(item.saved_as)}/${action}?${query.toString()}`);
    await Promise.all([refreshSession(), refreshCaptured(true), refreshSamples(true)]);
    notify(action === "approve_personal" ? `Clip added to ${targetWord} positives.` : action === "mark_negative" ? `Clip added to ${targetWord} negatives.` : "Clip discarded.");
  } catch (error) {
    reportError(error, "Review action failed.");
  } finally {
    setBusy("review", false);
  }
}

export async function removeSample(item: AudioItem, bucket: SampleBucket): Promise<void> {
  if (!window.confirm(`Remove ${item.saved_as} from ${bucket} samples?`)) return;
  setBusy("review", true);
  try {
    await request(`/api/samples/${bucket}/${encodeURIComponent(item.saved_as)}`, { method: "DELETE" });
    await refreshSamples(true);
    notify("Sample removed.");
  } catch (error) {
    reportError(error, "Sample removal failed.");
  } finally {
    setBusy("review", false);
  }
}

export async function revertSample(item: AudioItem, bucket: SampleBucket): Promise<void> {
  if (!window.confirm(`Revert ${item.saved_as} to its pre-trim version?`)) return;
  const form = new FormData();
  form.append("bucket", bucket);
  form.append("file_name", item.saved_as);
  setBusy("review", true);
  try {
    await request("/api/samples/revert", { method: "POST", body: form });
    await refreshSamples(true);
    notify("Original sample restored.");
  } catch (error) {
    reportError(error, "Sample revert failed.");
  } finally {
    setBusy("review", false);
  }
}

export async function clearSamples(bucket: SampleBucket): Promise<void> {
  const count = bucket === "personal" ? personalCount.value : negativeCount.value;
  if (!count || !window.confirm(`Clear ${count} ${bucket} sample${count === 1 ? "" : "s"}?`)) return;
  setBusy("review", true);
  try {
    await postJson(bucket === "personal" ? "/api/reset_recordings" : "/api/reset_negative_samples");
    await Promise.all([refreshSession(), refreshSamples(true), refreshCaptured(true)]);
    notify(`${titleCase(bucket)} samples cleared.`);
  } catch (error) {
    reportError(error, "Samples could not be cleared.");
  } finally {
    setBusy("review", false);
  }
}

function applyAuto(payload: AutoTrainPayload, populate: boolean): void {
  trainer.auto = payload || {};
  const incoming = { ...defaultAutoForm(), ...(payload.config || {}) };
  if (Array.isArray(payload.config?.generated_negative_phrases)) {
    incoming.generated_negative_phrases = payload.config.generated_negative_phrases.join("\n");
  }
  if (!incoming.wake_phrase) incoming.wake_phrase = trainer.session.raw_phrase || "";
  if (!incoming.language) incoming.language = trainer.session.language || "en";
  if (!incoming.english_accent) incoming.english_accent = trainer.session.english_accent || "mixed";
  if (!String(incoming.language).toLowerCase().startsWith("en")) incoming.english_accent = "mixed";
  for (const key of Object.keys(incoming) as Array<keyof AutoTrainForm>) {
    if (populate || trainer.autoForm[key] === autoBaseline[key]) Object.assign(trainer.autoForm, { [key]: incoming[key] });
  }
  autoBaseline = { ...incoming };
}

export async function refreshAuto(populate = false): Promise<AutoTrainPayload> {
  const payload = await getJson<AutoTrainPayload>("/api/auto_train");
  applyAuto(payload, populate);
  return payload;
}

export async function saveAuto(): Promise<void> {
  if (![
    trainer.autoForm.train_microwakeword,
    trainer.autoForm.train_openwakeword,
    trainer.autoForm.train_nanowakeword_cnn,
    trainer.autoForm.train_nanowakeword_gru,
  ].some(Boolean)) {
    notify("Select at least one model for Auto Training.", "warning");
    return;
  }
  setBusy("auto", true);
  try {
    const payload = await putJson<AutoTrainPayload>("/api/auto_train", trainer.autoForm);
    applyAuto(payload, true);
    notify(payload.config?.enabled ? "Auto Training saved and enabled." : "Auto Training saved.");
  } catch (error) {
    reportError(error, "Auto Training settings failed to save.");
  } finally {
    setBusy("auto", false);
  }
}

export async function runAutoAction(action: "review_now" | "train_now" | "notify_now"): Promise<void> {
  setBusy("auto", true);
  try {
    const payload = await postJson<AutoTrainPayload>("/api/auto_train/action", { action });
    applyAuto(payload, false);
    if (action === "train_now") {
      trainer.trainingStages = [];
      const config = trainer.auto.config || trainer.autoForm;
      trainer.trainingTargets = [
        ...(config.train_microwakeword ? ["MicroWakeWord"] : []),
        ...(config.train_openwakeword ? ["openWakeWord"] : []),
        ...(config.train_nanowakeword_cnn ? ["NanoWakeWord CNN"] : []),
        ...(config.train_nanowakeword_gru ? ["NanoWakeWord GRU"] : []),
      ];
      trainer.training = { ...emptyTraining(), running: true };
      trainer.activeView = "training";
      beginTrainingPoll();
    }
    notify(action === "review_now" ? `${Number(payload.queued || 0)} clips queued for review.` : action === "train_now" ? "Training started." : "Wake word published.");
  } catch (error) {
    reportError(error, "Auto Training action failed.");
  } finally {
    setBusy("auto", false);
  }
}

export async function claimTater(taterUrl: string, pairingCode: string): Promise<boolean> {
  setBusy("link", true);
  try {
    await postJson("/api/tater_link/claim", { tater_url: taterUrl.trim(), pairing_code: pairingCode.trim() });
    trainer.autoForm.tater_url = taterUrl.trim();
    await refreshAuto(false);
    notify("Trainer linked securely to Tater.");
    return true;
  } catch (error) {
    reportError(error, "Tater link failed.");
    return false;
  } finally {
    setBusy("link", false);
  }
}

export async function unlinkTater(): Promise<void> {
  if (!window.confirm("Unlink this trainer from Tater?")) return;
  setBusy("auto", true);
  try {
    await postJson("/api/tater_link/unlink");
    await refreshAuto(false);
    notify("Trainer unlinked from Tater.", "warning");
  } catch (error) {
    reportError(error, "Tater unlink failed.");
  } finally {
    setBusy("auto", false);
  }
}

export async function refreshWakeWords(quiet = false): Promise<void> {
  if (!quiet) setBusy("firmware", true);
  try {
    const payload = await getJson<JsonRecord>("/api/trained_wake_words/catalog");
    trainer.wakeWords = Array.isArray(payload.wake_words) ? payload.wake_words : [];
  } finally {
    if (!quiet) setBusy("firmware", false);
  }
}

export async function refreshWakeWordLibrary(quiet = false, live = false): Promise<void> {
  if (!quiet) setBusy("library", true);
  try {
    const payload = await getJson<JsonRecord>(`/api/wake_word_library${live ? "?include_usage=false" : ""}`);
    const previous = new Map(trainer.wakeWordLibrary.map(word => [word.safe_word, word]));
    trainer.wakeWordLibrary = Array.isArray(payload.words)
      ? payload.words.map((word: WakeWordLibraryItem) => ({ ...previous.get(word.safe_word), ...word })) : [];
    const histories = await Promise.all(trainer.wakeWordLibrary.map(async (word) => {
      const signature = JSON.stringify([word.run_count, word.published_run_id, word.latest_run]);
      if (live && historySignatures.get(word.safe_word) === signature) return trainer.modelHistory.filter(run => run.safe_word === word.safe_word);
      try {
        const result = await getJson<JsonRecord>(`/api/wake_word_library/${encodeURIComponent(word.safe_word)}/history`);
        historySignatures.set(word.safe_word, signature);
        return Array.isArray(result.runs) ? result.runs as TrainingRun[] : [];
      } catch {
        return trainer.modelHistory.filter(run => run.safe_word === word.safe_word);
      }
    }));
    trainer.modelHistory = histories.flat().sort((left, right) =>
      String(right.completed_at || "").localeCompare(String(left.completed_at || "")),
    );
  } finally {
    if (!quiet) setBusy("library", false);
  }
}

export async function useSavedWord(word: WakeWordLibraryItem): Promise<void> {
  if (trainer.session.safe_word && trainer.session.safe_word !== word.safe_word) {
    notify("Stop the current session before switching saved wake words.", "warning");
    return;
  }
  if (trainer.session.safe_word === word.safe_word) {
    notify(`${word.phrase} is already active.`);
    return;
  }
  trainer.phrase = word.phrase;
  if (word.language) trainer.language = word.language;
  if (word.english_accent) trainer.englishAccent = word.english_accent;
  if (word.tts_mode) trainer.ttsMode = word.tts_mode;
  ensureSupportedTtsMode();
  await startSession();
  await Promise.all([refreshSamples(true), refreshWakeWordLibrary(true)]);
}

export async function publishSavedVersion(word: WakeWordLibraryItem, run: TrainingRun): Promise<void> {
  if (trainer.training.running) { notify("Wait for training to finish before changing the published version.", "warning"); return; }
  setBusy("version-publish", true);
  try {
    await postJson(`/api/wake_word_library/${encodeURIComponent(word.safe_word)}/runs/${encodeURIComponent(run.run_id)}/publish`);
    await Promise.all([refreshWakeWords(true), refreshWakeWordLibrary(true)]);
    notify(`Version ${run.version || run.run_id} is now current for ${word.phrase}. Publish through Tater when you want satellites to use it.`);
  } catch (error) {
    reportError(error, "The published version could not be changed.");
  } finally { setBusy("version-publish", false); }
}

export async function refreshManagedData(quiet = false): Promise<ManagedDataPayload> {
  if (!quiet) setBusy("data", true);
  try {
    const payload = await getJson<ManagedDataPayload>("/api/data");
    trainer.managedData = { ...emptyManagedData(), ...payload };
    lastDataRefresh = Date.now();
    return payload;
  } finally {
    if (!quiet) setBusy("data", false);
  }
}

export function chooseDataDirectory(): void {
  const handler = window.webkit?.messageHandlers?.wakeWordTrainerStorage;
  if (!macDataFolderPickerAvailable || !handler) {
    notify("Open this page in the WakeWord Trainer Mac app to change the data folder.", "warning");
    return;
  }
  handler.postMessage({ action: "choose" });
}

export async function deleteManagedData(item: ManagedDataItem): Promise<void> {
  if (!item.file_count) return;
  const details = `${formatBytes(item.size_bytes)} · ${Number(item.file_count).toLocaleString()} file${item.file_count === 1 ? "" : "s"}`;
  const rebuild = item.rebuild_note ? `\n\n${item.rebuild_note}` : "";
  if (!window.confirm(`Permanently delete ${item.label} (${details})?${rebuild}\n\nThis cannot be undone.`)) return;
  setBusy("data-delete", true);
  try {
    const payload = await request<ManagedDataPayload>(`/api/data/${encodeURIComponent(item.id)}`, { method: "DELETE" });
    trainer.managedData = { ...emptyManagedData(), ...payload };
    await Promise.allSettled([
      refreshSession(),
      refreshSamples(true),
      refreshCaptured(true),
      refreshWakeWords(true),
    ]);
    notify(`${item.label} deleted. ${formatBytes(item.size_bytes)} released.`);
  } catch (error) {
    reportError(error, `${item.label} could not be deleted.`);
  } finally {
    setBusy("data-delete", false);
  }
}

export async function copyWakeWord(url: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(url);
    notify("Wake-word JSON URL copied.");
  } catch (error) {
    reportError(error, "Clipboard unavailable.");
  }
}

export async function startTraining(): Promise<void> {
  if (![
    trainer.trainMicroWakeWord,
    trainer.trainOpenWakeWord,
    trainer.trainNanoWakeWordCnn,
    trainer.trainNanoWakeWordGru,
  ].some(Boolean)) {
    notify("Select at least one wake-word model to train.", "warning");
    return;
  }
  await Promise.all([refreshSession(), refreshSamples(true)]);
  let allowNoPersonal = false;
  if (!personalCount.value) {
    allowNoPersonal = window.confirm("No positive samples are saved. Train anyway without personal voices?");
    if (!allowNoPersonal) return;
  }
  trainer.trainingTargets = [
    ...(trainer.trainMicroWakeWord ? ["MicroWakeWord", "Wake boundary"] : []),
    ...(trainer.trainOpenWakeWord ? ["openWakeWord"] : []),
    ...(trainer.trainNanoWakeWordCnn ? ["NanoWakeWord CNN"] : []),
    ...(trainer.trainNanoWakeWordGru ? ["NanoWakeWord GRU"] : []),
  ];
  setBusy("training-start", true);
  trainer.trainingStages = [];
  trainer.training = { running: true, exit_code: null, log_lines: ["Waiting for training output…"] };
  trainer.activeView = "training";
  try {
    await postJson("/api/train", {
      allow_no_personal: allowNoPersonal,
      train_microwakeword: trainer.trainMicroWakeWord,
      train_openwakeword: trainer.trainOpenWakeWord,
      train_nanowakeword_cnn: trainer.trainNanoWakeWordCnn,
      train_nanowakeword_gru: trainer.trainNanoWakeWordGru,
      generated_negative_phrases: trainer.generatedNegativePhrases,
    });
    beginTrainingPoll();
  } catch (error) {
    trainer.training = { running: false, exit_code: 1, log_lines: [error instanceof Error ? error.message : String(error)] };
    reportError(error, "Training could not start.");
  } finally {
    setBusy("training-start", false);
  }
}

export async function startBoundaryTraining(): Promise<void> {
  if (!trainer.session.safe_word) {
    notify("Select or start a saved wake word first.", "warning");
    return;
  }
  const saved = trainer.wakeWordLibrary.find((word) => word.safe_word === trainer.session.safe_word);
  if (!saved?.boundary_only_ready) {
    notify("Run one full MicroWakeWord training for this word before training its boundary only.", "warning");
    return;
  }
  trainer.trainingTargets = ["Wake boundary"];
  setBusy("boundary-start", true);
  trainer.trainingStages = [];
  trainer.training = { running: true, exit_code: null, log_lines: ["Preparing cached data for boundary-only training…"] };
  trainer.activeView = "training";
  try {
    await postJson("/api/train", {
      allow_no_personal: true,
      train_boundary_only: true,
      train_microwakeword: false,
      train_openwakeword: false,
      train_nanowakeword_cnn: false,
      train_nanowakeword_gru: false,
      generated_negative_phrases: trainer.generatedNegativePhrases,
    });
    beginTrainingPoll();
  } catch (error) {
    trainer.training = { running: false, exit_code: 1, log_lines: [error instanceof Error ? error.message : String(error)] };
    reportError(error, "Boundary training could not start.");
  } finally {
    setBusy("boundary-start", false);
  }
}

export async function retryOpenWakeWord(safeWord = ""): Promise<void> {
  trainer.trainingTargets = ["openWakeWord"];
  setBusy("oww-retry", true);
  trainer.trainingStages = [];
  trainer.training = { running: true, exit_code: null, log_lines: ["Starting openWakeWord-only retry…"] };
  trainer.activeView = "training";
  try {
    await postJson("/api/train_openwakeword", { safe_word: safeWord });
    beginTrainingPoll();
  } catch (error) {
    trainer.training = { running: false, exit_code: 1, log_lines: [error instanceof Error ? error.message : String(error)] };
    reportError(error, "openWakeWord retry could not start.");
  } finally {
    setBusy("oww-retry", false);
  }
}

export async function revalidateNanoWakeWord(architecture: string, safeWord = ""): Promise<void> {
  const normalized = architecture.toLowerCase();
  trainer.trainingTargets = [`NanoWakeWord ${normalized.toUpperCase()}`];
  setBusy("nww-revalidate", true);
  trainer.trainingStages = [];
  trainer.training = { running: true, exit_code: null, log_lines: [`Revalidating existing NanoWakeWord ${normalized.toUpperCase()} candidate…`] };
  trainer.activeView = "training";
  try {
    await postJson("/api/revalidate_nanowakeword", { architecture: normalized, safe_word: safeWord });
    beginTrainingPoll();
  } catch (error) {
    trainer.training = { running: false, exit_code: 1, log_lines: [error instanceof Error ? error.message : String(error)] };
    reportError(error, "NanoWakeWord candidate revalidation could not start.");
  } finally {
    setBusy("nww-revalidate", false);
  }
}

function applyLiveTraining(payload: TrainingState): void {
  // Local start/stop requests own their optimistic state until they finish.
  if (isBusy("session") || ["training-start", "boundary-start", "oww-retry", "nww-revalidate"].some(isBusy)) return;
  const wasRunning = trainer.training.running;
  if (!wasRunning && payload.running) {
    trainer.trainingStages = [];
    trainer.trainingTargets = [];
  }
  applyTraining({ ...emptyTraining(), ...payload });
  if (wasRunning && !payload.running) {
    liveUpdates?.refresh();
    if (payload.exit_code !== null) notify(payload.exit_code === 0 ? "Training finished successfully." : `Training ended with exit ${payload.exit_code}.`, payload.exit_code === 0 ? "success" : "error");
  }
}

async function synchronizeLive(options: { status: boolean; panels: boolean }): Promise<void> {
  // Mutations already refresh their own panels. Let the next tick reconcile them.
  if (isBusy()) return;
  const tasks: Promise<unknown>[] = [];
  if (options.status) tasks.push(getJson<JsonRecord>("/api/train_status").then(payload => applyLiveTraining(payload.training || emptyTraining())));
  if (options.panels) {
    tasks.push(refreshSession(false), refreshSamples(true), refreshCaptured(true), refreshAuto(false), refreshWakeWords(true), refreshWakeWordLibrary(true, true));
    // Storage inspection walks large feature directories; never run it on every event.
    if (trainer.activeView === "data" && Date.now() - lastDataRefresh >= 30000) {
      lastDataRefresh = Date.now();
      void refreshManagedData(true).catch(() => undefined);
    }
  }
  await Promise.allSettled(tasks);
}

export function beginTrainingPoll(): void {
  liveUpdates?.refresh();
}

let lifecycle = 0;
export async function initializeTrainer(): Promise<void> {
  const generation = ++lifecycle;
  setBusy("bootstrap", true);
  try {
    await Promise.allSettled([
      refreshSession(),
      refreshSamples(true),
      refreshCaptured(true),
      refreshAuto(true),
      refreshWakeWords(true),
      refreshWakeWordLibrary(true),
    ]);
    ensureSupportedTtsMode();
    if (trainer.training.running) trainer.activeView = "training";
    trainer.initialized = true;
  } finally {
    setBusy("bootstrap", false);
    if (generation === lifecycle) {
      liveUpdates?.stop();
      liveUpdates = createLiveUpdates({
        synchronize: synchronizeLive,
        training: payload => applyLiveTraining(payload as TrainingState),
        state: state => { trainer.liveState = state; },
      });
      liveUpdates.start();
    }
  }
}

export function disposeTrainer(): void {
  lifecycle += 1;
  liveUpdates?.stop();
  liveUpdates = null;
}

export function formatTimestamp(value: unknown): string {
  if (!value) return "";
  const parsed = new Date(String(value));
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
}

export function formatBytes(value: unknown): string {
  const bytes = Math.max(0, Number(value) || 0);
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && amount >= 1024; index += 1) {
    amount /= 1024;
    unit = units[index];
  }
  return `${amount >= 10 ? amount.toFixed(1) : amount.toFixed(2)} ${unit}`;
}

export function describeFormat(info: JsonRecord | undefined): string {
  if (!info) return "16 kHz · mono · 16-bit WAV";
  const rate = Number(info.sample_rate || info.sample_rate_hz || 16000);
  const channels = Number(info.channels || 1) === 1 ? "mono" : `${info.channels} channels`;
  const bits = Number(info.bits_per_sample || info.sample_width_bits || 16);
  return `${Math.round(rate / 1000)} kHz · ${channels} · ${bits}-bit`;
}

export function captureTone(item: AudioItem): { label: string; tone: string } {
  if (item.blocked_by_vad) return { label: "Blocked by VAD", tone: "warning" };
  const type = String(item.event_type || "").toLowerCase();
  if (type.includes("close")) return { label: item.capture_label || "Close miss", tone: "warning" };
  if (type.includes("false")) return { label: item.capture_label || "False trigger", tone: "error" };
  if (type.includes("wake") || type.includes("detect")) return { label: item.capture_label || "Wake trigger", tone: "success" };
  return { label: item.capture_label || "Captured", tone: "neutral" };
}

export function itemAudioUrl(item: AudioItem, bucket: SampleBucket | "captured"): string {
  return item.audio_url || `/api/audio/${bucket}/${encodeURIComponent(item.saved_as)}`;
}
