import { computed, reactive } from "vue";
import { getJson, postJson, putJson, request, type JsonRecord } from "./api";
import type {
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
  ViewName,
  WakeWordItem,
} from "./types";

const emptyTraining = (): TrainingState => ({ running: false, exit_code: null, log_lines: [] });
const emptySamples = (): SamplesPayload => ({ personal: [], negative: [], personal_count: 0, negative_count: 0 });
const emptyCaptured = (): CapturedPayload => ({ items: [], captured_count: 0, personal_count: 0, negative_count: 0 });
const emptyManagedData = (): ManagedDataPayload => ({ items: [], total_size_bytes: 0, total_file_count: 0 });

const defaultAutoForm = (): AutoTrainForm => ({
  enabled: false,
  wake_phrase: "",
  language: "en",
  stt_engine: "faster_whisper",
  minimum_transcript_chars: 2,
  delete_confirmed_wakes: false,
  promote_close_misses: false,
  schedule_hours: 24,
  minimum_new_negatives: 3,
  advertised_base_url: "",
  tater_url: "http://127.0.0.1:8501",
  notify_satellites: true,
});

export const trainer = reactive({
  activeView: "trainer" as ViewName,
  initialized: false,
  busy: new Set<string>(),
  phrase: "",
  language: "en",
  ttsMode: "hybrid",
  languages: [{ code: "en", label: "English (en)", engines: ["omnivoice"] }] as LanguageOption[],
  session: {} as SessionPayload,
  samples: emptySamples(),
  captured: emptyCaptured(),
  training: emptyTraining(),
  auto: {} as AutoTrainPayload,
  autoForm: defaultAutoForm(),
  wakeWords: [] as WakeWordItem[],
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

let autoTimer = 0;
let trainingTimer = 0;

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

function applySession(payload: SessionPayload): void {
  trainer.session = payload || {};
  if (Array.isArray(payload.available_languages) && payload.available_languages.length) {
    trainer.languages = payload.available_languages;
  }
  if (payload.raw_phrase) trainer.phrase = payload.raw_phrase;
  if (payload.language) trainer.language = payload.language;
  if (payload.tts_mode) trainer.ttsMode = payload.tts_mode;
  if (payload.training) trainer.training = payload.training;
}

export async function refreshSession(): Promise<SessionPayload> {
  const payload = await getJson<SessionPayload>("/api/session");
  applySession(payload);
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
      tts_mode: trainer.ttsMode,
    });
    applySession(payload);
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
  if (trainingTimer) {
    window.clearInterval(trainingTimer);
    trainingTimer = 0;
  }
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
  setBusy("review", true);
  try {
    await postJson(`/api/captured_audio/${encodeURIComponent(item.saved_as)}/${action}`);
    await Promise.all([refreshSession(), refreshCaptured(true), refreshSamples(true)]);
    notify(action === "approve_personal" ? "Clip added to personal samples." : action === "mark_negative" ? "Clip marked negative." : "Clip discarded.");
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
  if (!populate) return;
  trainer.autoForm = { ...defaultAutoForm(), ...(payload.config || {}) };
  if (!trainer.autoForm.wake_phrase) trainer.autoForm.wake_phrase = trainer.session.raw_phrase || "";
  if (!trainer.autoForm.language) trainer.autoForm.language = trainer.session.language || "en";
}

export async function refreshAuto(populate = false): Promise<AutoTrainPayload> {
  const payload = await getJson<AutoTrainPayload>("/api/auto_train");
  applyAuto(payload, populate);
  return payload;
}

export async function saveAuto(): Promise<void> {
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
      trainer.consoleOpen = true;
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

export async function refreshManagedData(): Promise<ManagedDataPayload> {
  setBusy("data", true);
  try {
    const payload = await getJson<ManagedDataPayload>("/api/data");
    trainer.managedData = { ...emptyManagedData(), ...payload };
    return payload;
  } finally {
    setBusy("data", false);
  }
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
  await Promise.all([refreshSession(), refreshSamples(true)]);
  let allowNoPersonal = false;
  if (!personalCount.value) {
    allowNoPersonal = window.confirm("No positive samples are saved. Train anyway without personal voices?");
    if (!allowNoPersonal) return;
  }
  setBusy("training-start", true);
  trainer.training = { running: true, exit_code: null, log_lines: ["Waiting for training output…"] };
  trainer.consoleOpen = true;
  try {
    await postJson("/api/train", { allow_no_personal: allowNoPersonal });
    beginTrainingPoll();
  } catch (error) {
    trainer.training = { running: false, exit_code: 1, log_lines: [error instanceof Error ? error.message : String(error)] };
    reportError(error, "Training could not start.");
  } finally {
    setBusy("training-start", false);
  }
}

export function beginTrainingPoll(): void {
  if (trainingTimer) return;
  const poll = async () => {
    try {
      const payload = await getJson<JsonRecord>("/api/train_status");
      trainer.training = { ...emptyTraining(), ...(payload.training || {}) };
      if (!trainer.training.running) {
        window.clearInterval(trainingTimer);
        trainingTimer = 0;
        await Promise.all([refreshSamples(true), refreshWakeWords(true)]);
        notify(trainer.training.exit_code === 0 ? "Training finished successfully." : `Training ended with exit ${trainer.training.exit_code}.`, trainer.training.exit_code === 0 ? "success" : "error");
      }
    } catch {
      // A temporary request failure should not stop the live poll.
    }
  };
  void poll();
  trainingTimer = window.setInterval(() => void poll(), 1500);
}

export async function initializeTrainer(): Promise<void> {
  setBusy("bootstrap", true);
  try {
    await Promise.allSettled([
      refreshSession(),
      refreshSamples(true),
      refreshCaptured(true),
      refreshAuto(true),
      refreshWakeWords(true),
    ]);
    ensureSupportedTtsMode();
    try {
      const payload = await getJson<JsonRecord>("/api/train_status");
      trainer.training = { ...emptyTraining(), ...(payload.training || {}) };
      if (trainer.training.running) {
        trainer.consoleOpen = true;
        beginTrainingPoll();
      }
    } catch {
      // Remaining panels can still function when status is temporarily unavailable.
    }
    autoTimer = window.setInterval(() => {
      if (trainer.activeView === "auto" && !isBusy("auto")) void refreshAuto(false).catch(() => undefined);
    }, 2500);
    trainer.initialized = true;
  } finally {
    setBusy("bootstrap", false);
  }
}

export function disposeTrainer(): void {
  window.clearInterval(autoTimer);
  window.clearInterval(trainingTimer);
  autoTimer = 0;
  trainingTimer = 0;
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
