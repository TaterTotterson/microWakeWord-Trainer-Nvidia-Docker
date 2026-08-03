<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from "vue";
import { request, type JsonRecord } from "../api";
import { notify, refreshSamples, trainer } from "../trainerStore";

const canvas = ref<HTMLCanvasElement | null>(null);
const audioBuffer = ref<AudioBuffer | null>(null);
const duration = ref(0);
const start = ref(0);
const end = ref(0);
const vadSegments = ref<Array<{ start: number; end: number }>>([]);
const loading = ref(false);
const saving = ref(false);

watch(() => trainer.trimItem, async (item) => {
  if (!item) {
    audioBuffer.value = null;
    return;
  }
  loading.value = true;
  try {
    const url = `/api/audio/${encodeURIComponent(trainer.trimBucket)}/${encodeURIComponent(item.saved_as)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error("Audio could not be loaded.");
    const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
    const context = new AudioContextCtor() as AudioContext;
    audioBuffer.value = await context.decodeAudioData(await response.arrayBuffer());
    duration.value = audioBuffer.value.duration;
    start.value = 0;
    end.value = duration.value;
    await context.close();
    try {
      const vad = await request<JsonRecord>(`/api/samples/${encodeURIComponent(trainer.trimBucket)}/${encodeURIComponent(item.saved_as)}/vad`, { method: "POST" });
      vadSegments.value = Array.isArray(vad.segments) ? vad.segments : [];
      if (vadSegments.value.length) {
        start.value = Math.max(0, Number(vadSegments.value[0].start || 0));
        end.value = Math.min(duration.value, Number(vadSegments.value[0].end || duration.value));
      }
    } catch {
      vadSegments.value = [];
    }
    await nextTick();
    draw();
  } catch (error) {
    notify(error instanceof Error ? error.message : "Audio could not be loaded.", "error");
    close();
  } finally {
    loading.value = false;
  }
}, { immediate: true });

watch([start, end], () => draw());

function close(): void {
  trainer.trimItem = null;
  audioBuffer.value = null;
  vadSegments.value = [];
}

function selectFirstVad(): void {
  const segment = vadSegments.value[0];
  if (!segment) return;
  start.value = Number(segment.start);
  end.value = Number(segment.end);
}

function draw(): void {
  const target = canvas.value;
  const buffer = audioBuffer.value;
  if (!target || !buffer || !duration.value) return;
  const rect = target.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = window.devicePixelRatio || 1;
  target.width = Math.round(rect.width * dpr);
  target.height = Math.round(rect.height * dpr);
  const context = target.getContext("2d");
  if (!context) return;
  context.scale(dpr, dpr);
  const width = rect.width;
  const height = rect.height;
  const middle = height / 2;
  const samples = buffer.getChannelData(0);
  const step = Math.max(1, Math.floor(samples.length / width));
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(222, 218, 212, .24)";
  context.lineWidth = 1;
  context.beginPath();
  for (let x = 0; x < width; x += 1) {
    let minimum = 1;
    let maximum = -1;
    for (let offset = 0; offset < step; offset += 1) {
      const value = samples[Math.floor(x) * step + offset] || 0;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    context.moveTo(x, middle + minimum * middle * 0.84);
    context.lineTo(x, middle + maximum * middle * 0.84);
  }
  context.stroke();
  const from = (start.value / duration.value) * width;
  const to = (end.value / duration.value) * width;
  context.fillStyle = "rgba(8, 8, 9, .66)";
  context.fillRect(0, 0, from, height);
  context.fillRect(to, 0, width - to, height);
  context.fillStyle = "rgba(255, 145, 52, .12)";
  context.fillRect(from, 0, to - from, height);
  context.strokeStyle = "#ff9134";
  context.lineWidth = 2;
  for (const x of [from, to]) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  context.strokeStyle = "rgba(68, 225, 165, .55)";
  for (const segment of vadSegments.value) {
    const x = (segment.start / duration.value) * width;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
}

function playSelection(): void {
  const buffer = audioBuffer.value;
  if (!buffer) return;
  const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
  const context = new AudioContextCtor() as AudioContext;
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start(0, start.value, Math.max(0.01, end.value - start.value));
  source.onended = () => void context.close();
}

async function wavBlob(): Promise<Blob> {
  const buffer = audioBuffer.value;
  if (!buffer) throw new Error("Audio is not loaded.");
  const startSample = Math.floor(start.value * buffer.sampleRate);
  const endSample = Math.min(Math.floor(end.value * buffer.sampleRate), buffer.length);
  const targetRate = 16000;
  let pcm: Float32Array;
  if (buffer.sampleRate === targetRate) {
    pcm = buffer.getChannelData(0).slice(startSample, endSample);
  } else {
    const frames = Math.max(1, Math.floor((endSample - startSample) * targetRate / buffer.sampleRate));
    const offline = new OfflineAudioContext(1, frames, targetRate);
    const source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start(0, start.value, end.value - start.value);
    pcm = (await offline.startRendering()).getChannelData(0);
  }
  const output = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(output);
  view.setUint32(0, 0x52494646, false);
  view.setUint32(4, 36 + pcm.length * 2, true);
  view.setUint32(8, 0x57415645, false);
  view.setUint32(12, 0x666d7420, false);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, targetRate, true);
  view.setUint32(28, targetRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  view.setUint32(36, 0x64617461, false);
  view.setUint32(40, pcm.length * 2, true);
  for (let index = 0; index < pcm.length; index += 1) {
    view.setInt16(44 + index * 2, Math.max(-32768, Math.min(32767, Math.round(pcm[index] * 32767))), true);
  }
  return new Blob([output], { type: "audio/wav" });
}

async function save(): Promise<void> {
  const item = trainer.trimItem;
  if (!item) return;
  saving.value = true;
  try {
    const form = new FormData();
    form.append("file", await wavBlob(), "trimmed.wav");
    form.append("bucket", trainer.trimBucket);
    form.append("source_file", item.saved_as);
    form.append("start_time", start.value.toFixed(3));
    form.append("end_time", end.value.toFixed(3));
    const result = await request<JsonRecord>("/api/samples/trim", { method: "POST", body: form });
    close();
    await refreshSamples(true);
    notify(result.message || "Trimmed sample saved.");
  } catch (error) {
    notify(error instanceof Error ? error.message : "Trim failed.", "error");
  } finally {
    saving.value = false;
  }
}

function redraw(): void {
  if (trainer.trimItem) draw();
}

window.addEventListener("resize", redraw);
onBeforeUnmount(() => window.removeEventListener("resize", redraw));
</script>

<template>
  <Teleport to="body">
    <div v-if="trainer.trimItem" class="modal-backdrop" @click.self="close">
      <section class="modal trim-modal" role="dialog" aria-modal="true" aria-label="Trim audio">
        <header class="modal-head">
          <div><span class="eyebrow">Audio editor</span><h2>Trim {{ trainer.trimItem.saved_as }}</h2></div>
          <button type="button" class="button ghost" @click="close">Close</button>
        </header>
        <p class="muted">Keep the spoken wake phrase and remove excess silence or noise. VAD markers appear in green.</p>
        <div v-if="loading" class="empty-state">Loading waveform…</div>
        <template v-else>
          <canvas ref="canvas" class="waveform" />
          <div class="range-grid">
            <label><span>Start · {{ start.toFixed(2) }}s</span><input v-model.number="start" type="range" min="0" :max="Math.max(0, end - .01)" step=".01" /></label>
            <label><span>End · {{ end.toFixed(2) }}s</span><input v-model.number="end" type="range" :min="Math.min(duration, start + .01)" :max="duration" step=".01" /></label>
          </div>
          <div class="row space">
            <span class="pill">Selection {{ Math.max(0, end - start).toFixed(2) }}s</span>
            <span v-if="vadSegments.length" class="pill success">{{ vadSegments.length }} speech segment{{ vadSegments.length === 1 ? "" : "s" }}</span>
          </div>
          <div class="row modal-actions">
            <button type="button" @click="playSelection">Play selection</button>
            <button v-if="vadSegments.length" type="button" @click="selectFirstVad">Select first VAD</button>
            <button type="button" class="button primary" :disabled="saving" @click="save">{{ saving ? "Saving…" : "Save trim" }}</button>
          </div>
        </template>
      </section>
    </div>
  </Teleport>
</template>
