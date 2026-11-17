<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { copyWakeWord, formatTimestamp, isBusy, publishSavedVersion, trainer } from "../trainerStore";
import type { TrainingRun, WakeWordLibraryItem } from "../types";
import UiIcon from "./UiIcon.vue";

const props = defineProps<{ word: WakeWordLibraryItem }>();
defineEmits<{ retrain: [word: WakeWordLibraryItem]; recordings: [word: WakeWordLibraryItem] }>();
const tab = ref("results");
const selectedId = ref("");
const compareId = ref("");
const runs = computed(() => trainer.modelHistory.filter(run => run.safe_word === props.word.safe_word));
const current = computed(() => runs.value.find(run => run.is_current));
const selected = computed(() => runs.value.find(run => run.run_id === selectedId.value));
const catalog = computed(() => trainer.wakeWords.find(word => word.key === props.word.safe_word));
const showingPublished = computed(() => !selected.value || selected.value.is_current);
const summary = computed(() => selected.value?.summary);
const links = computed(() => selected.value ? { bundle: selected.value.json_url || "", esphome: selected.value.esphome_json_url || "" } : { bundle: String(catalog.value?.json_url || catalog.value?.url || catalog.value?.jsonUrl || ""), esphome: String(catalog.value?.esphome_json_url || catalog.value?.esphomeJsonUrl || "") });
const compare = computed(() => runs.value.find(run => run.run_id === compareId.value));
const comparisonModels = computed(() => Array.from(new Set([...(current.value?.summary?.models || []), ...(compare.value?.summary?.models || [])].map(model => model.id))));
watch(() => props.word.safe_word, () => { selectedId.value = ""; tab.value = "results"; compareId.value = ""; }, { immediate: true });
watch(runs, () => {
  if (!selectedId.value || !runs.value.some(run => run.run_id === selectedId.value)) selectedId.value = current.value?.run_id || (props.word.has_published_bundle ? "" : runs.value[0]?.run_id || "");
}, { immediate: true });
function version(run?: TrainingRun): string { return run ? `v${run.version || runs.value.length - runs.value.indexOf(run)}` : "Current publication"; }
function showRun(run: TrainingRun): void { selectedId.value = run.run_id; tab.value = "results"; compareId.value = ""; }
function artifactUrls(): Record<string, string> {
  if (selected.value) return selected.value.artifact_urls || {};
  const word = catalog.value;
  if (!word) return {};
  const entries: Array<[string, string]> = [["MicroWakeWord", String(word.model_url || "")], ["openWakeWord", String(word.openwakeword_model_url || "")], ["Wake boundary", String(word.wake_boundary_model_url || "")]];
  for (const [architecture, url] of Object.entries(word.nanowakeword_models || {})) entries.push([`NanoWakeWord ${architecture.toUpperCase()}`, url]);
  return Object.fromEntries(entries.filter(([, url]) => url));
}
</script>

<template>
  <button type="button" class="workspace-back" @click="trainer.activeView = 'home'"><UiIcon name="back" />Back to start</button>
  <header class="workspace-heading"><div><span class="eyebrow">Saved wake word</span><div class="row"><h2>{{ word.phrase }}</h2><span v-if="word.language" class="pill">{{ word.language }}</span></div></div><button type="button" :disabled="trainer.training.running || isBusy('session')" @click="$emit('retrain', word)"><UiIcon name="refresh" />Train a new version</button></header>
  <div class="version-toolbar">
    <div><div class="row"><label v-if="runs.length > 1" class="row">Viewing version<select v-model="selectedId" aria-label="Viewing version"><option v-if="!current && !selectedId" value="">Current publication</option><option v-for="run in runs" :key="run.run_id" :value="run.run_id">{{ version(run) }}{{ run.is_current ? ' · Current' : '' }}{{ run === runs[0] ? ' · Latest run' : '' }}</option></select></label><strong v-else>{{ version(selected) }}</strong><span v-if="summary" class="pill" :class="summary.tone">{{ summary.overall }}</span></div><small v-if="selected">Trained {{ formatTimestamp(selected.completed_at) }} · {{ selected.run_kind === 'boundary' ? 'Boundary only' : 'Model training' }}</small><small v-else>Published JSON links · no matching saved quality report</small></div>
    <div class="row"><span v-if="showingPublished" class="pill success">Current publication</span><template v-else><button type="button" class="primary" :disabled="!selected?.restorable || trainer.training.running || isBusy('version-publish')" @click="selected && publishSavedVersion(word, selected)"><UiIcon name="check" />{{ isBusy('version-publish') ? 'Publishing…' : `Use ${version(selected)} as current` }}</button><small>{{ version(current) }} is current</small></template></div>
  </div>
  <div class="workspace-subnav" role="tablist" aria-label="Wake word workspace"><button v-for="item in [{id:'results',label:'Results & JSON'}, {id:'versions',label:`Versions · ${runs.length}`}, {id:'recordings',label:'Recordings'}]" :id="`word-tab-${item.id}`" :key="item.id" type="button" role="tab" aria-controls="word-panel" :aria-selected="tab === item.id" @click="tab = item.id">{{ item.label }}</button></div>
  <section id="word-panel" role="tabpanel" :aria-labelledby="`word-tab-${tab}`" class="stack">
    <template v-if="tab === 'results'">
      <div class="json-link-grid">
        <section v-for="item in [{key:'bundle',label:'Tater bundle JSON',description:'Use in Tater or the Home Assistant bridge.',url:links.bundle}, {key:'esphome',label:'ESPHome JSON',description:'Use in your ESPHome micro_wake_word setup.',url:links.esphome}]" :key="item.key" class="panel json-link-card"><h3><UiIcon name="file" />{{ item.label }}</h3><p>{{ item.description }}</p><template v-if="item.url"><code>{{ item.url }}</code><div class="row"><button type="button" class="primary" @click="copyWakeWord(item.url)"><UiIcon name="copy" />Copy {{ item.key === 'bundle' ? 'bundle JSON URL' : 'ESPHome URL' }}</button><a class="button" :href="item.url" target="_blank" rel="noreferrer"><UiIcon name="external" />View JSON</a></div><small>{{ selected ? `${version(selected)} · version-specific link` : 'Current published link' }}</small></template><p v-else class="workspace-note">{{ item.key === 'esphome' ? 'No MicroWakeWord model was published in this version.' : 'No model was published by this run yet.' }}</p></section>
      </div>
      <section class="panel results-panel"><header class="workspace-section-title"><div><h3>Training results{{ selected ? ` · ${version(selected)}` : '' }}</h3><p>{{ summary?.message || 'Quality results are unavailable for this publication. No quality rating has been inferred.' }}</p></div><span v-if="summary" class="pill" :class="summary.tone">{{ summary.overall }}</span></header><article v-for="model in summary?.models || []" :key="model.id" class="result-model"><header><div><h3>{{ model.label }}</h3><p>{{ model.message }}</p></div><span class="pill" :class="model.tone">{{ model.rating }}</span></header><dl class="result-metrics"><div v-for="metric in model.metrics" :key="metric.label"><dt>{{ metric.label }}</dt><dd>{{ metric.value }}</dd></div></dl></article><p class="workspace-note">Each detector reports its own validation metrics. Missing measurements stay unavailable.</p></section>
      <div class="row space"><small>{{ runs.length }} saved runs · earlier results and JSON links are retained</small><button type="button" @click="tab = 'versions'"><UiIcon name="history" />View version history</button></div>
      <details class="workspace-advanced"><summary>Raw model files & technical details</summary><div class="stack"><div v-for="(url, name) in artifactUrls()" :key="name" class="row space"><span>{{ name }}</span><a :href="url" target="_blank" rel="noreferrer">{{ String(name).endsWith('.json') ? 'View artifact' : 'Download model' }}</a></div><p v-if="!selected && catalog?.wake_boundary_available" class="workspace-note">Wake boundary model · wake boundary · threshold {{ catalog.wake_boundary_threshold }}</p><p v-if="selected && !selected.restorable" class="workspace-note">This run did not publish a usable model. Its recorded results remain in history.</p></div></details>
    </template>
    <template v-else-if="tab === 'versions'">
      <section class="panel"><header class="workspace-section-title"><div><h3>Saved versions</h3><p>Open any run for its exact results and immutable JSON links.</p></div><span class="pill">{{ runs.length }} retained</span></header><div v-if="!runs.length" class="empty-state">Completed training attempts will appear here. Your current JSON links are still available under Results & JSON.</div><article v-for="run in runs" :key="run.run_id" class="workspace-version-row"><div><strong>{{ version(run) }}</strong><span v-if="run.is_current" class="pill success">Current</span><span v-else-if="run === runs[0]" class="pill">Latest run</span></div><div><strong>{{ formatTimestamp(run.completed_at) }}</strong><small>{{ run.run_kind === 'boundary' ? 'Boundary only' : run.selected_models?.join(' + ') }}</small></div><div><span class="pill" :class="run.summary?.tone || ''">{{ run.summary?.overall || 'Results unavailable' }}</span><small v-if="!run.restorable">No usable published model</small></div><div class="row"><button type="button" @click="showRun(run)">View results</button><button type="button" :disabled="!current || current.run_id === run.run_id" @click="compareId = run.run_id">Compare</button></div></article></section>
      <section v-if="compare" class="panel"><header class="workspace-section-title"><div><h3>Compare with current</h3><p>Training datasets can change between runs. Compare matching metrics with that context.</p></div><button type="button" @click="compareId = ''">Close comparison</button></header><div class="comparison-grid"><article v-for="run in [current, compare].filter(Boolean)" :key="run!.run_id"><header class="row"><h3>{{ version(run) }}</h3><span class="pill">{{ run?.is_current ? 'Current' : 'Comparing' }}</span></header><div v-for="id in comparisonModels" :key="id" class="comparison-model"><h3>{{ run?.summary?.models.find(model => model.id === id)?.label || id }}</h3><p>{{ run?.summary?.models.find(model => model.id === id)?.rating || 'Not evaluated' }}</p><dl><div v-for="metric in run?.summary?.models.find(model => model.id === id)?.metrics || []" :key="metric.label"><dt>{{ metric.label }}</dt><dd>{{ metric.value }}</dd></div></dl></div><button v-if="run && !run.is_current" type="button" @click="showRun(run)">View {{ version(run) }}</button></article></div></section>
    </template>
    <section v-else class="panel"><header class="workspace-section-title"><div><h3>Recordings for “{{ word.phrase }}”</h3><p>{{ word.personal_count || 0 }} personal recordings · {{ word.negative_count || 0 }} reviewed negatives</p></div><button type="button" :disabled="trainer.training.running && trainer.session.safe_word !== word.safe_word" @click="$emit('recordings', word)"><UiIcon name="plus" />Manage recordings</button></header><p class="workspace-note">Review, upload, trim, or restore recordings in the sample manager. Existing versions keep their original models and results.</p></section>
  </section>
</template>
