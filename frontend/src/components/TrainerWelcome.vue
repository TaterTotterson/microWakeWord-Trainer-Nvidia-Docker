<script setup lang="ts">
import { computed } from "vue";
import { formatBytes, trainer } from "../trainerStore";
import type { WakeWordLibraryItem } from "../types";
import UiIcon from "./UiIcon.vue";

defineProps<{ library?: boolean }>();
defineEmits<{ create: []; open: [word: WakeWordLibraryItem] }>();
const words = computed(() => trainer.wakeWordLibrary);
function summary(word: WakeWordLibraryItem) {
  return trainer.modelHistory.find(run => run.safe_word === word.safe_word && run.is_current)?.summary
    || (!word.published_run_id ? word.latest_run?.summary : undefined);
}
function metric(word: WakeWordLibraryItem): string {
  const model = summary(word)?.models.find(item => item.id === "micro");
  return model?.metrics.filter(item => ["Recall", "False accepts"].includes(item.label)).map(item => `${item.label} ${item.value}`).join(" · ") || "";
}
</script>

<template>
  <div class="welcome-workspace">
    <header v-if="!library" class="welcome-intro"><span class="eyebrow">Wake Word Trainer</span><h2>What would you like to do?</h2><p>Train a new phrase, or find the right version of one you already use.</p></header>
    <header v-else class="workspace-heading"><div><span class="eyebrow">Your library</span><h2>Wake words</h2><p>Each phrase keeps its results, JSON links, recordings, and saved versions.</p></div><button type="button" class="primary" :disabled="trainer.training.running" @click="$emit('create')"><UiIcon name="plus" />New wake word</button></header>
    <div v-if="!library" class="welcome-tasks">
      <section class="welcome-task"><UiIcon name="plus" /><h3>Train a new word</h3><p>Choose a phrase, add recordings, and create your first trained version.</p><button type="button" class="primary" :disabled="trainer.training.running" @click="$emit('create')"><UiIcon name="next" />Create a wake word</button></section>
      <section class="welcome-task"><UiIcon name="folder" /><h3>Use an existing word</h3><p>Review its results, choose a version, and get its bundle or ESPHome JSON.</p><button type="button" @click="trainer.activeView = 'firmware'"><UiIcon name="next" />Browse wake words</button></section>
    </div>
    <div class="workspace-section-title"><h3>Your wake words</h3><small>{{ words.length }} saved {{ words.length === 1 ? 'word' : 'words' }}</small></div>
    <div v-if="!words.length" class="panel empty-state">Your first wake word will appear here with its recordings, models, and training history.</div>
    <article v-for="word in words" :key="word.safe_word" class="workspace-word-row">
      <div class="workspace-word-icon"><UiIcon :name="word.has_published_bundle ? 'wave' : 'file'" /></div>
      <div><div class="row"><h3>{{ word.phrase }}</h3><span class="pill" :class="summary(word)?.tone || ''">{{ summary(word)?.overall || (word.has_published_bundle ? 'Published · results unavailable' : 'Draft') }}</span></div><p>{{ word.run_count }} saved {{ word.run_count === 1 ? 'run' : 'runs' }} · {{ word.personal_count || 0 }} recordings · {{ formatBytes(word.dataset_size_bytes) }} retained</p></div>
      <div class="workspace-word-actions"><button type="button" @click="$emit('open', word)"><UiIcon name="external" />Open word</button><small v-if="metric(word)">MicroWakeWord · {{ metric(word) }}</small></div>
    </article>
    <p v-if="trainer.training.running" class="workspace-note">A training run is in progress. You can open words and use their existing JSON links.</p>
  </div>
</template>
