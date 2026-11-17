<script setup lang="ts">
defineProps<{ name: string }>();
</script>

<template>
  <svg class="ui-icon" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path v-if="name === 'home'" d="m3 10 9-7 9 7v10H3V10Zm6 10v-7h6v7" />
    <path v-else-if="name === 'wave'" d="M4 9v6m4-10v14m4-17v20m4-17v14m4-10v6" />
    <path v-else-if="name === 'plus'" d="M12 5v14M5 12h14" />
    <path v-else-if="name === 'back'" d="M20 12H4m7-7-7 7 7 7" />
    <path v-else-if="name === 'next'" d="M4 12h16m-7-7 7 7-7 7" />
    <path v-else-if="name === 'refresh'" d="M20 7v5h-5M4 17v-5h5m-4-5a8 8 0 0 1 13-3l2 3M4 17l2 3a8 8 0 0 0 13-3" />
    <path v-else-if="name === 'activity'" d="M2 12h4l3-9 6 18 3-9h4" />
    <path v-else-if="name === 'download'" d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" />
    <path v-else-if="name === 'external'" d="M14 3h7v7m0-7-11 11M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4" />
    <path v-else-if="name === 'folder'" d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z" />
    <template v-else-if="name === 'file'"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z" /><path d="M14 2v6h6M8 13h8M8 17h5" /></template>
    <template v-else-if="name === 'copy'"><rect x="8" y="8" width="13" height="13" rx="2" /><path d="M16 8V3H3v13h5" /></template>
    <path v-else-if="name === 'inbox'" d="m3 4-2 12v5h22v-5L21 4H3Zm-2 12h7l2 3h4l2-3h7" />
    <template v-else-if="name === 'history'"><path d="M3 3v6h6M3 9a9 9 0 1 1-1 7M12 7v5l4 2" /></template>
    <template v-else-if="name === 'check'"><circle cx="12" cy="12" r="9" /><path d="m8 12 3 3 5-6" /></template>
    <path v-else-if="name === 'link'" d="m10 13 4-4m-6 6-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m2 2 2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0" />
    <template v-else-if="name === 'stop'"><rect x="4" y="4" width="16" height="16" rx="2" /></template>
    <circle v-else cx="12" cy="12" r="8" />
  </svg>
</template>
