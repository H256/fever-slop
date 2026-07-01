<script setup lang="ts">
import { FolderKanban, Settings } from "lucide-vue-next";
</script>

<template>
  <div class="studio-shell">
    <aside class="sidebar">
      <RouterLink class="brand" to="/">
        <span class="brand-mark">FS</span>
        <span>FeverSlop Studio</span>
      </RouterLink>
      <nav>
        <RouterLink to="/"><FolderKanban :size="18" /> Projects</RouterLink>
        <RouterLink to="/settings"><Settings :size="18" /> Settings</RouterLink>
      </nav>
    </aside>
    <main class="main-pane">
      <RouterView />
    </main>
  </div>
</template>
