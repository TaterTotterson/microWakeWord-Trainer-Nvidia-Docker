import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const trainerVersion = readFileSync(
  resolve(import.meta.dirname, "../VERSION"),
  "utf8",
).trim() || "development";

export default defineConfig({
  plugins: [vue()],
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
    __TRAINER_VERSION__: JSON.stringify(trainerVersion),
  },
  build: {
    outDir: resolve(import.meta.dirname, "../static/ui"),
    emptyOutDir: true,
    lib: {
      entry: resolve(import.meta.dirname, "src/main.ts"),
      formats: ["es"],
      fileName: () => "trainer-ui.js",
    },
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) =>
          assetInfo.name?.endsWith(".css") ? "trainer-ui.css" : "[name][extname]",
      },
    },
  },
});
