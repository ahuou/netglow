import { defineConfig } from "vite";

// Static build: relative base so dist/ works from any folder or sub-path (GitHub Pages, a USB stick, file server).
export default defineConfig({
  base: "./",
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
