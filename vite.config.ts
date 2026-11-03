import { defineConfig } from "vite";

// Static build. Default base is relative, so dist/ works from any folder or sub-path (a USB stick, a file server).
// The GitHub Pages workflow builds with NETGLOW_BASE=/netglow/ so asset URLs are absolute under the project path.
export default defineConfig({
  base: process.env.NETGLOW_BASE ?? "./",
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
