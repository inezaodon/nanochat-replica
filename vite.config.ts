import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const gpt2WeightsProxy = {
  "/models/gpt2-small/weights.f32.bin": {
    target:
      "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/browser-gpt2-weights.f32.bin",
    changeOrigin: true,
    rewrite: () => "",
  },
  "/nanochat-replica/models/gpt2-small/weights.f32.bin": {
    target:
      "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/browser-gpt2-weights.f32.bin",
    changeOrigin: true,
    rewrite: () => "",
  },
};

export default defineConfig(({ mode }) => ({
  // GitHub Pages serves at /<repo>/, so we build with base='/nanochat-replica/' in that mode.
  base: mode === "github" ? "/nanochat-replica/" : "/",
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: gpt2WeightsProxy,
  },
  preview: {
    port: 5174,
    strictPort: true,
    proxy: gpt2WeightsProxy,
  },
}));

