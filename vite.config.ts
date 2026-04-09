import { defineConfig } from "vite";

export default defineConfig(({ mode }) => ({
  // GitHub Pages serves at /<repo>/, so we build with base='/nanochat-replica/' in that mode.
  base: mode === "github" ? "/nanochat-replica/" : "/",
  server: {
    port: 5173,
    strictPort: true,
  },
  preview: {
    port: 5174,
    strictPort: true,
  },
}));

