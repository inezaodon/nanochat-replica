/**
 * Flat files uploaded by `.github/workflows/publish-gpt2-web-bundle.yml` to tag `gpt2-web-v1`.
 * Used when `public/models/gpt2-small` is absent (e.g. fresh clone, GitHub Pages without postinstall cache).
 *
 * Browsers cannot trigger GitHub Actions (no repo token); publishing the release is a maintainer step.
 */
export const GPT2_RELEASE_FLAT = {
  manifest:
    "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/browser-gpt2-manifest.json",
  tokenizer:
    "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/browser-gpt2-tokenizer.json",
  weights:
    "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/browser-gpt2-weights.f32.bin",
} as const;

export async function gpt2ReleaseFlatReachable(): Promise<boolean> {
  try {
    const r = await fetch(GPT2_RELEASE_FLAT.manifest, { method: "GET", cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}
