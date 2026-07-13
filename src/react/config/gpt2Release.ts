/**
 * Same-origin gpt2-small paths. Manifest + tokenizer ship in `public/models/gpt2-small/`.
 * Weights (~622MB) are proxied to the GitHub release (Vercel rewrite; Vite dev proxy)
 * so the browser never cross-origin-fetches release-assets.githubusercontent.com (no CORS).
 */
export function gpt2ModelUrls(baseUrl: string) {
  const prefix = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const dir = `${prefix}models/gpt2-small`;
  return {
    manifest: `${dir}/manifest.json`,
    tokenizer: `${dir}/tokenizer.json`,
    weights: `${dir}/weights.f32.bin`,
  } as const;
}

/** GitHub release tag used by Vercel/Vite proxies for the weights file only. */
export const GPT2_RELEASE_WEIGHTS =
  "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/browser-gpt2-weights.f32.bin";

export async function gpt2BundleReachable(baseUrl: string): Promise<boolean> {
  try {
    const urls = gpt2ModelUrls(baseUrl);
    const [manifestRes, weightsRes] = await Promise.all([
      fetch(urls.manifest, { method: "GET", cache: "no-store" }),
      fetch(urls.weights, { method: "HEAD", cache: "no-store" }),
    ]);
    return manifestRes.ok && (weightsRes.ok || weightsRes.status === 206);
  } catch {
    return false;
  }
}
