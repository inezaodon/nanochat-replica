/**
 * gpt2-small load paths. Manifest + tokenizer ship in `public/models/gpt2-small/`.
 * Weights (~622MB): Vercel/local dev use `/api/gpt2-weights` edge stream;
 * GitHub Pages serves bundled `weights.f32.bin` from deploy.
 */
export function gpt2ModelUrls(baseUrl: string) {
  const prefix = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const dir = `${prefix}models/gpt2-small`;
  return {
    manifest: `${dir}/manifest.json`,
    tokenizer: `${dir}/tokenizer.json`,
    weights: gpt2WeightsUrl(prefix),
  } as const;
}

function gpt2WeightsUrl(prefix: string): string {
  if (typeof window !== "undefined" && window.location.hostname.endsWith("github.io")) {
    return `${prefix}models/gpt2-small/weights.f32.bin`;
  }
  return `${prefix}api/gpt2-weights`;
}

/** GitHub release tag — source for edge proxy and CI fetch. */
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
