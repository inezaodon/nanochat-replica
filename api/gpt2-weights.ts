export const config = { runtime: "edge" };

const RELEASE_URL =
  "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/browser-gpt2-weights.f32.bin";

/** Stream gpt2-small weights from GitHub release without passing 302/CORS to the browser. */
export default async function handler(request: Request): Promise<Response> {
  const range = request.headers.get("Range");
  const upstream = await fetch(RELEASE_URL, {
    redirect: "follow",
    headers: range ? { Range: range } : undefined,
  });

  if (!upstream.ok && upstream.status !== 206) {
    return new Response(`Upstream weights fetch failed: ${upstream.status}`, {
      status: upstream.status,
    });
  }

  const headers = new Headers();
  headers.set("Content-Type", "application/octet-stream");
  headers.set("Accept-Ranges", "bytes");
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  const len = upstream.headers.get("Content-Length");
  if (len) headers.set("Content-Length", len);
  const contentRange = upstream.headers.get("Content-Range");
  if (contentRange) headers.set("Content-Range", contentRange);

  return new Response(upstream.body, { status: upstream.status, headers });
}
