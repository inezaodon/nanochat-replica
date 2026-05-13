#!/usr/bin/env node
/**
 * Fetches the pre-built gpt2-small browser bundle (manifest + tokenizer + weights.f32.bin)
 * into public/models/gpt2-small/ so `npm run dev` works without running Python.
 *
 * Default source: GitHub Release asset on this repo (created by CI workflow once).
 * Override: GPT2_WEB_BUNDLE_URL=https://.../gpt2-small-web.tgz
 * Skip:     SKIP_GPT2_WEB_FETCH=1
 */
import { createWriteStream } from "node:fs";
import { mkdir, stat, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const OUT = join(ROOT, "public", "models", "gpt2-small");
const MANIFEST = join(OUT, "manifest.json");
const WEIGHTS = join(OUT, "weights.f32.bin");

const DEFAULT_TGZ_URL =
  "https://github.com/inezaodon/nanochat-replica/releases/download/gpt2-web-v1/gpt2-small-web.tgz";

async function pathExists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

async function weightsLookComplete() {
  if (!(await pathExists(MANIFEST)) || !(await pathExists(WEIGHTS))) return false;
  const st = await stat(WEIGHTS);
  return st.size > 50_000_000; // expect ~0.5GB f32; avoid treating a partial stub as done
}

function downloadWithCurl(url, dest) {
  const r = spawnSync("curl", ["-fL", "--retry", "3", "--connect-timeout", "20", "-o", dest, url], {
    stdio: "inherit",
  });
  return r.status === 0;
}

async function downloadWithNode(url, dest) {
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  const body = res.body;
  if (!body) throw new Error("empty response body");
  if (typeof Readable.fromWeb !== "function") {
    throw new Error("Node.js 18+ required for streaming fetch when curl is unavailable");
  }
  await pipeline(Readable.fromWeb(body), createWriteStream(dest));
}

async function downloadFile(url, dest) {
  try {
    await rm(dest, { force: true });
  } catch {
    /* ignore */
  }
  if (downloadWithCurl(url, dest)) return;
  const probe = spawnSync("curl", ["-V"], { encoding: "utf8" });
  if (probe.error?.code === "ENOENT") {
    console.log("[ensure-gpt2-web] curl not found; using Node fetch (streaming)…");
  } else {
    console.log("[ensure-gpt2-web] curl failed; using Node fetch (streaming)…");
  }
  await downloadWithNode(url, dest);
}

function extractTgz(tgzPath, extractIntoModelsDir) {
  const r = spawnSync("tar", ["-xzf", tgzPath, "-C", extractIntoModelsDir], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`tar exited ${r.status}`);
}

async function main() {
  if (process.env.SKIP_GPT2_WEB_FETCH === "1") {
    console.log("[ensure-gpt2-web] SKIP_GPT2_WEB_FETCH=1 — skipping.");
    return;
  }
  if (await weightsLookComplete()) {
    console.log("[ensure-gpt2-web] public/models/gpt2-small already present — skipping.");
    return;
  }

  const url = process.env.GPT2_WEB_BUNDLE_URL || DEFAULT_TGZ_URL;
  console.log(`[ensure-gpt2-web] downloading bundle…\n  ${url}`);

  await mkdir(OUT, { recursive: true });
  const tmp = join(tmpdir(), `gpt2-small-web-${Date.now()}.tgz`);

  try {
    await downloadFile(url, tmp);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("404") || msg.includes("HTTP 404")) {
      console.warn(
        "[ensure-gpt2-web] Release bundle not found (404). Either:\n" +
          "  • Ask the maintainer to run GitHub Actions → “Publish GPT-2 web bundle”, or\n" +
          "  • Run locally: npm run prepare:gpt2-web (needs Python + torch)\n" +
          "npm install will continue without the browser GPT-2 weights.",
      );
      process.exit(0);
    }
    console.warn("[ensure-gpt2-web] download failed:", msg);
    process.exit(0);
  }

  const modelsDir = join(ROOT, "public", "models");
  await mkdir(modelsDir, { recursive: true });
  try {
    await rm(OUT, { recursive: true, force: true });
  } catch {
    /* ignore */
  }
  await mkdir(modelsDir, { recursive: true });

  try {
    extractTgz(tmp, modelsDir);
  } catch (e) {
    console.warn("[ensure-gpt2-web] extract failed:", e instanceof Error ? e.message : e);
    process.exit(0);
  } finally {
    try {
      await rm(tmp, { force: true });
    } catch {
      /* ignore */
    }
  }

  if (!(await weightsLookComplete())) {
    console.warn("[ensure-gpt2-web] after extract, weights still missing or too small — leaving as-is.");
    process.exit(0);
  }
  console.log("[ensure-gpt2-web] ready: public/models/gpt2-small/");
}

main().catch((e) => {
  console.warn("[ensure-gpt2-web] unexpected:", e);
  process.exit(0);
});
