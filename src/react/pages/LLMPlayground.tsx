import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Gpt2TiktokenTokenizer } from "../../core/gpt2Tokenizer";
import { RegexBPETokenizer } from "../../core/tokenizer";
import { createTinyGPTWeb, loadTensors, TinyGPTWeb } from "../../core/inferTinyGPT";
import { fetchArrayBuffer, fetchJSON, WebManifest } from "../../core/webModel";
import { GPT2_RELEASE_FLAT, gpt2ReleaseFlatReachable } from "../config/gpt2Release";
import { useReveal } from "../hooks/useReveal";

type ModelPreset = "tiny-gpt" | "gpt2-small";

/** Vite `base` (e.g. `/` or `/nanochat-replica/`) + `models/<name>`. */
function modelsDir(name: "tiny-gpt" | "gpt2-small"): string {
  const b = import.meta.env.BASE_URL;
  const prefix = b.endsWith("/") ? b : `${b}/`;
  return `${prefix}models/${name}`;
}

/** Maintainer: builds release `gpt2-web-v1` (tarball + flat files for browser Load). */
const GPT2_BUNDLE_PUBLISH_WORKFLOW =
  "https://github.com/inezaodon/nanochat-replica/actions/workflows/publish-gpt2-web-bundle.yml";

const GPT2_FAIL_PREFIX = "[gpt2-small]";

function isLargeBrowserModel(manifest: { config: { vocab_size: number; n_embd: number } }): boolean {
  return manifest.config.vocab_size >= 4096 || manifest.config.n_embd >= 256;
}

function yieldToMain(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export function LLMPlayground() {
  const [preset, setPreset] = useState<ModelPreset>("tiny-gpt");
  const [prompt, setPrompt] = useState("Hello from a tiny GPT.");
  const [status, setStatus] = useState<string>("Checking which model bundles are available…");
  const [out, setOut] = useState<string>("");
  const [model, setModel] = useState<TinyGPTWeb | null>(null);
  const gpt2TokenizerRef = useRef<Gpt2TiktokenTokenizer | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const generateAbortRef = useRef<AbortController | null>(null);
  const [maxNewTokens, setMaxNewTokens] = useState(60);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [seed, setSeed] = useState(42);
  const revealMain = useReveal();

  const showGpt2Recovery = useMemo(() => {
    if (preset !== "gpt2-small") return false;
    const s = status;
    return (
      s.startsWith(GPT2_FAIL_PREFIX) ||
      s.includes("404") ||
      s.includes("Failed to fetch") ||
      s.includes("bundle missing") ||
      s.includes("has no browser files yet")
    );
  }, [preset, status]);

  const switchToTinyGptClear = useCallback(() => {
    setPreset("tiny-gpt");
    setModel(null);
    setOut("");
    setStatus("Switched to tiny-gpt — press Load model to use the checked-in /models/tiny-gpt bundle.");
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const localManifest = `${modelsDir("gpt2-small")}/manifest.json`;
        const r = await fetch(localManifest, { cache: "no-store" });
        if (cancelled) return;
        if (r.ok) {
          setPreset("gpt2-small");
          setStatus(
            "gpt2-small bundle found on this host — press Load model. (tiny-gpt is a small char-LM for demos.)",
          );
          return;
        }
        if (await gpt2ReleaseFlatReachable()) {
          if (cancelled) return;
          setPreset("gpt2-small");
          setStatus(
            "gpt2-small is not under public/models, but the GitHub release files exist — press Load model to pull weights in the browser (large download; may take a minute).",
          );
          return;
        }
        if (!cancelled) {
          setStatus(
            "gpt2-small not on disk and the GitHub release has no browser files yet. A maintainer must run Publish GPT-2 web bundle once (link below). Developers: npm run fetch:gpt2-web or npm run prepare:gpt2-web.",
          );
        }
      } catch {
        if (!cancelled) {
          setStatus(
            "Could not probe gpt2-small (offline?). Defaulting to tiny-gpt. When online, try again or run npm run fetch:gpt2-web.",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
      generateAbortRef.current?.abort();
      gpt2TokenizerRef.current?.free();
      gpt2TokenizerRef.current = null;
    };
  }, []);

  async function loadModel() {
    setLoading(true);
    setStatus("Loading manifest/tokenizer/weights…");
    gpt2TokenizerRef.current?.free();
    gpt2TokenizerRef.current = null;
    try {
      type TokJson = {
        tokenizer_type?: string;
        merges?: Record<string, number>;
        vocab: Record<string, string>;
        special_tokens: Record<string, number>;
        pattern?: string;
      };

      if (preset === "tiny-gpt") {
        const base = modelsDir("tiny-gpt");
        const manifest = await fetchJSON<WebManifest>(`${base}/manifest.json`);
        const tokObj = await fetchJSON<TokJson>(`${base}/tokenizer.json`);
        let tokenizer: RegexBPETokenizer | Gpt2TiktokenTokenizer;
        if (tokObj.tokenizer_type === "gpt2_tiktoken" || manifest.tokenizer_type === "gpt2_tiktoken") {
          const g2 = new Gpt2TiktokenTokenizer();
          gpt2TokenizerRef.current = g2;
          tokenizer = g2;
        } else {
          tokenizer = RegexBPETokenizer.fromJSON(tokObj);
        }
        setStatus("Downloading / reading weights…");
        await yieldToMain();
        const buf = await fetchArrayBuffer(`${base}/${manifest.weights}`);
        setStatus("Mapping weight buffer (may pause briefly)…");
        await yieldToMain();
        const tensors = loadTensors(buf, manifest);
        await yieldToMain();
        const m = createTinyGPTWeb(manifest, tokenizer, tensors);
        setModel(m);
        if (isLargeBrowserModel(manifest)) setMaxNewTokens((n) => Math.min(n, 20));
        setStatus(
          `Loaded ${preset}: vocab=${manifest.config.vocab_size}, layers=${manifest.config.n_layer}, heads=${manifest.config.n_head}, embd=${manifest.config.n_embd}`,
        );
        return;
      }

      const gpt2Local = modelsDir("gpt2-small");
      let manifest: WebManifest;
      let tokUrl: string;
      let weightsUrl: string;
      let sourceNote = "";

      try {
        manifest = await fetchJSON<WebManifest>(`${gpt2Local}/manifest.json`);
        tokUrl = `${gpt2Local}/tokenizer.json`;
        weightsUrl = `${gpt2Local}/${manifest.weights}`;
      } catch {
        setStatus(
          "No local gpt2-small folder — loading manifest, tokenizer, and weights from GitHub release gpt2-web-v1 (large; first time can take several minutes)…",
        );
        manifest = await fetchJSON<WebManifest>(GPT2_RELEASE_FLAT.manifest);
        tokUrl = GPT2_RELEASE_FLAT.tokenizer;
        weightsUrl = GPT2_RELEASE_FLAT.weights;
        sourceNote = " — source: GitHub release";
      }

      const tokObj = await fetchJSON<TokJson>(tokUrl);
      let tokenizer: RegexBPETokenizer | Gpt2TiktokenTokenizer;
      if (tokObj.tokenizer_type === "gpt2_tiktoken" || manifest.tokenizer_type === "gpt2_tiktoken") {
        const g2 = new Gpt2TiktokenTokenizer();
        gpt2TokenizerRef.current = g2;
        tokenizer = g2;
      } else {
        tokenizer = RegexBPETokenizer.fromJSON(tokObj);
      }

      setStatus(
        `Downloading weights (~500MB f32)${sourceNote} — tab may look frozen until this finishes…`,
      );
      await yieldToMain();
      const buf = await fetchArrayBuffer(weightsUrl);
      setStatus("Mapping weight buffer into tensors (may pause 10–30s)…");
      await yieldToMain();
      const tensors = loadTensors(buf, manifest);
      await yieldToMain();
      const m = createTinyGPTWeb(manifest, tokenizer, tensors);
      setModel(m);
      setMaxNewTokens((n) => Math.min(n, 20));
      setStatus(
        `Loaded gpt2-small${sourceNote}: vocab=${manifest.config.vocab_size}, layers=${manifest.config.n_layer}, heads=${manifest.config.n_head}, embd=${manifest.config.n_embd}. Use ≤20 new tokens per click — full GPT-2 in JS is slow.`,
      );
    } catch (e) {
      const msg = (e as Error).message;
      if (preset === "gpt2-small") {
        setStatus(
          `${GPT2_FAIL_PREFIX} ${msg} This page cannot start GitHub Actions (no repo token). If the release is missing, open Publish GPT-2 web bundle (below) once. If your browser blocks cross-origin downloads, run npm run fetch:gpt2-web or npm run prepare:gpt2-web locally.`,
        );
      } else {
        setStatus(msg);
      }
      setModel(null);
    } finally {
      setLoading(false);
    }
  }

  const busy = loading || generating;
  const isGpt2Loaded = model !== null && isLargeBrowserModel(model.manifest);
  const maxTokensCap = isGpt2Loaded || preset === "gpt2-small" ? 32 : 512;

  async function generate() {
    if (!model) {
      setStatus("Load the model first.");
      return;
    }
    generateAbortRef.current?.abort();
    const ac = new AbortController();
    generateAbortRef.current = ac;
    setGenerating(true);
    const cap = Math.min(maxNewTokens, maxTokensCap);
    const large = isLargeBrowserModel(model.manifest);
    setStatus(
      large
        ? `Generating (0/${cap}) — gpt2-small in plain JS; each token can take several seconds…`
        : `Generating (0/${cap})…`,
    );
    try {
      const text = await model.generateAsync(
        prompt,
        { maxNewTokens: cap, temperature, topK, seed },
        {
          signal: ac.signal,
          onProgress: (step, max) => {
            setStatus(
              large
                ? `Generating (${step}/${max}) — keep this tab open; closing cancels…`
                : `Generating (${step}/${max})…`,
            );
          },
        },
      );
      setOut(text);
      const cfg = model.manifest.config;
      const toy =
        cfg.vocab_size < 4096 || cfg.n_embd < 256
          ? " (Toy character-scale model — fluent English is not expected. Use gpt2-small for full GPT-2.)"
          : large
            ? " (gpt2-small runs entirely in the browser — use fewer tokens if this felt slow.)"
            : "";
      setStatus(`Done.${toy}`);
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setStatus("Generation cancelled.");
      } else {
        setStatus((e as Error).message);
      }
    } finally {
      setGenerating(false);
      generateAbortRef.current = null;
    }
  }

  function cancelGenerate() {
    generateAbortRef.current?.abort();
  }

  return (
    <div className="play-layout play-layout--solo stack-gap">
        <section ref={revealMain.ref} className={`card lift-reveal ${revealMain.active ? "is-visible" : ""}`}>
          <div className="cardH">
            <h2>GPT demo</h2>
            <div className="cardH-meta">Load weights, tune sampling, stream tokens into the output pane.</div>
          </div>
          <div className="cardB stack-gap">
            <div className="field">
              <label htmlFor="llm-prompt">Prompt</label>
              <textarea id="llm-prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} spellCheck={false} />
            </div>
            <div className="row">
              <div className="field mb-0">
                <label htmlFor="llm-max">Max new tokens</label>
                <input
                  id="llm-max"
                  type="number"
                  value={maxNewTokens}
                  min={1}
                  max={maxTokensCap}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                />
              </div>
              <div className="field mb-0">
                <label htmlFor="llm-seed">Seed</label>
                <input id="llm-seed" type="number" value={seed} min={0} max={999999} onChange={(e) => setSeed(Number(e.target.value))} />
              </div>
            </div>
            <div className="row">
              <div className="field mb-0">
                <label htmlFor="llm-temp">Temperature</label>
                <input
                  id="llm-temp"
                  type="number"
                  step="0.05"
                  value={temperature}
                  min={0.1}
                  max={2.0}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                />
              </div>
              <div className="field mb-0">
                <label htmlFor="llm-topk">Top‑K</label>
                <input id="llm-topk" type="number" value={topK} min={0} max={200} onChange={(e) => setTopK(Number(e.target.value))} />
              </div>
            </div>
            <div className="field mb-0">
              <label htmlFor="llm-preset">Model bundle</label>
              <select id="llm-preset" value={preset} onChange={(e) => setPreset(e.target.value as ModelPreset)}>
                <option value="tiny-gpt">tiny-gpt (trained in-repo)</option>
                <option value="gpt2-small">gpt2-small (ND weights-v1 — ships via release bundle or prepare:gpt2-web)</option>
              </select>
            </div>
            <div className="btn-row">
              <button type="button" className="primary" disabled={busy} onClick={generate}>
                {generating ? "Generating…" : "Generate"}
              </button>
              {generating ? (
                <button type="button" onClick={cancelGenerate}>
                  Cancel
                </button>
              ) : null}
              <button type="button" disabled={busy} onClick={loadModel}>
                {loading ? "Loading…" : "Load model"}
              </button>
              <button type="button" onClick={() => setPrompt("")}>
                Clear prompt
              </button>
            </div>
            <div
              className={`status-line ${
                status.startsWith(GPT2_FAIL_PREFIX) ||
                status.includes("Failed to fetch") ||
                status.includes("404") ||
                status.includes("bundle missing") ||
                status.includes("has no browser files yet")
                  ? "status-line--error"
                  : ""
              }`}
              role="status"
            >
              {status}
            </div>
            {showGpt2Recovery ? (
              <div className="status-actions stack-gap">
                <p className="muted mb-0">
                  <strong>This tab cannot run GitHub Actions</strong> (there is no deploy secret in the browser). After a maintainer runs{" "}
                  <a href={GPT2_BUNDLE_PUBLISH_WORKFLOW} target="_blank" rel="noreferrer">
                    Publish GPT-2 web bundle
                  </a>{" "}
                  once, <em>Load model</em> will pull <span className="mono">gpt2-small</span> from the{" "}
                  <span className="mono">gpt2-web-v1</span> release when local <span className="mono">public/models/gpt2-small</span> is missing. For
                  offline dev or if the browser blocks those downloads: <span className="mono">npm run fetch:gpt2-web</span> or{" "}
                  <span className="mono">npm run prepare:gpt2-web</span>, then restart <span className="mono">npm run dev</span>.
                </p>
                <div className="btn-row">
                  <button type="button" className="primary" onClick={switchToTinyGptClear}>
                    Switch to tiny-gpt &amp; clear error
                  </button>
                  <button type="button" disabled={busy} onClick={loadModel}>
                    {loading ? "Loading…" : "Load model"}
                  </button>
                </div>
              </div>
            ) : null}
            <div className="field mb-0">
              <label htmlFor="llm-out">Output</label>
              <textarea id="llm-out" className="mono output-area" value={out} onChange={(e) => setOut(e.target.value)} spellCheck={false} />
            </div>
          </div>
        </section>

      <p className="muted mb-0" style={{ fontSize: 14 }}>
        How the model works and the train/export pipeline are on{" "}
        <a href="#/architecture">Architecture</a>.
      </p>
    </div>
  );
}
