import React, { useEffect, useMemo, useRef, useState } from "react";
import { Gpt2TiktokenTokenizer } from "../../core/gpt2Tokenizer";
import { RegexBPETokenizer } from "../../core/tokenizer";
import { createTinyGPTWeb, loadTensors, TinyGPTWeb } from "../../core/inferTinyGPT";
import { fetchArrayBuffer, fetchJSON, WebManifest } from "../../core/webModel";
import { useReveal } from "../hooks/useReveal";

type ModelPreset = "tiny-gpt" | "gpt2-small";

const GPT2_MANIFEST = "/models/gpt2-small/manifest.json";

export function LLMPlayground() {
  const [preset, setPreset] = useState<ModelPreset>("tiny-gpt");
  const [prompt, setPrompt] = useState("Hello from a tiny GPT.");
  const [status, setStatus] = useState<string>("Checking which model bundles are available…");
  const [out, setOut] = useState<string>("");
  const [model, setModel] = useState<TinyGPTWeb | null>(null);
  const gpt2TokenizerRef = useRef<Gpt2TiktokenTokenizer | null>(null);
  const [loading, setLoading] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(60);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [seed, setSeed] = useState(42);
  const [showDiagramFallback, setShowDiagramFallback] = useState(false);

  const revealMain = useReveal();
  const revealHow = useReveal();
  const revealPipe = useReveal();

  const help = useMemo(
    () => [
      "1) Install deps: pip install -r requirements.txt && npm install",
      "2) Tiny model — train: python -m llm.train --data data/training_corpus.txt --device cpu",
      "2b) Bigger corpus: python -m llm.expand_corpus --out data/corpus_expanded.txt --include-local data/training_corpus.txt --hf-preset wikitext-103 ag_news",
      "3) Tiny model — export: python -m llm.export_web --ckpt checkpoints/tiny-gpt/model.pt --tokenizer checkpoints/tiny-gpt/tokenizer.json --out_dir public/models/tiny-gpt",
      "4) ND GPT-2 release — download + export: python -m llm.prepare_course_model --out_dir public/models/gpt2-small",
      "5) Run web: npm run dev — if gpt2-small exists under public/models, the preset switches automatically; press Load model.",
    ],
    [],
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(GPT2_MANIFEST, { cache: "no-store" });
        if (cancelled) return;
        if (r.ok) {
          setPreset("gpt2-small");
          setStatus(
            "gpt2-small bundle found under public/models — press Load model. (tiny-gpt in the dropdown is a toy char-LM for demos.)",
          );
        } else {
          setStatus(
            "Only tiny-gpt is checked in. For fluent English in the browser, run: python -m llm.prepare_course_model --out_dir public/models/gpt2-small — then refresh and Load model.",
          );
        }
      } catch {
        if (!cancelled) {
          setStatus(
            "Could not probe for gpt2-small (offline?). Default is tiny-gpt — a small char-LM, not GPT-2. Use prepare_course_model when online.",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
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
      const base = preset === "tiny-gpt" ? "/models/tiny-gpt" : "/models/gpt2-small";
      const manifest = await fetchJSON<WebManifest>(`${base}/manifest.json`);
      const tokObj = await fetchJSON<{
        tokenizer_type?: string;
        merges?: Record<string, number>;
        vocab: Record<string, string>;
        special_tokens: Record<string, number>;
        pattern?: string;
      }>(`${base}/tokenizer.json`);

      let tokenizer: RegexBPETokenizer | Gpt2TiktokenTokenizer;
      if (tokObj.tokenizer_type === "gpt2_tiktoken" || manifest.tokenizer_type === "gpt2_tiktoken") {
        const g2 = new Gpt2TiktokenTokenizer();
        gpt2TokenizerRef.current = g2;
        tokenizer = g2;
      } else {
        tokenizer = RegexBPETokenizer.fromJSON(tokObj);
      }

      const buf = await fetchArrayBuffer(`${base}/${manifest.weights}`);
      const tensors = loadTensors(buf, manifest);
      const m = createTinyGPTWeb(manifest, tokenizer, tensors);
      setModel(m);
      setStatus(
        `Loaded ${preset}: vocab=${manifest.config.vocab_size}, layers=${manifest.config.n_layer}, heads=${manifest.config.n_head}, embd=${manifest.config.n_embd}`,
      );
    } catch (e) {
      const msg = (e as Error).message;
      const hint =
        preset === "gpt2-small"
          ? `${msg} If the bundle is missing, run: python -m llm.prepare_course_model --out_dir public/models/gpt2-small`
          : msg;
      setStatus(hint);
      setModel(null);
    } finally {
      setLoading(false);
    }
  }

  function generate() {
    if (!model) {
      setStatus("Load the model first.");
      return;
    }
    setStatus("Generating…");
    const text = model.generate(prompt, { maxNewTokens, temperature, topK, seed });
    setOut(text);
    const cfg = model.manifest.config;
    const toy =
      cfg.vocab_size < 4096 || cfg.n_embd < 256
        ? " (Toy character-scale model — fluent English is not expected. Use gpt2-small preset after `prepare_course_model` for GPT-2.)"
        : "";
    setStatus(`Done.${toy}`);
  }

  return (
    <div className="play-layout">
      <div className="play-main stack-gap">
        <section ref={revealMain.ref} className={`card lift-reveal ${revealMain.active ? "is-visible" : ""}`}>
          <div className="cardH">
            <h2>In-browser inference</h2>
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
                  max={512}
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
                <option value="gpt2-small">gpt2-small (ND weights-v1 — run prepare_course_model)</option>
              </select>
            </div>
            <div className="btn-row">
              <button type="button" className="primary" disabled={loading} onClick={generate}>
                Generate
              </button>
              <button type="button" disabled={loading} onClick={loadModel}>
                {loading ? "Loading…" : "Load model"}
              </button>
              <button type="button" onClick={() => setPrompt("")}>
                Clear prompt
              </button>
            </div>
            <div className="status-line" role="status">
              {status}
            </div>
            <div className="field mb-0">
              <label htmlFor="llm-out">Output</label>
              <textarea id="llm-out" className="mono output-area" value={out} onChange={(e) => setOut(e.target.value)} spellCheck={false} />
            </div>
          </div>
        </section>
      </div>

      <aside className="play-aside" aria-label="Reference">
        <section ref={revealHow.ref} className={`card lift-reveal ${revealHow.active ? "is-visible" : ""}`}>
          <div className="cardH">
            <h2>How it works</h2>
            <div className="cardH-meta">Sampling loop</div>
          </div>
          <div className="cardB stack-gap">
            <p className="muted mb-0">
              Each step appends one token: forward pass → logits → temperature / top‑k → sample → extend context.
            </p>
            <div className="diagram-frame">
              {!showDiagramFallback ? (
                <img
                  src="https://jalammar.github.io/images/t/transformer_decoding_3.gif"
                  alt="Transformer decoding tokens one at a time"
                  loading="lazy"
                  onError={() => setShowDiagramFallback(true)}
                />
              ) : (
                <div className="mono" style={{ padding: 14, fontSize: 12, lineHeight: 1.55, background: "rgba(9,9,11,.65)" }}>
                  prompt → tokenizer → token IDs → transformer → logits → top‑k / temperature → next token → repeat
                </div>
              )}
            </div>
            <ul className="how-list">
              <li>
                <strong>Context</strong> — effective window is the bundle&apos;s <span className="mono">block_size</span>{" "}
                (128 tiny-gpt, 1024 gpt2-small); max new tokens caps continuation length.
              </li>
              <li>
                <strong>Controls</strong> — higher temperature adds randomness; top‑k trims the tail of the distribution.
              </li>
              <li>
                <strong>Seed</strong> — fixed seed makes runs reproducible for teaching and demos.
              </li>
            </ul>
          </div>
        </section>

        <section ref={revealPipe.ref} className={`card lift-reveal ${revealPipe.active ? "is-visible" : ""}`}>
          <div className="cardH">
            <h2>Pipeline</h2>
            <div className="cardH-meta">From corpus to browser</div>
          </div>
          <div className="cardB">
            <div className="mono pipeline">
              {help.map((x) => (
                <div key={x}>{x}</div>
              ))}
            </div>
          </div>
        </section>
      </aside>
    </div>
  );
}
