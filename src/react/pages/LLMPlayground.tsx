import React, { useEffect, useMemo, useRef, useState } from "react";
import { Gpt2TiktokenTokenizer } from "../../core/gpt2Tokenizer";
import { RegexBPETokenizer } from "../../core/tokenizer";
import { createTinyGPTWeb, loadTensors, TinyGPTWeb } from "../../core/inferTinyGPT";
import { fetchArrayBuffer, fetchJSON, WebManifest } from "../../core/webModel";

type ModelPreset = "tiny-gpt" | "gpt2-small";

export function LLMPlayground() {
  const [preset, setPreset] = useState<ModelPreset>("tiny-gpt");
  const [prompt, setPrompt] = useState("Hello from a tiny GPT.");
  const [status, setStatus] = useState<string>("Model not loaded yet.");
  const [out, setOut] = useState<string>("");
  const [model, setModel] = useState<TinyGPTWeb | null>(null);
  const gpt2TokenizerRef = useRef<Gpt2TiktokenTokenizer | null>(null);
  const [loading, setLoading] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(60);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [seed, setSeed] = useState(42);
  const [showDiagramFallback, setShowDiagramFallback] = useState(false);

  const help = useMemo(
    () => [
      "1) Install deps: pip install -r requirements.txt && npm install",
      "2) Tiny model — train: python -m llm.train --data data/training_corpus.txt --device cpu",
      "2b) Bigger corpus: python -m llm.expand_corpus --out data/corpus_expanded.txt --include-local data/training_corpus.txt --hf-preset wikitext-103 ag_news",
      "3) Tiny model — export: python -m llm.export_web --ckpt checkpoints/tiny-gpt/model.pt --tokenizer checkpoints/tiny-gpt/tokenizer.json --out_dir public/models/tiny-gpt",
      "4) ND GPT-2 release — download + export: python -m llm.prepare_course_model --out_dir public/models/gpt2-small",
      "5) Run web: npm run dev (pick preset, then Load model)",
    ],
    [],
  );

  useEffect(() => {
    return () => {
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
      setStatus((e as Error).message);
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
    setStatus("Done.");
  }

  return (
    <div className="grid">
      <section className="card">
        <div className="cardH">
          <h2>Small LLM</h2>
          <div className="muted">Interact with the tiny GPT</div>
        </div>
        <div className="cardB">
          <label>Prompt</label>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} spellCheck={false} />
          <div className="row" style={{ marginTop: 10 }}>
            <div>
              <label>Max new tokens</label>
              <input
                type="number"
                value={maxNewTokens}
                min={1}
                max={512}
                onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              />
            </div>
            <div>
              <label>Seed</label>
              <input type="number" value={seed} min={0} max={999999} onChange={(e) => setSeed(Number(e.target.value))} />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <div>
              <label>Temperature</label>
              <input
                type="number"
                step="0.05"
                value={temperature}
                min={0.1}
                max={2.0}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </div>
            <div>
              <label>Top‑K</label>
              <input type="number" value={topK} min={0} max={200} onChange={(e) => setTopK(Number(e.target.value))} />
            </div>
          </div>
          <div style={{ marginTop: 10 }}>
            <label>Model bundle</label>
            <select
              value={preset}
              onChange={(e) => setPreset(e.target.value as ModelPreset)}
              style={{ marginLeft: 8 }}
            >
              <option value="tiny-gpt">tiny-gpt (trained in-repo)</option>
              <option value="gpt2-small">gpt2-small (ND weights-v1 release, run prepare_course_model)</option>
            </select>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
            <button className="primary" disabled={loading} onClick={generate}>
              Generate
            </button>
            <button disabled={loading} onClick={loadModel}>
              {loading ? "Loading…" : "Load model"}
            </button>
            <button onClick={() => setPrompt("")}>Clear</button>
          </div>
          <div className="muted" style={{ marginTop: 10 }}>
            {status}
          </div>
          <div style={{ height: 10 }} />
          <label>Output</label>
          <textarea className="mono" value={out} onChange={(e) => setOut(e.target.value)} spellCheck={false} />
        </div>
      </section>

      <section className="card">
        <div className="cardH">
          <h2>How it works</h2>
          <div className="muted">Under the hood</div>
        </div>
        <div className="cardB">
          <p className="muted" style={{ marginTop: 0, marginBottom: 12 }}>
            Each generated token is sampled from logits produced by a forward pass over the current context, then appended
            to the prompt and repeated.
          </p>
          <div style={{ marginBottom: 12 }}>
            {!showDiagramFallback ? (
              <img
                src="https://jalammar.github.io/images/t/transformer_decoding_3.gif"
                alt="Illustration of a transformer decoding tokens"
                style={{ width: "100%", borderRadius: 12, border: "1px solid rgba(148,163,184,.4)" }}
                onError={() => setShowDiagramFallback(true)}
              />
            ) : (
              <div
                className="mono"
                style={{
                  border: "1px solid rgba(148,163,184,.4)",
                  borderRadius: 12,
                  padding: 12,
                  lineHeight: 1.5,
                  background: "rgba(15,23,42,.45)",
                  fontSize: 12,
                }}
              >{"prompt -> tokenizer -> token IDs -> transformer forward pass -> logits -> top-k/temperature sampling -> next token -> repeat"}</div>
            )}
          </div>
          <ul style={{ fontSize: 13 }}>
            <li>
              <strong>Context window</strong>: last {maxNewTokens} new tokens; model context is the bundle&apos;s{" "}
              <span className="mono">block_size</span> (128 for tiny-gpt, 1024 for gpt2-small).
            </li>
            <li>
              <strong>Generation controls</strong>: adjust temperature and top‑k to explore different creative modes.
            </li>
            <li>
              <strong>Determinism</strong>: set a fixed seed to reproduce the same outputs for demos.
            </li>
          </ul>
        </div>
      </section>

      <section className="card">
        <div className="cardH">
          <h2>How to retrain</h2>
          <div className="muted">GPU friendly</div>
        </div>
        <div className="cardB">
          <div className="mono">
            {help.map((x) => (
              <div key={x}>{x}</div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

