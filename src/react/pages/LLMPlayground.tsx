import React, { useMemo, useState } from "react";
import { RegexBPETokenizer } from "../../core/tokenizer";
import { createTinyGPTWeb, loadTensors, TinyGPTWeb } from "../../core/inferTinyGPT";
import { fetchArrayBuffer, fetchJSON, WebManifest } from "../../core/webModel";

export function LLMPlayground() {
  const [prompt, setPrompt] = useState("Hello from a tiny GPT.");
  const [status, setStatus] = useState<string>("Model not loaded yet.");
  const [out, setOut] = useState<string>("");
  const [model, setModel] = useState<TinyGPTWeb | null>(null);
  const [loading, setLoading] = useState(false);
  const [maxNewTokens, setMaxNewTokens] = useState(60);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [seed, setSeed] = useState(42);

  const help = useMemo(
    () => [
      "1) Install deps: pip install -r requirements.txt",
      "2) Train: python -m llm.train --data data/shakespeare.txt --device cuda (or mps/cpu)",
      "3) Export: python -m llm.export_web --ckpt checkpoints/tiny-gpt/model.pt --tokenizer checkpoints/tiny-gpt/tokenizer.json --out_dir public/models/tiny-gpt",
      "4) Run web: npm run dev (then Load model below)",
    ],
    [],
  );

  async function loadModel() {
    setLoading(true);
    setStatus("Loading manifest/tokenizer/weights…");
    try {
      const manifest = await fetchJSON<WebManifest>("/models/tiny-gpt/manifest.json");
      const tokObj = await fetchJSON<{
        merges: Record<string, number>;
        vocab: Record<string, string>;
        special_tokens: Record<string, number>;
        pattern?: string;
      }>("/models/tiny-gpt/tokenizer.json");
      const tokenizer = RegexBPETokenizer.fromJSON(tokObj);
      const buf = await fetchArrayBuffer(`/models/tiny-gpt/${manifest.weights}`);
      const tensors = loadTensors(buf, manifest);
      const m = createTinyGPTWeb(manifest, tokenizer, tensors);
      setModel(m);
      setStatus(
        `Loaded tiny-gpt: vocab=${manifest.config.vocab_size}, layers=${manifest.config.n_layer}, heads=${manifest.config.n_head}, embd=${manifest.config.n_embd}`,
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
                max={256}
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
          <div style={{ marginBottom: 12 }}>
            <img
              src="https://jalammar.github.io/images/t/transformer_decoding_3.gif"
              alt="Illustration of a transformer decoding tokens"
              style={{ width: "100%", borderRadius: 12, border: "1px solid rgba(148,163,184,.4)" }}
            />
          </div>
          <ul style={{ fontSize: 13 }}>
            <li>
              <strong>Context window</strong>: last {maxNewTokens} tokens within a block of 128 positions.
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

