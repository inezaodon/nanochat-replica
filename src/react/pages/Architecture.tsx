import React, { useMemo, useState } from "react";
import { useReveal } from "../hooks/useReveal";

export function Architecture() {
  const revealHow = useReveal();
  const revealPipe = useReveal();
  const [showDiagramFallback, setShowDiagramFallback] = useState(false);

  const help = useMemo(
    () => [
      "1) Install deps: pip install -r requirements.txt && npm install",
      "2) Tiny model — train: python -m llm.train --data data/training_corpus.txt --device cpu",
      "2b) Bigger corpus: python -m llm.expand_corpus --out data/corpus_expanded.txt --include-local data/training_corpus.txt --hf-preset wikitext-103 ag_news",
      "3) Tiny model — export: python -m llm.export_web --ckpt checkpoints/tiny-gpt/model.pt --tokenizer checkpoints/tiny-gpt/tokenizer.json --out_dir public/models/tiny-gpt",
      "4) ND GPT-2 — maintainer runs Actions → Publish GPT-2 web bundle once (release gpt2-web-v1). Then Load model loads from disk or, if missing, straight from that release in the browser (large download).",
      "5) Run web: npm run dev — gpt2-small preset auto-selects when local or release files exist; press Load model.",
    ],
    [],
  );

  return (
    <div className="architecture-page">
      <section className="card card--flush home-section">
        <div className="cardB">
          <div className="hero-grid">
            <div>
              <p className="hero-kicker">Small-scale transformer</p>
              <h1 className="hero-title">
                A tiny GPT you can <span className="gradient-text">train and run</span> in your browser.
              </h1>
              <p className="hero-lede">
                Built from first principles: BPE tokenizer, transformer blocks, and a GPT-style decoder you train on your
                own text (default mix in <span className="mono">data/training_corpus.txt</span>), wrapped in a focused
                React playground.
              </p>
              <ul className="hero-list">
                <li>
                  <strong>GPT demo</strong> —{" "}
                  <a href="#/overview">load weights and sample</a> entirely in the browser.
                </li>
                <li>
                  <strong>Reproduce locally</strong> — Python training and export scripts ship with the repo.
                </li>
                <li>
                  <strong>Three.js lab</strong> —{" "}
                  <a href="#/three">orbiting 3D scene</a> (tone-mapped lighting, orbit controls).
                </li>
                <li>
                  <strong>Labs &amp; flashcards</strong> — on <a href="#/">Home</a>.
                </li>
                <li>
                  <strong>Architecture</strong> — how the stack works (this page).
                </li>
              </ul>
            </div>
            <figure className="media-card">
              <img
                src="https://images.unsplash.com/photo-1529101091764-c3526daf38fe?auto=format&fit=crop&w=900&q=80"
                alt="Abstract visualization suggesting neural computation"
                width={600}
                height={376}
                loading="lazy"
              />
              <figcaption>Tiny LM stack · tokenizer · decoder · browser inference</figcaption>
            </figure>
          </div>
        </div>
      </section>

      <div className="architecture-sections" aria-label="Deep dive sections">
        <section className="card home-section">
          <div className="cardB">
            <p className="section-idx">§1 — Corpus and tokenizer</p>
            <h2 className="slab-title">From raw text to token IDs the trainer and the tab both understand.</h2>
            <p className="slab-lede">
              You shape a corpus, optionally normalize it, then learn merges (BPE-style or byte-level) straight from
              that data. Export the vocab and merge table to JSON once: the Python trainer and the TypeScript runtime
              load the same artifact so optimization in PyTorch and sampling in the browser never disagree on token
              boundaries.
            </p>
            <ul className="slab-list">
              <li>
                <strong>Corpus</strong> — default mix in{" "}
                <span className="mono">data/training_corpus.txt</span>; swap in course notes, logs, or streamed
                sources when you want the model to specialize.
              </li>
              <li>
                <strong>Tokenizer</strong> — train merges on your text or import a preset; reuse identical encode/decode
                in export scripts and the web bundle.
              </li>
              <li>
                <strong>Vocab footprint</strong> — keep it tiny for teaching demos, or scale to{" "}
                <span className="mono">gpt2-small</span> when you need full GPT-2 behavior in the playground.
              </li>
            </ul>
          </div>
        </section>

        <section className="card home-section">
          <div className="cardB">
            <p className="section-idx">§2 — Architecture</p>
            <h2 className="slab-title">A GPT-style decoder: embeddings, blocks, language head.</h2>
            <p className="slab-lede">
              Once tokens are embedded, the model is the familiar stack—token and position embeddings, stacked
              transformer blocks with multi-head attention and MLP, then a projection to vocabulary logits. Train in
              PyTorch, run the same graph in the browser.
            </p>
            <ul className="slab-list">
              <li>
                <strong>Attention</strong> — SDPA-style paths where available, with a manual fallback for portability.
              </li>
              <li>
                <strong>Sampling</strong> — temperature and top‑k in TypeScript over the live logits stream.
              </li>
            </ul>
          </div>
        </section>

        <section className="card home-section">
          <div className="cardB">
            <p className="section-idx">§3 — Export and browser</p>
            <h2 className="slab-title">Ship weights once, then load them like any other static asset.</h2>
            <p className="slab-lede">
              The playground pulls a flat buffer plus a tiny manifest—no API server in the loop. Bundle the small
              checkpoint for lectures, or run <span className="mono">prepare_course_model</span> when you want the
              full GPT-2 stack in the tab.
            </p>
            <div className="grid" style={{ marginTop: 20 }}>
              <div>
                <div className="field-label">Web</div>
                <div className="mono">
                  <div>npm install</div>
                  <div>npm run dev</div>
                </div>
              </div>
              <div>
                <div className="field-label">Train</div>
                <div className="mono">
                  <div>python -m llm.train --device cpu</div>
                  <div>python -m llm.export_web …</div>
                </div>
              </div>
            </div>
            <p className="muted" style={{ marginTop: 18, marginBottom: 0, fontSize: 14 }}>
              Open <a href="#/overview">GPT demo</a> when you are ready to load a bundle and sample.
            </p>
          </div>
        </section>
      </div>

      <div className="arch-ref-grid">
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
                <div className="mono diagram-fallback">
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
      </div>
    </div>
  );
}
