import React from "react";
import { ExamFactsSlider } from "../components/ExamFactsSlider";

export function Home() {
  return (
    <>
      <section className="card card--flush" style={{ marginBottom: 22 }}>
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
                  <strong>Small LLM</strong> — load weights and sample entirely in the browser.
                </li>
                <li>
                  <strong>Reproduce locally</strong> — Python training and export scripts ship with the repo.
                </li>
                <li>
                  <strong>Three.js lab</strong> —{" "}
                  <a href="#/three">orbiting 3D scene</a> (tone-mapped lighting, orbit controls).
                </li>
                <li>
                  <strong>Labs</strong> — static course exports live under <em>Course labs</em>.
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

      <section className="card" style={{ marginBottom: 22 }} aria-label="Practice answer key flashcards">
        <div className="cardB">
          <p className="flashcard-decks-lede muted">
            Two decks from the course keys—each carousel only includes facts from that packet. Click a card to flip
            between the prompt side and the answer side.
          </p>
          <div className="flashcard-decks-grid">
            <div className="flashcard-deck">
              <ExamFactsSlider
                filterSource="exam01"
                deckTitle="Exam 01 — practice packet (answer key)"
                deckSubtitle="Facts auto-extracted from the Exam 01 practice answer key."
              />
            </div>
            <div className="flashcard-deck">
              <ExamFactsSlider
                filterSource="exam02"
                deckTitle="Practice Packet 02 — answer key"
                deckSubtitle="Facts auto-extracted from the Packet 02 answer key PDF."
              />
            </div>
          </div>
        </div>
      </section>

      <div className="sticky-stack" aria-label="Deep dive sections">
        <div className="stack-slab" style={{ zIndex: 10 }}>
          <section className="card">
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
        </div>

        <div className="stack-slab" style={{ zIndex: 11 }}>
          <section className="card">
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
                  <strong>Attention</strong> — SDPA-style paths where available, with a manual
                  fallback for portability.
                </li>
                <li>
                  <strong>Sampling</strong> — temperature and top‑k in TypeScript over the live
                  logits stream.
                </li>
              </ul>
            </div>
          </section>
        </div>

        <div className="stack-slab" style={{ zIndex: 12 }}>
          <section className="card">
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
                Open <a href="#/llm">Small LLM</a> when you are ready to load a bundle and sample.
              </p>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
