import React from "react";

export function Home() {
  return (
    <>
      <section className="card" style={{ marginBottom: 18 }}>
        <div className="cardB" style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ flex: "1 1 260px", minWidth: 0 }}>
            <p className="muted" style={{ textTransform: "uppercase", letterSpacing: "0.18em", marginBottom: 6 }}>
              Shakespeare‑sized transformer
            </p>
            <h1
              style={{
                margin: 0,
                fontSize: 30,
                lineHeight: 1.1,
                letterSpacing: "-0.05em",
              }}
            >
              A tiny GPT you can{" "}
              <span
                style={{
                  background: "linear-gradient(90deg,#38bdf8,#22c55e)",
                  WebkitBackgroundClip: "text",
                  color: "transparent",
                }}
              >
                train and run
              </span>{" "}
              in your browser.
            </h1>
            <p style={{ marginTop: 14, maxWidth: 520 }}>
              Built from first principles: BPE tokenizer, transformer blocks, and a GPT‑style decoder trained on
              Shakespeare, all wrapped in a modern React experience.
            </p>
            <ul style={{ marginTop: 14, paddingLeft: 18 }}>
              <li>
                <strong>Small LLM tab</strong> – talk to the tiny GPT directly in your browser.
              </li>
              <li>
                <strong>Reproduce locally</strong> – Python training + export scripts are included.
              </li>
              <li>
                <strong>Labs</strong> – original course notebooks are preserved under <em>Course labs</em>.
              </li>
            </ul>
          </div>
          <div style={{ flex: "0 0 260px", textAlign: "center" }}>
            <div
              style={{
                borderRadius: 18,
                overflow: "hidden",
                border: "1px solid rgba(148,163,184,.4)",
                background:
                  "radial-gradient(circle at 0 0,rgba(56,189,248,.35),transparent 55%),radial-gradient(circle at 100% 100%,rgba(16,185,129,.35),transparent 55%),#020617",
              }}
            >
              <img
                src="https://images.unsplash.com/photo-1529101091764-c3526daf38fe?auto=format&fit=crop&w=900&q=80"
                alt="Abstract visualization of a neural network"
                style={{ width: "100%", height: 180, objectFit: "cover", display: "block" }}
              />
              <div style={{ padding: "10px 12px 12px" }}>
                <p style={{ margin: 0, fontSize: 12 }} className="muted">
                  Tiny GPT with 2 layers · 12 heads · BPE tokenizer
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="grid">
        <section className="card">
          <div className="cardH">
            <h2>What’s inside</h2>
            <div className="muted">Architecture</div>
          </div>
          <div className="cardB">
            <ul>
              <li>
                <strong>Tokenizer</strong>: Regex‑based BPE trained on Shakespeare, exported to JSON and shared between
                Python and TypeScript.
              </li>
              <li>
                <strong>Model</strong>: GPT‑style decoder (embedding + positional + transformer blocks + lm_head) with
                12‑head attention.
              </li>
              <li>
                <strong>Inference</strong>: Pure TypeScript forward pass and sampling, running entirely in the browser.
              </li>
            </ul>
          </div>
        </section>

        <section className="card">
          <div className="cardH">
            <h2>Quickstart</h2>
            <div className="muted">Run + retrain</div>
          </div>
          <div className="cardB">
            <div className="mono" style={{ fontSize: 12 }}>
              <div># web demo</div>
              <div>npm install</div>
              <div>npm run dev</div>
            </div>
            <div style={{ height: 10 }} />
            <div className="mono" style={{ fontSize: 12 }}>
              <div># train tiny GPT</div>
              <div>python -m venv .venv</div>
              <div>source .venv/bin/activate</div>
              <div>pip install -r requirements.txt</div>
              <div>python -m llm.train --device cpu</div>
            </div>
          </div>
        </section>
      </div>
    </>
  );
}

