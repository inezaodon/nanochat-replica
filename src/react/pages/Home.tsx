import React from "react";

export function Home() {
  return (
    <div className="grid">
      <section className="card">
        <div className="cardH">
          <h2>Project</h2>
          <div className="muted">Small LLM + Web UI</div>
        </div>
        <div className="cardB">
          <p>
            This repo is evolving from the Lab 1/2/3 concepts into a small, trainable GPT-style language model with a
            responsive React UI.
          </p>
          <ul>
            <li>
              <strong>Train</strong> a tiny GPT in Python (GPU if available)
            </li>
            <li>
              <strong>Export</strong> weights to a browser-friendly format
            </li>
            <li>
              <strong>Run inference</strong> in the browser for demos
            </li>
          </ul>
          <p className="muted">
            Open <strong>Small LLM</strong> to generate text once weights are exported.
          </p>
        </div>
      </section>

      <section className="card">
        <div className="cardH">
          <h2>Status</h2>
          <div className="muted">Next steps</div>
        </div>
        <div className="cardB">
          <div className="mono">
            <div>1) Add training pipeline (PyTorch)</div>
            <div>2) Export weights → public/models/</div>
            <div>3) Implement JS inference + sampling</div>
          </div>
        </div>
      </section>
    </div>
  );
}

