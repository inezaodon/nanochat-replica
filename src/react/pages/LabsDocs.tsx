import React from "react";
import { useReveal } from "../hooks/useReveal";

export function LabsDocs() {
  const { ref, active } = useReveal();

  return (
    <section ref={ref} className={`card lift-reveal ${active ? "is-visible" : ""}`}>
      <div className="cardH">
        <h2>Course labs</h2>
        <div className="cardH-meta">Static HTML exports bundled with the repo</div>
      </div>
      <div className="cardB">
        <ul className="doc-list">
          <li>
            <a href="/Lab_01_Tokenization.html">Lab 01 — Tokenization</a>
          </li>
          <li>
            <a href="/Lab_02_Embedding.html">Lab 02 — Embedding</a>
          </li>
          <li>
            <a href="/Lab_03_Transformer_Block.html">Lab 03 — Transformer block</a>
          </li>
          <li>
            <a href="/legacy/GPT2_Replica_12Heads.html">GPT‑2 replica (12 heads)</a>
          </li>
        </ul>
        <p className="muted doc-footnote">
          These notebooks are preserved as standalone pages. Interactive training and the browser demo live under{" "}
          <a href="#/llm">Small LLM</a>.
        </p>
      </div>
    </section>
  );
}
