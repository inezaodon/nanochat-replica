import React from "react";
import { useReveal } from "../hooks/useReveal";

const LAB05_RLHF_SPACED = encodeURI("/Lab_05_RLHF 20.15.47.html");

export function LabsDocs() {
  const { ref, active } = useReveal();

  return (
    <section ref={ref} className={`card lift-reveal ${active ? "is-visible" : ""}`}>
      <div className="cardH">
        <h2>Course labs</h2>
        <div className="cardH-meta">Static HTML exports bundled with the repo</div>
      </div>
      <div className="cardB">
        <div className="labs-course-stack">
          <div className="lab-group">
            <h3 className="lab-group-title">Earlier labs</h3>
            <div className="lab-links">
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
            </div>
          </div>

          <div className="lab-group">
            <h3 className="lab-group-title">Lab 04 — RAG foundations</h3>
            <p className="lab-blurb">
              Retrieval-augmented generation grounds model outputs in external text: chunk documents, embed queries and passages, retrieve top matches, and inject them into the prompt. Core concepts include the RAG pipeline, dense retrieval, indexing, and context windows.
            </p>
            <div className="lab-links">
              <ul className="doc-list">
                <li>
                  <a href="/Lab_04_RAG_Foundations.html">Course intro — RAG foundations (written for this site)</a>
                </li>
                <li>
                  <a href="/Lab_04_Everything_Together.html">Notebook export — end-to-end RAG and related pieces</a>
                </li>
              </ul>
            </div>
          </div>

          <div className="lab-group">
            <h3 className="lab-group-title">Lab 05 — Systems, alignment, and agents</h3>
            <p className="lab-blurb">
              Goes beyond a minimal RAG demo into system design, supervised fine-tuning (SFT), preference learning and RLHF, and agent-style tool use. Core concepts include multi-stage RAG, instruction tuning, reward modeling, RLHF, and agent loops with retrieval or tools.
            </p>
            <div className="lab-links">
              <ul className="doc-list">
                <li>
                  <a href="/Lab_05_RAG_Systems.html">Course intro — RAG systems and related ideas (written for this site)</a>
                </li>
                <li>
                  <a href="/Lab_05_RAG.html">Notebook export — RAG variants and retrieval practice</a>
                </li>
                <li>
                  <a href="/Lab_05_AGENT.html">Notebook export — agents, tools, and orchestration</a>
                </li>
                <li>
                  <a href="/Lab_05_SFT.html">Notebook export — supervised fine-tuning (SFT)</a>
                </li>
                <li>
                  <a href="/Lab_05_SFT-2.html">Notebook export — SFT follow-on / extensions</a>
                </li>
                <li>
                  <a href="/Lab_05_RLHF.html">Notebook export — RLHF and preference learning</a>
                </li>
                <li>
                  <a href={LAB05_RLHF_SPACED}>Notebook export — RLHF (alternate dated export)</a>
                </li>
              </ul>
            </div>
          </div>
        </div>

        <p className="muted doc-footnote">
          These notebooks are preserved as standalone pages. Interactive training and the browser demo live under{" "}
          <a href="#/llm">Small LLM</a>.
        </p>
      </div>
    </section>
  );
}
