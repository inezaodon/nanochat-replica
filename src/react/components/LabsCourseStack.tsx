import React from "react";

const LAB05_RLHF_SPACED = encodeURI("/Lab_05_RLHF 20.15.47.html");

export function LabsCourseStack({ showFootnote = true }: { showFootnote?: boolean }) {
  return (
    <>
        <div className="labs-course-stack">
          <div className="lab-group">
            <h3 className="lab-group-title">Lab 01 — Tokenization</h3>
            <p className="lab-blurb">
              How raw text becomes token IDs: character sets (ASCII, Unicode, UTF-8), corpora, simple n-gram views of
              language, vocabularies, OOV issues, and byte-pair encoding (BPE) including merges and GPT-style special
              tokens. Core concepts: tokens vs words, subwords, tokenizer training vs inference, and why tokenization
              affects what the model can count or spell.
            </p>
            <div className="lab-links">
              <ul className="doc-list">
                <li>
                  <a href="/Lab_01_Tokenization.html">Lab 01 — Tokenization (notebook export)</a>
                </li>
              </ul>
            </div>
          </div>

          <div className="lab-group">
            <h3 className="lab-group-title">Lab 02 — Embeddings</h3>
            <p className="lab-blurb">
              From one-hot rows to learned dense vectors: embedding matrices, lookup by token ID, geometry (distance,
              similarity), and how embeddings sit at the input of a sequence model. Core concepts: high-dimensional
              representation, sparsity vs dense features, and embeddings as the bridge between discrete tokens and
              continuous math.
            </p>
            <div className="lab-links">
              <ul className="doc-list">
                <li>
                  <a href="/Lab_02_Embedding.html">Lab 02 — Embedding (notebook export)</a>
                </li>
              </ul>
            </div>
          </div>

          <div className="lab-group">
            <h3 className="lab-group-title">Lab 03 — Transformer block</h3>
            <p className="lab-blurb">
              The modular core of modern LMs: self-attention (queries, keys, values, masking), multi-head attention,
              position-wise feed-forward layers, residuals, and normalization. Core concepts: causal vs bidirectional
              attention, depth as repeated blocks, and how one block mixes information across the sequence.
            </p>
            <div className="lab-links">
              <ul className="doc-list">
                <li>
                  <a href="/Lab_03_Transformer_Block.html">Lab 03 — Transformer block (notebook export)</a>
                </li>
              </ul>
            </div>
          </div>

          <div className="lab-group">
            <h3 className="lab-group-title">Lab 04 — RAG foundations</h3>
            <p className="lab-blurb">
              Retrieval-augmented generation grounds model outputs in external text: chunk documents, embed queries and
              passages, retrieve top matches, and inject them into the prompt. Core concepts include the RAG pipeline,
              dense retrieval, indexing, and context windows.
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
              Goes beyond a minimal RAG demo into system design, supervised fine-tuning (SFT), preference learning and
              RLHF, and agent-style tool use. Core concepts include multi-stage RAG, instruction tuning, reward
              modeling, RLHF, and agent loops with retrieval or tools.
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

          <div className="lab-group lab-group--other">
            <h3 className="lab-group-title">Other resources</h3>
            <p className="lab-blurb">
              Legacy visualizations and interactive pages that accompany the course material but are not numbered lab
              notebooks.
            </p>
            <div className="lab-links">
              <ul className="doc-list">
                <li>
                  <a href="/legacy/GPT2_Replica_12Heads.html">GPT‑2 replica (12 heads) — legacy attention visualization</a>
                </li>
                <li>
                  <a href="#/three">Three.js lab — spatial intuition demo (orbit controls, scene legend on page)</a>
                </li>
                <li>
                  <a href="#/overview">Small LLM — in-browser inference and tokenizer/manifest pipeline</a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      {showFootnote ? (
        <p className="muted doc-footnote">
          Interactive training and browser inference live under <a href="#/overview">GPT demo</a>. The{" "}
          <a href="#/three">Three.js lab</a> is a separate spatial demo for teaching, not a graded lab export.
        </p>
      ) : null}
    </>
  );
}
