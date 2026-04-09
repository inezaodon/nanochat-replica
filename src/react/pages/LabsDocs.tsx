import React from "react";

export function LabsDocs() {
  return (
    <div className="card">
      <div className="cardH">
        <h2>Docs</h2>
        <div className="muted">Legacy lab exports</div>
      </div>
      <div className="cardB">
        <ul>
          <li>
            <a href="/Lab_01_Tokenization.html">Lab_01_Tokenization.html</a>
          </li>
          <li>
            <a href="/Lab_02_Embedding.html">Lab_02_Embedding.html</a>
          </li>
          <li>
            <a href="/Lab_03_Transformer_Block.html">Lab_03_Transformer_Block.html</a>
          </li>
          <li>
            <a href="/legacy/GPT2_Replica_12Heads.html">GPT2_Replica_12Heads.html</a>
          </li>
        </ul>
        <p className="muted">These are static notebook exports; the LLM work lives in the React app.</p>
      </div>
    </div>
  );
}

