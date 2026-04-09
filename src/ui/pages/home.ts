export function renderHome(container: HTMLElement) {
  container.innerHTML = `
    <div class="grid">
      <section class="card">
        <div class="cardH">
          <h2>Shippable replica</h2>
          <div class="muted">Interactive, browser-only</div>
        </div>
        <div class="cardB">
          <p>
            This project reuses the ideas from your Lab 01/02/03 notebooks and turns them into a small web app:
          </p>
          <ul>
            <li><strong>Tokenizer</strong>: BPE on bytes (+ optional regex pre-splitting)</li>
            <li><strong>Embedding</strong>: token embedding lookup</li>
            <li><strong>Transformer</strong>: LayerNorm + self-attention + causal masking</li>
            <li><strong>GPT‑2 demo</strong>: 12 attention heads visualized</li>
          </ul>
          <p class="muted">
            Use the nav to open each interactive page. The original lab HTML exports are preserved under “Docs”.
          </p>
        </div>
      </section>

      <section class="card">
        <div class="cardH">
          <h2>How to run</h2>
          <div class="muted">Vite</div>
        </div>
        <div class="cardB">
          <div class="mono">
            <div>npm install</div>
            <div>npm run dev</div>
            <div style="margin-top:10px;">npm run build</div>
            <div>npm run preview</div>
          </div>
          <p class="muted" style="margin-top:10px;">
            Build outputs a static site to <span class="mono">dist/</span> suitable for Netlify, GitHub Pages, S3, etc.
          </p>
        </div>
      </section>
    </div>
  `;
}

