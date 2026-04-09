import { mulberry32, TransformerBlock } from "../../core/transformer";
import { RegexBPETokenizer } from "../../core/tokenizer";
import { EmbeddingLayer } from "../../core/embedding";

function heatColor(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  const c0 = [30, 41, 59];
  const c1 = [96, 165, 250];
  const c2 = [52, 211, 153];
  const mid = 0.5;
  const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
  let r: number, g: number, b: number;
  if (t < mid) {
    const u = t / mid;
    r = lerp(c0[0], c1[0], u);
    g = lerp(c0[1], c1[1], u);
    b = lerp(c0[2], c1[2], u);
  } else {
    const u = (t - mid) / (1 - mid);
    r = lerp(c1[0], c2[0], u);
    g = lerp(c1[1], c2[1], u);
    b = lerp(c1[2], c2[2], u);
  }
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

function drawHeatmap(canvas: HTMLCanvasElement, A: Float32Array, T: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const pad = 24;
  const cell = 16;
  canvas.width = Math.max(220, pad + T * cell + 8);
  canvas.height = Math.max(220, pad + T * cell + 8);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(15,23,42,0.25)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // cells
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const v = A[i * T + j];
      ctx.fillStyle = heatColor(v);
      ctx.fillRect(pad + j * cell, pad + i * cell, cell - 1, cell - 1);
    }
  }
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.strokeRect(pad - 1, pad - 1, T * cell + 1, T * cell + 1);

  ctx.font =
    "10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
  ctx.fillStyle = "rgba(229,231,235,0.7)";
  for (let t = 0; t < T; t++) {
    ctx.fillText(String(t), pad + t * cell + 3, 16);
    ctx.fillText(String(t), 6, pad + t * cell + 12);
  }
}

export function renderTransformerPage(container: HTMLElement) {
  container.innerHTML = `
    <div class="grid">
      <section class="card">
        <div class="cardH">
          <h2>Transformer block (Lab 03)</h2>
          <div class="muted">LayerNorm → Self-Attention → Residual</div>
        </div>
        <div class="cardB">
          <label for="prompt">Prompt</label>
          <textarea id="prompt" spellcheck="false">&lt;|sos|&gt;Thou art more lovely and more temperate&lt;|eos|&gt;</textarea>

          <div class="row" style="margin-top:10px;">
            <div>
              <label for="embedDim">d_model</label>
              <input id="embedDim" type="number" min="24" max="192" step="12" value="48" />
            </div>
            <div>
              <label for="nHead">n_head</label>
              <input id="nHead" type="number" min="1" max="12" step="1" value="12" />
            </div>
          </div>

          <div class="row" style="margin-top:10px;">
            <div>
              <label for="maxTokens">Max tokens</label>
              <input id="maxTokens" type="number" min="2" max="24" step="1" value="12" />
            </div>
            <div>
              <label for="seed">Seed</label>
              <input id="seed" type="number" min="0" max="999999" value="42" />
            </div>
          </div>

          <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">
            <button class="primary" id="runBtn">Run 1 block</button>
          </div>
          <div id="meta" class="muted" style="margin-top:10px;"></div>
        </div>
      </section>

      <section class="card">
        <div class="cardH">
          <h2>Attention head</h2>
          <div class="muted">Pick a head to visualize</div>
        </div>
        <div class="cardB">
          <div class="row">
            <div>
              <label for="headIdx">Head</label>
              <input id="headIdx" type="number" min="1" max="12" value="1" />
            </div>
            <div>
              <label for="causal">Masking</label>
              <select id="causal">
                <option value="1" selected>Causal (GPT-style)</option>
                <option value="0">Non-causal</option>
              </select>
            </div>
          </div>
          <div style="height:10px;"></div>
          <canvas id="map"></canvas>
          <div style="height:10px;"></div>
          <div class="muted mono" id="shape"></div>
        </div>
      </section>
    </div>
  `;

  const tok = new RegexBPETokenizer();

  function setMeta(msg: string) {
    const m = container.querySelector("#meta");
    if (m) m.textContent = msg;
  }

  function run() {
    const prompt = (container.querySelector("#prompt") as HTMLTextAreaElement).value;
    const dModel = Number((container.querySelector("#embedDim") as HTMLInputElement).value || 48);
    const nHead = Number((container.querySelector("#nHead") as HTMLInputElement).value || 12);
    const maxTokens = Number((container.querySelector("#maxTokens") as HTMLInputElement).value || 12);
    const seed = Number((container.querySelector("#seed") as HTMLInputElement).value || 42);
    const headIdx = Number((container.querySelector("#headIdx") as HTMLInputElement).value || 1);
    const causal = (container.querySelector("#causal") as HTMLSelectElement).value === "1";

    if (dModel % nHead !== 0) {
      setMeta("d_model must be divisible by n_head.");
      return;
    }

    tok.train(prompt, 1024);
    const ids = tok.encode(prompt).slice(0, maxTokens);

    const vocabSize = 2048;
    const emb = new EmbeddingLayer(vocabSize, dModel, seed);
    const X = emb.forward(ids.map((x) => x % vocabSize)); // (T, dModel)
    const T = ids.length;

    // Block weights seeded separately from embedding.
    const rng = mulberry32((seed ^ 0x13579bdf) >>> 0);
    const block = new TransformerBlock(rng, dModel, nHead);
    const { AHeads } = block.forward(X, T, causal);

    const h = Math.max(1, Math.min(nHead, headIdx)) - 1;
    const A = AHeads[h];
    const canvas = container.querySelector("#map") as HTMLCanvasElement;
    drawHeatmap(canvas, A, T);
    const shape = container.querySelector("#shape");
    if (shape) shape.textContent = `A (head ${h + 1}) shape: ${T}×${T}`;
    setMeta(`tokens=${T}, d_model=${dModel}, n_head=${nHead}, causal=${causal ? "on" : "off"}`);
  }

  container.querySelector("#runBtn")?.addEventListener("click", run);
  container.querySelector("#headIdx")?.addEventListener("change", run);
  container.querySelector("#causal")?.addEventListener("change", run);

  setMeta("ready");
  run();
}

