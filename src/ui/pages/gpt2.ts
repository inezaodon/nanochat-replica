import { mulberry32, TransformerBlock } from "../../core/transformer";
import { EmbeddingLayer } from "../../core/embedding";
import { RegexBPETokenizer } from "../../core/tokenizer";

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

function drawHeatmap(canvas: HTMLCanvasElement, A: Float32Array, T: number, tokensHint: string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const pad = 26;
  const cell = 14;
  canvas.width = Math.max(200, pad + T * cell + 8);
  canvas.height = Math.max(200, pad + T * cell + 20);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(15,23,42,0.25)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
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
    ctx.fillText(String(t), pad + t * cell + 2, 16);
    ctx.fillText(String(t), 6, pad + t * cell + 11);
  }
  ctx.fillStyle = "rgba(229,231,235,0.55)";
  ctx.fillText(tokensHint, pad, canvas.height - 8);
}

export function renderGPT2HeadsPage(container: HTMLElement) {
  container.innerHTML = `
    <div class="card">
      <div class="cardH">
        <h2>GPT‑2 style attention (12 heads)</h2>
        <div class="muted">Choose a layer, see all 12 heads</div>
      </div>
      <div class="cardB">
        <div class="row">
          <div>
            <label for="prompt">Prompt</label>
            <textarea id="prompt" spellcheck="false">The quick brown fox jumps over the lazy dog.</textarea>
          </div>
          <div>
            <label for="nLayer">n_layer</label>
            <input id="nLayer" type="number" min="1" max="24" value="12" />
            <div style="height:10px;"></div>
            <label for="layerIdx">View layer</label>
            <select id="layerIdx"></select>
            <div style="height:10px;"></div>
            <label for="maxTokens">Max tokens</label>
            <input id="maxTokens" type="number" min="2" max="24" value="16" />
            <div style="height:10px;"></div>
            <label for="seed">Seed</label>
            <input id="seed" type="number" min="0" max="999999" value="42" />
            <div style="height:10px;"></div>
            <button class="primary" id="runBtn">Run</button>
          </div>
        </div>

        <div id="meta" class="muted" style="margin-top:10px;"></div>
        <div id="heads" style="margin-top:12px; display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:10px;"></div>
      </div>
    </div>
  `;

  const N_HEAD = 12;
  const D_MODEL = 48;
  const tok = new RegexBPETokenizer();

  function setMeta(msg: string) {
    const m = container.querySelector("#meta");
    if (m) m.textContent = msg;
  }

  function updateLayerSelect(nLayer: number) {
    const sel = container.querySelector("#layerIdx") as HTMLSelectElement;
    const prev = sel.value;
    sel.innerHTML = "";
    for (let i = 0; i < nLayer; i++) {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = `Layer ${i + 1} / ${nLayer}`;
      sel.appendChild(o);
    }
    if (prev && Number(prev) < nLayer) sel.value = prev;
  }

  function run() {
    const prompt = (container.querySelector("#prompt") as HTMLTextAreaElement).value;
    const nLayer = Math.max(1, Math.min(24, Number((container.querySelector("#nLayer") as HTMLInputElement).value || 12)));
    const maxTokens = Math.max(2, Math.min(24, Number((container.querySelector("#maxTokens") as HTMLInputElement).value || 16)));
    const seed = Math.max(0, Math.min(999999, Number((container.querySelector("#seed") as HTMLInputElement).value || 42)));
    (container.querySelector("#nLayer") as HTMLInputElement).value = String(nLayer);
    (container.querySelector("#maxTokens") as HTMLInputElement).value = String(maxTokens);
    (container.querySelector("#seed") as HTMLInputElement).value = String(seed);

    updateLayerSelect(nLayer);
    const layerIdx = Math.max(0, Math.min(nLayer - 1, Number((container.querySelector("#layerIdx") as HTMLSelectElement).value || 0)));
    (container.querySelector("#layerIdx") as HTMLSelectElement).value = String(layerIdx);

    tok.train(prompt, 2048);
    const ids = tok.encode(prompt).slice(0, maxTokens);
    const T = ids.length;

    const vocabSize = 2048;
    const emb = new EmbeddingLayer(vocabSize, D_MODEL, seed);
    const X = emb.forward(ids.map((x) => x % vocabSize));

    // Build all layers (so "view layer" is meaningful)
    const layers: TransformerBlock[] = [];
    for (let l = 0; l < nLayer; l++) {
      const rng = mulberry32((seed ^ (0x9e3779b9 * (l + 1))) >>> 0);
      layers.push(new TransformerBlock(rng, D_MODEL, N_HEAD));
    }

    const { AHeads } = layers[layerIdx].forward(X, T, true);
    setMeta(`tokens=${T}, d_model=${D_MODEL}, n_head=${N_HEAD}, layer=${layerIdx + 1}/${nLayer}, causal=on`);

    const hint = ids.slice(0, 6).map((x) => String(x)).join(" ");
    const heads = container.querySelector("#heads") as HTMLDivElement;
    heads.innerHTML = "";
    for (let h = 0; h < N_HEAD; h++) {
      const card = document.createElement("div");
      card.className = "card";
      card.style.boxShadow = "none";
      const hTop = document.createElement("div");
      hTop.className = "cardH";
      hTop.innerHTML = `<h2>Head ${h + 1}</h2><div class="muted mono">${T}×${T}</div>`;
      const hBody = document.createElement("div");
      hBody.className = "cardB";
      const canvas = document.createElement("canvas");
      canvas.style.width = "100%";
      canvas.style.borderRadius = "12px";
      hBody.appendChild(canvas);
      card.appendChild(hTop);
      card.appendChild(hBody);
      heads.appendChild(card);
      drawHeatmap(canvas, AHeads[h], T, hint);
    }
  }

  (container.querySelector("#runBtn") as HTMLButtonElement).addEventListener("click", run);
  (container.querySelector("#nLayer") as HTMLInputElement).addEventListener("change", () => {
    updateLayerSelect(Number((container.querySelector("#nLayer") as HTMLInputElement).value || 12));
    run();
  });
  (container.querySelector("#layerIdx") as HTMLSelectElement).addEventListener("change", run);

  updateLayerSelect(12);
  setMeta("ready");
  run();
}

