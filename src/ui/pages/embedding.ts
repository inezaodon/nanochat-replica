import { RegexBPETokenizer } from "../../core/tokenizer";
import { EmbeddingLayer } from "../../core/embedding";

export function renderEmbeddingPage(container: HTMLElement) {
  container.innerHTML = `
    <div class="grid">
      <section class="card">
        <div class="cardH">
          <h2>Embedding (Lab 02)</h2>
          <div class="muted">W[tokens] lookup</div>
        </div>
        <div class="cardB">
          <label for="prompt">Prompt</label>
          <textarea id="prompt" spellcheck="false">&lt;|sos|&gt;hello embedding world&lt;|eos|&gt;</textarea>

          <div class="row" style="margin-top:10px;">
            <div>
              <label for="vocabSize">Vocab size</label>
              <input id="vocabSize" type="number" min="260" max="50000" value="2048" />
            </div>
            <div>
              <label for="embedDim">Embedding dim</label>
              <input id="embedDim" type="number" min="4" max="512" value="64" />
            </div>
          </div>

          <div class="row" style="margin-top:10px;">
            <div>
              <label for="seed">Seed</label>
              <input id="seed" type="number" min="0" max="999999" value="42" />
            </div>
            <div>
              <label for="showN">Show first N values</label>
              <input id="showN" type="number" min="4" max="64" value="10" />
            </div>
          </div>

          <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">
            <button class="primary" id="runBtn">Embed</button>
          </div>
          <div id="meta" class="muted" style="margin-top:10px;"></div>
        </div>
      </section>

      <section class="card">
        <div class="cardH">
          <h2>Output</h2>
          <div class="muted">Token IDs and vectors</div>
        </div>
        <div class="cardB">
          <label>Token IDs</label>
          <textarea id="ids" class="mono" spellcheck="false"></textarea>
          <div style="height:10px;"></div>
          <label>Embeddings (per token, first N dims)</label>
          <textarea id="vecs" class="mono" spellcheck="false"></textarea>
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
    const vocabSize = Number((container.querySelector("#vocabSize") as HTMLInputElement).value || 2048);
    const embedDim = Number((container.querySelector("#embedDim") as HTMLInputElement).value || 64);
    const seed = Number((container.querySelector("#seed") as HTMLInputElement).value || 42);
    const showN = Math.max(1, Math.min(embedDim, Number((container.querySelector("#showN") as HTMLInputElement).value || 10)));

    tok.train(prompt, Math.max(260, Math.min(4096, vocabSize))); // small local train for demo
    const ids = tok.encode(prompt).map((x) => x % vocabSize);

    const emb = new EmbeddingLayer(vocabSize, embedDim, seed);
    const Y = emb.forward(ids); // shape (T, embedDim) flattened

    (container.querySelector("#ids") as HTMLTextAreaElement).value = JSON.stringify(ids);

    const lines: string[] = [];
    for (let t = 0; t < ids.length; t++) {
      const row: string[] = [];
      const base = t * embedDim;
      for (let i = 0; i < showN; i++) row.push(Y[base + i].toFixed(4));
      lines.push(`t${t} id=${ids[t]} [${row.join(", ")}${showN < embedDim ? ", …" : ""}]`);
    }
    (container.querySelector("#vecs") as HTMLTextAreaElement).value = lines.join("\n");

    setMeta(`shape: (T=${ids.length}, D=${embedDim})`);
  }

  container.querySelector("#runBtn")?.addEventListener("click", run);
  setMeta("ready");
}

