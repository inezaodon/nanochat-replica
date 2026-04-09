import { RegexBPETokenizer } from "../../core/tokenizer";

export function renderTokenizerPage(container: HTMLElement) {
  container.innerHTML = `
    <div class="grid">
      <section class="card">
        <div class="cardH">
          <h2>Tokenizer (Lab 01)</h2>
          <div class="muted">Character-level tokenization</div>
        </div>
        <div class="cardB">
          <label for="tokText">Training text</label>
          <textarea id="tokText" spellcheck="false">to be or not to be, that is the question</textarea>

          <div class="row" style="margin-top:10px;">
            <div>
              <label for="vocabSize">Max vocab size</label>
              <input id="vocabSize" type="number" min="260" max="4096" value="400" />
            </div>
            <div>
              <label for="prompt">Encode prompt</label>
              <input id="prompt" type="text" value="<|sos|>to be or not to be<|eos|>" />
            </div>
          </div>

          <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">
            <button class="primary" id="trainBtn">Train tokenizer</button>
            <button id="encodeBtn">Encode</button>
            <button id="decodeBtn">Decode</button>
          </div>

          <div id="tokMeta" class="muted" style="margin-top:10px;"></div>
        </div>
      </section>

      <section class="card">
        <div class="cardH">
          <h2>Output</h2>
          <div class="muted">IDs / decoded text</div>
        </div>
        <div class="cardB">
          <label>Token IDs</label>
          <textarea id="ids" class="mono" spellcheck="false"></textarea>
          <div style="height:10px;"></div>
          <label>Decoded</label>
          <textarea id="decoded" class="mono" spellcheck="false"></textarea>
        </div>
      </section>
    </div>
  `;

  const $ = (id: string) => container.querySelector(`#${id}`) as HTMLElement | null;

  const tok = new RegexBPETokenizer();
  let trained = false;

  function setMeta(msg: string) {
    const m = $("tokMeta");
    if (m) m.textContent = msg;
  }

  function train() {
    const text = (container.querySelector("#tokText") as HTMLTextAreaElement).value;
    const vocabSize = Number((container.querySelector("#vocabSize") as HTMLInputElement).value || 400);
    tok.train(text, vocabSize);
    trained = true;
    setMeta(`trained: vocab=${tok.vocab.size}`);
  }

  function encode() {
    if (!trained) train();
    const prompt = (container.querySelector("#prompt") as HTMLInputElement).value;
    const ids = tok.encode(prompt);
    (container.querySelector("#ids") as HTMLTextAreaElement).value = JSON.stringify(ids);
    (container.querySelector("#decoded") as HTMLTextAreaElement).value = tok.decode(ids);
  }

  function decode() {
    if (!trained) train();
    const raw = (container.querySelector("#ids") as HTMLTextAreaElement).value.trim();
    try {
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) throw new Error("not array");
      const ids = parsed.map((x) => Number(x));
      (container.querySelector("#decoded") as HTMLTextAreaElement).value = tok.decode(ids);
    } catch {
      setMeta("Could not parse IDs; expected JSON like [1,2,3].");
      return;
    }
  }

  $("trainBtn")?.addEventListener("click", train);
  $("encodeBtn")?.addEventListener("click", encode);
  $("decodeBtn")?.addEventListener("click", decode);

  setMeta("not trained yet");
}

