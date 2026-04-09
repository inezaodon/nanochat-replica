import { renderHome } from "./pages/home";
import { renderTokenizerPage } from "./pages/tokenizer";
import { renderEmbeddingPage } from "./pages/embedding";
import { renderTransformerPage } from "./pages/transformer";
import { renderGPT2HeadsPage } from "./pages/gpt2";

type Route = "home" | "tokenizer" | "embedding" | "transformer" | "gpt2" | "docs";

function css(): string {
  return `
  :root{
    color-scheme:dark;
    --bg:#0b0f17;--panel:#111827;--panel2:#0f172a;--text:#e5e7eb;--muted:#9ca3af;
    --border:rgba(255,255,255,.12);--accent:#60a5fa;--shadow:0 10px 30px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;background:var(--bg);color:var(--text)}
  a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
  .wrap{max-width:1200px;margin:0 auto;padding:20px 16px 60px}
  .topbar{
    position:sticky;top:0;z-index:10;
    backdrop-filter:blur(10px);
    background:rgba(11,15,23,.75);
    border-bottom:1px solid var(--border);
  }
  .nav{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:12px 16px;max-width:1200px;margin:0 auto}
  .brand{font-weight:900;letter-spacing:-.02em;margin-right:6px}
  .pill{border:1px solid var(--border);background:rgba(255,255,255,.04);padding:7px 10px;border-radius:999px;font-size:13px}
  .pill.active{border-color:rgba(96,165,250,.55);background:rgba(96,165,250,.14)}
  .card{border:1px solid var(--border);background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));border-radius:16px;box-shadow:var(--shadow);overflow:hidden}
  .cardH{padding:14px 14px 10px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;gap:10px;align-items:baseline;background:rgba(255,255,255,.02)}
  .cardH h2{margin:0;font-size:14px;font-weight:900;letter-spacing:.02em;text-transform:uppercase}
  .cardB{padding:14px}
  .muted{color:var(--muted)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start}
  @media(max-width:980px){.grid{grid-template-columns:1fr}}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace}
  input,textarea,select,button{
    border:1px solid var(--border);background:rgba(15,23,42,.7);color:var(--text);
    border-radius:12px;padding:10px 12px;font:inherit;outline:none
  }
  textarea{width:100%;min-height:84px;resize:vertical;line-height:1.35}
  input[type="number"],select{width:100%}
  button{cursor:pointer;font-weight:800;background:rgba(255,255,255,.06)}
  button.primary{border-color:rgba(96,165,250,.55);background:linear-gradient(180deg,rgba(96,165,250,.30),rgba(96,165,250,.15))}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  @media(max-width:520px){.row{grid-template-columns:1fr}}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
  `;
}

function routeFromHash(): Route {
  const h = (location.hash || "").replace(/^#\/?/, "");
  if (h === "tokenizer") return "tokenizer";
  if (h === "embedding") return "embedding";
  if (h === "transformer") return "transformer";
  if (h === "gpt2") return "gpt2";
  if (h === "docs") return "docs";
  return "home";
}

function navLink(route: Route, label: string, active: boolean): HTMLElement {
  const a = document.createElement("a");
  a.href = route === "home" ? "#/" : `#/${route}`;
  a.className = `pill${active ? " active" : ""}`;
  a.textContent = label;
  return a;
}

function renderDocs(container: HTMLElement) {
  container.innerHTML = `
    <div class="card">
      <div class="cardH"><h2>Lab HTML exports</h2><div class="muted">Original notebooks as static references</div></div>
      <div class="cardB">
        <ul>
          <li><a href="/Lab_01_Tokenization.html">Lab_01_Tokenization.html</a></li>
          <li><a href="/Lab_02_Embedding.html">Lab_02_Embedding.html</a></li>
          <li><a href="/Lab_03_Transformer_Block.html">Lab_03_Transformer_Block.html</a></li>
          <li><a href="/legacy/GPT2_Replica_12Heads.html">GPT2_Replica_12Heads.html</a></li>
        </ul>
        <p class="muted">These are kept as-is; the interactive demos live in the other tabs.</p>
      </div>
    </div>
  `;
}

export function renderApp(root: HTMLElement | null) {
  if (!root) throw new Error("Missing #app element");

  const style = document.createElement("style");
  style.textContent = css();
  document.head.appendChild(style);

  const top = document.createElement("div");
  top.className = "topbar";
  const nav = document.createElement("div");
  nav.className = "nav";
  top.appendChild(nav);

  const wrap = document.createElement("div");
  wrap.className = "wrap";
  const main = document.createElement("main");
  wrap.appendChild(main);

  root.replaceChildren(top, wrap);

  function doRender() {
    const r = routeFromHash();
    nav.replaceChildren(
      Object.assign(document.createElement("div"), { className: "brand", textContent: "nanochat-replica" }),
      navLink("home", "Home", r === "home"),
      navLink("tokenizer", "Tokenizer (Lab 1)", r === "tokenizer"),
      navLink("embedding", "Embedding (Lab 2)", r === "embedding"),
      navLink("transformer", "Transformer (Lab 3)", r === "transformer"),
      navLink("gpt2", "GPT‑2 (12 heads)", r === "gpt2"),
      navLink("docs", "Docs", r === "docs"),
    );

    main.innerHTML = "";
    if (r === "home") renderHome(main);
    else if (r === "tokenizer") renderTokenizerPage(main);
    else if (r === "embedding") renderEmbeddingPage(main);
    else if (r === "transformer") renderTransformerPage(main);
    else if (r === "gpt2") renderGPT2HeadsPage(main);
    else renderDocs(main);
  }

  window.addEventListener("hashchange", doRender);
  doRender();
}

