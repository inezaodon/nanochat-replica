import React, { useEffect, useMemo, useState } from "react";
import { Home } from "./pages/Home";
import { LLMPlayground } from "./pages/LLMPlayground";
import { LabsDocs } from "./pages/LabsDocs";

type Route = "home" | "llm" | "docs";

function routeFromHash(): Route {
  const h = (location.hash || "").replace(/^#\/?/, "");
  if (h === "llm") return "llm";
  if (h === "docs") return "docs";
  return "home";
}

function useRoute(): Route {
  const [r, setR] = useState<Route>(() => routeFromHash());
  useEffect(() => {
    const on = () => setR(routeFromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return r;
}

export function App() {
  const route = useRoute();

  const NavLink = useMemo(
    () =>
      function NavLink({ to, label }: { to: Route; label: string }) {
        const active = route === to;
        const href = to === "home" ? "#/" : `#/${to}`;
        return (
          <a className={`pill ${active ? "active" : ""}`} href={href}>
            {label}
          </a>
        );
      },
    [route],
  );

  return (
    <>
      <style>{css}</style>
      <div className="topbar">
        <div className="nav">
          <div className="brand">nanochat-replica</div>
          <NavLink to="home" label="Home" />
          <NavLink to="llm" label="Small LLM" />
          <NavLink to="docs" label="Docs" />
        </div>
      </div>
      <div className="wrap">
        {route === "home" && <Home />}
        {route === "llm" && <LLMPlayground />}
        {route === "docs" && <LabsDocs />}
      </div>
    </>
  );
}

const css = `
:root{
  color-scheme:dark;
  --bg:#0b0f17;--panel:#111827;--panel2:#0f172a;--text:#e5e7eb;--muted:#9ca3af;
  --border:rgba(255,255,255,.12);--accent:#60a5fa;--shadow:0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;background:var(--bg);color:var(--text)}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.wrap{max-width:1200px;margin:0 auto;padding:20px 16px 60px}
.topbar{position:sticky;top:0;z-index:10;backdrop-filter:blur(10px);background:rgba(11,15,23,.75);border-bottom:1px solid var(--border);}
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
input,textarea,select,button{border:1px solid var(--border);background:rgba(15,23,42,.7);color:var(--text);border-radius:12px;padding:10px 12px;font:inherit;outline:none}
textarea{width:100%;min-height:96px;resize:vertical;line-height:1.35}
button{cursor:pointer;font-weight:800;background:rgba(255,255,255,.06)}
button.primary{border-color:rgba(96,165,250,.55);background:linear-gradient(180deg,rgba(96,165,250,.30),rgba(96,165,250,.15))}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:520px){.row{grid-template-columns:1fr}}
label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
`;

