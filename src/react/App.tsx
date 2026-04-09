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
          <div className="brand">
            nanochat<span className="brand-accent">.replica</span>
          </div>
          <div className="nav-links">
            <NavLink to="home" label="Overview" />
            <NavLink to="llm" label="Small LLM demo" />
            <NavLink to="docs" label="Course labs" />
          </div>
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
  --bg:#020617;
  --panel:#020617;
  --panel2:#020617;
  --text:#e5e7eb;
  --muted:#9ca3af;
  --border:rgba(148,163,184,.4);
  --accent:#38bdf8;
  --accent-soft:rgba(56,189,248,.16);
  --shadow:0 18px 45px rgba(15,23,42,.75);
}
*{box-sizing:border-box}
body{
  margin:0;
  font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
  background:
    radial-gradient(900px 600px at 10% 0%,rgba(56,189,248,.25),transparent 55%),
    radial-gradient(900px 600px at 90% 0%,rgba(129,140,248,.25),transparent 55%),
    radial-gradient(1000px 700px at 50% 100%,rgba(16,185,129,.18),transparent 60%),
    var(--bg);
  color:var(--text);
}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px 80px}
.topbar{
  position:sticky;top:0;z-index:20;
  backdrop-filter:blur(18px);
  background:linear-gradient(to bottom,rgba(15,23,42,.92),rgba(15,23,42,.65) 60%,transparent);
  border-bottom:1px solid rgba(15,23,42,.8);
}
.nav{
  display:flex;justify-content:space-between;align-items:center;
  gap:16px;flex-wrap:wrap;
  padding:14px 18px 10px;
  max-width:1200px;margin:0 auto;
}
.brand{
  font-weight:900;letter-spacing:-.09em;
  font-size:22px;
}
.brand-accent{
  letter-spacing:-.02em;
  background:linear-gradient(90deg,#38bdf8,#22c55e);
  -webkit-background-clip:text;
  color:transparent;
}
.nav-links{
  display:flex;gap:8px;flex-wrap:wrap;
}
.pill{
  border:1px solid rgba(148,163,184,.45);
  background:rgba(15,23,42,.85);
  padding:7px 12px;
  border-radius:999px;
  font-size:13px;
  display:inline-flex;
  align-items:center;
  gap:6px;
}
.pill.active{
  border-color:var(--accent);
  background:var(--accent-soft);
}
.card{
  border:1px solid rgba(148,163,184,.45);
  background:radial-gradient(circle at top left,rgba(56,189,248,.26),transparent 55%),rgba(15,23,42,.92);
  border-radius:20px;
  box-shadow:var(--shadow);
  overflow:hidden;
}
.cardH{padding:16px 18px 12px;border-bottom:1px solid rgba(148,163,184,.26);display:flex;justify-content:space-between;gap:10px;align-items:baseline;background:rgba(15,23,42,.96)}
.cardH h2{margin:0;font-size:14px;font-weight:900;letter-spacing:.02em;text-transform:uppercase}
.cardB{padding:18px}
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

