import React, { Suspense, useEffect, useMemo, useState } from "react";
import "./app.css";
import { Home } from "./pages/Home";
import { LLMPlayground } from "./pages/LLMPlayground";
import { LabsDocs } from "./pages/LabsDocs";

const ThreePlayground = React.lazy(async () => {
  const m = await import("./pages/ThreePlayground");
  return { default: m.ThreePlayground };
});

type Route = "home" | "llm" | "docs" | "three";

function routeFromHash(): Route {
  const h = (location.hash || "").replace(/^#\/?/, "");
  if (h === "llm") return "llm";
  if (h === "docs") return "docs";
  if (h === "three") return "three";
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
      <header className="topbar">
        <div className="nav-shell">
          <div className="brand">
            nanochat<span className="brand-accent">.replica</span>
          </div>
          <nav className="nav-links" aria-label="Primary">
            <NavLink to="home" label="Overview" />
            <NavLink to="llm" label="Small LLM" />
            <NavLink to="docs" label="Course labs" />
            <NavLink to="three" label="Three.js" />
          </nav>
        </div>
      </header>
      <div className="wrap">
        {route === "home" && <Home />}
        {route === "llm" && <LLMPlayground />}
        {route === "docs" && <LabsDocs />}
        {route === "three" && (
          <Suspense
            fallback={
              <div className="card cardB muted" style={{ minHeight: 200 }}>
                Loading WebGL…
              </div>
            }
          >
            <ThreePlayground />
          </Suspense>
        )}
      </div>
    </>
  );
}
