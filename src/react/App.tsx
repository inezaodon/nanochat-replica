import React, { Suspense, useEffect, useMemo, useState } from "react";
import "./app.css";
import { Landing } from "./pages/Landing";
import { LLMPlayground } from "./pages/LLMPlayground";
import { Architecture } from "./pages/Architecture";
const ThreePlayground = React.lazy(async () => {
  const m = await import("./pages/ThreePlayground");
  return { default: m.ThreePlayground };
});

type Route = "home" | "overview" | "architecture" | "three" | "docs" | "llm";

function routeFromHash(): Route {
  const h = (location.hash || "").replace(/^#\/?/, "");
  if (h === "overview" || h === "llm") return "overview";
  if (h === "architecture") return "architecture";
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
        const active =
          route === to || (to === "overview" && route === "llm") || (to === "home" && route === "docs");
        const href =
          to === "home" ? "#/" : to === "overview" ? "#/overview" : to === "docs" ? "#/" : `#/${to}`;
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
            <NavLink to="home" label="Home" />
            <NavLink to="overview" label="Overview" />
            <NavLink to="architecture" label="Architecture" />
            <NavLink to="three" label="Three.js" />
          </nav>
        </div>
      </header>
      <div className="wrap">
        {(route === "home" || route === "docs") && <Landing />}
        {route === "overview" && <LLMPlayground />}
        {route === "architecture" && <Architecture />}
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
