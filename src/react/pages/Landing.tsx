import React from "react";
import { useReveal } from "../hooks/useReveal";
import { LabsCourseStack } from "../components/LabsCourseStack";
import { PracticeFlashcardsSection } from "../components/PracticeFlashcardsSection";

export function Landing() {
  const { ref, active } = useReveal();

  return (
    <>
      <section ref={ref} className={`card home-section lift-reveal ${active ? "is-visible" : ""}`}>
        <div className="cardH">
          <h2>Course labs</h2>
          <div className="cardH-meta">Static HTML exports bundled with the repo (Labs 01–05), plus a few in-app tools.</div>
        </div>
        <div className="cardB">
          <LabsCourseStack />
        </div>
      </section>
      <PracticeFlashcardsSection />
    </>
  );
}
