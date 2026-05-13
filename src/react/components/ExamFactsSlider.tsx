import React, { useCallback, useId, useMemo, useState } from "react";
import { EXAM_FACTS, type ExamFact } from "../data/examFacts";

function wrapIndex(i: number, len: number): number {
  if (len === 0) return 0;
  return ((i % len) + len) % len;
}

export function ExamFactsSlider() {
  const facts = useMemo(() => EXAM_FACTS, []);
  const [index, setIndex] = useState(0);
  const headingId = useId();
  const liveId = useId();
  const panelId = useId();

  const current: ExamFact | undefined = facts[index];
  const count = facts.length;

  const goPrev = useCallback(() => {
    setIndex((i) => wrapIndex(i - 1, count));
  }, [count]);

  const goNext = useCallback(() => {
    setIndex((i) => wrapIndex(i + 1, count));
  }, [count]);

  const goTo = useCallback((i: number) => {
    setIndex(wrapIndex(i, count));
  }, [count]);

  if (!current || count === 0) return null;

  const positionLabel = `Fact ${index + 1} of ${count}`;

  return (
    <section
      className="exam-slider"
      aria-labelledby={headingId}
      aria-roledescription="carousel"
      aria-label="Study facts from practice answer keys"
    >
      <div className="exam-slider__header">
        <h2 className="exam-slider__title" id={headingId}>
          Course study facts
        </h2>
        <p className="exam-slider__subtitle muted">
          Short ideas from the Building ChatGPT practice answer keys—aligned with tokenizer, transformer, and training
          topics in this project.
        </p>
      </div>

      <div className="exam-slider__frame">
        <p className="exam-slider__source" aria-live="off">
          <span className="exam-slider__source-pill">{current.sourceLabel}</span>
        </p>

        <div
          id={panelId}
          className="exam-slider__panel"
          role="group"
          aria-roledescription="slide"
          aria-label={positionLabel}
        >
          <p className="exam-slider__fact">{current.text}</p>
        </div>

        <div id={liveId} className="exam-slider__visually-hidden" aria-live="polite" aria-atomic="true">
          {positionLabel}. {current.sourceLabel}. {current.text}
        </div>

        <div className="exam-slider__controls">
          <button type="button" className="exam-slider__nav" onClick={goPrev} aria-controls={panelId}>
            Previous fact
          </button>
          <button type="button" className="exam-slider__nav" onClick={goNext} aria-controls={panelId}>
            Next fact
          </button>
        </div>

        <div className="exam-slider__dots" role="tablist" aria-label="Jump to a fact">
          {facts.map((f, i) => {
            const selected = i === index;
            return (
              <button
                key={f.id}
                type="button"
                role="tab"
                aria-selected={selected}
                tabIndex={0}
                aria-controls={panelId}
                className={`exam-slider__dot${selected ? " exam-slider__dot--active" : ""}`}
                onClick={() => goTo(i)}
              >
                <span className="exam-slider__visually-hidden">
                  Fact {i + 1}: {f.sourceLabel}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
