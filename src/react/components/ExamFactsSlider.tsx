import React, { useCallback, useEffect, useId, useMemo, useState } from "react";
import { EXAM_FACTS, type ExamFact, type ExamFactSource } from "../data/examFacts";

function wrapIndex(i: number, len: number): number {
  if (len === 0) return 0;
  return ((i % len) + len) % len;
}

/** Strip bracket tag; split Q/A on first `?` when an answer follows, else first sentence vs rest. */
function parseQuestionAnswer(raw: string): { question: string; answer: string } {
  const stripped = raw.replace(/^\[[^\]]+\]\s*/, "").trim();
  const qIdx = stripped.indexOf("?");
  if (qIdx !== -1 && qIdx < stripped.length - 2) {
    const after = stripped.slice(qIdx + 1).trim();
    if (after.length > 0) {
      return {
        question: stripped.slice(0, qIdx + 1).trim(),
        answer: after,
      };
    }
  }
  const dot = stripped.search(/\.\s+[A-Za-z0-9"']/);
  if (dot > 30 && dot < 480) {
    return {
      question: stripped.slice(0, dot + 1).trim(),
      answer: stripped.slice(dot + 2).trim(),
    };
  }
  return { question: stripped, answer: stripped };
}

export interface ExamFactsSliderProps {
  filterSource: ExamFactSource;
  deckTitle: string;
  /** Shown under the deck title */
  deckSubtitle?: string;
}

export function ExamFactsSlider({ filterSource, deckTitle, deckSubtitle }: ExamFactsSliderProps) {
  const facts = useMemo(
    () => EXAM_FACTS.filter((f) => f.source === filterSource),
    [filterSource],
  );
  const [index, setIndex] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const headingId = useId();
  const liveId = useId();
  const panelId = useId();
  const jumpInputId = useId();

  useEffect(() => {
    setIndex(0);
  }, [filterSource]);

  useEffect(() => {
    setShowAnswer(false);
  }, [index]);

  const current: ExamFact | undefined = facts[index];
  const count = facts.length;

  const goPrev = useCallback(() => {
    setIndex((i) => wrapIndex(i - 1, count));
  }, [count]);

  const goNext = useCallback(() => {
    setIndex((i) => wrapIndex(i + 1, count));
  }, [count]);

  const goTo = useCallback(
    (i: number) => {
      setIndex(wrapIndex(i, count));
    },
    [count],
  );

  const toggleFace = useCallback(() => {
    setShowAnswer((v) => !v);
  }, []);

  if (!current || count === 0) return null;

  const positionLabel = `Card ${index + 1} of ${count}`;
  const { question, answer } = parseQuestionAnswer(current.text);
  const backText = answer.length > 0 ? answer : current.text.replace(/^\[[^\]]+\]\s*/, "").trim();
  const faceLabel = showAnswer ? "Answer" : "Question";

  return (
    <section
      className="exam-slider"
      aria-labelledby={headingId}
      aria-roledescription="carousel"
      aria-label={deckTitle}
    >
      <div className="exam-slider__header">
        <h2 className="exam-slider__title" id={headingId}>
          {deckTitle}
        </h2>
        {deckSubtitle ? (
          <p className="exam-slider__subtitle muted">{deckSubtitle}</p>
        ) : null}
      </div>

      <div className="exam-slider__frame">
        <p className="exam-slider__source" aria-live="off">
          <span className="exam-slider__source-pill">{current.sourceLabel}</span>
        </p>

        <button
          type="button"
          className="flashcard-deck__card"
          id={panelId}
          aria-label={`${faceLabel}. Click to show ${showAnswer ? "question" : "answer"}.`}
          onClick={toggleFace}
        >
          <p className="flashcard-deck__card-kicker">{faceLabel}</p>
          <p className="flashcard-deck__card-body exam-slider__fact">
            {showAnswer ? backText : question}
          </p>
          <p className="flashcard-deck__card-hint muted">Click card to flip</p>
        </button>

        <div id={liveId} className="exam-slider__visually-hidden" aria-live="polite" aria-atomic="true">
          {positionLabel}. {current.sourceLabel}. {showAnswer ? backText : question}
        </div>

        <div className="exam-slider__controls">
          <button type="button" className="exam-slider__nav" onClick={goPrev} aria-controls={panelId}>
            Previous card
          </button>
          <button type="button" className="exam-slider__nav" onClick={goNext} aria-controls={panelId}>
            Next card
          </button>
        </div>

        {count <= 48 ? (
          <div className="exam-slider__dots" role="tablist" aria-label="Jump to a card">
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
                    Card {i + 1}: {f.sourceLabel}
                  </span>
                </button>
              );
            })}
          </div>
        ) : (
          <div className="exam-slider__jump">
            <label className="exam-slider__jump-label muted" htmlFor={jumpInputId}>
              Jump to card ({index + 1} / {count})
            </label>
            <input
              id={jumpInputId}
              className="exam-slider__jump-range"
              type="range"
              min={1}
              max={count}
              value={index + 1}
              onChange={(e) => goTo(Number(e.target.value) - 1)}
            />
          </div>
        )}
      </div>
    </section>
  );
}
