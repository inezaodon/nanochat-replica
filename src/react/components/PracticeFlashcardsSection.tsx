import React from "react";
import { ExamFactsSlider } from "./ExamFactsSlider";

export function PracticeFlashcardsSection() {
  return (
    <section className="card home-section" aria-label="Practice answer key flashcards">
        <div className="cardH">
          <h2>Practice flashcards</h2>
          <div className="cardH-meta">Answer keys · flip or step through cards</div>
        </div>
        <div className="cardB">
          <p className="flashcard-decks-lede muted">
            Two decks from the course keys—each carousel only includes facts from that packet. Click a card to flip
            between the prompt side and the answer side.
          </p>
          <div className="flashcard-decks-grid">
            <div className="flashcard-deck">
              <ExamFactsSlider
                filterSource="exam01"
                deckTitle="Exam 01 — practice packet (answer key)"
                deckSubtitle="Facts auto-extracted from the Exam 01 practice answer key."
              />
            </div>
            <div className="flashcard-deck">
              <ExamFactsSlider
                filterSource="exam02"
                deckTitle="Practice Packet 02 — answer key"
                deckSubtitle="Facts auto-extracted from the Packet 02 answer key PDF."
              />
            </div>
          </div>
        </div>
      </section>
  );
}
