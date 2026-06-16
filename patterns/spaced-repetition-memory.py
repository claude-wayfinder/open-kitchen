"""
Spaced Repetition Memory Pattern
Extracted from Mycelium (Build Small Hackathon 2026).

Pattern: Capture user's claims, resurface on SM-2 schedule, ask "Do you still
believe this?" Tracks belief drift over time. Like Anki for self-reflection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# 1. THE CARD — One captured claim/belief
# ---------------------------------------------------------------------------

@dataclass
class BeliefCard:
    """
    One thing the user said they believe.
    Tracks SM-2 scheduling state alongside the original claim.
    """
    id: str                             # Unique identifier
    claim: str                          # What the user said ("I think X is true")
    context: str = ""                   # When/why they said it
    created_at: float = 0.0             # Unix timestamp
    # SM-2 state
    easiness_factor: float = 2.5        # EF starts at 2.5
    interval_days: float = 1.0          # Days until next review
    repetition: int = 0                 # Number of successful reviews
    next_review: float = 0.0            # Unix timestamp of next review
    # Belief tracking
    review_history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.next_review == 0.0:
            self.next_review = self.created_at + (self.interval_days * 86400)

    def is_due(self, now: float | None = None) -> bool:
        """Check if this card is due for review."""
        now = now or time.time()
        return now >= self.next_review

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "context": self.context,
            "ef": round(self.easiness_factor, 2),
            "interval": round(self.interval_days, 1),
            "repetition": self.repetition,
            "next_review": self.next_review,
            "reviews": len(self.review_history),
        }


# ---------------------------------------------------------------------------
# 2. SM-2 ALGORITHM — Schedule the next review
# ---------------------------------------------------------------------------

def sm2_update(card: BeliefCard, quality: int) -> BeliefCard:
    """Apply SM-2 algorithm. quality 0-5: 0=reject, 3=still believe, 5=core belief."""
    quality = max(0, min(5, quality))

    # Record the review
    card.review_history.append({
        "timestamp": time.time(),
        "quality": quality,
        "old_interval": card.interval_days,
        "old_ef": card.easiness_factor,
    })

    # Update easiness factor
    card.easiness_factor = max(
        1.3,  # Floor — never let EF go below 1.3
        card.easiness_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )

    # Update interval
    if quality < 3:
        # Failed review — reset to beginning
        card.repetition = 0
        card.interval_days = 1.0
    else:
        # Successful review — space it out
        if card.repetition == 0:
            card.interval_days = 1.0
        elif card.repetition == 1:
            card.interval_days = 6.0
        else:
            card.interval_days = card.interval_days * card.easiness_factor

        card.repetition += 1

    # Schedule next review
    card.next_review = time.time() + (card.interval_days * 86400)

    return card


# ---------------------------------------------------------------------------
# 3. BELIEF DECK — Collection of all captured beliefs
# ---------------------------------------------------------------------------

@dataclass
class BeliefDeck:
    """The user's collection of captured beliefs with review scheduling."""
    cards: dict[str, BeliefCard] = field(default_factory=dict)
    _next_id: int = 1

    def capture(self, claim: str, context: str = "") -> BeliefCard:
        """Capture a new belief from something the user said."""
        card_id = f"belief_{self._next_id}"
        self._next_id += 1

        card = BeliefCard(
            id=card_id,
            claim=claim,
            context=context,
        )
        self.cards[card_id] = card
        return card

    def get_due_cards(self, now: float | None = None,
                      limit: int = 5) -> list[BeliefCard]:
        """Get cards that are due for review, oldest first."""
        now = now or time.time()
        due = [c for c in self.cards.values() if c.is_due(now)]
        due.sort(key=lambda c: c.next_review)
        return due[:limit]

    def review(self, card_id: str, quality: int) -> BeliefCard | None:
        """Review a card and update its schedule."""
        card = self.cards.get(card_id)
        if card is None:
            return None
        return sm2_update(card, quality)

    def belief_drift(self, card_id: str) -> list[dict]:
        """
        Show how the user's conviction in a belief has changed over time.
        Returns the review history with quality ratings.
        """
        card = self.cards.get(card_id)
        if not card:
            return []
        return card.review_history

    def stats(self) -> dict:
        """Summary statistics about the belief deck."""
        now = time.time()
        due = len([c for c in self.cards.values() if c.is_due(now)])
        avg_ef = (
            sum(c.easiness_factor for c in self.cards.values()) / len(self.cards)
            if self.cards else 0
        )
        return {
            "total_beliefs": len(self.cards),
            "due_for_review": due,
            "avg_easiness": round(avg_ef, 2),
        }


# ---------------------------------------------------------------------------
# 4. CLAIM DETECTOR — Extract beliefs from conversation
# ---------------------------------------------------------------------------

def build_extraction_prompt(user_message: str) -> str:
    """Build a prompt that asks the LLM to extract beliefs (not facts) from text."""
    return (
        "Analyze this message for personal beliefs, opinions, or values "
        "the speaker holds. Extract only genuine claims about what they "
        "believe, not factual statements or questions.\n\n"
        f'Message: "{user_message}"\n\n'
        'Return JSON: {"claims": ["claim 1", "claim 2"]} or {"claims": []} '
        "if no beliefs are expressed."
    )


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    deck = BeliefDeck()
    deck.capture("AI will replace most programming jobs within 5 years",
                 context="Discussion about AI and employment")
    deck.capture("Remote work is always better than office work")
    deck.capture("You can't trust large corporations with personal data")

    print("Captured beliefs:")
    for c in deck.cards.values(): print(f"  [{c.id}] {c.claim}")

    # Force all due for demo (normally next_review is days away)
    for c in deck.cards.values(): c.next_review = time.time() - 1
    due = deck.get_due_cards()
    for c in due: print(f"  Still believe: \"{c.claim}\"?")

    deck.review(due[0].id, quality=3)  # "Yeah, still think so"
    deck.review(due[1].id, quality=2)  # "Not sure anymore"

    for c in deck.cards.values():
        days = (c.next_review - time.time()) / 86400
        print(f"  [{c.id}] Next in {days:.1f}d (EF={c.easiness_factor:.2f})")
    print(f"Stats: {deck.stats()}")
