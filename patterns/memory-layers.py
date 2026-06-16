"""
Three-Layer Memory System
Extracted from World Simulator (Build Small Hackathon 2026).

Pattern: Raw tick log (capped) + semantic episodes (by emotional weight) +
LLM-authored rolling summary (rewritten each turn). Recency, salience, and
continuity in a fixed token budget.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# LAYER 1: RAW TICK LOG — A capped ring buffer of recent events
# ---------------------------------------------------------------------------
# This is the "short-term memory." Exact events, in order, capped at N entries.
# When full, oldest entries fall off. Cheap, fast, no LLM needed.

@dataclass
class TickLog:
    """Ring buffer of raw events. Fixed capacity."""
    capacity: int = 50
    _buffer: deque = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self):
        self._buffer = deque(maxlen=self.capacity)

    def append(self, tick: int, event: str, metadata: dict | None = None) -> None:
        """Record a raw event."""
        self._buffer.append({
            "tick": tick,
            "time": time.time(),
            "event": event,
            "meta": metadata or {},
        })

    def recent(self, n: int | None = None) -> list[dict]:
        """Get the most recent N events (or all if N is None)."""
        items = list(self._buffer)
        return items[-n:] if n else items

    def to_prompt(self, n: int = 20) -> str:
        """Format recent events for inclusion in an LLM prompt."""
        events = self.recent(n)
        if not events:
            return "[No recent events]"
        lines = [f"Tick {e['tick']}: {e['event']}" for e in events]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LAYER 2: SEMANTIC EPISODES — Sorted by emotional weight
# ---------------------------------------------------------------------------
# These are the "important memories." Each episode has an emotional weight
# (intensity, surprise, significance). The list stays sorted so the most
# impactful moments are always at the top. Capped to a max count.

@dataclass
class Episode:
    """One memorable event, compressed and weighted."""
    tick: int
    summary: str                    # Short description of what happened
    emotional_weight: float         # 0.0 = mundane, 1.0 = life-changing
    tags: list[str] = field(default_factory=list)   # "combat", "betrayal", etc.
    participants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tick": self.tick,
            "summary": self.summary,
            "weight": self.emotional_weight,
            "tags": self.tags,
        }


@dataclass
class EpisodeMemory:
    """Collection of significant episodes, sorted by emotional weight."""
    max_episodes: int = 30
    _episodes: list[Episode] = field(default_factory=list)

    def add(self, episode: Episode) -> None:
        """Insert an episode, maintain sort order, enforce cap."""
        self._episodes.append(episode)
        self._episodes.sort(key=lambda e: e.emotional_weight, reverse=True)
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[:self.max_episodes]

    def top(self, n: int = 10) -> list[Episode]:
        """Get the N most emotionally weighted episodes."""
        return self._episodes[:n]

    def by_tag(self, tag: str) -> list[Episode]:
        """Filter episodes by tag."""
        return [e for e in self._episodes if tag in e.tags]

    def to_prompt(self, n: int = 10) -> str:
        """Format top episodes for inclusion in an LLM prompt."""
        episodes = self.top(n)
        if not episodes:
            return "[No significant memories]"
        lines = []
        for ep in episodes:
            lines.append(f"- (weight {ep.emotional_weight:.1f}) {ep.summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LAYER 3: ROLLING NARRATIVE SUMMARY — LLM-authored, rewritten each turn
# ---------------------------------------------------------------------------
# This is the "self-identity." The LLM writes a summary of who it is and
# what its story is so far. It gets rewritten every N ticks, incorporating
# new events from Layers 1 and 2.

@dataclass
class NarrativeSummary:
    """LLM-authored rolling summary, rewritten periodically."""
    text: str = "No history yet."
    last_updated_tick: int = 0
    update_interval: int = 10   # Rewrite every N ticks

    def needs_update(self, current_tick: int) -> bool:
        return (current_tick - self.last_updated_tick) >= self.update_interval

    def build_rewrite_prompt(self, entity_name: str,
                             tick_log: TickLog,
                             episodes: EpisodeMemory) -> str:
        """
        Build the prompt that asks the LLM to rewrite the narrative.
        Feed it the current summary + recent events + top episodes.
        """
        return (
            f"You are writing the internal narrative memory for {entity_name}.\n\n"
            f"CURRENT NARRATIVE:\n{self.text}\n\n"
            f"RECENT EVENTS:\n{tick_log.to_prompt(20)}\n\n"
            f"KEY MEMORIES:\n{episodes.to_prompt(10)}\n\n"
            f"Rewrite the narrative in first person, under 150 words. "
            f"Incorporate new events. Drop details that no longer matter. "
            f"Keep the voice consistent."
        )

    def update(self, new_text: str, current_tick: int) -> None:
        """Replace the narrative with the LLM's new version."""
        self.text = new_text
        self.last_updated_tick = current_tick


# ---------------------------------------------------------------------------
# COMPOSITE: The full three-layer memory for one entity
# ---------------------------------------------------------------------------

@dataclass
class EntityMemory:
    """All three layers for one entity, composable into a single prompt block."""
    entity_name: str
    tick_log: TickLog = field(default_factory=TickLog)
    episodes: EpisodeMemory = field(default_factory=EpisodeMemory)
    narrative: NarrativeSummary = field(default_factory=NarrativeSummary)

    def record_event(self, tick: int, event: str,
                     emotional_weight: float = 0.0,
                     tags: list[str] | None = None) -> None:
        """Record an event to the tick log, and optionally as an episode."""
        self.tick_log.append(tick, event)

        # Auto-promote to episode if emotionally significant
        if emotional_weight > 0.3:
            self.episodes.add(Episode(
                tick=tick,
                summary=event,
                emotional_weight=emotional_weight,
                tags=tags or [],
            ))

    def to_prompt(self, recent_ticks: int = 15, top_episodes: int = 8) -> str:
        """Compose all three layers into one prompt-ready block."""
        return (
            f"=== NARRATIVE ===\n{self.narrative.text}\n\n"
            f"=== KEY MEMORIES ===\n{self.episodes.to_prompt(top_episodes)}\n\n"
            f"=== RECENT EVENTS ===\n{self.tick_log.to_prompt(recent_ticks)}"
        )


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    memory = EntityMemory(entity_name="Alice")

    # Simulate some ticks
    memory.record_event(1, "Alice woke up in the forest.", 0.1)
    memory.record_event(2, "Alice found a rusty sword.", 0.4, tags=["loot"])
    memory.record_event(3, "A wolf attacked. Alice fought it off.", 0.8, tags=["combat"])
    memory.record_event(4, "Alice rested by the river.", 0.1)
    memory.record_event(5, "Alice met Bob, a traveling merchant.", 0.5, tags=["social"])
    memory.record_event(6, "Bob betrayed Alice and stole the sword.", 0.95, tags=["betrayal"])

    # See what the LLM would receive as context
    print(memory.to_prompt())
    print()

    # Check if narrative needs rewriting
    print(f"Narrative needs update: {memory.narrative.needs_update(6)}")

    # The rewrite prompt you'd send to the LLM:
    prompt = memory.narrative.build_rewrite_prompt(
        "Alice", memory.tick_log, memory.episodes
    )
    print(f"\nRewrite prompt preview:\n{prompt[:300]}...")
