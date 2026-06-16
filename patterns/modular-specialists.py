"""
Modular Specialists (Shared Latent Channel)
Source: ModuleMind (Build Small Hackathon 2026)

Pattern: Multiple small neural nets (specialists) communicate through a
shared latent state vector. Each specialist reads the shared state, does
its thing, and writes back. A coordinator reads the shared state to decide
which specialist fires next.

This is NOT a mixture-of-experts with a router. The shared channel is
persistent state that accumulates context across specialists. Think of it
as a whiteboard that everyone reads and writes to.

Use this for: modular agent architectures, multi-skill systems, ensemble
coordination, any setup where you want specialized small models that
share context without direct communication.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import hashlib


@dataclass
class SharedState:
    """
    The shared latent channel. All specialists read from and write to this.
    In a real system this might be a tensor, embedding, or structured dict.
    Here we use a dict for clarity -- swap in your latent representation.
    """
    data: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    history: list[dict] = field(default_factory=list)

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from shared state."""
        return self.data.get(key, default)

    def write(self, key: str, value: Any, writer: str):
        """Write a value to shared state. Tracked for audit."""
        old = self.data.get(key)
        self.data[key] = value
        self.version += 1
        self.history.append({
            "version": self.version,
            "writer": writer,
            "key": key,
            "old": old,
            "new": value,
            "timestamp": time.time(),
        })

    def snapshot(self) -> dict:
        return dict(self.data)


# =============================================================
# Specialist: One small model with a focused capability
# =============================================================

@dataclass
class Specialist:
    """
    A single specialist module. Reads shared state, performs its
    specialty, writes results back to shared state.
    """
    name: str
    capability: str  # What this specialist does (for coordinator)
    # The actual work function. Takes shared state, returns updates.
    process_fn: Callable[[dict], dict]
    # Which shared state keys this specialist reads
    reads: list[str] = field(default_factory=list)
    # Which shared state keys this specialist writes
    writes: list[str] = field(default_factory=list)
    call_count: int = 0

    def run(self, shared: SharedState) -> dict:
        """
        Execute this specialist. Reads from shared state, processes,
        writes results back. Returns the updates it made.
        """
        # Gather inputs from shared state
        inputs = {key: shared.read(key) for key in self.reads}

        # Run the specialist's logic
        outputs = self.process_fn(inputs)

        # Write outputs to shared state
        for key, value in outputs.items():
            if key in self.writes:
                shared.write(key, value, writer=self.name)

        self.call_count += 1
        return outputs


# =============================================================
# Coordinator: Reads shared state, picks next specialist
# =============================================================

class Coordinator:
    """
    Reads the shared state and decides which specialist should fire next.
    This is the 'brain' -- but it only routes, it doesn't do the work.
    Replace select_next() with your own logic (rule-based, LLM, learned).
    """

    def __init__(self):
        self.specialists: dict[str, Specialist] = {}
        self.execution_order: list[str] = []

    def register(self, specialist: Specialist):
        """Add a specialist to the roster."""
        self.specialists[specialist.name] = specialist

    def select_next(self, shared: SharedState) -> Optional[Specialist]:
        """
        Choose the next specialist to run based on shared state.
        Override this with your routing logic.
        Default: round-robin through all specialists.
        """
        names = list(self.specialists.keys())
        if not names:
            return None
        # Simple round-robin. Replace with priority/need-based selection.
        idx = len(self.execution_order) % len(names)
        return self.specialists[names[idx]]

    def step(self, shared: SharedState) -> Optional[dict]:
        """Run one coordination step: select specialist and execute."""
        specialist = self.select_next(shared)
        if not specialist:
            return None

        result = specialist.run(shared)
        self.execution_order.append(specialist.name)
        return {"specialist": specialist.name, "outputs": result}

    def run_cycle(self, shared: SharedState, max_steps: int = 10) -> list[dict]:
        """
        Run a full coordination cycle until max_steps or no work remains.
        """
        results = []
        for _ in range(max_steps):
            result = self.step(shared)
            if result is None:
                break
            results.append(result)
        return results


# =============================================================
# Example: Build a simple multi-specialist system
# =============================================================

def make_sentiment_specialist() -> Specialist:
    """Specialist that analyzes sentiment from raw text."""
    def process(inputs: dict) -> dict:
        text = inputs.get("raw_text", "")
        # Stub -- replace with actual model inference
        positive_words = {"good", "great", "love", "happy", "excellent"}
        negative_words = {"bad", "terrible", "hate", "sad", "awful"}
        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        score = (pos - neg) / max(pos + neg, 1)
        return {"sentiment_score": round(score, 2), "sentiment_done": True}

    return Specialist(
        name="sentiment",
        capability="Analyze emotional tone of text",
        process_fn=process,
        reads=["raw_text"],
        writes=["sentiment_score", "sentiment_done"],
    )


def make_topic_specialist() -> Specialist:
    """Specialist that extracts topics from raw text."""
    def process(inputs: dict) -> dict:
        text = inputs.get("raw_text", "")
        # Stub -- replace with actual model inference
        words = text.lower().split()
        # Fake topic extraction: just grab unique long words
        topics = [w for w in set(words) if len(w) > 5][:3]
        return {"topics": topics, "topics_done": True}

    return Specialist(
        name="topics",
        capability="Extract key topics from text",
        process_fn=process,
        reads=["raw_text"],
        writes=["topics", "topics_done"],
    )


def make_summary_specialist() -> Specialist:
    """Specialist that writes a summary using outputs from other specialists."""
    def process(inputs: dict) -> dict:
        sentiment = inputs.get("sentiment_score", 0)
        topics = inputs.get("topics", [])
        tone = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        summary = f"Text is {tone} (score: {sentiment}), covering: {', '.join(topics) or 'no topics extracted'}"
        return {"summary": summary, "summary_done": True}

    return Specialist(
        name="summarizer",
        capability="Combine specialist outputs into a summary",
        process_fn=process,
        reads=["sentiment_score", "topics"],
        writes=["summary", "summary_done"],
    )


# --- Demo ---
if __name__ == "__main__":
    # Set up the shared state with initial input
    shared = SharedState()
    shared.write("raw_text", "I love this excellent product but the delivery was terrible", writer="user")

    # Register specialists
    coord = Coordinator()
    coord.register(make_sentiment_specialist())
    coord.register(make_topic_specialist())
    coord.register(make_summary_specialist())

    # Run a full cycle
    results = coord.run_cycle(shared, max_steps=3)

    print("Execution order:", coord.execution_order)
    print("Final shared state:", json.dumps(shared.snapshot(), indent=2))
    print(f"State version: {shared.version}")
    print(f"\nSummary: {shared.read('summary')}")
