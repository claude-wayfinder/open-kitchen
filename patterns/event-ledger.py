"""
Append-Only Event Ledger for Multi-Agent Coordination
Extracted from Multi-Agent Lab (Build Small Hackathon 2026).

Pattern: Agents never talk directly. Everything flows through one immutable,
append-only log. Memory is filtered views into the ledger. No cycles, no
deadlocks, full replay from the log alone.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# 1. EVENT — The atomic unit of the ledger
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """One immutable record in the ledger."""
    id: int                         # Monotonic, assigned by the ledger
    timestamp: float                # time.time() when appended
    source: str                     # Which agent wrote this ("planner", "coder", etc.)
    event_type: str                 # What happened ("task_created", "result", "error")
    payload: dict[str, Any]         # The actual data (freeform)
    parent_id: int | None = None    # Optional: which event this responds to

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# 2. LEDGER — The single source of truth
# ---------------------------------------------------------------------------

class EventLedger:
    """
    Thread-safe, append-only event log.

    In production, back this with SQLite or a file. This in-memory version
    is the skeleton — the interface is what matters.
    """

    def __init__(self, max_events: int = 10_000):
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._next_id = 1
        self._max_events = max_events

    def append(
        self,
        source: str,
        event_type: str,
        payload: dict[str, Any],
        parent_id: int | None = None,
    ) -> Event:
        """
        Write a new event to the ledger. Returns the created event.
        This is the ONLY way to add data. There is no update, no delete.
        """
        with self._lock:
            event = Event(
                id=self._next_id,
                timestamp=time.time(),
                source=source,
                event_type=event_type,
                payload=payload,
                parent_id=parent_id,
            )
            self._events.append(event)
            self._next_id += 1

            # Safety cap — in production, archive old events instead of dropping
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

            return event

    # --- VIEWS: Filtered reads into the ledger ---

    def get_all(self) -> list[Event]:
        """Full ledger. Use sparingly — prefer filtered views."""
        with self._lock:
            return list(self._events)

    def get_by_source(self, source: str) -> list[Event]:
        """Everything one agent wrote."""
        with self._lock:
            return [e for e in self._events if e.source == source]

    def get_by_type(self, event_type: str) -> list[Event]:
        """All events of a given type, across all agents."""
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def get_since(self, event_id: int) -> list[Event]:
        """Everything after a given event ID. For polling / catch-up."""
        with self._lock:
            return [e for e in self._events if e.id > event_id]

    def get_thread(self, root_id: int) -> list[Event]:
        """Follow a chain of parent_id links from a root event."""
        with self._lock:
            thread = []
            ids_in_thread = {root_id}
            for e in self._events:
                if e.id == root_id or e.parent_id in ids_in_thread:
                    thread.append(e)
                    ids_in_thread.add(e.id)
            return thread

    def get_last_n(self, n: int = 10) -> list[Event]:
        """Most recent N events. Good for building agent context windows."""
        with self._lock:
            return list(self._events[-n:])

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# 3. AGENT VIEW — What each agent sees
# ---------------------------------------------------------------------------

class AgentView:
    """
    A filtered lens into the ledger for one agent.
    Agents don't see the raw ledger — they see a view tailored to their role.
    """

    def __init__(self, ledger: EventLedger, agent_name: str,
                 watch_types: list[str] | None = None):
        self.ledger = ledger
        self.agent_name = agent_name
        self.watch_types = watch_types or []
        self._last_seen_id = 0

    def write(self, event_type: str, payload: dict,
              parent_id: int | None = None) -> Event:
        """Write an event as this agent."""
        return self.ledger.append(
            source=self.agent_name,
            event_type=event_type,
            payload=payload,
            parent_id=parent_id,
        )

    def poll(self) -> list[Event]:
        """Get new events since last poll, filtered to watched types."""
        new_events = self.ledger.get_since(self._last_seen_id)
        if self.watch_types:
            new_events = [e for e in new_events
                          if e.event_type in self.watch_types]
        if new_events:
            self._last_seen_id = max(e.id for e in new_events)
        return new_events

    def my_events(self) -> list[Event]:
        """Everything this agent has written."""
        return self.ledger.get_by_source(self.agent_name)

    def recent_context(self, n: int = 20) -> str:
        """Build a text summary of recent events for the agent's LLM prompt."""
        events = self.ledger.get_last_n(n)
        lines = []
        for e in events:
            lines.append(f"[{e.source}] {e.event_type}: "
                         f"{json.dumps(e.payload, default=str)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ledger = EventLedger()
    planner = AgentView(ledger, "planner", watch_types=["result", "error"])
    coder = AgentView(ledger, "coder", watch_types=["task_created"])

    task = planner.write("task_created", {"description": "Sort a list", "priority": "high"})
    print(f"Task created: event #{task.id}")

    new_tasks = coder.poll()
    print(f"Coder sees {len(new_tasks)} new task(s)")

    coder.write("code_submitted", {"code": "sorted(items)"}, parent_id=task.id)
    thread = ledger.get_thread(task.id)
    print(f"Thread: {len(thread)} events")
    print(f"Context:\n{planner.recent_context(10)}")
