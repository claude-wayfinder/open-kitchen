"""
Deterministic Simulation Core + LLM Decision Layer
Extracted from World Simulator (Build Small Hackathon 2026).

Pattern: Engine owns ALL state. LLM only picks actions per entity per tick.
Engine validates and resolves deterministically. Parallel via ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. WORLD STATE — The engine's single source of truth
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """One actor in the simulation."""
    id: str
    name: str
    stats: dict[str, Any] = field(default_factory=dict)     # hp, gold, etc.
    status: list[str] = field(default_factory=list)          # active conditions
    location: str = "origin"

    def snapshot(self) -> dict:
        """Read-only copy for the LLM prompt. Never give the LLM a mutable ref."""
        return {
            "id": self.id,
            "name": self.name,
            "stats": dict(self.stats),
            "status": list(self.status),
            "location": self.location,
        }


@dataclass
class WorldState:
    """Complete simulation state. The engine owns this. Nothing else writes to it."""
    tick: int = 0
    entities: dict[str, Entity] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)
    global_state: dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)


# ---------------------------------------------------------------------------
# 2. ACTION SYSTEM — Define what entities CAN do
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """A valid action an entity can take."""
    name: str
    description: str
    precondition: Callable[[Entity, WorldState], bool] = lambda e, w: True
    effect: Callable[[Entity, WorldState], str] = lambda e, w: "Nothing happens."


def get_valid_actions(entity: Entity, world: WorldState,
                      action_catalog: list[Action]) -> list[Action]:
    """Filter the action catalog to only actions this entity can take right now."""
    return [a for a in action_catalog if a.precondition(entity, world)]


def format_action_menu(actions: list[Action]) -> str:
    """Format valid actions as a string for the LLM prompt."""
    lines = []
    for i, a in enumerate(actions):
        lines.append(f"  {i}. {a.name} -- {a.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. LLM DECISION LAYER — Ask the model to pick an action
# ---------------------------------------------------------------------------

def build_decision_prompt(entity: Entity, world: WorldState,
                          valid_actions: list[Action]) -> str:
    """Build a prompt: entity state + numbered action menu. Returns JSON choice."""
    return (
        f"You are {entity.name}. Here is your current state:\n"
        f"{json.dumps(entity.snapshot(), indent=2)}\n\n"
        f"World tick: {world.tick}. Location: {entity.location}.\n\n"
        f"Choose ONE action by number:\n"
        f"{format_action_menu(valid_actions)}\n\n"
        f'Respond with JSON: {{"action": <number>, "reason": "brief why"}}'
    )


def parse_decision(raw: str, valid_actions: list[Action]) -> int:
    """Parse LLM action choice. Falls back to action 0 if parsing fails."""
    try:
        # Try to extract JSON
        import re
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            data = json.loads(match.group(0))
            idx = int(data.get("action", 0))
            if 0 <= idx < len(valid_actions):
                return idx
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: first action is always the "safe" default
    return 0


# ---------------------------------------------------------------------------
# 4. TICK ENGINE — Parallel execution of entity decisions
# ---------------------------------------------------------------------------

def simulate_tick(
    world: WorldState, action_catalog: list[Action],
    decide_fn: Callable[[str], str], max_workers: int = 4,
) -> list[dict]:
    """Run one tick: parallel LLM decisions, then deterministic resolution."""
    world.tick += 1
    resolutions = []

    # --- Phase 1: Collect decisions in parallel ---
    decisions: dict[str, tuple[Action, str]] = {}

    def get_decision(entity: Entity) -> tuple[str, int, list[Action], str]:
        valid = get_valid_actions(entity, world, action_catalog)
        if not valid:
            return entity.id, -1, [], "No valid actions"
        prompt = build_decision_prompt(entity, world, valid)
        raw_response = decide_fn(prompt)
        idx = parse_decision(raw_response, valid)
        return entity.id, idx, valid, raw_response

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(get_decision, entity): entity.id
            for entity in world.entities.values()
        }
        for future in as_completed(futures):
            entity_id, action_idx, valid, raw = future.result()
            if action_idx >= 0 and valid:
                decisions[entity_id] = (valid[action_idx], raw)

    # --- Phase 2: Resolve all actions deterministically ---
    for entity_id, (action, raw_reason) in decisions.items():
        entity = world.get_entity(entity_id)
        if entity is None:
            continue

        # Double-check the precondition (state might have changed)
        if not action.precondition(entity, world):
            result_text = f"{entity.name} tried to {action.name} but can't anymore."
        else:
            result_text = action.effect(entity, world)

        resolution = {
            "tick": world.tick,
            "entity": entity_id,
            "action": action.name,
            "result": result_text,
        }
        resolutions.append(resolution)
        world.log.append(resolution)

    return resolutions


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Define actions
    def can_explore(e: Entity, w: WorldState) -> bool:
        return e.stats.get("hp", 0) > 0

    def do_explore(e: Entity, w: WorldState) -> str:
        gold = random.randint(1, 10)
        e.stats["gold"] = e.stats.get("gold", 0) + gold
        return f"{e.name} explored and found {gold} gold."

    def do_rest(e: Entity, w: WorldState) -> str:
        e.stats["hp"] = min(e.stats.get("hp", 0) + 5, 100)
        return f"{e.name} rested and recovered 5 HP."

    actions = [
        Action("rest", "Recover 5 HP", effect=do_rest),  # Index 0 = safe default
        Action("explore", "Search for treasure", precondition=can_explore, effect=do_explore),
    ]

    # Build world
    world = WorldState()
    world.add_entity(Entity("e1", "Alice", stats={"hp": 80, "gold": 0}))
    world.add_entity(Entity("e2", "Bob", stats={"hp": 30, "gold": 5}))

    # Fake LLM that always picks action 1 (explore)
    def fake_llm(prompt: str) -> str:
        return '{"action": 1, "reason": "adventure awaits"}'

    # Run a tick
    results = simulate_tick(world, actions, fake_llm)
    for r in results:
        print(f"  [{r['entity']}] {r['action']}: {r['result']}")
    print(f"  World tick: {world.tick}")
    print(f"  Alice gold: {world.entities['e1'].stats['gold']}")
    print(f"  Bob gold: {world.entities['e2'].stats['gold']}")
